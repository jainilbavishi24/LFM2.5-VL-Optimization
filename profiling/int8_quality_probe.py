"""Which INT8 scheme keeps the model's answers for the first token (prefill + vision)? Simulated on the real model
before writing kernels: every targeted nn.Linear is replaced by a fake-quantized one that computes exactly what an
INT8 kernel would (integer products are exact in FP32), then compared with the unmodified BF16 model.

Schemes (weights: symmetric INT8, one scale per output row, round-to-nearest, as lfm2opt.quant.quantize_int8):
  w8a16    INT8 weights, BF16 activations (dequantized weights, BF16 GEMM)
  w8a8     + activations INT8 with one dynamic scale per token (INT8 tensor-core GEMM)
  w8a8_sq  + SmoothQuant (alpha 0.5): per input channel x / s, W * s, s from calibration prompts that are not in the
           test set (variants.CALIB_PROMPTS); layers that share an input (q|k|v, w1|w3) share s, so it can be applied
           inside the kernel that quantizes that input
Scopes: vision (SigLIP2 encoder layers), lm (LFM2 layers; lm_head always weight-only, as in decode), all, or one
layer kind (vqkv, vout, fc1, fc2, qkv, o, in, out, w13, w2) to find the sensitive ones.

Metrics, per scheme x scope, over fixed workloads + an evenly spaced sample of benchmark images (data/eval):
  * teacher-forced: the BF16 model's greedy 64-token answer is fed through the quantized model in one forward pass
    (= prefill numerics at every position): argmax agreement with the BF16 forward, median max|dlogit|
  * first token of the benchmark question (what the 5 benchmarks mostly measure): argmax agreement
  * vision features vs an FP32 copy of the tower: relative error (BF16 itself is ~4 %)

    python profiling/int8_quality_probe.py [--per-bench 6] [--gen 64] [--configs ...]
"""

import argparse
import copy
import io
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from lfm2opt.env import gpu_short_name  # noqa: E402
from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.quant import quantize_int8  # noqa: E402
from lfm2opt.variants import CALIB_PROMPTS  # noqa: E402
from lfm2opt.workloads import IMAGE_PROMPT, WORKLOADS  # noqa: E402

KINDS = {  # module-name suffix -> (kind, input group for SmoothQuant)
    "self_attn.q_proj": ("qkv", "attn_in"), "self_attn.k_proj": ("qkv", "attn_in"), "self_attn.v_proj": ("qkv", "attn_in"),
    "self_attn.out_proj": ("o", "o_in"), "conv.in_proj": ("in", "conv_in"), "conv.out_proj": ("out", "out_in"),
    "feed_forward.w1": ("w13", "ffn_in"), "feed_forward.w3": ("w13", "ffn_in"), "feed_forward.w2": ("w2", "w2_in"),
}
VKINDS = {"self_attn.q_proj": "vqkv", "self_attn.k_proj": "vqkv", "self_attn.v_proj": "vqkv", "self_attn.out_proj": "vout",
          "mlp.fc1": "fc1", "mlp.fc2": "fc2"}
TEXT_PROMPTS = ["Explain in detail how a bicycle works.", "What are the main causes of climate change?",
                "Summarize the plot of Romeo and Juliet."]
DEFAULT_CONFIGS = ("w8a16:vision,w8a16:lm,w8a16:all,w8a8:vision,w8a8:lm,w8a8:all,"
                   "w8a8:vqkv,w8a8:vout,w8a8:fc1,w8a8:fc2,w8a8:qkv,w8a8:o,w8a8:in,w8a8:out,w8a8:w13,w8a8:w2,"
                   "w8a8_sq:vision,w8a8_sq:lm,w8a8_sq:all")


def targets(net):
    """name -> (module, scope, kind, smooth group) for every linear an INT8 prefill / vision path would quantize."""
    out = {}
    for name, m in net.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if name == "lm_head":
            out[name] = (m, "lm", "head", None)
        elif name.startswith("model.vision_tower.") and ".encoder.layers." in name:
            suf = next((s for s in VKINDS if name.endswith(s)), None)
            if suf:
                prefix = name[: -len(suf)]
                grp = prefix + ("vattn_in" if VKINDS[suf] == "vqkv" else suf)
                out[name] = (m, "vision", VKINDS[suf], grp)
        elif name.startswith("model.language_model.layers."):
            suf = next((s for s in KINDS if name.endswith(s)), None)
            if suf:
                kind, g = KINDS[suf]
                out[name] = (m, "lm", kind, name[: -len(suf)] + g)
    return out


@torch.no_grad()
def calibrate(net, processor, tg):
    """Per input channel max |x| of every target linear over the calibration prompts (prefill only)."""
    amax, hooks = {}, []
    for name, (m, *_rest) in tg.items():
        def hook(mod, args, name=name):
            a = args[0].detach().float().abs().reshape(-1, args[0].shape[-1]).amax(0)
            amax[name] = torch.maximum(amax[name], a) if name in amax else a
        hooks.append(m.register_forward_pre_hook(hook))
    img = WORKLOADS["img_small"].image()
    try:
        for i, p in enumerate(CALIB_PROMPTS):
            net(**build_inputs(processor, img if i % 2 == 0 else None, p))
    finally:
        for h in hooks:
            h.remove()
    return amax


def smooth_scales(tg, amax, alpha=0.5):
    """SmoothQuant s_j = max|x_j|^a / max|w_j|^(1-a), shared by the linears that read the same input."""
    groups = {}
    for name, (m, scope, kind, grp) in tg.items():
        if grp is not None:
            groups.setdefault(grp, []).append(name)
    s = {}
    for names in groups.values():
        xa = torch.stack([amax[n] for n in names]).amax(0)
        wa = torch.cat([tg[n][0].weight.float().abs() for n in names]).amax(0)
        sc = (xa.clamp_min(1e-5).pow(alpha) / wa.clamp_min(1e-5).pow(1 - alpha)).clamp(1e-2, 1e2)
        for n in names:
            s[n] = sc
    return s


class Patch:
    """Context manager: fake-quantize the selected linears."""

    def __init__(self, tg, scheme, scope, smooth=None):
        self.sel = {n: v for n, v in tg.items()
                    if (scope == "all" or v[1] == scope or v[2] == scope or (scope == "lm" and v[2] == "head"))}
        if scope not in ("all", "lm"):  # a single kind or the vision scope: lm_head stays BF16
            self.sel = {n: v for n, v in self.sel.items() if v[2] != "head"}
        self.scheme, self.smooth = scheme, smooth

    def __enter__(self):
        for name, (m, scope, kind, grp) in self.sel.items():
            a8 = self.scheme.startswith("w8a8") and kind != "head"
            s = self.smooth.get(name) if (self.smooth is not None and a8) else None
            w = m.weight.float() * (s[None, :] if s is not None else 1)
            q, sc = quantize_int8(w)
            if not a8:  # W8A16: dequantized BF16 weights, BF16 GEMM
                wd = (q.float() * sc[:, None]).to(m.weight.dtype)
                m.forward = (lambda x, m=m, wd=wd: F.linear(x, wd, m.bias))
                continue
            wd = q.float() * sc[:, None]
            b = m.bias.float() if m.bias is not None else None
            inv = (1.0 / s) if s is not None else None

            def fwd(x, wd=wd, b=b, inv=inv):
                xf = x.float() if inv is None else x.float() * inv
                sx = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
                xf = torch.round(xf / sx).clamp(-127, 127) * sx
                return F.linear(xf, wd, b).to(x.dtype)
            m.forward = fwd
        return self

    def __exit__(self, *exc):
        for m, *_ in self.sel.values():
            del m.forward
        torch.cuda.empty_cache()


def bench_items(per_bench):
    from eval_quality import DATA, load_bench, subsample
    items = []
    for b in ("mmstar", "pope", "ocrbench", "chartqa", "textvqa"):
        try:
            items += [dict(bench=b, image=Image.open(io.BytesIO(it["image"])).convert("RGB"), q=it["prompt"])
                      for it in subsample(load_bench(b), per_bench)]
        except Exception as e:  # data not downloaded on this machine
            print(f"[probe] skip {b}: {e!r} ({DATA})")
    return items


def vfeats(vt, inp):
    out = vt(pixel_values=inp["pixel_values"].to(next(vt.parameters()).dtype),
             pixel_attention_mask=inp["pixel_attention_mask"], spatial_shapes=inp["spatial_shapes"])
    h, m = out.last_hidden_state.float(), inp["pixel_attention_mask"]
    return torch.cat([h[t, : int(m[t].sum())] for t in range(h.shape[0])])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bench", type=int, default=6)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS, help="scheme:scope,...")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net, processor = load()
    vt = net.model.vision_tower
    vt32 = copy.deepcopy(vt).float()
    tg = targets(net)
    print(f"[probe] {len(tg)} linears: " + ", ".join(f"{k}={sum(v[2] == k for v in tg.values())}"
                                                       for k in dict.fromkeys(v[2] for v in tg.values())))

    # cases: (image or None, teacher-forced prompt, first-token question or None)
    cases = [(WORKLOADS[w].image(), IMAGE_PROMPT, None) for w in ("img_small", "img_hd", "img_fhd")]
    cases += [(None, p, None) for p in TEXT_PROMPTS]
    cases += [(it["image"], IMAGE_PROMPT, it["q"]) for it in bench_items(args.per_bench)]
    print(f"[probe] {len(cases)} cases")

    t0 = time.time()
    ref = []
    for img, tp, q in cases:  # BF16 reference: greedy answer, its one-pass logits, first token, vision features
        inp = build_inputs(processor, img, tp)
        T = inp["input_ids"].shape[1]
        gen = net.generate(**inp, max_new_tokens=args.gen, min_new_tokens=args.gen, do_sample=False,
                           pad_token_id=processor.tokenizer.pad_token_id)[:, T:]
        seq = torch.cat([inp["input_ids"], gen[:, :-1]], 1)
        extra = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        lg = net(input_ids=seq, attention_mask=torch.ones_like(seq), logits_to_keep=args.gen, **extra).logits[0].float()
        r = dict(inp=inp, seq=seq, extra=extra, gen=gen[0], logits=lg.cpu(), argmax=lg.argmax(-1),
                 decode_agree=(lg.argmax(-1) == gen[0]).float().mean().item())
        if q is not None:
            r["qinp"] = build_inputs(processor, img, q)
            r["qtok"] = net(**r["qinp"], logits_to_keep=1).logits[0, -1].argmax().item()
        if img is not None:
            r["f32"] = vfeats(vt32, inp)
            f = vfeats(vt, inp)
            r["bf16_vrel"] = ((f - r["f32"]).norm() / r["f32"].norm()).item()
        ref.append(r)
    del vt32
    print(f"[probe] reference done in {time.time() - t0:.0f} s; BF16 one-pass vs decode argmax agreement "
          f"{100 * statistics.mean(r['decode_agree'] for r in ref):.2f} %; BF16 vision rel err vs FP32 "
          f"{100 * statistics.mean(r['bf16_vrel'] for r in ref if 'bf16_vrel' in r):.2f} %")

    smooth = None
    results = []
    for cfg in args.configs.split(","):
        scheme, scope = cfg.split(":")
        if scheme == "w8a8_sq" and smooth is None:
            smooth = smooth_scales(tg, calibrate(net, processor, tg))
        agree, dmax, kl, qok, qn, vrel = [], [], [], 0, 0, []
        with Patch(tg, scheme, scope, smooth if scheme == "w8a8_sq" else None):
            for r in ref:
                lg = net(input_ids=r["seq"], attention_mask=torch.ones_like(r["seq"]), logits_to_keep=args.gen,
                         **r["extra"]).logits[0].float()
                rl = r["logits"].to(lg.device)
                agree.append((lg.argmax(-1) == r["argmax"]).float().mean().item())
                dmax.append((lg - rl).abs().amax(-1).median().item())
                kl.append(F.kl_div(lg.log_softmax(-1), rl.log_softmax(-1), log_target=True, reduction="none")
                          .sum(-1).mean().item())
                if "qinp" in r:
                    qn += 1
                    qok += int(net(**r["qinp"], logits_to_keep=1).logits[0, -1].argmax().item() == r["qtok"])
                if "f32" in r and scope not in ("lm", "qkv", "o", "in", "out", "w13", "w2"):
                    f = vfeats(vt, r["inp"])
                    vrel.append(((f - r["f32"]).norm() / r["f32"].norm()).item())
        res = dict(scheme=scheme, scope=scope, tf_agree=100 * statistics.mean(agree), tf_agree_min=100 * min(agree),
                   dlogit_median=statistics.median(dmax), kl_mean=statistics.mean(kl),
                   first_token_agree=100 * qok / qn if qn else None, n_first=qn,
                   vision_rel_err=100 * statistics.mean(vrel) if vrel else None)
        results.append(res)
        print(f"[probe] {scheme:>8}:{scope:<7} teacher-forced agree {res['tf_agree']:6.2f} % (min {res['tf_agree_min']:5.1f})"
              f"  median max|dlogit| {res['dlogit_median']:.3f}  KL {res['kl_mean']:.4f}"
              + (f"  first token {res['first_token_agree']:5.1f} % of {qn}" if qn else "")
              + (f"  vision rel err {res['vision_rel_err']:.2f} %" if vrel else ""), flush=True)

    out = Path(args.out) if args.out else ROOT / "profiling" / "reports" / \
        f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}_int8_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(cases=len(cases), gen=args.gen, decode_agree_bf16=100 * statistics.mean(r["decode_agree"] for r in ref),
                   vision_rel_err_bf16=100 * statistics.mean(r["bf16_vrel"] for r in ref if "bf16_vrel" in r),
                   results=results)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[probe] -> {out}")


if __name__ == "__main__":
    main()
