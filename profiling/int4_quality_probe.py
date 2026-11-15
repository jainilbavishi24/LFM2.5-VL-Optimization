"""Which 4-bit weight scheme keeps the model's answers? Simulated on the real model (fake quantization: every LM
projection and lm_head computes with its dequantized 4-bit weights), compared with the unmodified BF16 model, on the
same cases and metrics as profiling/int8_quality_probe.py.

Schemes  <method>_<format>_g<group>[c][+<kinds kept in INT8>]:
  method   rtn (round-to-nearest) | gptq (lfm2opt.gptq: error-compensated rounding with calibration Hessians from
           COCO test2017 answers, no overlap with any evaluation set)
  format   int4 (symmetric, one fp32 scale per group: what kernels/lfm2_decode.cu gemv_w4 runs)
           nvfp4 (E2M1 values, FP8 E4M3 scale per 16, fp32 per tensor: Blackwell's 4-bit format)
  c        scale clipping searched per group (squared error)
  +head,w2 those weight kinds stay INT8 (per-row scale), e.g. gptq_int4_g128c+head
  w8       INT8 per row (the shipped extension), as the "acceptable" reference

Stricter than the runtime: the teacher-forced pass also runs the prompt through the 4-bit weights (the decode-only
runtime keeps BF16 prefill weights), so this measures "4-bit everywhere in the language model".

Metrics: teacher-forced argmax agreement (64 tokens), median max|dlogit|, KL vs BF16, first token of the benchmark
question, and per weight kind the relative output error on the calibration inputs tr(dW H dW^T) / tr(W H W^T).

    python profiling/int4_quality_probe.py [--per-bench 20] [--schemes ...]
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "profiling"))
from int8_quality_probe import TEXT_PROMPTS, bench_items  # noqa: E402
from lfm2opt.env import collect_env, gpu_short_name  # noqa: E402
from lfm2opt.gptq import Int4, NvFp4, collect_hessians, decode_linears, gptq, output_error, rtn  # noqa: E402
from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.quant import quantize_int8  # noqa: E402
from lfm2opt.workloads import IMAGE_PROMPT, WORKLOADS  # noqa: E402

DEFAULT_SCHEMES = ("w8,rtn_int4_g128,rtn_int4_g128c,rtn_int4_g64c,rtn_int4_g32c,rtn_nvfp4_g16,"
                   "gptq_int4_g128c,gptq_int4_g64c,gptq_int4_g32c,gptq_nvfp4_g16,gptq_int4_g128c+head,gptq_int4_g64c+head")


def kind(name):
    if name == "lm_head":
        return "head"
    for suf, k in ((".q_proj", "qkv"), (".k_proj", "qkv"), (".v_proj", "qkv"), ("attn.out_proj", "o"), ("conv.in_proj", "in"),
                   ("conv.out_proj", "out"), (".w1", "w13"), (".w3", "w13"), (".w2", "w2")):
        if name.endswith(suf):
            return k
    raise KeyError(name)


def quantized_weights(lins, hs, scheme):
    """{name: dequantized weight (bf16)}, {kind: [relative output errors]}"""
    base, _, keep8 = scheme.partition("+")
    keep8 = set(filter(None, keep8.split(",")))
    out, err = {}, {}
    for name, m in lins.items():
        W = m.weight
        if base == "w8" or kind(name) in keep8:
            q, s = quantize_int8(W)
            Wq = q.float() * s[:, None]
        else:
            method, fmt, g = base.split("_")
            clip, group = g.endswith("c"), int(g[1:].rstrip("c"))
            qz = Int4(clip) if fmt == "int4" else NvFp4(clip)
            Wq, _ = gptq(W, hs[name][0], group, qz) if method == "gptq" else rtn(W, group, qz)
        err.setdefault(kind(name), []).append(output_error(W, Wq, hs[name][0]))
        out[name] = Wq.to(W.dtype)
    return out, err


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bench", type=int, default=20)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--calib-images", type=int, default=96)
    ap.add_argument("--schemes", default=DEFAULT_SCHEMES)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    net, processor = load()
    lins = decode_linears(net)

    t0 = time.time()
    hs = collect_hessians(net, processor, args.calib_images)
    rows = hs["lm_head"][1]
    print(f"[probe4] calibration: {len(lins)} linears, {rows} decode-position rows, {time.time() - t0:.0f} s", flush=True)

    cases = [(WORKLOADS[w].image(), IMAGE_PROMPT, None) for w in ("img_small", "img_hd", "img_fhd")]
    cases += [(None, p, None) for p in TEXT_PROMPTS]
    cases += [(it["image"], IMAGE_PROMPT, it["q"]) for it in bench_items(args.per_bench)]
    ref = []
    for img, tp, q in cases:                                      # BF16 reference
        inp = build_inputs(processor, img, tp)
        T = inp["input_ids"].shape[1]
        gen = net.generate(**inp, max_new_tokens=args.gen, min_new_tokens=args.gen, do_sample=False,
                           pad_token_id=processor.tokenizer.pad_token_id)[:, T:]
        seq = torch.cat([inp["input_ids"], gen[:, :-1]], 1)
        extra = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        lg = net(input_ids=seq, attention_mask=torch.ones_like(seq), logits_to_keep=args.gen, **extra).logits[0].float()
        r = dict(seq=seq, extra=extra, logits=lg.cpu(), argmax=lg.argmax(-1),
                 decode_agree=(lg.argmax(-1) == gen[0]).float().mean().item())
        if q is not None:
            r["qinp"] = build_inputs(processor, img, q)
            r["qtok"] = net(**r["qinp"], logits_to_keep=1).logits[0, -1].argmax().item()
        ref.append(r)
    floor = 100 * statistics.mean(r["decode_agree"] for r in ref)
    print(f"[probe4] {len(cases)} cases; BF16 noise floor (one-pass vs decode argmax) {floor:.2f} %", flush=True)

    results = []
    for scheme in args.schemes.split(","):
        t1 = time.time()
        wq, err = quantized_weights(lins, hs, scheme)
        for name, m in lins.items():
            m.forward = (lambda x, m=m, w=wq[name]: F.linear(x, w, m.bias))
        agree, dmax, kl, qok, qn = [], [], [], 0, 0
        try:
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
        finally:
            for m in lins.values():
                del m.forward
            del wq
            torch.cuda.empty_cache()
        res = dict(scheme=scheme, tf_agree=100 * statistics.mean(agree), tf_agree_min=100 * min(agree),
                   dlogit_median=statistics.median(dmax), kl_mean=statistics.mean(kl),
                   first_token_agree=100 * qok / qn if qn else None, n_first=qn,
                   out_err={k: 100 * statistics.mean(v) for k, v in err.items()}, seconds=time.time() - t1)
        results.append(res)
        print(f"[probe4] {scheme:<24} teacher-forced agree {res['tf_agree']:6.2f} % (min {res['tf_agree_min']:5.1f})  "
              f"median max|dlogit| {res['dlogit_median']:.3f}  KL {res['kl_mean']:.4f}  first token "
              f"{res['first_token_agree']:5.1f} % of {qn}  out err % " +
              " ".join(f"{k}={v:.2f}" for k, v in res["out_err"].items()), flush=True)

    out = Path(args.out) if args.out else ROOT / "profiling" / "reports" / \
        f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}_int4_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(env=collect_env(), args=vars(args), cases=len(cases), calib_rows=rows,
                                   bf16_noise_floor=floor, results=results), indent=2))
    print(f"[probe4] -> {out}")


if __name__ == "__main__":
    main()
