"""GPTQ (Frantar et al., ICLR 2023) for the 4-bit decode weights: no training, a few minutes on one GPU.

Round-to-nearest (RTN) rounds every weight on its own. GPTQ rounds the input columns of a weight matrix one at a
time and, after each, spreads that column's rounding error over the columns not yet rounded, weighted by the inverse
of the layer's input Hessian H = sum x x^T. The result minimizes the layer's *output* error on the calibration inputs
instead of the weight error. The storage format is unchanged (q in [-8, 7] times one scale per group of `group` inputs,
lfm2opt.quant), so kernels/lfm2_decode.cu gemv_w4 runs GPTQ weights as they are.

Calibration (no overlap with any evaluation set): COCO *test2017* images (no labels; none of our benchmarks uses
them) with generic instructions, plus text-only prompts. The BF16 model answers each greedily, then one forward pass
over prompt + answer records the inputs of every decode weight at the answer positions only: the decode GEMVs never
see prompt or image-token activations (prefill keeps its own weights), so those rows would only bias H.

    from lfm2opt.gptq import gptq_decode_weights
    pre = gptq_decode_weights(net, processor, group=128)     # {StaticDecoder key: (packed, scale)}, cached on disk
"""

import io
import os
from pathlib import Path

import torch

from .env import REPO_ROOT

CALIB_PARQUET = REPO_ROOT / "data" / "calib" / "coco_test0_256.parquet"   # 256 test2017 images, scripts/download_eval_data.sh
IMAGE_PROMPTS = [
    "Give a detailed description of this picture.", "What is happening in this picture?",
    "List the objects you can see and where they are.", "Write a short caption for this image.",
    "Which colors dominate this image?", "Is there any text in this image? If so, write it down.",
    "How many people are in this image, and what are they doing?", "What might have happened just before this photo?",
]
TEXT_PROMPTS = [
    "Write a short story about a lighthouse keeper.", "How does a refrigerator keep food cold?",
    "List five tips for learning a new language.", "Explain the difference between weather and climate.",
    "Write a Python function that checks whether a number is prime.", "What is the Pythagorean theorem? Give an example.",
    "Describe the water cycle step by step.", "Give me a recipe for a simple vegetable soup.",
    "Why is the sky blue?", "Summarize the history of the printing press.",
    "Compare trains and airplanes for travelling 500 km.", "Write a polite email asking for a deadline extension.",
    "What are black holes?", "Explain how vaccines work.", "Translate 'good morning, how are you?' into French and Spanish.",
    "Plan a one-day visit to a museum for a family with two children.",
]


# ------------------------------------------------------------------------------------------- 4-bit quantizers
class Int4:
    """Symmetric INT4, q in [-8, 7], one fp32 scale per group; clip: scale = absmax * a / 7 with a in 0.70..1.00 picked
    per group by squared error (as lfm2opt.quant.quantize_int4(clip_search=True))."""

    name = "int4"

    def __init__(self, clip=True):
        self.clip = clip

    def prepare(self, W):
        pass

    def scale(self, wg):                                                  # wg [N, G] -> [N]
        amax = wg.abs().amax(-1).clamp_min(1e-12)
        if not self.clip:
            return amax / 7
        best, scale = None, None
        for a in torch.linspace(0.70, 1.00, 16).tolist():
            s = amax * a / 7
            err = (self.q(wg, s[:, None]) - wg).pow(2).sum(-1)
            if best is None:
                best, scale = err, s
            else:
                better = err < best
                best, scale = torch.where(better, err, best), torch.where(better, s, scale)
        return scale

    @staticmethod
    def q(w, s):
        return torch.round(w / s).clamp(-8, 7) * s


class NvFp4:
    """NVFP4 (Blackwell's 4-bit format): E2M1 values {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign, one FP8 E4M3 scale per
    block of 16, and one fp32 scale per tensor that keeps the block scales inside E4M3's range."""

    name = "nvfp4"
    GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def __init__(self, clip=False):
        self.clip = clip

    def prepare(self, W):
        self.g = (W.abs().amax() / (448.0 * 6.0)).clamp_min(1e-12)
        self.grid = self.GRID.to(W.device)
        self.mid = (self.grid[1:] + self.grid[:-1]) / 2

    def _fp8(self, s):
        return (s / self.g).clamp(2**-9, 448.0).to(torch.float8_e4m3fn).float() * self.g

    def scale(self, wg):
        amax = wg.abs().amax(-1).clamp_min(1e-12)
        if not self.clip:
            return self._fp8(amax / 6)
        best, scale = None, None
        for a in torch.linspace(0.70, 1.00, 16).tolist():
            s = self._fp8(amax * a / 6)
            err = (self.q(wg, s[:, None]) - wg).pow(2).sum(-1)
            if best is None:
                best, scale = err, s
            else:
                better = err < best
                best, scale = torch.where(better, err, best), torch.where(better, s, scale)
        return scale

    def q(self, w, s):
        v = w / s
        idx = torch.bucketize(v.abs(), self.mid)
        return torch.sign(v) * self.grid[idx] * s


# ------------------------------------------------------------------------------------------- RTN and GPTQ
@torch.no_grad()
def rtn(W, group, quantizer):
    """Round-to-nearest. -> (dequantized W [N, K] fp32, scales [N, K / group])"""
    W = W.float()
    N, K = W.shape
    quantizer.prepare(W)
    wg = W.view(N, K // group, group)
    s = torch.stack([quantizer.scale(wg[:, j]) for j in range(K // group)], 1)
    return quantizer.q(wg, s[..., None]).view(N, K), s


@torch.no_grad()
def gptq(W, H, group, quantizer, blocksize=128, damp=0.01):
    """GPTQ with group-wise scales taken from the (already error-updated) weights of each group when the group starts.
    H: [K, K] input Hessian (fp32). -> (dequantized W [N, K] fp32, scales [N, K / group])"""
    W = W.float().clone()
    N, K = W.shape
    assert K % group == 0 and blocksize % group == 0
    quantizer.prepare(W)
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(K, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    scales = torch.zeros(N, K // group, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1, Hb = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E1 = torch.zeros_like(W1)
        s = None
        for i in range(i2 - i1):
            if (i1 + i) % group == 0:
                s = quantizer.scale(W1[:, i:i + group])
                scales[:, (i1 + i) // group] = s
            w = W1[:, i]
            q = quantizer.q(w, s)
            Q[:, i1 + i] = q
            err = (w - q) / Hb[i, i]
            W1[:, i:] -= err[:, None] * Hb[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, scales


def pack_int4(Wq, scales):
    """Dequantized INT4 weights + their scales -> (packed uint8 [N, K/2], scales) in lfm2opt.quant's layout."""
    N, K = Wq.shape
    group = K // scales.shape[1]
    q = torch.round(Wq.view(N, K // group, group) / scales[..., None]).clamp(-8, 7).to(torch.int16).view(N, K) + 8
    return (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8).contiguous(), scales.contiguous()


# ------------------------------------------------------------------------------------------- calibration
def decode_linears(net):
    """HF module name -> module, for every weight the decode GEMVs read (LM projections and lm_head)."""
    out = {}
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Linear) and (name == "lm_head" or name.startswith("model.language_model.layers.")):
            out[name] = m
    return out


def calib_samples(n_images=96):
    """[(PIL image or None, prompt)]: n_images COCO test2017 images (evenly spaced in the shard) x rotating
    instructions, then the text-only prompts."""
    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(CALIB_PARQUET, columns=["image"])
    assert n_images <= t.num_rows, f"{CALIB_PARQUET} has {t.num_rows} images"
    idx = torch.linspace(0, t.num_rows - 1, n_images).round().long().tolist()
    out = []
    for j, i in enumerate(idx):
        v = t.slice(i, 1).column("image")[0].as_py()
        raw = v["bytes"] if isinstance(v, dict) else v
        out.append((Image.open(io.BytesIO(raw)).convert("RGB"), IMAGE_PROMPTS[j % len(IMAGE_PROMPTS)]))
    return out + [(None, p) for p in TEXT_PROMPTS]


@torch.no_grad()
def collect_hessians(net, processor, n_images=96, gen=128):
    """{HF linear name: (H = sum x x^T over decode-position inputs [K, K] fp32, rows)}. Linears that read the same
    input (q/k/v, w1/w3) share one H."""
    from .model import build_inputs

    lins = decode_linears(net)
    owner = {}                                                          # input-sharing modules -> one recorder
    for name in lins:
        for a, b in ((".k_proj", ".q_proj"), (".v_proj", ".q_proj"), (".w3", ".w1")):
            if name.endswith(a):
                owner[name] = name[: -len(a)] + b
    H, rows, state = {}, {}, {"start": None}

    def hook(name):
        def f(mod, args):
            if state["start"] is None:
                return
            x = args[0].detach().reshape(-1, args[0].shape[-1])[state["start"]:].float()
            H[name] = H.get(name, 0) + x.t() @ x
            rows[name] = rows.get(name, 0) + x.shape[0]
        return f

    hooks = [m.register_forward_pre_hook(hook(n)) for n, m in lins.items() if n not in owner]
    try:
        for img, prompt in calib_samples(n_images):
            inp = build_inputs(processor, img, prompt)
            T = inp["input_ids"].shape[1]
            out = net.generate(**inp, max_new_tokens=gen, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
            extra = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
            state["start"] = T - 1                                      # inputs that produced answer tokens 1..n
            net(input_ids=out[:, :-1], attention_mask=torch.ones_like(out[:, :-1]), **extra)
            state["start"] = None
    finally:
        for h in hooks:
            h.remove()
    return {n: (H[owner.get(n, n)], rows[owner.get(n, n)]) for n in lins}


# ------------------------------------------------------------------------------------------- runtime weights
def _key_modules(net):
    """StaticDecoder.lin key -> HF linears whose rows it concatenates (same order as runtime._prepare_fused)."""
    lm = net.model.language_model
    cfg = lm.config
    p = "model.language_model.layers."
    keys = {}
    for i in range(cfg.num_hidden_layers):
        if cfg.layer_types[i] == "full_attention":
            keys[("qkv", i)] = [f"{p}{i}.self_attn.{s}_proj" for s in "qkv"]
            keys[("o", i)] = [f"{p}{i}.self_attn.out_proj"]
        else:
            keys[("in", i)] = [f"{p}{i}.conv.in_proj"]
            keys[("out", i)] = [f"{p}{i}.conv.out_proj"]
        keys[("w13", i)] = [f"{p}{i}.feed_forward.w1", f"{p}{i}.feed_forward.w3"]
        keys[("w2", i)] = [f"{p}{i}.feed_forward.w2"]
    keys[("head", 0)] = ["lm_head"]
    return keys


def gptq_decode_weights(net, processor, group=128, n_images=96, fmt="int4"):
    """GPTQ INT4 for every decode weight, keyed like StaticDecoder.lin: {key: (packed uint8, fp32 scales)}.
    Cached in models/gptq/ (git-ignored; LFM2_GPTQ_CACHE overrides the directory, e.g. node-local scratch):
    calibration + GPTQ take a few minutes once per configuration."""
    assert fmt == "int4", "the runtime GEMV reads INT4 (NVFP4 is evaluated in the quality probe only)"
    name = getattr(net.config, "_name_or_path", "model").rstrip("/").split("/")[-1]
    cache = Path(os.environ.get("LFM2_GPTQ_CACHE", REPO_ROOT / "models" / "gptq")) / f"{name}_int4_g{group}_n{n_images}.pt"
    if cache.exists():
        return torch.load(cache, map_location="cuda")
    mods = dict(net.named_modules())
    hs = collect_hessians(net, processor, n_images)
    out = {}
    for key, names in _key_modules(net).items():
        W = torch.cat([mods[n].weight for n in names], 0)
        Wq, s = gptq(W, hs[names[0]][0], group, Int4(clip=True))
        out[key] = pack_int4(Wq, s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def output_error(W, Wq, H):
    """Relative output error on the calibration inputs: tr(dW H dW^T) / tr(W H W^T)."""
    d = (W.float() - Wq)
    return (torch.einsum("nk,kj,nj->", d, H, d) / torch.einsum("nk,kj,nj->", W.float(), H, W.float())).item()
