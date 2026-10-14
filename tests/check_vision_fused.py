"""Fused SigLIP2 encoder (lfm2opt.vision) vs HF: image features at the valid patches, against an FP32 reference.

The fused path should be as close to FP32 as HF's own BF16 path is (it reorders BF16 rounding, it does not drop
precision). Also checks greedy generation (64 tokens) against HF on the benchmark workloads.

    python tests/check_vision_fused.py [--opts attn=cudnn_mask,...]
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.vision import FusedSiglip2Encoder  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402


def feats(out, mask):
    h = out.last_hidden_state.float()
    return torch.cat([h[t, : int(mask[t].sum())] for t in range(h.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opts", default="")
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()
    opts = {}
    for kv in filter(None, args.opts.split(",")):
        k, v = kv.split("=")
        opts[k] = v if k == "attn" else v not in ("0", "false")

    net, proc = load()
    vt = net.model.vision_tower
    vt32 = copy.deepcopy(vt).float()
    enc = FusedSiglip2Encoder(vt, **opts)
    ok = True
    for name in ("img_small", "img_hd", "img_fhd"):
        wl = WORKLOADS[name]
        inp = build_inputs(proc, wl.image(), wl.prompt)
        kw = dict(pixel_values=inp["pixel_values"], pixel_attention_mask=inp["pixel_attention_mask"],
                  spatial_shapes=inp["spatial_shapes"])
        m = inp["pixel_attention_mask"]
        with torch.no_grad():
            ref = feats(vt32(**{**kw, "pixel_values": kw["pixel_values"].float()}), m)
            hf = feats(vt(**kw), m)
            ours = feats(enc(**kw), m)
        for tag, f in (("HF bf16", hf), ("fused", ours)):
            cos = F.cosine_similarity(f, ref, dim=-1)
            rel = (f - ref).norm() / ref.norm()
            print(f"{name:>9} {tag:>8} vs fp32: rel err {rel:.4e}  cos min {cos.min():.6f} mean {cos.mean():.6f}")
        d = (ours - hf).norm() / hf.norm()
        print(f"{name:>9}   fused vs HF bf16: rel diff {d:.4e}")

        # end to end: greedy generation with HF's vision tower vs ours (same HF LM)
        with torch.inference_mode():
            a = net.generate(**inp, max_new_tokens=args.gen, do_sample=False)
            orig = vt.forward
            vt.forward = enc
            b = net.generate(**inp, max_new_tokens=args.gen, do_sample=False)
            vt.forward = orig
        T = inp["input_ids"].shape[1]
        a, b = a[0, T:].tolist(), b[0, T:].tolist()
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"{name:>9}   generation: identical for {n}/{min(len(a), len(b))} tokens")
        ok &= bool(rel < 2 * ((hf - ref).norm() / ref.norm()) + 1e-3)
    print("OK" if ok else "FAIL: fused error > 2x HF bf16 error")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
