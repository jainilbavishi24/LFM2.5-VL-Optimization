"""TTFT components, HF vs ours, with CUDA events (median of N): vision encoder and LM prefill timed separately.

  vision  : HF SigLIP2 tower (SDPA default / cuDNN attention) vs FusedSiglip2Encoder (lfm2opt.vision, O7d)
  prefill : HF language model on the merged embeddings (no cache) vs StaticDecoder._prefill_fused (O3b)

    python profiling/ttft_probe.py [--workloads text,img_small,img_hd,img_fhd] [--repeats 20]
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.runtime import StaticDecoder  # noqa: E402
from lfm2opt.vision import FusedSiglip2Encoder  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402


def cuda_ms(fn, repeats):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="text,img_small,img_hd,img_fhd")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--int8", action="store_true", help="ours = the INT8 extension (vision + prefill GEMMs on INT8)")
    args = ap.parse_args()

    net, proc = load()
    vt = net.model.vision_tower
    enc = FusedSiglip2Encoder(vt, int8=args.int8)
    dec = StaticDecoder(net, fused=True, gemv="int8" if args.int8 else "bf16", fused_prefill=True, prefill_int8=args.int8)
    lm = net.model.language_model
    cudnn = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    with torch.inference_mode():
        for name in args.workloads.split(","):
            wl = WORKLOADS[name]
            inp = build_inputs(proc, wl.image(), wl.prompt)
            ids = inp["input_ids"]
            mm = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
            r = {}
            if "pixel_values" in inp:
                kw = dict(pixel_values=inp["pixel_values"], pixel_attention_mask=inp["pixel_attention_mask"],
                          spatial_shapes=inp["spatial_shapes"])
                r["vision_hf"] = cuda_ms(lambda: vt(**kw), args.repeats)

                def hf_cudnn():
                    with sdpa_kernel(cudnn, set_priority=True):
                        vt(**kw)
                r["vision_hf_cudnn"] = cuda_ms(hf_cudnn, args.repeats)
                r["vision_fused"] = cuda_ms(lambda: enc(**kw), args.repeats)
            emb = dec._embed(ids, **mm)
            T = ids.shape[1]
            r["prefill_hf"] = cuda_ms(lambda: lm(inputs_embeds=emb, use_cache=True), args.repeats)
            s = dec._get(1, T + 2, 1)
            r["prefill_fused"] = cuda_ms(lambda: _prefill_from_emb(dec, emb, s), args.repeats)
            print(f"{name:>9} T={T:<5} " + "  ".join(f"{k}={v:.2f}ms" for k, v in r.items()), flush=True)


def _prefill_from_emb(dec, emb, s):
    """_prefill_fused without the vision tower (embeddings given), to time the LM part alone."""
    orig = dec._embed
    dec._embed = lambda *a, **k: emb
    try:
        ids = torch.zeros(emb.shape[:2], dtype=torch.long, device=emb.device)
        return dec._prefill_fused(ids, s)
    finally:
        dec._embed = orig


if __name__ == "__main__":
    main()
