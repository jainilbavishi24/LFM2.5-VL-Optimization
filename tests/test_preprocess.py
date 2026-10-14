"""GPU image preprocessing (lfm2opt.preprocess) vs the HF CPU processor: same tokens, same pixel_values.

    python tests/test_preprocess.py [--model LiquidAI/LFM2.5-VL-3B]   # exit code 1 on any mismatch beyond the tolerance
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.model import build_inputs, resolve_model_path  # noqa: E402
from lfm2opt.preprocess import GpuImageProcessor  # noqa: E402
from lfm2opt.workloads import SOURCE_IMAGE  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

# (width, height): single tile (down/upscaled), multi-tile + thumbnail, portrait, odd sizes, very large
SIZES = [(640, 480), (1280, 720), (1920, 1080), (200, 150), (64, 64), (512, 512), (720, 1280), (1000, 777),
         (333, 1500), (3000, 2000), (4032, 3024), (800, 600), (1024, 1024), (2048, 512)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-VL-1.6B", help="the 3B's processor resizes bicubic")
    args = ap.parse_args()
    path, rev = resolve_model_path(args.model)
    proc = AutoProcessor.from_pretrained(path, revision=rev)
    gproc = GpuImageProcessor.wrap(proc)
    src = Image.open(SOURCE_IMAGE).convert("RGB")
    rng = np.random.default_rng(0)
    fails = 0
    for w, h in SIZES:
        for kind in ("photo", "noise"):
            img = src.resize((w, h), Image.Resampling.BICUBIC) if kind == "photo" else \
                Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            ref = build_inputs(proc, img, "Describe this image.")
            got = build_inputs(gproc, img, "Describe this image.")
            ok_ids = torch.equal(ref["input_ids"].cpu(), got["input_ids"].cpu())
            ok_shape = ref["pixel_values"].shape == got["pixel_values"].shape
            ok_meta = ok_shape and torch.equal(ref["pixel_attention_mask"].cpu(), got["pixel_attention_mask"].cpu()) \
                and torch.equal(ref["spatial_shapes"].cpu(), got["spatial_shapes"].cpu())
            if ok_shape:
                d = (ref["pixel_values"].float() - got["pixel_values"].float()).abs()
                exact = (d == 0).float().mean().item()
                maxd = d.max().item()
            else:
                exact, maxd = 0.0, float("inf")
            # bit-exact expected; allow one uint8 level (2/255 after normalize) on a tiny fraction as a fallback
            ok = ok_ids and ok_meta and (maxd == 0 or (maxd <= 2 / 255 + 1e-6 and exact > 0.999))
            fails += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {w:>4}x{h:<4} {kind:>5} tiles={tuple(got['pixel_values'].shape)} "
                  f"ids={ok_ids} meta={ok_meta} exact={exact * 100:.4f}% maxdiff={maxd:.5f}")
    print(f"{fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
