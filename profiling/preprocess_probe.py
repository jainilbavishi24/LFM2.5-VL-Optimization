"""Where does the CPU preprocessing time go? (chat template, tokenizer, image processor stages, H2D copy)

Times the HF processor on the benchmark workloads, with the image processor on the CPU (default) and with
HF's own `device="cuda"` option (same torchvision ops, run on the GPU), and optionally our fused CUDA path.

    python profiling/preprocess_probe.py [--workloads img_small,img_hd,img_fhd] [--repeats 20] [--ours]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.model import build_inputs, resolve_model_path  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


def timeit(fn, repeats):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="img_small,img_hd,img_fhd")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--ours", action="store_true", help="also time lfm2opt.preprocess (fused CUDA kernels)")
    args = ap.parse_args()

    path, rev = resolve_model_path("LiquidAI/LFM2.5-VL-1.6B")
    proc = AutoProcessor.from_pretrained(path, revision=rev)
    ip = proc.image_processor
    print(f"image processor: {type(ip).__name__}, torch threads {torch.get_num_threads()}")
    gpu_proc = None
    if args.ours:
        from lfm2opt.preprocess import GpuImageProcessor
        gpu_proc = GpuImageProcessor.wrap(proc)

    for name in args.workloads.split(","):
        wl = WORKLOADS[name]
        img = wl.image()
        r = {}
        r["full_cpu"] = timeit(lambda: build_inputs(proc, img, wl.prompt), args.repeats)
        r["imgproc_cpu"] = timeit(lambda: ip(images=[[img]], return_tensors="pt"), args.repeats)
        r["imgproc_cpu_to_gpu"] = timeit(lambda: ip(images=[[img]], return_tensors="pt")["pixel_values"].cuda(),
                                         args.repeats)
        r["imgproc_hf_cuda"] = timeit(lambda: ip(images=[[img]], return_tensors="pt", device="cuda"), args.repeats)
        r["text_only"] = timeit(lambda: build_inputs(proc, None, wl.prompt), args.repeats)
        if gpu_proc is not None:
            r["full_ours"] = timeit(lambda: build_inputs(gpu_proc, img, wl.prompt), args.repeats)
            r["imgproc_ours"] = timeit(lambda: gpu_proc.image_processor(images=[[img]], return_tensors="pt"),
                                       args.repeats)
        out = ip(images=[[img]], return_tensors="pt")
        print(f"{name:>9} {img.size} tiles={out['pixel_values'].shape[0]} "
              + "  ".join(f"{k}={v:.2f}ms" for k, v in r.items()))


if __name__ == "__main__":
    main()
