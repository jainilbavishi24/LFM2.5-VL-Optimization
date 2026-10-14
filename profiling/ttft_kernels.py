"""Kernel-time breakdown of one 1080p first token with the full BF16 stack (PyTorch profiler, PDL off):
what is left besides the GEMMs (attention, our fused kernels, PyTorch glue)?

    LFM2_PDL=0 python profiling/ttft_kernels.py [--workload img_fhd]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.preprocess import GpuImageProcessor  # noqa: E402
from lfm2opt.variants import apply_variant  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402

CATS = [  # (category, substrings of the kernel name), first match wins
    ("attention (flash/cuDNN)", ("flash", "fmha", "attention", "cudnn")),  # before GEMM: flash names contain "cutlass"
    ("GEMM (cuBLAS/CUTLASS)", ("gemm", "cutlass", "cublas", "sm90_", "sm100_", "nvjet")),
    ("ours: fused kernels", ("add_layernorm", "add_rmsnorm", "silu_mul", "shortconv", "qk_norm", "resize_")),
    ("PyTorch elementwise/copy", ("elementwise", "copy", "cat", "index", "fill", "unrolled")),
    ("PyTorch reduce/other", ("reduce", "upsample", "softmax", "layer_norm", "argmax")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="img_fhd")
    args = ap.parse_args()
    net, proc = load()
    net = apply_variant("fused_gemv_pf_vfused", net, proc)
    proc = GpuImageProcessor.wrap(proc)
    wl = WORKLOADS[args.workload]
    img = wl.image()
    gen = dict(max_new_tokens=1, min_new_tokens=1, do_sample=False)
    with torch.inference_mode():
        for _ in range(3):
            net.generate(**build_inputs(proc, img, wl.prompt), **gen)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            net.generate(**build_inputs(proc, img, wl.prompt), **gen)
            torch.cuda.synchronize()
    agg, n = defaultdict(float), defaultdict(int)
    for e in p.key_averages():
        if e.device_type.name != "CUDA" and getattr(e, "self_device_time_total", 0) == 0:
            continue
        t = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if t <= 0:
            continue
        name = e.key.lower()
        cat = next((c for c, keys in CATS if any(k in name for k in keys)), "other")
        agg[cat] += t / 1e3
        n[cat] += e.count
    tot = sum(agg.values())
    print(f"{args.workload}: {sum(n.values())} kernels, {tot:.1f} ms kernel time")
    for c, t in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<28}{n[c]:6d} kernels {t:8.2f} ms {100 * t / tot:6.1f} %")


if __name__ == "__main__":
    main()
