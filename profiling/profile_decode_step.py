"""Kernel breakdown of one decode step of our runtime (O1 / fused), via the PyTorch profiler.

The step runs eagerly here (same kernels the CUDA graph replays), so every kernel is attributed:

    python profiling/profile_decode_step.py --fused --workload img_small --steps 20
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.runtime import StaticDecoder  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402


def category(name: str) -> str:
    n = name.lower()
    for key, cat in [("gemv", "gemv/gemm"), ("gemm", "gemv/gemm"), ("cutlass", "gemv/gemm"), ("internal::kernel", "gemv/gemm"),
                     ("sm90", "gemv/gemm"), ("sm100", "gemv/gemm"), ("sm120", "gemv/gemm"), ("splitk", "gemv/gemm"),
                     ("add_rmsnorm", "ours: add_rmsnorm"), ("shortconv", "ours: shortconv"), ("qk_norm_rope", "ours: qk_rope_kv"),
                     ("attention_split", "ours: attention"), ("attn_combine", "ours: attention"), ("decode_attention", "ours: attention"),
                     ("silu_mul", "ours: silu_mul"), ("argmax", "argmax/topk"), ("reduce", "reduce"), ("embedding", "embedding"),
                     ("index", "index/copy"), ("copy", "index/copy"), ("fill", "index/copy")]:
        if key in n:
            return cat
    return "other elementwise"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--workload", default="img_small")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    net, processor = load()
    dec = StaticDecoder(net, fused=args.fused)
    wl = WORKLOADS[args.workload]
    inputs = build_inputs(processor, wl.image(), wl.prompt)
    model_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    _, s = dec.prefill(inputs["input_ids"], args.steps + 8, attention_mask=inputs.get("attention_mask"), **model_inputs)
    step = dec._step_fused if args.fused else dec._step
    for _ in range(3):
        step(s)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step(s)
        torch.cuda.synchronize()
    per_kernel = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            k = per_kernel[e.name]
            k[0] += 1
            k[1] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    cats = defaultdict(lambda: [0, 0.0])
    for name, (n, us) in per_kernel.items():
        c = cats[category(name)]
        c[0] += n
        c[1] += us
    total_n = sum(n for n, _ in per_kernel.values()) / args.steps
    total_us = sum(us for _, us in per_kernel.values()) / args.steps
    print(f"{'fused' if args.fused else 'O1'} {args.workload}: {total_n:.0f} kernels/step, kernel time {total_us / 1e3:.3f} ms/step")
    for c, (n, us) in sorted(cats.items(), key=lambda kv: -kv[1][1]):
        print(f"  {c:22s} {n / args.steps:6.1f} kernels  {us / args.steps:8.1f} us  {100 * us / args.steps / total_us:5.1f} %")
    print("top kernels:")
    for name, (n, us) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {us / args.steps:8.1f} us  {n / args.steps:5.1f}x  {name[:110]}")


if __name__ == "__main__":
    main()
