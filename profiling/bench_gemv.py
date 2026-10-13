"""Batch-1 GEMV microbenchmark at LFM2.5-VL-1.6B's decode shapes: cuBLAS (F.linear) vs our BF16 / INT8 kernel.

Each configuration cycles through enough distinct weight matrices (> 400 MB) that nothing is served from the
96 MB L2, so the numbers are DRAM-bound like in the real decode step.

    python profiling/bench_gemv.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.kernels import ext  # noqa: E402

K = ext()
SHAPES = {"qkv": (3072, 2048), "o / conv out": (2048, 2048), "conv in": (6144, 2048), "w1|w3": (16384, 2048),
          "w2": (2048, 8192), "lm_head": (65536, 2048)}


def timed(fn, mats, reps=3):
    for m in mats[:2]:
        fn(m)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        for m in mats:
            fn(m)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * len(mats))  # us per call


@torch.no_grad()
def main():
    print(f"{'matrix':14s} {'N x K':>12s} | {'cuBLAS bf16':>16s} | {'ours bf16 (auto)':>18s} | int8 rows=1 / 2 / 4 / auto | int4 (auto)")
    for name, (N, Kd) in SHAPES.items():
        n_mats = max(2, int(400e6 // (N * Kd * 2)) + 1)
        Ws = [torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(n_mats)]
        x = torch.randn(1, Kd, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(1, N, device="cuda", dtype=torch.bfloat16)
        Wq, Sq = [], []
        for W in Ws:
            s = W.float().abs().amax(1) / 127
            Wq.append(torch.round(W.float() / s[:, None]).clamp(-127, 127).to(torch.int8))
            Sq.append(s)
        qs = list(zip(Wq, Sq))
        from lfm2opt.quant import quantize_int4

        q4 = [quantize_int4(W) for W in Ws]
        b = N * Kd * 2
        t_cub = timed(lambda W: F.linear(x, W), Ws)
        t_bf = timed(lambda W: K.gemv(W, None, x, y), Ws)
        t_i8 = [timed(lambda q, r=r: K.gemv(q[0], q[1], x, y, r), qs) for r in (1, 2, 4, 0)]
        t_i4 = timed(lambda q: K.gemv_w4(q[0], q[1], x, y), q4)
        tb = lambda t, nbytes: f"{t:6.1f}us {nbytes / t / 1e6:4.2f}TB/s"  # noqa: E731
        print(f"{name:14s} {N:>6d}x{Kd:<5d} | {tb(t_cub, b)} | {tb(t_bf, b):>18s} | "
              + " / ".join(f"{t:5.1f}us {b / 2 / t / 1e6:4.2f}" for t in t_i8)
              + f" | int4 {t_i4:5.1f}us ({b * 0.53 / 2 / t_i4 / 1e6:4.2f}TB/s)")
        del Ws, Wq, Sq, qs, q4
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
