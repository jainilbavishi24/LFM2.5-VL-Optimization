"""How close are the first-token GEMMs to the GPU's BF16 tensor-core peak?

Times every GEMM shape of a 1080p request (SigLIP2 encoder: 9216 patches; LM prefill: 2327 tokens) with cuBLAS
(F.linear, as our fused vision/prefill paths call it) and compares the achieved TFLOPS with a large square GEMM
(best of several sizes = the practically achievable peak; long ones hit the 300 W power limit), CUDA events, median of N.

    python profiling/gemm_efficiency.py [--repeats 50]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F


def tflops(M, K, N, repeats, gelu=False, bias=True):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16) if bias else None
    fn = (lambda: torch._addmm_activation(b, x, w.t(), use_gelu=True)) if gelu else (lambda: F.linear(x, w, b))
    for _ in range(5):
        fn()
    ts = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ms = statistics.median(ts)
    return ms, 2 * M * K * N / ms / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()
    # long square GEMMs hit the 300 W board power limit and clock down, so take the best of several sizes
    sq = {n: tflops(n, n, n, args.repeats, bias=False)[1] for n in (2048, 3072, 4096, 6144, 8192)}
    peak = max(sq.values())
    print("square GEMMs: " + "  ".join(f"{n}^3 {t:.0f}" for n, t in sq.items()) + f" TFLOPS -> peak {peak:.0f}")
    # (name, M, K, N, count per request, gelu epilogue); SigLIP2 linears have biases, the LFM2 LM's have none
    shapes = [
        ("vision qkv", 9216, 1152, 3456, 27, False), ("vision out", 9216, 1152, 1152, 27, False),
        ("vision fc1+gelu", 9216, 1152, 4304, 27, True), ("vision fc2", 9216, 4304, 1152, 27, False),
        ("LM attn qkv", 2327, 2048, 3072, 6, False), ("LM attn out", 2327, 2048, 2048, 6, False),
        ("LM conv in", 2327, 2048, 6144, 10, False), ("LM conv out", 2327, 2048, 2048, 10, False),
        ("LM w1|w3", 2327, 2048, 16384, 16, False), ("LM w2", 2327, 8192, 2048, 16, False),
    ]
    tot_ms = tot_flop = 0.0
    print(f"{'GEMM':<16}{'M x K x N':>20}{'ms':>8}{'TFLOPS':>8}{'% peak':>8}{'x/req':>6}{'ms/req':>8}")
    for name, M, K, N, n, gelu in shapes:
        ms, tf = tflops(M, K, N, args.repeats, gelu, bias=name.startswith("vision"))
        tot_ms += ms * n
        tot_flop += 2 * M * K * N * n
        print(f"{name:<16}{f'{M}x{K}x{N}':>20}{ms:8.3f}{tf:8.0f}{100 * tf / peak:7.0f}%{n:6d}{ms * n:8.2f}")
    print(f"all first-token GEMMs: {tot_ms:.1f} ms per 1080p request, {tot_flop / tot_ms / 1e9:.0f} TFLOPS "
          f"({100 * tot_flop / tot_ms / 1e9 / peak:.0f} % of the best square GEMM)")


if __name__ == "__main__":
    main()
