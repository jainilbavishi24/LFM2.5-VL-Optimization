"""Can INT8 speed up the first token? Every prefill / vision GEMM shape, BF16 vs INT8, with CUDA events (median of N).

  bf16     : F.linear on BF16 weights (what our fused vision / prefill call today; fc1 with the GELU epilogue)
  w8a16    : INT8 weights dequantized to BF16 (one pass over W), then the same BF16 GEMM: saves memory, not time
  int_mm   : torch._int_mm alone (cuBLASLt INT8 tensor cores, int32 out): the W8A8 upper bound
  w8a8     : per-token activation quantization + int_mm + dequant epilogue, all as plain torch ops (unfused)
  w8a8 fused (estimate): int_mm + one extra pass over the int32 output (read 4 B + write 2 B per element at the
             measured copy bandwidth), i.e. quantization folded into the preceding norm kernel and dequant (+ bias,
             GELU) into an epilogue kernel, as the real implementation would do

    python profiling/int8_gemm_probe.py [--repeats 30]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F


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


def copy_bandwidth(repeats):
    src = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    return 2 * src.numel() / cuda_ms(lambda: dst.copy_(src), repeats) / 1e9  # GB/s (read + write)


def quant_rows(x):
    s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return torch.round(x.float() / s).clamp(-127, 127).to(torch.int8), s


def probe(M, K, N, bias, gelu, repeats, bw):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16) if bias else None
    wq, sw = quant_rows(w)
    sw = sw.view(1, N)
    act = (lambda y: F.gelu(y, approximate="tanh")) if gelu else (lambda y: y)

    def bf16():
        if gelu:
            return torch._addmm_activation(b, x, w.t(), use_gelu=True)
        return F.linear(x, w, b)

    def w8a16():
        wd = (wq.to(torch.bfloat16) * sw.view(N, 1).to(torch.bfloat16))
        if gelu:
            return torch._addmm_activation(b, x, wd.t(), use_gelu=True)
        return F.linear(x, wd, b)

    def int_mm():
        return torch._int_mm(xq, wq.t())

    def w8a8():
        q, sx = quant_rows(x)
        y = torch._int_mm(q, wq.t()).float() * sx * sw
        if b is not None:
            y = y + b.float()
        return act(y).to(torch.bfloat16)

    xq, _ = quant_rows(x)
    r = dict(bf16=cuda_ms(bf16, repeats), w8a16=cuda_ms(w8a16, repeats), int_mm=cuda_ms(int_mm, repeats),
             w8a8=cuda_ms(w8a8, repeats))
    r["w8a8_fused_est"] = r["int_mm"] + M * N * 6 / bw / 1e6
    # accuracy of the W8A8 product on random data (sanity check of the int path, not of the model)
    ref = F.linear(x.float(), w.float(), b.float() if b is not None else None)
    r["w8a8_rel_err"] = ((w8a8().float() - act(ref)).norm() / act(ref).norm()).item()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    bw = copy_bandwidth(args.repeats)
    print(f"GPU {torch.cuda.get_device_name()}  copy bandwidth {bw:.0f} GB/s")
    # (name, M, K, N, count per request, bias, gelu): SigLIP2 linears have biases, the LFM2 LM's have none.
    # 1080p = 9 tiles (9216 patches) + 2327 prompt tokens; VGA = 1 tile (1024 patches) + ~270 tokens.
    vision = [("vision qkv", 1152, 3456, False), ("vision out", 1152, 1152, False),
              ("vision fc1+gelu", 1152, 4304, True), ("vision fc2", 4304, 1152, False)]
    lm = [("LM attn qkv", 2048, 3072, 6), ("LM attn out", 2048, 2048, 6), ("LM conv in", 2048, 6144, 10),
          ("LM conv out", 2048, 2048, 10), ("LM w1|w3", 2048, 16384, 16), ("LM w2", 8192, 2048, 16)]
    for label, Mv, Ml in (("1080p", 9216, 2327), ("VGA", 1024, 272)):
        shapes = [(n, Mv, K, N, 27, True, g) for n, K, N, g in vision] + [(n, Ml, K, N, c, False, False) for n, K, N, c in lm]
        tot = dict(bf16=0.0, w8a16=0.0, int_mm=0.0, w8a8=0.0, w8a8_fused_est=0.0)
        print(f"\n== {label}: ms per GEMM (x count per request) ==")
        print(f"{'GEMM':<16}{'M x K x N':>18}{'bf16':>8}{'w8a16':>8}{'int_mm':>8}{'w8a8':>8}{'fused~':>8}"
              f"{'int/bf16':>9}{'x':>4}{'rel err':>9}")
        for name, M, K, N, n, bias, gelu in shapes:
            r = probe(M, K, N, bias, gelu, args.repeats, bw)
            for k in tot:
                tot[k] += r[k] * n
            print(f"{name:<16}{f'{M}x{K}x{N}':>18}{r['bf16']:8.3f}{r['w8a16']:8.3f}{r['int_mm']:8.3f}{r['w8a8']:8.3f}"
                  f"{r['w8a8_fused_est']:8.3f}{r['bf16'] / r['int_mm']:8.2f}x{n:4d}{r['w8a8_rel_err']:9.4f}")
        print(f"{'per request':<34}" + "".join(f"{tot[k]:8.2f}" for k in ("bf16", "w8a16", "int_mm", "w8a8", "w8a8_fused_est"))
              + f"   (ms; w8a8 fused est. = {100 * (1 - tot['w8a8_fused_est'] / tot['bf16']):.0f} % less GEMM time)")


if __name__ == "__main__":
    main()
