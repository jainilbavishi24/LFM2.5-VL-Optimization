"""INT8 prefill / vision GEMMs (kernels/int8_gemm.cu) vs PyTorch references, then their speed vs cuBLAS.

Correctness, random data at the model's shapes plus odd M (tails):
  quant_rows  must equal torch's per-row absmax quantization exactly
  gemm_i8     integer products are exact, so it must equal the fp64 product of the same int8 operands with the same
              fp32 scaling (bit-exact without GELU; GELU: <= 1 bf16 ulp from torch's tanh GELU); grouped scales: within
              5 % of the bf16 rounding error of the exact (fp64) result
  producers   add_layernorm_q / add_rmsnorm_q / silu_mul_q must equal the unfused bf16 kernel + quantization exactly
  gemm_w8     fp32 accumulation like cuBLAS: its error vs an fp64 reference must be on the level of cuBLAS BF16's
              error on the dequantized weights
Speed: every first-token GEMM shape at 1080p and VGA, CUDA events, median.

    python tests/test_int8_gemm.py [--no-speed]
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.int8 import quantize_rows  # noqa: E402
from lfm2opt.kernels import ext, int8_ext, vision_ext  # noqa: E402
from lfm2opt.quant import quantize_int8  # noqa: E402

torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
E = int8_ext()
results = []


def ref_quant(x):
    a = x.float().abs().amax(-1).clamp_min(1e-8)
    s = a / torch.full_like(a, 127.0)  # true division (tensor / scalar would multiply by 1/127: 1 ulp off)
    return torch.round(x.float() / s[:, None]).clamp(-127, 127).to(torch.int8), s


def bf16_ulps(a, b):
    """max distance in bf16 units in the last place (same sign assumed for nonzero values)."""
    ia = a.view(torch.int16).int()
    ib = b.view(torch.int16).int()
    ia = torch.where(ia < 0, -32768 - ia, ia)
    ib = torch.where(ib < 0, -32768 - ib, ib)
    return (ia - ib).abs().max().item()


def check(name, ok, msg):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:44s} {msg}")


def test_quant_rows(M, K):
    x = torch.randn(M, K, device=dev, dtype=bf) * torch.rand(M, 1, device=dev, dtype=bf) * 4
    x[0, 3] = 37.0  # an outlier
    q, s = quantize_rows(x)
    rq, rs = ref_quant(x)
    check(f"quant_rows {M}x{K}", torch.equal(q, rq) and torch.equal(s, rs),
          f"q mismatches {(q != rq).sum().item()}, scale max|d| {(s - rs).abs().max().item():.3g}")


def ref_quant_groups(x):
    xg = x.float().unflatten(-1, (-1, 64))
    a = xg.abs().amax(-1).clamp_min(1e-8)
    s = a / torch.full_like(a, 127.0)
    q = torch.round(xg / s[..., None]).clamp(-127, 127).to(torch.int8).flatten(-2)
    return q, s.t().contiguous()  # [K/64, M]


def test_quant_groups(M, K):
    x = torch.randn(M, K, device=dev, dtype=bf) * torch.rand(M, 1, device=dev, dtype=bf) * 4
    x[0, 70] = 37.0
    q, s = quantize_rows(x, 64)
    rq, rs = ref_quant_groups(x)
    check(f"quant_rows grouped {M}x{K}", torch.equal(q, rq) and torch.equal(s, rs),
          f"q mismatches {(q != rq).sum().item()}, scale max|d| {(s - rs).abs().max().item():.3g}")


def test_i8_grouped(M, K, N, bias, gelu):
    x = torch.randn(M, K, device=dev, dtype=bf)
    x[:, 5] *= 30  # outlier channel
    w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
    b = torch.randn(N, device=dev, dtype=bf) if bias else None
    xq, sxT = quantize_rows(x, 64)
    wq, sw = quantize_int8(w)
    y = torch.empty(M, N, device=dev, dtype=bf)
    E.gemm_i8(xq, sxT, wq, sw, b, y, gelu)
    r = torch.zeros(M, N, device=dev, dtype=torch.float64)
    for g in range(K // 64):  # exact integer group sums with the same scales, in fp64
        part = xq[:, g * 64:(g + 1) * 64].double() @ wq[:, g * 64:(g + 1) * 64].double().t()
        r = r + part * sxT[g].double()[:, None]
    r = r * sw.double()[None, :]
    if b is not None:
        r = r + b.double()
    if gelu:
        r = F.gelu(r, approximate="tanh")
    # the fp32 fold (one fma per group) differs from fp64 by ~1e-7 relative to the partial sums, which is many bf16
    # ulps on outputs near zero after cancellation: compare norms with the unavoidable bf16 output rounding instead
    e_ours = ((y.double() - r).norm() / r.norm()).item()
    e_round = ((r.to(bf).double() - r).norm() / r.norm()).item()
    check(f"gemm_i8 grouped {M}x{K}x{N}{' bias' if bias else ''}{' gelu' if gelu else ''}", e_ours <= 1.05 * e_round + 1e-6,
          f"rel err {e_ours:.3e} (bf16 rounding of the exact result {e_round:.3e})")


def test_fused_producers():
    """add_layernorm_q / add_rmsnorm_q / silu_mul_q == the unfused bf16 kernel followed by quant_rows (exactly)."""
    V, D = vision_ext(), ext()
    for grouped in (0, 64):
        tag = "grouped" if grouped else "per row"
        rows, H = 300, 1152
        h0, y = torch.randn(rows, H, device=dev, dtype=bf), torch.randn(rows, H, device=dev, dtype=bf)
        w, b = (torch.rand(H, device=dev) + 0.5).to(bf), torch.randn(H, device=dev, dtype=bf)
        h1, h2, x = h0.clone(), h0.clone(), torch.empty_like(h0)
        V.add_layernorm(h1, y, w, b, x, 1e-6)
        rq, rs = quantize_rows(x, grouped)
        q, s = torch.empty_like(rq), torch.empty_like(rs)
        E.add_layernorm_q(h2, y, w, b, q, s, 1e-6)
        check(f"add_layernorm_q {tag}", torch.equal(h1, h2) and torch.equal(q, rq) and torch.equal(s, rs),
              f"q mismatches {(q != rq).sum().item()}, h equal {torch.equal(h1, h2)}")
        rows, H = 257, 2048
        h0, y = torch.randn(rows, H, device=dev, dtype=bf), torch.randn(rows, H, device=dev, dtype=bf)
        w = (torch.rand(H, device=dev) + 0.5).to(bf)
        h1, h2, x = h0.clone(), h0.clone(), torch.empty_like(h0)
        D.add_rmsnorm(h1, y, w, x, 1e-5)
        rq, rs = quantize_rows(x, grouped)
        q, s = torch.empty_like(rq), torch.empty_like(rs)
        E.add_rmsnorm_q(h2, y, w, q, s, 1e-5)
        check(f"add_rmsnorm_q {tag}", torch.equal(h1, h2) and torch.equal(q, rq) and torch.equal(s, rs),
              f"q mismatches {(q != rq).sum().item()}, h equal {torch.equal(h1, h2)}")
        I = 8192
        g = torch.randn(rows, 2 * I, device=dev, dtype=bf) * 2
        a = torch.empty(rows, I, device=dev, dtype=bf)
        D.silu_mul(g, a)
        rq, rs = quantize_rows(a, 64) if grouped else ref_quant(a)  # per-row quant_rows needs K <= 2048
        q, s = torch.empty_like(rq), torch.empty_like(rs)
        E.silu_mul_q(g, q, s)
        check(f"silu_mul_q {tag}", torch.equal(q, rq) and torch.equal(s, rs), f"q mismatches {(q != rq).sum().item()}")


def test_i8(M, K, N, bias, gelu):
    x = torch.randn(M, K, device=dev, dtype=bf)
    w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
    b = torch.randn(N, device=dev, dtype=bf) if bias else None
    xq, sx = quantize_rows(x)
    wq, sw = quantize_int8(w)
    y = torch.empty(M, N, device=dev, dtype=bf)
    E.gemm_i8(xq, sx, wq, sw, b, y, gelu)
    acc = (xq.double() @ wq.double().t()).float()  # exact integer sums
    r = acc * sx[:, None] * sw[None, :]
    if b is not None:
        r = r + b.float()
    if gelu:
        r = F.gelu(r, approximate="tanh")
    r = r.to(bf)
    u = bf16_ulps(y, r)
    check(f"gemm_i8 {M}x{K}x{N}{' bias' if bias else ''}{' gelu' if gelu else ''}", u <= (1 if gelu else 0),
          f"max {u} bf16 ulp vs exact reference")


def test_w8(M, K, N, bias, gelu):
    x = torch.randn(M, K, device=dev, dtype=bf)
    w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
    b = torch.randn(N, device=dev, dtype=bf) if bias else None
    wq, sw = quantize_int8(w)
    y = torch.empty(M, N, device=dev, dtype=bf)
    E.gemm_w8(x, wq, sw, b, y, gelu)
    r64 = (x.double() @ wq.double().t()) * sw.double()[None, :]
    if b is not None:
        r64 = r64 + b.double()
    wd = (wq.float() * sw[:, None]).to(bf)  # cuBLAS on dequantized BF16 weights, for the error scale
    rc = torch._addmm_activation(b, x, wd.t(), use_gelu=True) if gelu else F.linear(x, wd, b)
    if gelu:
        r64 = F.gelu(r64, approximate="tanh")
    e_ours = ((y.double() - r64).norm() / r64.norm()).item()
    e_cub = ((rc.double() - r64).norm() / r64.norm()).item()
    check(f"gemm_w8 {M}x{K}x{N}{' bias' if bias else ''}{' gelu' if gelu else ''}", e_ours <= 1.5 * e_cub + 1e-4,
          f"rel err {e_ours:.3e} (cuBLAS on dequantized W {e_cub:.3e})")


def cuda_ms(fn, repeats=30):
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


def speed():
    vision = [("vision qkv", 1152, 3456, False), ("vision out", 1152, 1152, False), ("vision fc1+gelu", 1152, 4304, True),
              ("vision fc2", 4304, 1152, False)]
    lm = [("LM attn qkv", 2048, 3072, 6), ("LM attn out", 2048, 2048, 6), ("LM conv in", 2048, 6144, 10),
          ("LM conv out", 2048, 2048, 10), ("LM w1|w3", 2048, 16384, 16), ("LM w2", 8192, 2048, 16)]
    for label, Mv, Ml in (("1080p", 9216, 2327), ("VGA", 1024, 272)):
        shapes = [(n, Mv, K, N, 27, True, g) for n, K, N, g in vision] + [(n, Ml, K, N, c, False, False) for n, K, N, c in lm]
        tot = dict(cublas=0.0, int_mm=0.0, i8=0.0, quant=0.0, w8=0.0, i8g=0.0, quantg=0.0)
        print(f"\n== speed, {label}: ms per GEMM ==")
        print(f"{'GEMM':<16}{'M x K x N':>18}{'cuBLAS':>8}{'int_mm':>8}{'ours i8':>8}{'+quant':>8}{'ours w8':>8}"
              f"{'i8 gain':>8}{'ours g64':>9}{'+quant':>8}{'g gain':>8}{'x':>4}")
        for name, M, K, N, n, bias, gelu in shapes:
            x = torch.randn(M, K, device=dev, dtype=bf)
            w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
            b = torch.randn(N, device=dev, dtype=bf) if bias else None
            xq, sx = quantize_rows(x) if K <= 2048 else ref_quant(x)
            Kg = K + (-K % 128)  # grouped: K padded to a multiple of 128 (vision fc2 4304 -> 4352)
            xg = F.pad(x, (0, Kg - K))
            xgq, sxg = quantize_rows(xg, 64)
            wgq, swg = quantize_int8(F.pad(w, (0, Kg - K)))
            wq, sw = quantize_int8(w)
            y = torch.empty(M, N, device=dev, dtype=bf)
            r = dict(
                cublas=cuda_ms(lambda: torch._addmm_activation(b, x, w.t(), use_gelu=True) if gelu else F.linear(x, w, b)),
                int_mm=cuda_ms(lambda: torch._int_mm(xq, wq.t())) if M > 16 else float("nan"),
                i8=cuda_ms(lambda: E.gemm_i8(xq, sx, wq, sw, b, y, gelu)),
                quant=cuda_ms(lambda: quantize_rows(x)) if K <= 2048 else float("nan"),
                w8=cuda_ms(lambda: E.gemm_w8(x, wq, sw, b, y, gelu)),
                i8g=cuda_ms(lambda: E.gemm_i8(xgq, sxg, wgq, swg, b, y, gelu)),
                quantg=cuda_ms(lambda: quantize_rows(xg, 64)))
            for k in tot:
                tot[k] += r[k] * n if r[k] == r[k] else 0
            print(f"{name:<16}{f'{M}x{K}x{N}':>18}{r['cublas']:8.3f}{r['int_mm']:8.3f}{r['i8']:8.3f}{r['quant']:8.3f}"
                  f"{r['w8']:8.3f}{r['cublas'] / r['i8']:7.2f}x{r['i8g']:9.3f}{r['quantg']:8.3f}{r['cublas'] / r['i8g']:7.2f}x{n:4d}")
        print(f"{'per request':<34}" + "".join(f"{tot[k]:8.2f}" for k in ("cublas", "int_mm", "i8", "quant", "w8"))
              + f"{'':8}{tot['i8g']:9.2f}{tot['quantg']:8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args()
    for M, K in ((1, 2048), (37, 1152), (2327, 2048), (9216, 1152)):
        test_quant_rows(M, K)
    for M, K, N, bias, gelu in ((9216, 1152, 3456, True, False), (9216, 1152, 4304, True, True), (2327, 2048, 16384, False, False),
                                (272, 2048, 3072, False, False), (17, 2048, 2048, False, False), (1, 1152, 1152, True, False),
                                (300, 1152, 1152, True, True)):
        test_i8(M, K, N, bias, gelu)
    for M, K in ((1, 2048), (37, 1152), (2327, 8192), (9216, 4352)):
        test_quant_groups(M, K)
    for M, K, N, bias, gelu in ((9216, 1152, 3456, True, False), (9216, 4352, 1152, True, False),
                                (2327, 8192, 2048, False, False), (1024, 1152, 4352, True, True), (17, 2048, 2048, False, False),
                                (1, 1152, 1152, True, False)):
        test_i8_grouped(M, K, N, bias, gelu)
    for M, K, N, bias, gelu in ((9216, 4304, 1152, True, False), (2327, 8192, 2048, False, False), (272, 8192, 2048, False, False),
                                (9, 2048, 6144, False, False), (1, 2048, 2048, False, False), (1024, 1152, 4304, True, True)):
        test_w8(M, K, N, bias, gelu)
    test_fused_producers()
    print(f"\n{sum(results)}/{len(results)} passed")
    if not args.no_speed:
        speed()
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
