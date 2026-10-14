"""Unit tests: each fused decode kernel (kernels/lfm2_decode.cu) vs a PyTorch reference written with the same ops
as transformers' modeling_lfm2.py. Random BF16 inputs at the model's real shapes.

    python tests/test_kernels.py          (needs a Python with headers: .venv-ref on the course server)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.kernels import ext  # noqa: E402

torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
K = ext()
H, I, NQ, NKV, D, EPS = 2048, 8192, 32, 8, 64, 1e-5
results = []


def rms(x, w, eps=EPS):
    xf = x.float()
    n = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(bf)
    return w * n


def report(name, out, ref):
    err = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    ok = err <= 2e-2 * max(scale, 1.0)
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} max|err| {err:.4g}  (ref max {scale:.3g})")


def test_add_rmsnorm(B=2):
    h, y = torch.randn(B, H, device=dev, dtype=bf), torch.randn(B, H, device=dev, dtype=bf)
    w = (torch.rand(H, device=dev) + 0.5).to(bf)
    ref_h = h + y
    ref_x = rms(ref_h, w)
    hk, xk = h.clone(), torch.empty_like(h)
    K.add_rmsnorm(hk, y, w, xk, EPS)
    report("add_rmsnorm (residual)", hk, ref_h)
    report("add_rmsnorm (normed)", xk, ref_x)
    x2 = torch.empty_like(h)
    K.add_rmsnorm(h.clone(), None, w, x2, EPS)
    report("rmsnorm (no add)", x2, rms(h, w))


def test_shortconv(B=2):
    bcx = torch.randn(B, 3 * H, device=dev, dtype=bf)
    state = torch.randn(B, H, 3, device=dev, dtype=bf)
    w = torch.randn(H, 3, device=dev, dtype=bf) * 0.5
    # reference = transformers causal_conv1d_update path
    BCx = bcx.view(B, 1, 3 * H).transpose(-1, -2)
    b_, c_, x_ = BCx.chunk(3, dim=-2)
    hx = b_ * x_
    st = state.clone()
    new = torch.cat([st, hx], dim=-1)
    st.copy_(new[:, :, -3:])
    conv = F.conv1d(new, w.unsqueeze(1), None, padding=0, groups=H)[:, :, -1:]
    ref = (c_ * conv).transpose(-1, -2).reshape(B, H)
    sk, yk = state.clone(), torch.empty(B, H, device=dev, dtype=bf)
    K.shortconv_step(bcx, sk, w, None, yk)
    report("shortconv_step (out)", yk, ref)
    report("shortconv_step (state)", sk, st)


def test_qk_rope_kv(B=2, max_len=512, pos=137):
    from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb

    qkv = torch.randn(B, (NQ + 2 * NKV) * D, device=dev, dtype=bf)
    qw, kw = (torch.rand(D, device=dev) + 0.5).to(bf), (torch.rand(D, device=dev) + 0.5).to(bf)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=dev, dtype=torch.float) / D))
    q, k, v = qkv.view(B, 1, NQ + 2 * NKV, D).split([NQ, NKV, NKV], dim=2)
    q = rms(q, qw).transpose(1, 2)
    k = rms(k, kw).transpose(1, 2)
    freqs = torch.tensor([[float(pos)]], device=dev)[..., None] * inv_freq
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(bf).expand(B, 1, D), emb.sin().to(bf).expand(B, 1, D)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    kc = torch.zeros(B, NKV, max_len, D, device=dev, dtype=bf)
    vc = torch.zeros_like(kc)
    qo = torch.empty(B, NQ, D, device=dev, dtype=bf)
    p = torch.tensor([pos], device=dev, dtype=torch.long)
    K.qk_norm_rope_kv(qkv, qw, kw, inv_freq, 1.0, p, qo, kc, vc, EPS)
    report("qk_norm_rope (q)", qo, q[:, :, 0])
    report("qk_norm_rope (k -> cache)", kc[:, :, pos], k[:, :, 0])
    report("qk_norm_rope (v -> cache)", vc[:, :, pos], v.transpose(1, 2)[:, :, 0])
    report("qk_norm_rope (cache untouched)", kc[:, :, pos + 1 :], torch.zeros_like(kc[:, :, pos + 1 :]))


def test_attention(B=2, max_len=3072, pos=2300):
    q = torch.randn(B, NQ, D, device=dev, dtype=bf)
    kc = torch.randn(B, NKV, max_len, D, device=dev, dtype=bf)
    vc = torch.randn(B, NKV, max_len, D, device=dev, dtype=bf)
    scale = D ** -0.5
    ref = F.scaled_dot_product_attention(q.unsqueeze(2), kc[:, :, : pos + 1], vc[:, :, : pos + 1], scale=scale,
                                         enable_gqa=True).squeeze(2)
    out = torch.empty_like(q)
    K.decode_attention(q, kc, vc, torch.tensor([pos], device=dev), out, scale)
    report(f"decode_attention (len {pos + 1})", out, ref)
    for S in (16, 4):
        ws = torch.empty(B, NQ, S, 66, device=dev, dtype=torch.float32)
        outs = torch.empty_like(q)
        K.decode_attention_split(q, kc, vc, torch.tensor([pos], device=dev), outs, ws, scale)
        report(f"attention_split S={S} (len {pos + 1})", outs, ref)
    ws = torch.empty(B, NQ, 16, 66, device=dev, dtype=torch.float32)
    outs1 = torch.empty_like(q)
    K.decode_attention_split(q, kc, vc, torch.tensor([0], device=dev), outs1, ws, scale)
    report("attention_split (len 1)", outs1, vc[:, :, :1].repeat_interleave(NQ // NKV, 1).squeeze(2))
    out1 = torch.empty_like(q)
    K.decode_attention(q, kc, vc, torch.tensor([0], device=dev), out1, scale)
    report("decode_attention (len 1)", out1, vc[:, :, :1].repeat_interleave(NQ // NKV, 1).squeeze(2))


def test_silu_mul(B=2):
    g = torch.randn(B, 2 * I, device=dev, dtype=bf)
    a, b = g.chunk(2, -1)
    ref = F.silu(a) * b
    y = torch.empty(B, I, device=dev, dtype=bf)
    K.silu_mul(g, y)
    report("silu_mul", y, ref)


def test_gemv():
    for N, Kd in [(3072, 2048), (6144, 2048), (2048, 2048), (16384, 2048), (2048, 8192), (65536, 2048)]:
        W = torch.randn(N, Kd, device=dev, dtype=bf) * 0.02
        x = torch.randn(1, Kd, device=dev, dtype=bf)
        y = torch.empty(1, N, device=dev, dtype=bf)
        K.gemv(W, None, x, y)
        report(f"gemv bf16 [{N}x{Kd}]", y, F.linear(x, W))
        scale = W.float().abs().amax(1) / 127
        Wq = torch.round(W.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
        K.gemv(Wq, scale, x, y)
        ref = F.linear(x.float(), Wq.float() * scale[:, None]).to(bf)
        report(f"gemv int8 [{N}x{Kd}]", y, ref)
        from lfm2opt.quant import dequantize_int4, quantize_int4

        for Bt in (2, 3, 4, 7, 8, 13):  # batched decode: chunks of 8 / 4 / 2 / 1 rows
            xb = torch.randn(Bt, Kd, device=dev, dtype=bf)
            yb = torch.empty(Bt, N, device=dev, dtype=bf)
            K.gemv(Wq, scale, xb, yb)
            report(f"gemv int8 B={Bt} [{N}x{Kd}]", yb, F.linear(xb.float(), Wq.float() * scale[:, None]).to(bf))
            K.gemv(W, None, xb, yb)
            report(f"gemv bf16 B={Bt} [{N}x{Kd}]", yb, F.linear(xb, W))
        Wp, S4 = quantize_int4(W)
        K.gemv_w4(Wp, S4, x, y)
        report(f"gemv int4 [{N}x{Kd}]", y, F.linear(x.float(), dequantize_int4(Wp, S4)).to(bf))


def test_gemv_epi():
    """O2d: GEMV + fused epilogue must be bit-identical to GEMV followed by silu_mul / shortconv_step."""
    torch.manual_seed(5)
    H, I = 2048, 8192
    x = torch.randn(1, H, device=dev, dtype=bf)
    for wt in ("bf16", "int8"):
        for epi, rows in ((0, 2 * I), (1, 3 * H)):
            W = (torch.randn(rows, H, device=dev) * 0.02).to(bf)
            scale = None
            if wt == "int8":
                scale = W.float().abs().amax(1) / 127
                W = torch.round(W.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
            g = torch.empty(1, rows, device=dev, dtype=bf)
            K.gemv(W, scale, x, g)
            if epi == 0:
                ref = torch.empty(1, I, device=dev, dtype=bf)
                K.silu_mul(g, ref)
                out = torch.empty_like(ref)
                K.gemv_epi(0, W, scale, x, out)
                ok = torch.equal(out, ref)
                name = f"gemv_epi swiglu {wt} [{rows}x{H}]"
            else:
                cw = (torch.randn(H, 3, device=dev) * 0.5).to(bf)
                cb = (torch.randn(H, device=dev) * 0.1).to(bf)
                st0 = torch.randn(1, H, 3, device=dev).to(bf)
                st_ref, st_out = st0.clone(), st0.clone()
                ref = torch.empty(1, H, device=dev, dtype=bf)
                K.shortconv_step(g, st_ref, cw, cb, ref)
                out = torch.empty_like(ref)
                K.gemv_epi(1, W, scale, x, out, st_out, cw, cb)
                ok = torch.equal(out, ref) and torch.equal(st_out, st_ref)
                name = f"gemv_epi shortconv {wt} [{rows}x{H}] (+state)"
            results.append(ok)
            print(f"{'PASS' if ok else 'FAIL'}  {name}  bit-identical={ok}  max|err| {(out.float() - ref.float()).abs().max():.3g}")


if __name__ == "__main__":
    for t in (test_add_rmsnorm, test_shortconv, test_qk_rope_kv, test_attention, test_silu_mul, test_gemv, test_gemv_epi):
        t()
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
