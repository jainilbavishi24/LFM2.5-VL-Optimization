"""Tests for lfm2opt.gptq (GPU):
  * identity Hessian -> GPTQ has nothing to compensate: identical to round-to-nearest (same scales)
  * correlated inputs -> GPTQ's output error is below RTN's
  * pack_int4 round-trips through lfm2opt.quant.dequantize_int4 exactly
  * NVFP4: every value is a signed E2M1 grid point times an FP8-representable block scale
  * the decode kernel (gemv_w4) on GPTQ-packed weights equals F.linear with the dequantized weights

    python tests/test_gptq.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.gptq import Int4, NvFp4, gptq, output_error, pack_int4, rtn  # noqa: E402
from lfm2opt.quant import dequantize_int4  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")


N, K = 256, 512
W = torch.randn(N, K, device=dev) * 0.02
for group in (32, 64, 128):
    for clip in (False, True):
        a, sa = rtn(W, group, Int4(clip))
        b, sb = gptq(W, torch.eye(K, device=dev), group, Int4(clip), damp=0.0)
        check(f"identity H == RTN (g{group}, clip={clip})", torch.equal(a, b) and torch.equal(sa, sb))

X = torch.randn(4096, 64, device=dev) @ torch.randn(64, K, device=dev) + 0.1 * torch.randn(4096, K, device=dev)
H = X.t() @ X
for group in (64, 128):
    e_rtn = output_error(W, rtn(W, group, Int4(True))[0], H)
    e_gptq = output_error(W, gptq(W, H, group, Int4(True))[0], H)
    check(f"GPTQ output error < RTN (g{group})", e_gptq < e_rtn, f"{100 * e_gptq:.3f} % vs {100 * e_rtn:.3f} %")

Wq, s = gptq(W, H, 128, Int4(True))
p, s2 = pack_int4(Wq, s)
check("pack_int4 round trip", torch.equal(dequantize_int4(p, s2), Wq))

qz = NvFp4()
Wn, sn = gptq(W, H, 16, qz)
v = (Wn.view(N, -1, 16) / sn[..., None]).abs()
on_grid = torch.isin(v.round(decimals=4), qz.grid.round(decimals=4)).all().item()
fp8_ok = torch.equal((sn / qz.g).to(torch.float8_e4m3fn).float() * qz.g, sn)
check("NVFP4 values on the E2M1 grid, FP8 block scales", on_grid and fp8_ok)
check("NVFP4 GPTQ output error < RTN", output_error(W, Wn, H) < output_error(W, rtn(W, 16, NvFp4())[0], H))

from lfm2opt.kernels import ext  # noqa: E402

for group in (64, 128):
    Wq, s = gptq(W, H, group, Int4(True))
    p, s = pack_int4(Wq, s)
    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    y = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
    ext().gemv_w4(p, s, x, y)
    ref = F.linear(x.float(), Wq)
    err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    check(f"gemv_w4 on GPTQ weights (g{group})", err < 1e-2, f"max rel err {err:.2e}")

print(f"{fails} failures")
sys.exit(1 if fails else 0)
