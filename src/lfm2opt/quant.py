"""Weight-only quantization for the decode GEMVs (O6). Round-to-nearest, symmetric; done once at load time.

* INT8: one fp32 scale per output row: W ~= q * s[n], q in [-127, 127].
* INT4: one fp32 scale per group of 128 inputs: W ~= q * s[n, k // 128], q in [-8, 7], packed two per byte
  (element 2j in the low nibble, 2j+1 in the high nibble, stored as q + 8) for kernels/lfm2_decode.cu gemv_w4.
"""

import torch

GROUP = 128


@torch.no_grad()
def quantize_int8(w: torch.Tensor):
    wf = w.float()
    scale = (wf.abs().amax(1) / 127).clamp_min(1e-12)
    q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


@torch.no_grad()
def quantize_int4(w: torch.Tensor, group: int = GROUP, clip_search: bool = False):
    """clip_search: per group, pick the scale (absmax * a / 7, a in 0.70..1.00) with the lowest squared error."""
    N, K = w.shape
    assert K % group == 0
    wg = w.float().view(N, K // group, group)
    amax = wg.abs().amax(-1).clamp_min(1e-12)                           # [N, K/group]
    scale = amax / 7
    if clip_search:
        best = torch.full_like(amax, float("inf"))
        for a in torch.linspace(0.70, 1.00, 16).tolist():
            s = amax * a / 7
            err = (torch.round(wg / s[..., None]).clamp(-8, 7) * s[..., None] - wg).pow(2).sum(-1)
            better = err < best
            best = torch.where(better, err, best)
            scale = torch.where(better, s, scale)
    q = torch.round(wg / scale[..., None]).clamp(-8, 7).to(torch.int16).view(N, K) + 8
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)           # [N, K/2]
    return packed.contiguous(), scale.contiguous()


@torch.no_grad()
def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor):
    p = packed.to(torch.int16)
    q = torch.stack([(p & 0xF) - 8, (p >> 4) - 8], dim=-1).view(packed.shape[0], -1).float()
    group = q.shape[1] // scale.shape[1]
    return (q.view(q.shape[0], -1, group) * scale[..., None]).view(q.shape[0], -1)


@torch.no_grad()
def quantize_int4_awq(w: torch.Tensor, x_calib: torch.Tensor, group: int = GROUP, grid: int = 20, clip_search: bool = True):
    """AWQ-style INT4: per input channel scale s = mean|x|^alpha (alpha searched on calibration inputs x_calib
    [M, K] by the output error ||x W^T - (x / s) Q(W s)^T||), then group-wise INT4 of W * s.
    Returns (packed, scales, inv_s) where inv_s [K] must multiply x before the GEMV."""
    wf = w.float()
    xc = x_calib.float()
    act = xc.abs().mean(0).clamp_min(1e-6)                              # [K]
    ref = xc @ wf.t()
    best = (float("inf"), None)
    for i in range(grid):
        alpha = i / grid
        s = act.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()                              # keep the scaled weights in range
        packed, scale = quantize_int4(wf * s[None, :], group, clip_search=False)
        wq = dequantize_int4(packed, scale) / s[None, :]
        err = (xc @ wq.t() - ref).pow(2).mean().item()
        if err < best[0]:
            best = (err, s)
    s = best[1]
    packed, scale = quantize_int4(wf * s[None, :], group, clip_search=clip_search)
    return packed, scale, (1.0 / s).contiguous()
