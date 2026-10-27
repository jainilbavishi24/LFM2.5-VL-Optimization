"""INT8 linear layers for prefill and the vision encoder (kernels/int8_gemm.cu): INT8 weights everywhere, so the
BF16 copies can be dropped (memory for the edge), and INT8 tensor cores where the activations allow it.

  a8  (W8A8):  activations quantized on the fly (quantize_rows) with one scale per token and group of 64 input
               channels (LFM2_INT8_GROUP=64, the default; 0 = one scale per token), INT8 x INT8 tensor-core GEMM.
               Per-group scales keep the outlier channels of the vision fc2 / LM w2 inputs (after GELU / SwiGLU) from
               coarsening the whole token: W8A8 everywhere then matches W8A16 quality (profiling/int8_quality_probe.py).
  a16 (W8A16): BF16 activations, INT8 weights converted in registers, BF16 tensor-core GEMM: for the layer kinds
               listed in LFM2_INT8_A16 (default none; e.g. "fc2,w2"), and the decode fallback for large batches.

Weights: symmetric INT8, one scale per output row (lfm2opt.quant.quantize_int8, as the decode GEMV).
"""

import os

import torch

from .kernels import int8_ext
from .quant import quantize_int8


def a16_kinds():
    """Layer kinds that keep BF16 activations (W8A16); the rest run W8A8."""
    return set(filter(None, os.environ.get("LFM2_INT8_A16", "").split(",")))


def act_group(kind=None, K=0):
    """Activation scale granularity of a W8A8 layer: 64 = per token and group of 64 channels (one per K stage of the
    kernel), 0 = per token. LFM2_INT8_GROUP=64 / 0 sets it for every kind; LFM2_INT8_GROUP_KINDS="fc2,w2" limits the
    groups to those kinds (the rest per token). Inputs wider than 2048 (fc2, w2) are always grouped."""
    g = int(os.environ.get("LFM2_INT8_GROUP", "64"))
    assert g in (0, 64), "the kernel folds one scale per 64-wide K stage"
    kinds = set(filter(None, os.environ.get("LFM2_INT8_GROUP_KINDS", "").split(",")))
    if K > 2048:
        return 64
    return g if (not kinds or kind in kinds) else 0


def qbuf(rows, K, group, device):
    """Empty (int8 [rows, K], scales) for a producer kernel that writes a W8A8 layer's input directly."""
    return (torch.empty(rows, K, device=device, dtype=torch.int8),
            torch.empty((K // 64, rows) if group else (rows,), device=device, dtype=torch.float32))


def quantize_rows(x2, group=0):
    """[M, K] bf16 -> (int8 [M, K], fp32 scales): [M] (absmax / 127 per row) or, group=64, [K / 64, M]."""
    M, K = x2.shape
    q = torch.empty(x2.shape, device=x2.device, dtype=torch.int8)
    s = torch.empty((K // 64, M) if group else (M,), device=x2.device, dtype=torch.float32)
    int8_ext().quant_rows(x2, q, s)
    return q, s


class Int8Linear:
    def __init__(self, weight=None, bias=None, a8=True, q=None, scale=None, pad_n=0, pad_k=0, kind=None):
        """From a BF16 weight [N, K] (quantized here), or from an already quantized (q, scale) pair. pad_n / pad_k
        append zero output rows (and zero bias) / zero input columns, e.g. to make K a multiple of 128 for the
        grouped activation scales (SigLIP2's 4304-wide MLP -> 4352: fc1 pad_n=48, fc2 pad_k=48)."""
        if q is None:
            weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))
            if bias is not None and pad_n:
                bias = torch.nn.functional.pad(bias, (0, pad_n))
            q, scale = quantize_int8(weight)
        self.q, self.s, self.bias, self.a8 = q, scale, bias, a8
        self.N, self.K = q.shape
        self.group = act_group(kind, self.K) if a8 else 0
        self.ext = int8_ext()

    def __call__(self, x, gelu=False, quant_out=False):
        """x: bf16 [..., K], or an already quantized (int8 [..., K], scales) pair from a fused producer kernel.
        quant_out: return the output as the next W8A8 GEMM's input (int8 [M, N], scales [N/64, M]), quantized inside
        this GEMM's epilogue (grouped W8A8 only, N % 128 == 0)."""
        if quant_out:
            xq, sx = x if isinstance(x, tuple) else quantize_rows(x.reshape(-1, self.K).contiguous(), 64)
            M = xq.numel() // self.K
            yq, ys = qbuf(M, self.N, 64, xq.device)
            self.ext.gemm_i8_q(xq, sx, self.q, self.s, self.bias, yq, ys, gelu)
            return yq, ys
        if isinstance(x, tuple):
            xq, sx = x
            y = torch.empty(xq.numel() // self.K, self.N, device=xq.device, dtype=torch.bfloat16)
            self.ext.gemm_i8(xq, sx, self.q, self.s, self.bias, y, gelu)
            return y.view(*xq.shape[:-1], self.N)
        x2 = x.reshape(-1, self.K)
        y = torch.empty(x2.shape[0], self.N, device=x.device, dtype=torch.bfloat16)
        if self.a8:
            xq, sx = quantize_rows(x2.contiguous(), self.group)
            self.ext.gemm_i8(xq, sx, self.q, self.s, self.bias, y, gelu)
        else:
            self.ext.gemm_w8(x2.contiguous(), self.q, self.s, self.bias, y, gelu)
        return y.view(*x.shape[:-1], self.N)
