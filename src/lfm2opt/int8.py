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


def act_group():
    """Activation scale granularity for W8A8: 64 = per token and group of 64 channels (kernel tile), 0 = per token."""
    g = int(os.environ.get("LFM2_INT8_GROUP", "64"))
    assert g in (0, 64), "the kernel folds one scale per 64-wide K stage"
    return g


def quantize_rows(x2, group=0):
    """[M, K] bf16 -> (int8 [M, K], fp32 scales): [M] (absmax / 127 per row) or, group=64, [K / 64, M]."""
    M, K = x2.shape
    q = torch.empty(x2.shape, device=x2.device, dtype=torch.int8)
    s = torch.empty((K // 64, M) if group else (M,), device=x2.device, dtype=torch.float32)
    int8_ext().quant_rows(x2, q, s)
    return q, s


class Int8Linear:
    def __init__(self, weight=None, bias=None, a8=True, q=None, scale=None, pad_n=0, pad_k=0):
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
        self.group = act_group() if a8 else 0
        self.ext = int8_ext()

    def __call__(self, x, gelu=False):
        x2 = x.reshape(-1, self.K)
        y = torch.empty(x2.shape[0], self.N, device=x.device, dtype=torch.bfloat16)
        if self.a8:
            xq, sx = quantize_rows(x2.contiguous(), self.group)
            self.ext.gemm_i8(xq, sx, self.q, self.s, self.bias, y, gelu)
        else:
            self.ext.gemm_w8(x2.contiguous(), self.q, self.s, self.bias, y, gelu)
        return y.view(*x.shape[:-1], self.N)
