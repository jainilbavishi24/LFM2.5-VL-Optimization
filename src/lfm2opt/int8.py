"""INT8 linear layers for prefill and the vision encoder (kernels/int8_gemm.cu): INT8 weights everywhere, so the
BF16 copies can be dropped (memory for the edge), and INT8 tensor cores where the activations allow it.

  a8  (W8A8):  activations quantized per token on the fly (quantize_rows), INT8 x INT8 tensor-core GEMM
  a16 (W8A16): BF16 activations, INT8 weights converted in registers, BF16 tensor-core GEMM. For the layers whose
               inputs have outlier channels: vision fc2 (after GELU) and LM w2 (after SwiGLU), which cost the most
               accuracy in W8A8 (profiling/int8_quality_probe.py). LFM2_INT8_A16="fc2,w2" (default) picks them.

Weights: symmetric INT8, one scale per output row (lfm2opt.quant.quantize_int8, as the decode GEMV).
"""

import os

import torch

from .kernels import int8_ext
from .quant import quantize_int8


def a16_kinds():
    """Layer kinds that keep BF16 activations (W8A16); the rest run W8A8."""
    return set(filter(None, os.environ.get("LFM2_INT8_A16", "fc2,w2").split(",")))


def quantize_rows(x2):
    """[M, K] bf16 -> (int8 [M, K], fp32 scale [M]), scale = absmax / 127 per row."""
    q = torch.empty(x2.shape, device=x2.device, dtype=torch.int8)
    s = torch.empty(x2.shape[0], device=x2.device, dtype=torch.float32)
    int8_ext().quant_rows(x2, q, s)
    return q, s


class Int8Linear:
    def __init__(self, weight=None, bias=None, a8=True, q=None, scale=None):
        """From a BF16 weight [N, K] (quantized here), or from an already quantized (q, scale) pair."""
        if q is None:
            q, scale = quantize_int8(weight)
        self.q, self.s, self.bias, self.a8 = q, scale, bias, a8
        self.N, self.K = q.shape
        self.ext = int8_ext()

    def __call__(self, x, gelu=False):
        x2 = x.reshape(-1, self.K)
        y = torch.empty(x2.shape[0], self.N, device=x.device, dtype=torch.bfloat16)
        if self.a8:
            xq, sx = quantize_rows(x2.contiguous())
            self.ext.gemm_i8(xq, sx, self.q, self.s, self.bias, y, gelu)
        else:
            self.ext.gemm_w8(x2.contiguous(), self.q, self.s, self.bias, y, gelu)
        return y.view(*x.shape[:-1], self.N)
