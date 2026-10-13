"""Model variants = the baseline plus our optimizations, applied to a loaded model in place.

Every optimization we build registers here, so the same benchmark / profiling scripts
measure it against the baseline with identical inputs:

    @register("fused_shortconv")
    def _(net, processor): ...patch modules...
"""

import torch

VARIANTS = {}


def register(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn

    return deco


@register("baseline")
def _baseline(net, processor):
    """Hugging Face transformers as shipped (eager PyTorch, SDPA attention)."""
    return net


@register("sdpa_repeat_kv")
def _sdpa_repeat_kv(net, processor):
    """O5 quick win: expand K/V to all query heads (repeat_kv) instead of SDPA's enable_gqa=True.

    On GPUs without a GQA-capable fused SDPA kernel (T4), enable_gqa forces the FP32 math backend;
    with repeated K/V the memory-efficient / flash kernels can run (sdpa_probe: 3.6-23x faster attention).
    Note: patches transformers globally for this process, so run this variant after `baseline`.
    """
    import transformers.integrations.sdpa_attention as sa

    sa.use_gqa_in_sdpa = lambda attention_mask, key, value: False
    return net


@register("compile_static")
def _compile_static(net, processor):
    """O0b reference: existing switches only = HF static cache + torch.compile(mode="reduce-overhead").

    generate() then auto-compiles the decode forward (prefill stays eager) and replays it as a CUDA graph.
    Each distinct cache length (prompt + max_new_tokens) is a new graph, so keep the grid small and
    warm up >= 2 times; run with --no-phases (forward hooks would break the compiled graph).
    """
    import torch
    from transformers import CompileConfig

    for knob in ("recompile_limit", "cache_size_limit"):  # name changed across torch versions
        if hasattr(torch._dynamo.config, knob):
            setattr(torch._dynamo.config, knob, 64)
    net.generation_config.cache_implementation = "static"
    net.generation_config.compile_config = CompileConfig(fullgraph=False, mode="reduce-overhead", dynamic=False)
    return net


@register("cudagraph")
def _cudagraph(net, processor):
    """O1: HF prefill, then our static decode loop replayed as one CUDA graph per token (src/lfm2opt/runtime.py).

    Greedy and fixed-length only (what the benchmark measures). Run with --no-phases: decode no longer calls
    the HF language-model module, so the phase hooks would only see the prefill.
    """
    from .runtime import StaticDecoder

    dec = StaticDecoder(net)

    def generate(input_ids=None, attention_mask=None, max_new_tokens=1, min_new_tokens=None, do_sample=False,
                 pad_token_id=None, **model_inputs):
        assert not do_sample, "cudagraph runtime is greedy only"
        return dec.generate(input_ids, max_new_tokens, attention_mask=attention_mask, **model_inputs)

    net.generate = generate
    net._static_decoder = dec
    return net


def _fused_variant(net, gemv):
    """O1 + O2-O4: the CUDA-graph decode runtime with our fused kernels (kernels/lfm2_decode.cu) and merged
    q|k|v and w1|w3 projections: ~140 kernels per step instead of ~700. Needs the CUDA extension (.venv-ref)."""
    from .runtime import StaticDecoder

    dec = StaticDecoder(net, fused=True, gemv=gemv)

    def generate(input_ids=None, attention_mask=None, max_new_tokens=1, min_new_tokens=None, do_sample=False,
                 pad_token_id=None, **model_inputs):
        assert not do_sample, "fused runtime is greedy only"
        return dec.generate(input_ids, max_new_tokens, attention_mask=attention_mask, **model_inputs)

    net.generate = generate
    net._static_decoder = dec
    return net


@register("fused")
def _fused(net, processor):
    """O1 + O2-O4 with cuBLAS GEMVs."""
    return _fused_variant(net, "cublas")


@register("fused_gemv")
def _fused_gemv(net, processor):
    """O1-O4 + our BF16 GEMV kernel (batch 1)."""
    return _fused_variant(net, "bf16")


@register("fused_w8")
def _fused_w8(net, processor):
    """O1-O4 + O6: INT8 weight-only (per-row scale) GEMV for every decode projection incl. lm_head (batch 1)."""
    return _fused_variant(net, "int8")


@register("fused_w4")
def _fused_w4(net, processor):
    """O1-O4 + O6b: INT4 weight-only (group-128 scales) GEMV for every decode projection incl. lm_head (batch 1)."""
    return _fused_variant(net, "int4")


def _vision_sdpa(net, backends):
    """O7a: run the SigLIP2 vision tower's SDPA calls with a preferred backend order (HF's default picks the
    sm80 memory-efficient kernel on Blackwell: ~1.1 ms per layer at 1080p, profiling/vision_attn_probe.py)."""
    from torch.nn.attention import sdpa_kernel

    vt = net.model.vision_tower
    orig = vt.forward

    def forward(*args, **kwargs):
        with sdpa_kernel(backends, set_priority=True):
            return orig(*args, **kwargs)

    vt.forward = forward
    return net


@register("vision_cudnn")
def _vision_cudnn(net, processor):
    """O7a: SigLIP2 attention via cuDNN (Blackwell-native) first, then flash / efficient / math."""
    from torch.nn.attention import SDPBackend

    return _vision_sdpa(net, [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                              SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


class FP8Linear(torch.nn.Module):
    """W8A8 FP8 (E4M3) linear for prefill-size inputs: weight scaled per output channel (once), activation per
    token (dynamic), product on FP8 tensor cores via torch._scaled_mm with row-wise scales, BF16 output."""

    FP8_MAX = 448.0

    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        sw = (w.abs().amax(1) / self.FP8_MAX).clamp_min(1e-12)                     # [N]
        self.register_buffer("wq", (w / sw[:, None]).to(torch.float8_e4m3fn).contiguous())
        self.register_buffer("sw", sw.view(1, -1).contiguous())
        self.bias = lin.bias
        self.out_features = lin.out_features

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        sx = (x2.float().abs().amax(1, keepdim=True) / self.FP8_MAX).clamp_min(1e-12)  # [M, 1]
        xq = (x2.float() / sx).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.wq.t(), scale_a=sx, scale_b=self.sw, bias=self.bias, out_dtype=x.dtype)
        return y.view(*shape[:-1], self.out_features)


def _vision_fp8(net):
    """O7b: every nn.Linear inside the SigLIP2 encoder layers -> FP8Linear (attention projections + MLP)."""
    layers = [m for m in net.model.vision_tower.modules() if type(m).__name__ == "Siglip2EncoderLayer"]
    assert layers, "no SigLIP2 encoder layers found"
    for layer in layers:
        for name, mod in list(layer.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(mod, cname, FP8Linear(child))
    return net


@register("vision_fp8")
def _vision_fp8_variant(net, processor):
    """O7b: FP8 W8A8 GEMMs in the vision encoder (BF16 elsewhere)."""
    return _vision_fp8(net)


@register("fused_w8_vfast")
def _fused_w8_vfast(net, processor):
    """O1-O6 (INT8 decode) + O7a (cuDNN vision attention) + O7b (FP8 vision GEMMs)."""
    _vision_cudnn(net, processor)
    _vision_fp8(net)
    return _fused_variant(net, "int8")


@register("fused_w8_vcudnn")
def _fused_w8_vcudnn(net, processor):
    """O1-O6 (INT8 decode) + O7a (cuDNN vision attention): the current best end-to-end configuration."""
    _vision_cudnn(net, processor)
    return _fused_variant(net, "int8")


def apply_variant(name: str, net, processor):
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; available: {sorted(VARIANTS)}")
    return VARIANTS[name](net, processor)
