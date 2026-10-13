"""Model variants = the baseline plus our optimizations, applied to a loaded model in place.

Every optimization we build registers here, so the same benchmark / profiling scripts
measure it against the baseline with identical inputs:

    @register("fused_shortconv")
    def _(net, processor): ...patch modules...
"""

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


def apply_variant(name: str, net, processor):
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; available: {sorted(VARIANTS)}")
    return VARIANTS[name](net, processor)
