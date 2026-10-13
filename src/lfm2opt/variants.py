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


def apply_variant(name: str, net, processor):
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; available: {sorted(VARIANTS)}")
    return VARIANTS[name](net, processor)
