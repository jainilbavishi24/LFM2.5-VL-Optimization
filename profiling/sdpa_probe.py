"""Which attention path does LFM2's language model actually take on this GPU, and what would the alternatives cost?

Answers plan questions Q5.1 / Q5.2 / O5a (docs/notes/optimization_plan.md) in a few minutes:
  1. Inside real generate(): how often transformers passes an attention mask to SDPA and whether it uses
     enable_gqa or repeat_kv (patched `use_gqa_in_sdpa`), for prefill and decode.
  2. Standalone, at the model's real shapes (32 query / 8 KV heads, head dim 64): for each SDPA backend
     (flash, efficient, cudnn, math) x {mask, no mask} x {enable_gqa, repeat_kv}: does it run, and how long?
     Plus which backend PyTorch picks by default (kernel names from the profiler).

  python profiling/sdpa_probe.py               # writes results/raw/<time>_<gpu>_sdpa_probe.json
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm2opt.env import REPO_ROOT, collect_env, gpu_short_name  # noqa: E402
from lfm2opt.model import build_inputs, load, resolve_dtype  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402

H_Q, H_KV, D = 32, 8, 64
BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH}


def in_model_usage(dtype: str) -> dict:
    """Count mask / GQA decisions transformers makes during one real generate()."""
    import transformers.integrations.sdpa_attention as sa

    calls = Counter()
    orig = sa.use_gqa_in_sdpa

    def spy(attention_mask, key, value):
        res = orig(attention_mask, key, value)
        phase = "prefill" if calls["n"] < 6 else "decode"  # first call of each of the 6 attention layers
        calls["n"] += 1
        calls[f"{phase}: mask={'none' if attention_mask is None else tuple(attention_mask.shape)[-2:]} "
              f"-> {'enable_gqa' if res else 'repeat_kv'}"] += 1
        return res

    sa.use_gqa_in_sdpa = spy
    try:
        net, processor = load(dtype=dtype)
        wl = WORKLOADS["img_small"]
        inputs = build_inputs(processor, wl.image(), wl.prompt)
        with torch.inference_mode():
            net.generate(**inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False,
                         pad_token_id=processor.tokenizer.pad_token_id)
        del net
        torch.cuda.empty_cache()
    finally:
        sa.use_gqa_in_sdpa = orig
    calls.pop("n", None)
    return dict(calls)


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def standalone(dtype: torch.dtype) -> list:
    rows = []
    for phase, q_len, kv_len in [("decode", 1, 256), ("decode", 1, 2304), ("prefill", 252, 252), ("prefill", 2327, 2327)]:
        q = torch.randn(1, H_Q, q_len, D, device="cuda", dtype=dtype)
        k = torch.randn(1, H_KV, kv_len, D, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        causal = torch.ones(q_len, kv_len, device="cuda", dtype=torch.bool).tril(diagonal=kv_len - q_len)
        mask = causal[None, None]
        for gqa_mode in ("enable_gqa", "repeat_kv"):
            kk, vv = (k, v) if gqa_mode == "enable_gqa" else (k.repeat_interleave(4, 1), v.repeat_interleave(4, 1))
            kw = {"enable_gqa": True} if gqa_mode == "enable_gqa" else {}
            for mask_mode in ("mask", "is_causal" if q_len > 1 else "none"):
                args = dict(attn_mask=mask) if mask_mode == "mask" else dict(is_causal=(mask_mode == "is_causal"))
                def call(backends=None):
                    if backends is None:
                        return F.scaled_dot_product_attention(q, kk, vv, **args, **kw)
                    with sdpa_kernel(backends):
                        return F.scaled_dot_product_attention(q, kk, vv, **args, **kw)
                row = dict(phase=phase, q_len=q_len, kv_len=kv_len, gqa=gqa_mode, mask=mask_mode)
                # default dispatch: which kernels run?
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    call(); torch.cuda.synchronize()
                row["default_kernels"] = sorted({e.name[:60] for e in prof.events() if e.device_type.name == "CUDA"})[:6]
                row["default_us"] = bench(lambda: call())
                for name, be in BACKENDS.items():
                    try:
                        row[f"{name}_us"] = round(bench(lambda: call([be])), 1)
                    except RuntimeError as e:
                        row[f"{name}_us"] = f"unsupported: {str(e).splitlines()[0][:80]}"
                rows.append(row)
                print(json.dumps(row))
    return rows


def main():
    dtype = resolve_dtype("auto")
    out = {"env": collect_env(), "dtype": str(dtype)}
    out["in_model"] = in_model_usage("auto")
    print("in-model SDPA decisions:", json.dumps(out["in_model"], indent=2))
    out["standalone"] = standalone(dtype)
    path = REPO_ROOT / "results" / "raw" / f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}_sdpa_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"[sdpa_probe] -> {path}")


if __name__ == "__main__":
    main()
