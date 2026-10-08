"""One instrumented generate() call, for Nsight Systems / Nsight Compute.

Warm-up runs happen first; the measured request is wrapped in cudaProfilerStart/Stop so
`nsys --capture-range=cudaProfilerApi` and `ncu --profile-from-start off` skip model loading and warm-up.
Phases are NVTX push/pop ranges (vision_tower, projector, lm_prefill, lm_decode, lm_head), so e.g.
`ncu --nvtx --nvtx-include "lm_decode/"` profiles only decode-step kernels.

`--start-at-decode-step N` starts the capture only at the N-th decode step (context = prompt + N tokens),
to profile late, long-context decode steps without capturing everything before them.

Used by profiling/nsys/profile_nsys.sh and profiling/ncu/profile_ncu.sh.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm2opt.instrument import PhaseInstrumentor  # noqa: E402
from lfm2opt.model import DEFAULT_MODEL, build_inputs, load  # noqa: E402
from lfm2opt.variants import apply_variant  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--workload", default="img_small", choices=list(WORKLOADS))
    ap.add_argument("--gen-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--start-at-decode-step", type=int, default=0,
                    help="start profiling at this decode step (0 = from the beginning of the request)")
    a = ap.parse_args()
    if a.start_at_decode_step >= a.gen_len:
        ap.error("--start-at-decode-step must be < --gen-len")

    net, processor = load(a.model, a.dtype, a.attn)
    net = apply_variant(a.variant, net, processor)
    wl = WORKLOADS[a.workload]
    inputs = build_inputs(processor, wl.image(), wl.prompt, a.batch_size)
    gen = dict(max_new_tokens=a.gen_len, min_new_tokens=a.gen_len, do_sample=False,
               pad_token_id=processor.tokenizer.pad_token_id)

    with torch.inference_mode():
        for _ in range(a.warmup):
            net.generate(**inputs, **gen)
        torch.cuda.synchronize()
        handle = None
        if a.start_at_decode_step > 0:
            # LM call 0 is the prefill, call k is decode step k. Registered before the NVTX hooks so the
            # profiler is already running when that step's range is pushed.
            calls = {"n": 0}

            def start_late(module, args):
                if calls["n"] == a.start_at_decode_step:
                    torch.cuda.synchronize()
                    torch.cuda.profiler.start()
                calls["n"] += 1

            handle = net.model.language_model.register_forward_pre_hook(start_late)
        with PhaseInstrumentor(net, modes=("nvtx",)).attached():
            if handle is None:
                torch.cuda.profiler.start()
            net.generate(**inputs, **gen)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        if handle is not None:
            handle.remove()
    print("[run_once] done")


if __name__ == "__main__":
    main()
