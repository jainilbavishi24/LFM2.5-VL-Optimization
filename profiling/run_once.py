"""One instrumented generate() call, for Nsight Systems / Nsight Compute.

Warm-up runs happen first; the measured request is wrapped in cudaProfilerStart/Stop so
`nsys --capture-range=cudaProfilerApi` and `ncu --profile-from-start off` skip model loading and warm-up.
Phases are NVTX push/pop ranges (vision_tower, projector, lm_prefill, lm_decode, lm_head), so e.g.
`ncu --nvtx --nvtx-include "lm_decode/"` profiles only decode-step kernels.

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
    a = ap.parse_args()

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
        with PhaseInstrumentor(net, modes=("nvtx",)).attached():
            torch.cuda.profiler.start()
            net.generate(**inputs, **gen)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
    print("[run_once] done")


if __name__ == "__main__":
    main()
