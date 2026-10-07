"""Kernel-level breakdown of one generate() call with the PyTorch profiler (works on any GPU, no root needed).

Every GPU kernel is attributed to
  * a phase:    vision_tower / projector / lm_prefill / lm_head_prefill / lm_decode / lm_head / other (embedding, sampling, glue)
                via the CPU-side launch inside the hook-based record_function ranges (lfm2opt.instrument)
  * a category: gemm, attention, conv, norm_reduce, elementwise, copy_index, other  (kernel-name regex)

Outputs in profiling/reports/<run_id>/:
  trace.json          chrome://tracing / Perfetto timeline
  ops_table.txt       torch key_averages, top ops by GPU time
  kernels.csv         per (phase, category, launching aten op, kernel name): count, total/avg time
  breakdown.csv       phase x category totals
  summary.json        per-decode-step stats: #kernels, kernel time, wall time, GPU-busy fraction

  python profiling/torch_profile.py --workload img_small --gen-len 32
"""

import argparse
import bisect
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm2opt.env import REPO_ROOT, collect_env, gpu_short_name  # noqa: E402
from lfm2opt.instrument import PhaseInstrumentor  # noqa: E402
from lfm2opt.model import DEFAULT_MODEL, build_inputs, describe_inputs, load  # noqa: E402
from lfm2opt.variants import apply_variant  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402

PHASES = ("vision_tower", "projector", "lm_prefill", "lm_head_prefill", "lm_decode", "lm_head")

CATEGORIES = [  # first match wins
    ("attention", r"flash|fmha|attention|attn|softmax"),
    ("gemm", r"gemm|gemv|cutlass|cublas|xmma|s1688|s16816|h1688|h884|wgmma|splitK|matmul|dot_kernel"),
    ("conv", r"conv"),
    ("norm_reduce", r"norm|reduce|Reduce"),
    ("copy_index", r"[Cc]opy|[Cc]at|[Ii]ndex|gather|scatter|fill|memcpy|memset|Memcpy|Memset|arange|where|nonzero"),
    ("elementwise", r"elementwise|vectorized|unrolled|silu|gelu|mul|add|pow|rsqrt|exp|neg"),
]


def categorize(name: str) -> str:
    for cat, pat in CATEGORIES:
        if re.search(pat, name):
            return cat
    return "other"


def analyze_trace(trace_path: Path, out_dir: Path) -> dict:
    events = json.loads(trace_path.read_text())["traceEvents"]
    launches = {}  # correlation id -> CPU launch timestamp
    annotations = []  # (start, end, phase)
    gen_range = None
    kernels = []
    cpu_ops = []  # (start, end, name): to find which aten op launched each kernel
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e["ts"]
        elif cat == "user_annotation":
            if e["name"] in PHASES:
                annotations.append((e["ts"], e["ts"] + e["dur"], e["name"]))
            elif e["name"] == "generate":
                gen_range = (e["ts"], e["ts"] + e["dur"])
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(e)
        elif cat == "cpu_op":
            cpu_ops.append((e["ts"], e["ts"] + e["dur"], e["name"]))

    annotations.sort()
    starts = [a[0] for a in annotations]

    cpu_ops.sort()
    op_starts = [o[0] for o in cpu_ops]

    def op_of(ts):
        """Innermost aten op whose CPU range contains the launch (latest-starting enclosing range)."""
        i = bisect.bisect_right(op_starts, ts) - 1
        for j in range(i, max(i - 64, -1), -1):
            s, t_end, name = cpu_ops[j]
            if s <= ts <= t_end:
                return name
        return "?"

    def phase_of(ts):
        i = bisect.bisect_right(starts, ts) - 1
        while i >= 0:  # innermost enclosing range (ranges here do not overlap, but be safe)
            s, t_end, name = annotations[i]
            if s <= ts <= t_end:
                return name
            i -= 1
        return "other"

    per_kernel = defaultdict(lambda: [0, 0.0])
    breakdown = defaultdict(lambda: [0, 0.0])
    decode_kernel_us, decode_kernel_n = 0.0, 0
    first_decode = min((a[0] for a in annotations if a[2] == "lm_decode"), default=None)
    for k in kernels:
        launch_ts = launches.get(k.get("args", {}).get("correlation"))
        phase = phase_of(launch_ts) if launch_ts is not None else "other"
        cat = categorize(k["name"])
        op = op_of(launch_ts) if launch_ts is not None else "?"
        per_kernel[(phase, cat, op, k["name"])][0] += 1
        per_kernel[(phase, cat, op, k["name"])][1] += k["dur"]
        breakdown[(phase, cat)][0] += 1
        breakdown[(phase, cat)][1] += k["dur"]
        if first_decode is not None and launch_ts is not None and launch_ts >= first_decode:
            decode_kernel_us += k["dur"]
            decode_kernel_n += 1

    with open(out_dir / "kernels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "category", "aten_op", "kernel", "count", "total_us", "avg_us"])
        for (ph, cat, op, name), (n, us) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1]):
            w.writerow([ph, cat, op, name[:160], n, round(us, 2), round(us / n, 3)])
    with open(out_dir / "breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "category", "count", "total_us"])
        for (ph, cat), (n, us) in sorted(breakdown.items()):
            w.writerow([ph, cat, n, round(us, 2)])

    n_decode = sum(1 for a in annotations if a[2] == "lm_decode")
    summary = {"n_kernels_total": len(kernels), "n_decode_steps": n_decode}
    if gen_range:
        summary["generate_wall_ms"] = (gen_range[1] - gen_range[0]) / 1e3
    if n_decode and gen_range:
        decode_wall_us = gen_range[1] - first_decode
        summary.update(
            decode_wall_ms_per_step=decode_wall_us / n_decode / 1e3,
            decode_kernel_ms_per_step=decode_kernel_us / n_decode / 1e3,
            decode_kernels_per_step=decode_kernel_n / n_decode,
            decode_gpu_busy_frac=decode_kernel_us / decode_wall_us,
        )
    phase_tot = defaultdict(float)
    cat_decode = defaultdict(float)
    for (ph, cat), (n, us) in breakdown.items():
        phase_tot[ph] += us / 1e3
        if ph == "lm_decode":
            cat_decode[cat] += us / 1e3
    summary["kernel_ms_by_phase"] = dict(sorted(phase_tot.items()))
    tot_dec = sum(cat_decode.values()) or 1.0
    summary["lm_decode_kernel_share_by_category"] = {c: round(v / tot_dec, 4) for c, v in
                                                      sorted(cat_decode.items(), key=lambda kv: -kv[1])}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--workload", default="img_small", choices=list(WORKLOADS))
    ap.add_argument("--gen-len", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(REPO_ROOT / "profiling" / "reports"))
    a = ap.parse_args()

    run_id = (f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}_torchprof_{a.variant}_{a.workload}_g{a.gen_len}"
              + (f"_{a.tag}" if a.tag else ""))
    out_dir = Path(a.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(collect_env(), indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(a), indent=2))

    net, processor = load(a.model, a.dtype, a.attn)
    net = apply_variant(a.variant, net, processor)
    wl = WORKLOADS[a.workload]
    inputs = build_inputs(processor, wl.image(), wl.prompt, a.batch_size)
    pad_id = processor.tokenizer.pad_token_id
    gen = dict(max_new_tokens=a.gen_len, min_new_tokens=a.gen_len, do_sample=False, pad_token_id=pad_id)

    with torch.inference_mode():
        for _ in range(a.warmup):
            net.generate(**inputs, **gen)
        torch.cuda.synchronize()
        instr = PhaseInstrumentor(net, modes=("record_function",))
        with instr.attached(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("generate"):
                net.generate(**inputs, **gen)
            torch.cuda.synchronize()

    t = time.perf_counter()
    trace = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace))
    sort_key = "self_device_time_total" if hasattr(prof.key_averages()[0], "self_device_time_total") else "self_cuda_time_total"
    (out_dir / "ops_table.txt").write_text(prof.key_averages().table(sort_by=sort_key, row_limit=60))
    summary = analyze_trace(trace, out_dir)
    summary.update(workload=a.workload, gen_len=a.gen_len, variant=a.variant,
                   **describe_inputs(inputs, net.config.image_token_id))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[torch_profile] analysis {time.perf_counter() - t:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
