#!/usr/bin/env bash
# Nsight Compute on our own CUDA kernels only (kernels/*.cu), for any command:
#
#   bash profiling/ncu/profile_ours.sh <label> <command...>
#   e.g. bash profiling/ncu/profile_ours.sh decode .venv-ref/bin/python profiling/profile_decode_step.py --fused --gemv bf16 --steps 2
#
# PDL is switched off (LFM2_PDL=0) so kernel durations do not overlap. Env: NCU_SET (default "full"), LAUNCHES (200).
# Writes profiling/reports/<run_id>/{ncu_ours.ncu-rep (git-ignored), ncu_ours_metrics.csv}
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/setup_profilers.sh" >/dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL="$1"; shift
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/[^A-Za-z0-9]//g')"
OUT="$ROOT/profiling/reports/$(date +%Y%m%d-%H%M%S)_${GPU}_ncu_ours_${LABEL}"
mkdir -p "$OUT"
KERNELS='regex:gemv|add_rmsnorm|shortconv|qk_norm_rope|decode_attention|attn_combine|silu_mul|resize_|add_layernorm'
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__registers_per_thread"
LFM2_PDL=0 ncu --target-processes all --kernel-name "$KERNELS" --launch-count "${LAUNCHES:-200}" \
    --set "${NCU_SET:-full}" --import-source no --metrics "$METRICS" --force-overwrite -o "$OUT/ncu_ours" "$@"
ncu --import "$OUT/ncu_ours.ncu-rep" --csv --page raw --metrics "$METRICS" > "$OUT/ncu_ours_metrics.csv"
python3 - "$OUT/ncu_ours_metrics.csv" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
units, rows = rows[0], rows[1:]                       # first data row holds the units
to_us = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}.get(units["gpu__time_duration.sum"], 1.0)
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["Kernel Name"].split("(")[0].replace("void ", ""), r["launch__grid_size"])].append(r)
avg = lambda rs, k: sum(float(r[k].replace(",", "")) for r in rs) / len(rs)
print(f"{'kernel':<58}{'grid':>8}{'n':>5}{'us':>9}{'DRAM %':>8}{'mem %':>8}{'SM %':>7}{'occ %':>7}")
for (name, grid), rs in sorted(agg.items(), key=lambda kv: -avg(kv[1], "gpu__time_duration.sum") * len(kv[1])):
    print(f"{name[:57]:<58}{grid:>8}{len(rs):>5}{avg(rs, 'gpu__time_duration.sum') * to_us:>9.2f}"
          f"{avg(rs, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'):>8.1f}"
          f"{avg(rs, 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'):>8.1f}"
          f"{avg(rs, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):>7.1f}"
          f"{avg(rs, 'sm__warps_active.avg.pct_of_peak_sustained_active'):>7.1f}")
PY
echo "-> $OUT"
