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
rows = list(csv.DictReader(open(sys.argv[1])))[1:]  # first data row holds the units
agg = collections.defaultdict(list)
for r in rows:
    name = r["Kernel Name"].split("(")[0].split("<")[0]
    agg[(name, r["launch__grid_size"])].append(r)
f = lambda r, k: float(r[k].replace(",", "")) if r.get(k) not in (None, "", "n/a") else float("nan")
print(f"{'kernel':<34}{'grid':>8}{'n':>4}{'us':>9}{'DRAM %':>8}{'mem %':>8}{'SM %':>7}{'GB/s':>8}")
for (name, grid), rs in sorted(agg.items(), key=lambda kv: -sum(f(r, 'gpu__time_duration.sum') for r in kv[1])):
    us = sum(f(r, "gpu__time_duration.sum") for r in rs) / len(rs) / 1e3
    by = sum(f(r, "dram__bytes_read.sum") + f(r, "dram__bytes_write.sum") for r in rs) / len(rs)
    print(f"{name[:33]:<34}{grid:>8}{len(rs):>4}{us:>9.2f}{sum(f(r,'dram__throughput.avg.pct_of_peak_sustained_elapsed') for r in rs)/len(rs):>8.1f}"
          f"{sum(f(r,'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed') for r in rs)/len(rs):>8.1f}"
          f"{sum(f(r,'sm__throughput.avg.pct_of_peak_sustained_elapsed') for r in rs)/len(rs):>7.1f}{by/us/1e3:>8.0f}")
PY
echo "-> $OUT"
