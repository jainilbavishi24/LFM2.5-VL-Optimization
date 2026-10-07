#!/usr/bin/env bash
# Nsight Compute: per-kernel hardware metrics for one phase of a generate() call.
#
#   bash profiling/ncu/profile_ncu.sh <decode|prefill|vision> [workload] [extra run_once.py args...]
#
#   decode  : kernels of decode steps (lm_decode + lm_head ranges), first LAUNCHES kernels (default 600 ~ 1-2 steps)
#   prefill : language-model prefill kernels
#   vision  : SigLIP2 vision tower + projector kernels
#
# Env: NCU_SET (default "full"; "basic" is ~5x faster), LAUNCHES (max kernels to profile),
#      NCU_LABEL (prefix for the phase in the output dir name, e.g. "deep_")
# Needs GPU performance-counter access (root, or NVreg_RestrictProfilingToAdminUsers=0),
# otherwise fails with ERR_NVGPUCTRPERM.
# Writes profiling/reports/<run_id>/{ncu_<phase>.ncu-rep (git-ignored), ncu_<phase>_metrics.csv}
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/setup_profilers.sh" >/dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE="${1:-decode}"; WL="${2:-img_small}"; shift $(( $# > 2 ? 2 : $# ))
case "$PHASE" in
    decode)  RANGES=(--nvtx-include "lm_decode/" --nvtx-include "lm_head/"); GEN=3 ;;
    prefill) RANGES=(--nvtx-include "lm_prefill/" --nvtx-include "lm_head_prefill/"); GEN=1 ;;
    vision)  RANGES=(--nvtx-include "vision_tower/" --nvtx-include "projector/"); GEN=1 ;;
    *) echo "phase must be decode|prefill|vision"; exit 1 ;;
esac
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/Tesla //; s/[^A-Za-z0-9]//g')"
OUT="${OUT_DIR:-$ROOT/profiling/reports/$(date +%Y%m%d-%H%M%S)_${GPU}_ncu_${NCU_LABEL:-}${PHASE}_${WL}}"
mkdir -p "$OUT"
REP="$OUT/ncu_${PHASE}"
# Always collected (also with NCU_SET=basic) and exported to the CSV: time, DRAM traffic / % of peak bandwidth,
# SM throughput, achieved occupancy, launch config.
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread"

ncu --target-processes all --profile-from-start off \
    --nvtx "${RANGES[@]}" \
    --launch-count "${LAUNCHES:-600}" \
    --set "${NCU_SET:-full}" --import-source no --metrics "$METRICS" \
    --force-overwrite -o "$REP" \
    python "$ROOT/profiling/run_once.py" --workload "$WL" --gen-len "$GEN" "$@"

ncu --import "$REP.ncu-rep" --csv --page raw --metrics "$METRICS" > "${REP}_metrics.csv"
echo "[ncu] -> $OUT"
ls -la "$OUT"
