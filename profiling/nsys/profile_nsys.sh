#!/usr/bin/env bash
# Nsight Systems timeline of one generate() call (model load + warm-up excluded).
#
#   bash profiling/nsys/profile_nsys.sh [workload] [gen_len] [extra run_once.py args...]
#   bash profiling/nsys/profile_nsys.sh img_small 32
#
# Writes profiling/reports/<run_id>/{*.nsys-rep, *_cuda_gpu_kern_sum.csv, *_nvtx_sum.csv, *_cuda_api_sum.csv}
# The .nsys-rep (git-ignored) opens in the Nsight Systems GUI on any laptop, no GPU needed.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/setup_profilers.sh" >/dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WL="${1:-img_small}"; GEN="${2:-32}"; shift $(( $# > 2 ? 2 : $# ))
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/Tesla //; s/[^A-Za-z0-9]//g')"
OUT="${OUT_DIR:-$ROOT/profiling/reports/$(date +%Y%m%d-%H%M%S)_${GPU}_nsys_${WL}_g${GEN}}"
mkdir -p "$OUT"
REP="$OUT/nsys_${WL}_g${GEN}"

nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o "$REP" \
    python "$ROOT/profiling/run_once.py" --workload "$WL" --gen-len "$GEN" "$@"

nsys stats --force-export=true --format csv \
    --report cuda_gpu_kern_sum --report nvtx_sum --report cuda_api_sum \
    --output "$REP" "$REP.nsys-rep" > /dev/null
echo "[nsys] -> $OUT"
ls -la "$OUT"
