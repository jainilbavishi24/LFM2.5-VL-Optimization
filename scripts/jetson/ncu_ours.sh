#!/usr/bin/env bash
# Nsight Compute on our kernels on the Jetson. Counter access needs root there; the board has a sudo rule for ncu only:
#   echo "gautam ALL=(ALL) NOPASSWD: /usr/local/cuda/bin/ncu" | sudo tee /etc/sudoers.d/paa-ncu   (remove: sudo rm ...)
#
#   bash scripts/jetson/ncu_ours.sh <label> <run_once.py args...>
#   bash scripts/jetson/ncu_ours.sh bf16_vga --variant fused_gemv_pf_vfused --workload img_small --gen-len 3 --warmup 1
#
# On the Orin (iGPU) the throughput counters read 0 and dram__ ones do not exist; kernel times are valid.
# The profiled Python runs as root but leaves nothing root-owned in the project: no .pyc files, HOME and the JIT kernel
# builds in /tmp copies. PDL is off so kernel times do not overlap. Writes profiling/reports/<run>/ncu_ours_metrics.csv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LABEL="$1"; shift
OUT="profiling/reports/$(date +%Y%m%d-%H%M%S)_Orin_ncu_ours_${LABEL}"
mkdir -p "$OUT"
EXT=/tmp/paa_root_ext; RHOME=/tmp/paa_root_home
rm -rf "$EXT" && cp -r "$HOME/.cache/torch_extensions" "$EXT" && mkdir -p "$RHOME"
KERNELS='regex:gemv|add_rmsnorm|shortconv|qk_norm_rope|decode_attention|attn_combine|silu_mul|resize_|add_layernorm'
METRICS="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__registers_per_thread"
REP="/tmp/paa_ncu_${LABEL}"
sudo -n /usr/local/cuda/bin/ncu --target-processes all --kernel-name "$KERNELS" --launch-count "${LAUNCHES:-300}" \
    --set "${NCU_SET:-basic}" --import-source no --metrics "$METRICS" --force-overwrite -o "$REP" \
    env HOME="$RHOME" TORCH_EXTENSIONS_DIR="$EXT" PYTHONDONTWRITEBYTECODE=1 LFM2_PDL=0 PYTHONWARNINGS=ignore \
        TRANSFORMERS_VERBOSITY=error PATH="$PWD/.venv-ref/bin:/usr/local/cuda/bin:/usr/bin:/bin" CUDA_HOME=/usr/local/cuda \
        "$PWD/.venv-ref/bin/python" profiling/run_once.py "$@"
/usr/local/cuda/bin/ncu --import "$REP.ncu-rep" --csv --page raw --metrics "$METRICS" > "$OUT/ncu_ours_metrics.csv"
cp "$REP.ncu-rep" "$OUT/ncu_ours.ncu-rep"
.venv-ref/bin/python - "$OUT/ncu_ours_metrics.csv" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
units, rows = rows[0], rows[1:]
to_us = {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}.get(units["gpu__time_duration.sum"], 1.0)
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["Kernel Name"].split("(")[0].replace("void ", ""), r["launch__grid_size"])].append(r)
avg = lambda rs, k: sum(float((r.get(k) or "0").replace(",", "")) for r in rs) / len(rs)  # Orin: no dram__ counters
print(f"{'kernel':<58}{'grid':>8}{'n':>5}{'us':>9}{'DRAM %':>8}{'mem %':>8}{'SM %':>7}{'occ %':>7}")
for (name, grid), rs in sorted(agg.items(), key=lambda kv: -avg(kv[1], "gpu__time_duration.sum") * len(kv[1])):
    print(f"{name[:57]:<58}{grid:>8}{len(rs):>5}{avg(rs, 'gpu__time_duration.sum') * to_us:>9.2f}"
          f"{avg(rs, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'):>8.1f}"
          f"{avg(rs, 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'):>8.1f}"
          f"{avg(rs, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):>7.1f}"
          f"{avg(rs, 'sm__warps_active.avg.pct_of_peak_sustained_active'):>7.1f}")
PY
echo "-> $OUT"
