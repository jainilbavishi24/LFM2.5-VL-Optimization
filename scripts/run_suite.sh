#!/usr/bin/env bash
# Full measurement suite for one GPU: benchmarks + profiling, one command, resumable by step.
#
#   bash scripts/run_suite.sh                          # everything
#   STEPS="bench_full torch_prof" bash scripts/run_suite.sh   # only some steps
#   VARIANT=fused_shortconv TAG=opt1 bash scripts/run_suite.sh  # same suite for an optimization
#
# Steps (in order):
#   bench_full    latency/energy grid: 4 workloads x gen {1,32,128,512} x 10 repeats      (~15 min on T4)
#   bench_batch   batch sweep 1..16 on img_small                                          (~5 min)
#   numerics      same requests in fp32 vs default dtype -> greedy-output agreement        (~3 min)
#   overhead      hooks on vs off, to bound the instrumentation overhead                  (~2 min)
#   torch_prof    PyTorch-profiler kernel breakdown: text / img_small / img_fhd, 32 tokens (~3 min)
#   nsys          Nsight Systems timelines: img_small and img_fhd, 32 tokens               (~2 min)
#   ncu           Nsight Compute, basic set + DRAM metrics: 1 decode step, vision, prefill  (~40 min on T4)
#   ncu_deep      Nsight Compute, full set: first ~150 decode kernels (conv + attn layers)  (~20 min on T4)
#   summarize     summary.md for every bench run of this suite
#
# Everything lands in results/raw/ and profiling/reports/; the log in results/raw/suite_<id>.log
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/setup_profilers.sh

STEPS="${STEPS:-bench_full bench_batch numerics overhead torch_prof nsys ncu ncu_deep summarize}"
VARIANT="${VARIANT:-baseline}"
TAG="${TAG:-$VARIANT}"
SUITE_ID="$(date +%Y%m%d-%H%M%S)_${TAG}"
LOG="results/raw/suite_${SUITE_ID}.log"
mkdir -p results/raw profiling/reports
FILTER='falling back to its reference|clean_up_tokenization|_warn_once|acc_events'

say() { echo "[suite $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() {  # run <step-name> <command...>
    local name="$1"; shift
    [[ " $STEPS " == *" $name "* ]] || return 0
    say "START $name: $*"
    local t0=$SECONDS
    "$@" 2>&1 | grep -vE "$FILTER" | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    say "END   $name rc=$rc ($((SECONDS - t0)) s)"
}

say "suite $SUITE_ID variant=$VARIANT steps: $STEPS"
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv | tee -a "$LOG"

B="python benchmarks/bench.py --variants $VARIANT"
run bench_full  $B --preset full  --tag "${TAG}_full"
run bench_batch $B --preset batch --tag "${TAG}_batch"
run numerics    $B --workloads text,img_small --gen-lens 64 --warmup 1 --repeats 1 --dtype fp32 --tag "${TAG}_fp32"
run numerics    $B --workloads text,img_small --gen-lens 64 --warmup 1 --repeats 1 --tag "${TAG}_dtype_default"
run overhead    $B --workloads img_small --gen-lens 1,128 --repeats 10 --no-phases --tag "${TAG}_nohooks"
run overhead    $B --workloads img_small --gen-lens 1,128 --repeats 10 --tag "${TAG}_hooks"

P="python profiling/torch_profile.py --variant $VARIANT --gen-len 32 --tag $TAG"
run torch_prof $P --workload text
run torch_prof $P --workload img_small
run torch_prof $P --workload img_fhd

run nsys bash profiling/nsys/profile_nsys.sh img_small 32 --variant "$VARIANT"
run nsys bash profiling/nsys/profile_nsys.sh img_fhd 32 --variant "$VARIANT"

# ncu: "basic" set + our DRAM/SM metrics over whole phases (one decode step is ~800 kernels in the baseline).
# The "full" set over 800 kernels needs >9 GB host RAM and >75 min on a T4 (OOM-killed on Colab), so the
# full set is only used in ncu_deep, on the first decode kernels (= 2 conv layers + 1 attention layer).
run ncu env NCU_SET=basic LAUNCHES="${NCU_DECODE_LAUNCHES:-800}" bash profiling/ncu/profile_ncu.sh decode img_small --variant "$VARIANT"
run ncu env NCU_SET=basic LAUNCHES="${NCU_VISION_LAUNCHES:-500}" bash profiling/ncu/profile_ncu.sh vision img_small --variant "$VARIANT"
run ncu env NCU_SET=basic LAUNCHES="${NCU_PREFILL_LAUNCHES:-500}" bash profiling/ncu/profile_ncu.sh prefill img_small --variant "$VARIANT"
run ncu_deep env NCU_SET=full LAUNCHES="${NCU_DEEP_LAUNCHES:-150}" NCU_LABEL=deep_ \
    bash profiling/ncu/profile_ncu.sh decode img_small --variant "$VARIANT"

if [[ " $STEPS " == *" summarize "* ]]; then
    for d in results/raw/*_"${TAG}"_*/; do
        [[ -f "$d/runs.jsonl" && ! -f "$d/summary.md" ]] && python benchmarks/summarize.py "$d" > /dev/null
    done
    say "summaries written"
fi
say "suite $SUITE_ID done; log: $LOG"
