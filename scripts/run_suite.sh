#!/usr/bin/env bash
# Measurement suite: benchmarks + profiling, steps run in priority order under an optional hard deadline.
#
#   bash scripts/run_suite.sh                                  # everything (SUITE=all), no deadline
#   SUITE=slot1 SLOT_MINUTES=105 bash scripts/run_suite.sh     # a GPU-slot preset that must end in 105 min
#   SUITE=slot1 SLOT_END=16:00 bash scripts/run_suite.sh       # ...or stop 12 min (SLOT_MARGIN_MIN) before 16:00
#   STEPS="bench_full torch_prof" bash scripts/run_suite.sh    # explicit steps (in this order)
#   VARIANT=fused_rmsnorm TAG=o2 SUITE=variant bash scripts/run_suite.sh
#
# Deadline logic (for our 2-hour GPU slots, docs/notes/gpu_sessions.md):
#   * a step starts only if its estimated duration fits before the deadline, otherwise it is SKIPPED (logged);
#   * every step runs under `timeout` (2x its estimate, capped by the time left), so nothing can eat the slot;
#   * the end of the log lists DONE / FAILED / SKIPPED steps, to carry over to the next slot.
# Estimates are T4 measurements (minutes); EST_SCALE=0.6 etc. once a faster GPU has been calibrated.
#
# Steps (estimate on T4):
#   bench_full 14  latency/energy grid: 4 workloads x gen {1,32,128,512} x 10 repeats (timing: run alone!)
#   bench_batch 4  batch sweep 1..16 on img_small (timing)
#   numerics 2     fp32 vs default dtype, greedy-output agreement
#   overhead 3     hooks on vs off (instrumentation overhead)
#   torch_prof 3   PyTorch-profiler kernel attribution: text / img_small / img_fhd, 32 tokens
#   nsys 2         Nsight Systems timelines: img_small, img_fhd
#   sdpa 4         which SDPA backend the LM attention uses + backend timings (profiling/sdpa_probe.py)
#   ncu_decode 28  ncu basic+DRAM: one decode step (~800 kernels), img_small
#   ncu_vision 16  ncu: vision tower + projector, img_small
#   ncu_prefill 28 ncu: LM prefill (~800 kernels), img_small
#   ncu_deep 18    ncu full set: first 150 decode kernels
#   ncu_step512 33 ncu: decode step 512 (context ~760)
#   ncu_fhd_decode 30 / ncu_fhd_vision 19 / ncu_fhd_prefill 30   ncu on the 9-tile 1080p input
#   summarize 1    summary.md for each benchmark run
# Backlog (slot presets append it so the GPU never idles; order = value):
#   bench_nohooks 14  full grid with hooks off (headline TPOT)    bench_long 10   1024-token generations (context growth)
#   sdpa_rkv 6        baseline vs sdpa_repeat_kv variant (O5)    bench_batch_big 8  batch 1/16/32/64
#   bench_repeat 14   full grid again (reproducibility)          model_3b 25     LFM2.5-VL-3B baseline (generalization)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/setup_profilers.sh

BACKLOG="ncu_prefill bench_nohooks bench_long sdpa_rkv summarize ncu_fhd_prefill ncu_fhd_vision ncu_fhd_decode ncu_step512 ncu_deep bench_batch_big bench_repeat model_3b summarize"
case "${SUITE:-all}" in
    all)     DEFAULT_STEPS="bench_full bench_batch numerics overhead torch_prof nsys sdpa ncu_decode ncu_vision $BACKLOG" ;;
    backlog) DEFAULT_STEPS="$BACKLOG" ;;
    timing)  DEFAULT_STEPS="bench_full bench_batch numerics overhead summarize" ;;
    profile) DEFAULT_STEPS="torch_prof nsys sdpa" ;;
    variant) DEFAULT_STEPS="bench_full torch_prof nsys summarize" ;;
    # GPU-slot presets: priority order; whatever does not fit is skipped and carried to the next slot
    # slot1 = Eval-1 essentials, then a BACKLOG so the GPU is never idle if the essentials finish early
    # (or if ncu is unavailable and its steps are skipped). Order = value; whatever doesn't fit is skipped.
    slot1)   DEFAULT_STEPS="bench_full bench_batch numerics overhead summarize torch_prof nsys sdpa ncu_decode ncu_vision $BACKLOG" ;;
    slot2)   DEFAULT_STEPS="$BACKLOG" ;;   # whatever slot 1 did not reach; already-done steps can be removed via STEPS=
    *) echo "unknown SUITE=$SUITE"; exit 1 ;;
esac
STEPS="${STEPS:-$DEFAULT_STEPS}"
VARIANT="${VARIANT:-baseline}"
TAG="${TAG:-$VARIANT}"
EST_SCALE="${EST_SCALE:-1.0}"
SUITE_ID="$(date +%Y%m%d-%H%M%S)_${TAG}"
LOG="results/raw/suite_${SUITE_ID}.log"
mkdir -p results/raw profiling/reports
FILTER='falling back to its reference|clean_up_tokenization|_warn_once|acc_events'
START_EPOCH=$(date +%s)
DEADLINE="${DEADLINE:-}"
if [[ -z "$DEADLINE" && -n "${SLOT_END:-}" ]]; then
    # Slot end (HH:MM) = forced logout on the course server: stop scheduling SLOT_MARGIN_MIN (default 12) minutes earlier
    end=$(date -d "$SLOT_END" +%s)
    if (( end < START_EPOCH )); then end=$(( end + 86400 )); fi
    DEADLINE=$(( end - ${SLOT_MARGIN_MIN:-12} * 60 ))
fi
if [[ -z "$DEADLINE" && -n "${SLOT_MINUTES:-}" ]]; then DEADLINE=$(( START_EPOCH + SLOT_MINUTES * 60 )); fi

declare -A EST=([bench_full]=14 [bench_batch]=4 [numerics]=2 [overhead]=3 [torch_prof]=3 [nsys]=2 [sdpa]=4
                [ncu_decode]=28 [ncu_vision]=16 [ncu_prefill]=28 [ncu_deep]=18 [ncu_step512]=33
                [ncu_fhd_decode]=30 [ncu_fhd_vision]=19 [ncu_fhd_prefill]=30 [summarize]=1
                [bench_nohooks]=14 [bench_long]=10 [sdpa_rkv]=6 [bench_batch_big]=8 [bench_repeat]=14 [model_3b]=25)
DONE=(); FAILED=(); SKIPPED=()

say() { echo "[suite $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
x() {  # run one command of the current step under the step's timeout; logs output
    timeout --signal=INT --kill-after=60 "${STEP_LIMIT}m" "$@" 2>&1 | grep -vE "$FILTER" | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    (( rc == 0 )) || STEP_RC=$rc
}

B="python benchmarks/bench.py --variants $VARIANT"
P="python profiling/torch_profile.py --variant $VARIANT --gen-len 32 --tag $TAG"
NCU="bash profiling/ncu/profile_ncu.sh"
step_body() {
    case "$1" in
        bench_full)  x $B --preset full --tag "${TAG}_full" ;;
        bench_batch) x $B --preset batch --tag "${TAG}_batch" ;;
        numerics)    x $B --workloads text,img_small --gen-lens 64 --warmup 1 --repeats 1 --dtype fp32 --tag "${TAG}_fp32"
                     x $B --workloads text,img_small --gen-lens 64 --warmup 1 --repeats 1 --tag "${TAG}_dtype_default" ;;
        overhead)    x $B --workloads img_small --gen-lens 1,128 --repeats 10 --no-phases --tag "${TAG}_nohooks"
                     x $B --workloads img_small --gen-lens 1,128 --repeats 10 --tag "${TAG}_hooks" ;;
        torch_prof)  for wl in text img_small img_fhd; do x $P --workload $wl; done ;;
        nsys)        for wl in img_small img_fhd; do x bash profiling/nsys/profile_nsys.sh $wl 32 --variant "$VARIANT"; done ;;
        sdpa)        x python profiling/sdpa_probe.py ;;
        # ncu caps cover whole phases (torch profiler counts: decode step ~790, vision 430-460 + projector 7-63,
        # prefill 795-845). The "full" section set over a whole step OOMs on 12 GB hosts -> only in ncu_deep.
        ncu_decode)  x env NCU_SET=basic LAUNCHES=800 $NCU decode img_small --variant "$VARIANT" ;;
        ncu_vision)  x env NCU_SET=basic LAUNCHES=600 $NCU vision img_small --variant "$VARIANT" ;;
        ncu_prefill) x env NCU_SET=basic LAUNCHES=1000 NCU_LABEL=full_ $NCU prefill img_small --variant "$VARIANT" ;;
        ncu_deep)    x env NCU_SET=full LAUNCHES=150 NCU_LABEL=deep_ $NCU decode img_small --variant "$VARIANT" ;;
        ncu_step512) x env NCU_SET=basic LAUNCHES=800 GEN=520 NCU_LABEL=step512_ $NCU decode img_small --variant "$VARIANT" \
                         --start-at-decode-step 512 --warmup 1 ;;
        ncu_fhd_decode)  x env NCU_SET=basic LAUNCHES=800 $NCU decode img_fhd --variant "$VARIANT" ;;
        ncu_fhd_vision)  x env NCU_SET=basic LAUNCHES=600 $NCU vision img_fhd --variant "$VARIANT" ;;
        ncu_fhd_prefill) x env NCU_SET=basic LAUNCHES=1000 NCU_LABEL=full_ $NCU prefill img_fhd --variant "$VARIANT" ;;
        # ---- backlog (runs only if the essentials finished early) ----
        bench_nohooks)   x $B --preset full --no-phases --tag "${TAG}_nohooks_full" ;;          # headline TPOT without hooks
        bench_long)      x $B --workloads text,img_small --gen-lens 1,1024 --repeats 5 --tag "${TAG}_long" ;;  # TPOT growth with context
        sdpa_rkv)        x python benchmarks/bench.py --variants baseline,sdpa_repeat_kv --workloads img_small,img_fhd \
                             --gen-lens 1,128 --repeats 5 --tag "${TAG}_sdpa_rkv" ;;           # O5 quick win vs baseline, same process
        bench_batch_big) x $B --workloads img_small --gen-lens 1,128 --batch-sizes 1,16,32,64 --repeats 3 --tag "${TAG}_batch_big" ;;
        bench_repeat)    x $B --preset full --tag "${TAG}_repeat" ;;                             # reproducibility across the slot
        model_3b)        x python benchmarks/bench.py --model LiquidAI/LFM2.5-VL-3B --variants "$VARIANT" --preset full \
                             --repeats 5 --tag "${TAG}_3b" ;;                                   # generalization (downloads 6.2 GB once)
        summarize)
            for d in results/raw/*_"${TAG}"_*/; do
                if [[ -f "$d/runs.jsonl" && ! -f "$d/summary.md" ]]; then x python benchmarks/summarize.py "$d"; fi
            done ;;
        *) say "unknown step $1"; STEP_RC=2 ;;
    esac
}

say "suite $SUITE_ID variant=$VARIANT steps: $STEPS"
if [[ -n "$DEADLINE" ]]; then say "deadline $(date -d @"$DEADLINE" +%H:%M:%S) ($(( (DEADLINE - START_EPOCH) / 60 )) min), EST_SCALE=$EST_SCALE"; fi
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv | tee -a "$LOG"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/[other GPU process] /' | tee -a "$LOG"

# Without GPU performance-counter access every ncu step fails: detect once, skip them.
NCU_OK=1
if [[ " $STEPS " == *" ncu_"* ]]; then
    if ! command -v ncu >/dev/null || ncu --launch-count 1 python -c "import torch; torch.ones(1, device='cuda').add_(1); torch.cuda.synchronize()" 2>&1 | grep -q ERR_NVGPUCTRPERM; then
        NCU_OK=0; say "WARNING: ncu unavailable or no counter access (ERR_NVGPUCTRPERM): ncu_* steps will be skipped"
    fi
fi

for s in $STEPS; do
    if [[ "$s" == ncu_* && $NCU_OK == 0 ]]; then SKIPPED+=("$s(no-ncu)"); continue; fi
    est=$(awk -v e="${EST[$s]:-10}" -v k="$EST_SCALE" 'BEGIN{v=e*k; print (v<1?1:int(v+0.5))}')
    STEP_LIMIT=$(( est * 2 > est + 5 ? est * 2 : est + 5 ))
    if [[ -n "$DEADLINE" ]]; then
        left=$(( (DEADLINE - $(date +%s)) / 60 ))
        if (( est > left )); then say "SKIP  $s (needs ~${est} min, ${left} min left)"; SKIPPED+=("$s"); continue; fi
        (( STEP_LIMIT > left )) && STEP_LIMIT=$left
    fi
    say "START $s (est ${est} min, limit ${STEP_LIMIT} min)"
    t0=$SECONDS; STEP_RC=0
    step_body "$s"
    if (( STEP_RC == 0 )); then DONE+=("$s"); else FAILED+=("$s(rc=$STEP_RC)"); fi
    say "END   $s rc=$STEP_RC ($(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s)"
done

say "DONE:    ${DONE[*]:-none}"
say "FAILED:  ${FAILED[*]:-none}"
say "SKIPPED: ${SKIPPED[*]:-none}"
say "suite $SUITE_ID done; log: $LOG"
