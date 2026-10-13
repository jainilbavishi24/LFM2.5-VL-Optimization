#!/usr/bin/env bash
# Unattended GPU-slot plan (runs ON the server): reference baselines O0a-O0d + carry-over, under one deadline.
#
#   DEADLINE=<unix epoch> [CARRY="steps"] bash scripts/remote/plan_refs.sh
#
# Items run in priority order. Each one is skipped if already done (marker in results/raw/plan_done/) or not
# ready yet (its venv / build / download is missing), and retried in a later pass, so the same command can be
# relaunched after a reconnect or in the next slot and simply continues. When everything is done, the GPU is
# kept busy with repeat measurements (reproducibility across sessions) until the deadline.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${DEADLINE:?DEADLINE (unix epoch) required}"
DATA="$(dirname "$ROOT")"
CC1D="$DATA/ext/cc1d"                 # causal-conv1d built for .venv-ref (PYTHONPATH-only, so .venv-ref = control)
LLAMA="$DATA/ext/llama.cpp"
GGUF="$DATA/ext/gguf"
MARK="results/raw/plan_done"; mkdir -p "$MARK"
say() { echo "[plan $(date +%H:%M:%S)] $*"; }
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
use() { deactivate 2>/dev/null || true; source "$1/bin/activate"; }
suite() {  # suite KEY=VAL ... : run_suite under the plan deadline; rc 3 if it had to skip steps (not finished)
    local before; before="$(ls results/raw/suite_*.log 2>/dev/null | wc -l)"
    env DEADLINE="$DEADLINE" EST_SCALE=0.5 "$@" bash scripts/run_suite.sh || return $?
    local log; log="$(ls -t results/raw/suite_*.log | head -1)"
    (( $(ls results/raw/suite_*.log | wc -l) > before )) || return 4
    grep "SKIPPED:" "$log" | tail -1 | grep -q "SKIPPED: none" || return 3
}

ref_ok()  { [[ -x .venv-ref/bin/python ]] && .venv-ref/bin/python -c "import torch, transformers" 2>/dev/null; }
cc1d_ok() { ref_ok && PYTHONPATH="$CC1D" .venv-ref/bin/python -c "import causal_conv1d" 2>/dev/null; }
ready() {
    case "$1" in
        carry)           [[ -n "${CARRY:-}" ]] ;;
        o0a_ctrl|o2|o6|o6b|o6_3b|o2_batch) ref_ok ;;
        o0a_cc1d|o0ab)   cc1d_ok ;;
        vllm|vllm_eager) [[ -x .venv-vllm/bin/python ]] && .venv-vllm/bin/python -c "import vllm" 2>/dev/null ;;
        llamacpp)        [[ -f "$LLAMA/CMakeLists.txt" && -f "$GGUF/LFM2.5-VL-1.6B-BF16.gguf" && -f "$GGUF/mmproj-LFM2.5-VL-1.6b-BF16.gguf" ]] \
                             && ! pgrep -u "$(whoami)" -f "[p]get.sh.*gguf" >/dev/null ;;
        *)               true ;;
    esac
}

run_item() {
    case "$1" in
        carry)    use .venv; suite SUITE=slot2 STEPS="$CARRY" ;;
        o0a_ctrl) use .venv-ref; suite TAG=o0a_ctrl STEPS="bench_nohooks bench_full summarize" ;;
        o0a_cc1d) use .venv-ref; suite PYTHONPATH="$CC1D" TAG=o0a_cc1d \
                      STEPS="bench_nohooks bench_full torch_prof nsys ncu_decode ncu_prefill summarize" ;;
        o0b)      use .venv-ref; suite EST_SCALE=1.0 VARIANT=compile_static TAG=o0b_compile STEPS="bench_nohooks summarize" ;;
        o0ab)     use .venv-ref; suite PYTHONPATH="$CC1D" EST_SCALE=1.0 VARIANT=compile_static TAG=o0ab_compile_cc1d \
                      STEPS="bench_nohooks summarize" ;;
        o1)       use .venv; suite VARIANT=cudagraph TAG=o1_cudagraph STEPS="bench_nohooks summarize" ;;
        o2)       use .venv-ref; suite VARIANT=fused TAG=o2_fused STEPS="bench_nohooks summarize" ;;
        o6)       use .venv-ref; suite VARIANT=fused_w8 TAG=o6_w8 STEPS="bench_nohooks summarize" ;;
        o6b)      use .venv-ref; suite VARIANT=fused_w4 TAG=o6b_w4 STEPS="bench_nohooks summarize" ;;
        o6_3b)    use .venv-ref; timeout --signal=INT "$(( $(left) < 25 ? $(left) : 25 ))m" python benchmarks/bench.py \
                      --model LiquidAI/LFM2.5-VL-3B --variants baseline,fused,fused_w8 --workloads text,img_small,img_fhd \
                      --gen-lens 1,128 --warmup 2 --repeats 5 --no-phases --tag 3b_fused ;;
        o2_batch) use .venv-ref; timeout --signal=INT "$(( $(left) < 15 ? $(left) : 15 ))m" python benchmarks/bench.py \
                      --variants baseline,fused --workloads img_small --gen-lens 1,128 --batch-sizes 1,4,16 --warmup 2 \
                      --repeats 5 --no-phases --tag fused_batch ;;
        vllm)     use .venv-vllm; timeout --signal=INT "$(( $(left) < 30 ? $(left) : 30 ))m" \
                      python benchmarks/ref_vllm.py --gen-lens 1,128,512 --repeats 5 ;;
        vllm_eager) use .venv-vllm; timeout --signal=INT "$(( $(left) < 20 ? $(left) : 20 ))m" \
                      python benchmarks/ref_vllm.py --gen-lens 1,128 --repeats 5 --eager --tag ref_vllm_eager ;;
        llamacpp) use .venv
                  [[ -x "$LLAMA/build/bin/llama-mtmd-cli" ]] || LLAMA="$LLAMA" bash benchmarks/ref_llamacpp.sh build
                  timeout --signal=INT "$(( $(left) < 30 ? $(left) : 30 ))m" env LLAMA="$LLAMA" GGUF_DIR="$GGUF" \
                      bash benchmarks/ref_llamacpp.sh ;;
        bl_night) use .venv; suite TAG=baseline_night STEPS="bench_nohooks bench_full summarize" ;;
    esac
}

ITEMS="${ITEMS:-carry o0a_ctrl o0a_cc1d o0b o0ab o1 vllm o2 o6 o6b o6_3b o2_batch llamacpp vllm_eager bl_night}"
# results/raw/plan_done/PAUSE: someone is using the GPU interactively -> do nothing (the launcher retries later)
if [[ -f "$MARK/PAUSE" ]]; then say "paused (remove $MARK/PAUSE to resume)"; exit 0; fi
say "deadline $(date -d @"$DEADLINE" +%H:%M) ($(left) min); items: $ITEMS"
for pass in 1 2 3; do
    pending=0
    for it in $ITEMS; do
        [[ -f "$MARK/$it" ]] && continue
        [[ -f "$MARK/PAUSE" ]] && { say "paused before $it"; break 2; }
        (( $(left) >= 4 )) || break 2
        if ! ready "$it"; then say "not ready: $it (pass $pass)"; pending=1; continue; fi
        say "START $it"; t0=$SECONDS
        run_item "$it" 2>&1; rc=$?
        say "END   $it rc=$rc ($(( (SECONDS - t0) / 60 )) min)"
        (( rc == 0 )) && date > "$MARK/$it"
    done
    (( pending )) || break
    (( $(left) >= 4 )) && { say "waiting 3 min for pending installs/downloads"; sleep 180; }
done
# Nothing left to do: keep the GPU busy with repeat measurements (run-to-run / session-to-session variance),
# unless results/raw/plan_done/NO_FILL exists (GPU wanted for interactive development instead)
while (( $(left) >= 8 )) && [[ ! -f "$MARK/NO_FILL" ]]; do
    use .venv; say "filler: baseline repeat"; suite TAG=baseline_fill STEPS="bench_nohooks"
done
say "plan finished ($(left) min left)"
