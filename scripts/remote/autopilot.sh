#!/usr/bin/env bash
# Unattended back-to-back GPU slots, driven from this machine (needs key login: ssh-copy-id once, no password).
#
#   bash scripts/remote/autopilot.sh 02:00 04:00          # slot ends (local time), in order
#
# For each slot end E: (re)connect the master as soon as the server lets us in, (re)launch
# scripts/remote/plan_refs.sh with DEADLINE = E - SLOT_MARGIN_MIN, sync results every 5 min, and relaunch the plan
# if the server killed it (it resumes from its done-markers). CARRY = slot-1 steps that never finished.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
R="bash scripts/remote/remote.sh"
REMOTE="${REMOTE:-gpu}"
MARGIN="${SLOT_MARGIN_MIN:-6}"
log() { echo "[auto $(date +%H:%M:%S)] $*"; }
alive() { ssh -O check "$REMOTE" 2>/dev/null; }
connect() {  # retry for up to ~15 min (login opens only once our booking has started)
    for _ in $(seq 1 45); do
        alive && return 0
        ssh -o BatchMode=yes -o ConnectTimeout=15 -fN "$REMOTE" 2>&1 | grep -vE '^(CS3406|Only the team|Your )' | head -2
        alive && { log "master connected"; return 0; }
        sleep 20
    done
    log "could not connect"; return 1
}
plan_running() { timeout 30 ssh "$REMOTE" "pgrep -u \$(whoami) -f '[p]lan_refs.sh' >/dev/null" 2>/dev/null; }
carry() {  # slot-1 steps without an 'END <step> rc=0' line
    local log; log="$(ls results/raw/suite_*_baseline.log 2>/dev/null | grep 20261013-2246 | head -1)"
    local all="bench_full bench_batch numerics overhead torch_prof nsys sdpa ncu_decode ncu_vision ncu_prefill bench_nohooks bench_long sdpa_rkv ncu_fhd_prefill ncu_fhd_vision ncu_fhd_decode ncu_step512 ncu_deep bench_batch_big bench_repeat model_3b"
    local out=""
    for s in $all; do grep -q "END   $s rc=0" "$log" 2>/dev/null || out="$out $s"; done
    echo "${out# }"
}

for slot_end in "$@"; do
    end=$(date -d "$slot_end" +%s); (( end <= $(date +%s) )) && end=$(( end + 86400 ))
    deadline=$(( end - MARGIN * 60 ))
    log "slot until $(date -d @"$end" +%H:%M); plan deadline $(date -d @"$deadline" +%H:%M)"
    while (( $(date +%s) < deadline - 240 )); do
        if ! alive; then connect || { sleep 60; continue; }; fi
        if ! plan_running; then
            $R sync > /dev/null 2>&1
            c="$(carry)"
            log "launching plan (carry: ${c:-none})"
            job="plan_$(date +%H%M)"
            $R bg "$job" "DEADLINE=$deadline CARRY='$c' bash scripts/remote/plan_refs.sh" 2>&1 | grep -v '^Your'
        fi
        for _ in 1 2 3 4 5; do sleep 60; alive || break; done
        $R sync > /dev/null 2>&1 && log "synced; $(timeout 30 ssh "$REMOTE" "grep -h '^\[plan' \$(ls -t /data/cs3406/team21/PAA_Project/jobs/plan_*.out | head -1) | tail -1" 2>/dev/null)"
    done
    # last sync before the slot ends, then wait for the boundary
    while (( $(date +%s) < end - 60 )); do sleep 30; done
    alive && $R sync > /dev/null 2>&1 && log "final sync for this slot done"
    while (( $(date +%s) < end + 20 )); do sleep 10; done
done
log "all slots done"
