#!/usr/bin/env bash
# Run a command on the Jetson only while it stays cool: our board has no fan (passive heatsink only).
# Waits until every thermal zone is below START_C, then runs the command and kills it if any zone passes MAX_C.
#
#   MAX_C=85 START_C=60 bash scripts/jetson/thermal_guard.sh <command...>
set -uo pipefail
MAX_C="${MAX_C:-85}"; START_C="${START_C:-60}"
hottest() { local m=0 t; for z in /sys/class/thermal/thermal_zone*/temp; do t=$(( $(cat "$z") / 1000 )); (( t > m )) && m=$t; done; echo "$m"; }
while (( $(hottest) >= START_C )); do echo "[guard] $(date +%T) $(hottest) C, waiting for < ${START_C} C"; sleep 30; done
echo "[guard] $(date +%T) start at $(hottest) C, limit ${MAX_C} C: $*"
setsid "$@" &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
    t=$(hottest); (( t > peak )) && peak=$t
    if (( t >= MAX_C )); then
        echo "[guard] $(date +%T) $t C >= ${MAX_C} C: stopping the job"
        kill -- -"$pid" 2>/dev/null; sleep 2; kill -9 -- -"$pid" 2>/dev/null
        echo "[guard] STOPPED_HOT peak ${peak} C"; exit 3
    fi
    sleep 5
done
wait "$pid"; rc=$?
echo "[guard] $(date +%T) done rc=$rc, peak ${peak} C"
exit "$rc"
