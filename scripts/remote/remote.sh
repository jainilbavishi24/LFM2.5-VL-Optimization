#!/usr/bin/env bash
# Drive the course GPU server (team21@10.4.25.63) from this machine during a booked 2-hour slot.
# Slot workflow, timings and failure playbook: docs/notes/gpu_sessions.md
#
# Server rules (course staff):
#   * login works ONLY during our slot ("Your account has expired" otherwise);
#   * at slot end we are logged out and ALL our processes stop (10-min warning first);
#   * logging out early also stops everything; nohup / screen / tmux do NOT survive.
# Hence: one SSH *master* connection, opened once per slot by a person (password), held open for the whole slot;
# every command here reuses it (~/.ssh/config "Host gpu": ControlMaster/ControlPersist). Jobs run with
# setsid+nohup only so that they don't depend on the short command that launched them; they still need the master.
#
#   connect                 (person, own terminal) open the master:  ssh -fN gpu   (password once)
#   bash scripts/remote/remote.sh status             # is the master connection alive?
#   bash scripts/remote/remote.sh check              # GPU model, memory, other users
#   bash scripts/remote/remote.sh push               # rsync code (no .git/models/results/third_party)
#   bash scripts/remote/remote.sh setup              # preflight + Python env + model weights + profilers (idempotent)
#   bash scripts/remote/remote.sh pushmodel|pushtools # no-internet fallbacks: weights / Nsight .debs from this machine
#   bash scripts/remote/remote.sh slot <preset> <HH:MM slot end>   # run_suite.sh SUITE=<preset>, stops 12 min before slot end
#   bash scripts/remote/remote.sh watch <job>        # sync results every 5 min + show progress until the job ends
#   bash scripts/remote/remote.sh bg <job> "<cmd>"   # any command detached; log in $RDIR/jobs/<job>.out
#   bash scripts/remote/remote.sh log <job> [n]      # last n lines of a job's log
#   bash scripts/remote/remote.sh jobs               # our running jobs on the server
#   bash scripts/remote/remote.sh stop <job>         # stop a job (SIGINT, then SIGKILL)
#   bash scripts/remote/remote.sh sync               # pull results/raw and profiling/reports (incremental)
#   bash scripts/remote/remote.sh run "<cmd>"        # short command in the project dir (attached)
#   bash scripts/remote/remote.sh shell              # interactive shell in the project dir
#   bash scripts/remote/remote.sh disconnect         # close the master at the very end (stops everything!)
#
# Env: REMOTE (default gpu), RDIR (default /data/cs3406/team21/PAA_Project)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-gpu}"
DATA="${DATA:-/data/cs3406/team21}"
RDIR="${RDIR:-$DATA/PAA_Project}"
SSH=(ssh "$REMOTE")
# Keep every cache on our team folder (home/quota elsewhere may be small or wiped)
ENVSETUP="export XDG_CACHE_HOME=$DATA/.cache UV_CACHE_DIR=$DATA/.cache/uv PIP_CACHE_DIR=$DATA/.cache/pip HF_HOME=$DATA/.cache/huggingface PATH=$DATA/.local/bin:\$PATH; cd $RDIR"

need_master() {
    if ! ssh -O check "$REMOTE" 2>/dev/null; then
        echo "No master connection to $REMOTE. In your own terminal run:  ssh -fN $REMOTE   (team password, once per slot)"
        exit 3
    fi
}
rrun() {  # short command on the server inside the project dir
    "${SSH[@]}" "mkdir -p $RDIR && bash -lc $(printf '%q' "$ENVSETUP && $1")"
}

cmd="${1:-}"
case "$cmd" in
    status) ssh -O check "$REMOTE" && echo "master connection OK" ;;
    disconnect) ssh -O exit "$REMOTE" ;;
    check)
        need_master
        "${SSH[@]}" 'echo "host: $(hostname)  user: $(whoami)  home: $HOME  date: $(date)"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv; echo "--- other GPU processes:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true; df -h '"$DATA"' | tail -1'
        ;;
    push)
        need_master
        { git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown; } | tr -d '\n' > "$ROOT/.git_commit"
        if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then echo -n "-dirty" >> "$ROOT/.git_commit"; fi
        "${SSH[@]}" "mkdir -p $RDIR"
        rsync -az --info=stats1 \
            --exclude .git/ --exclude third_party/ --exclude models/ --exclude results/raw/ \
            --exclude profiling/reports/ --exclude '__pycache__/' --exclude .venv/ --exclude '*.pdf' \
            "$ROOT/" "$REMOTE:$RDIR/"
        ;;
    pushmodel)  # fallback when the server has no internet: weights from our machine (scripts/download_model.sh here first)
        need_master
        rsync -az --info=progress2 "$ROOT/models/" "$REMOTE:$RDIR/models/"
        ;;
    pushtools)  # fallback when the server has no internet: Nsight .debs downloaded here into third_party/nsight/
        need_master
        "${SSH[@]}" "mkdir -p $RDIR/third_party/nsight"
        rsync -az --info=progress2 "$ROOT/third_party/nsight/" "$REMOTE:$RDIR/third_party/nsight/"
        ;;
    setup)
        need_master
        # preflight twice: the second run (inside the venv) also tests Nsight Compute counter access
        rrun "bash scripts/preflight.sh; bash scripts/setup_env.sh && source .venv/bin/activate && bash scripts/download_model.sh && bash scripts/setup_profilers.sh && bash scripts/preflight.sh"
        ;;
    bg)
        need_master
        job="$2"; c="$3"
        script="$ENVSETUP && mkdir -p jobs && source .venv/bin/activate 2>/dev/null; ($c) > jobs/$job.out 2>&1; echo JOB_EXIT=\$? >> jobs/$job.out"
        "${SSH[@]}" "mkdir -p $RDIR/jobs && setsid nohup bash -lc $(printf '%q' "$script") > /dev/null 2>&1 < /dev/null & echo started job $job"
        ;;
    slot)
        preset="$2"; slot_end="$3"   # HH:MM (our local time, IST) when the server logs us out
        job="${preset}_$(date +%H%M)"
        # Deadline computed HERE as a Unix epoch: independent of the server's time zone (it may run on UTC).
        now=$(date +%s); end=$(date -d "$slot_end" +%s)
        if (( end <= now )); then end=$(( end + 86400 )); fi       # e.g. a slot ending at 00:00
        deadline=$(( end - ${SLOT_MARGIN_MIN:-12} * 60 ))
        echo "slot ends $(date -d @"$end" '+%a %H:%M'); job stops scheduling at $(date -d @"$deadline" '+%H:%M') (local time)"
        echo "server clock: $("${SSH[@]}" 'date -u +%H:%M:%S') UTC   local: $(date -u +%H:%M:%S) UTC  (should match within seconds)"
        "$0" bg "$job" "SUITE=$preset DEADLINE=$deadline EST_SCALE=${EST_SCALE:-1.0} ${STEPS:+STEPS='$STEPS'} bash scripts/run_suite.sh"
        echo "next: bash $0 watch $job"
        ;;
    watch)
        job="$2"
        while true; do
            if ! ssh -O check "$REMOTE" 2>/dev/null; then
                echo "[$(date +%H:%M)] MASTER CONNECTION LOST: reopen it (ssh -fN $REMOTE); the job on the server has stopped"; exit 3
            fi
            "$0" sync > /dev/null 2>&1 || echo "[watch] sync failed (will retry)"
            out="$("${SSH[@]}" "grep -E '^\[suite|^JOB_EXIT=' $RDIR/jobs/$job.out | tail -3" 2>/dev/null || echo '[watch] ssh failed')"
            echo "[$(date +%H:%M)] synced; $out"
            if grep -q '^JOB_EXIT=' <<< "$out"; then "$0" sync > /dev/null; echo "[watch] job $job finished, results synced"; break; fi
            sleep 300
        done
        ;;
    log) need_master; "${SSH[@]}" "tail -n ${3:-20} $RDIR/jobs/$2.out" ;;
    # '[r]un_suite' etc.: the bracket stops the pattern from matching the remote shell that runs pgrep/pkill itself
    jobs) need_master; "${SSH[@]}" "pgrep -u \$(whoami) -fa '[r]un_suite|[b]ench.py|[t]orch_profile|[r]un_once|[n]cu |[n]sys |[s]dpa_probe' || echo 'no jobs'" ;;
    stop) need_master; "${SSH[@]}" "pkill -INT -u \$(whoami) -f '[r]un_suite.sh'; sleep 20; pkill -KILL -u \$(whoami) -f '[r]un_suite.sh|[b]ench.py|[t]orch_profile.py|[r]un_once.py|[s]dpa_probe.py'; echo stopped" ;;
    sync)
        need_master
        for d in results/raw profiling/reports; do
            mkdir -p "$ROOT/$d"
            rsync -az --info=stats1 "$REMOTE:$RDIR/$d/" "$ROOT/$d/"
        done
        ;;
    run) need_master; rrun "source .venv/bin/activate 2>/dev/null; $2" ;;
    shell) need_master; ssh -t "$REMOTE" "cd $RDIR && exec bash -l" ;;
    *) sed -n '2,32p' "$0"; exit 1 ;;
esac
