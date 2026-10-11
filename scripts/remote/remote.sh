#!/usr/bin/env bash
# Drive the course GPU server (L40) over SSH from this machine. Mirrors scripts/colab/colab.sh.
#
#   bash scripts/remote/remote.sh check               # can we connect? GPU visible? who else is on it?
#   bash scripts/remote/remote.sh push                # rsync code (no .git/models/results/third_party)
#   bash scripts/remote/remote.sh setup               # preflight + Python env + model weights + profilers
#   bash scripts/remote/remote.sh bg <job> "<cmd>"    # run detached in tmux (survives SSH drops); log in ~/jobs/<job>.out
#   bash scripts/remote/remote.sh log <job> [n]       # last n lines of a job's log
#   bash scripts/remote/remote.sh jobs                # running tmux jobs
#   bash scripts/remote/remote.sh sync                # pull results/raw and profiling/reports (incremental, safe to repeat)
#   bash scripts/remote/remote.sh run "<cmd>"         # run a short command in the project dir (attached)
#   bash scripts/remote/remote.sh shell               # interactive shell in the project dir
#
# Env: REMOTE  ssh host or ~/.ssh/config alias (default: l40)
#      RDIR    project dir on the server (default: ~/PAA_Project)
#
# Set up once in ~/.ssh/config (values from the course staff):
#   Host l40
#       HostName <server address>
#       User <username>
#       Port <port, usually 22>
#       IdentityFile ~/.ssh/id_ed25519
#       ServerAliveInterval 30
#       ServerAliveCountMax 6
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-l40}"
RDIR="${RDIR:-PAA_Project}"   # relative to the remote home directory
SSH=(ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 "$REMOTE")

rrun() {  # run a command on the server inside the project dir (login shell, so conda/modules load)
    "${SSH[@]}" "mkdir -p $RDIR && cd $RDIR && bash -lc $(printf '%q' "$1")"
}

case "${1:-}" in
    check)
        "${SSH[@]}" 'echo "host: $(hostname)  user: $(whoami)"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,power.limit --format=csv; echo "--- other GPU processes:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true'
        ;;
    push)
        { git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown; } | tr -d '\n' > "$ROOT/.git_commit"
        if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then echo -n "-dirty" >> "$ROOT/.git_commit"; fi
        rsync -az --info=stats1 \
            --exclude .git/ --exclude third_party/ --exclude models/ --exclude results/raw/ \
            --exclude profiling/reports/ --exclude '__pycache__/' --exclude .venv/ --exclude '*.pdf' \
            "$ROOT/" "$REMOTE:$RDIR/"
        ;;
    setup)
        # preflight twice: the second run (inside the venv) also tests Nsight Compute counter access
        rrun "bash scripts/preflight.sh; bash scripts/setup_env.sh && source .venv/bin/activate && bash scripts/download_model.sh && bash scripts/setup_profilers.sh && bash scripts/preflight.sh"
        ;;
    bg)
        job="$2"; cmd="$3"
        # tmux keeps the job alive if SSH drops; output also goes to ~/jobs/<job>.out for `log`
        # explicit bash (PIPESTATUS), whatever the server's default shell is
        script="cd $RDIR && source .venv/bin/activate 2>/dev/null; ($cmd) 2>&1 | tee ~/jobs/$job.out; echo JOB_EXIT=\${PIPESTATUS[0]} >> ~/jobs/$job.out"
        "${SSH[@]}" "mkdir -p ~/jobs && tmux new-session -d -s $job bash -lc $(printf '%q' "$script")" \
            && echo "started job $job on $REMOTE (tmux attach -t $job to watch)"
        ;;
    log) "${SSH[@]}" "tail -n ${3:-20} ~/jobs/$2.out" ;;
    jobs) "${SSH[@]}" "tmux ls 2>/dev/null || echo 'no jobs'" ;;
    sync)
        for d in results/raw profiling/reports; do
            mkdir -p "$ROOT/$d"
            rsync -az --info=stats1 "$REMOTE:$RDIR/$d/" "$ROOT/$d/"
        done
        ;;
    run) rrun "source .venv/bin/activate 2>/dev/null; $2" ;;
    shell) ssh -t "$REMOTE" "cd $RDIR && exec bash -l" ;;
    *) sed -n '2,25p' "$0"; exit 1 ;;
esac
