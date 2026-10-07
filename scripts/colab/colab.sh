#!/usr/bin/env bash
# Drive a Google Colab T4 session from this machine (needs the authenticated `colab` CLI).
#
#   bash scripts/colab/colab.sh up                 # create session (T4) + push code + install deps + download model
#   bash scripts/colab/colab.sh push               # re-upload code after local edits (keeps remote results/models)
#   bash scripts/colab/colab.sh run "<command>"    # run a shell command in /content/PAA_Project on the VM
#   bash scripts/colab/colab.sh pull <dir>         # download a remote dir (relative to project) into the same local path
#                                                  #   PULL_EXCLUDE='*.ncu-rep' skips matching files
#   bash scripts/colab/colab.sh down               # stop the session (ALWAYS do this when finished)
#
# Env: SESSION (default paa-t4), GPU (default T4), TIMEOUT for run (default 7200 s)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${SESSION:-paa-t4}"
GPU="${GPU:-T4}"
REMOTE_DIR=/content/PAA_Project
C=(colab --auth=oauth2)

remote() {  # remote "<shell command>" [timeout]
    "${C[@]}" exec -s "$SESSION" -f "$ROOT/scripts/colab/remote_run.py" --timeout "${2:-${TIMEOUT:-7200}}" \
        --env "CMD=$1" --env "REMOTE_DIR=$REMOTE_DIR"
}

push() {
    local tarball; tarball="$(mktemp --suffix=.tar.gz)"
    # .git is not shipped; record the commit so results on the VM stay traceable
    { git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown; } | tr -d '\n' > "$ROOT/.git_commit"
    if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then echo -n "-dirty" >> "$ROOT/.git_commit"; fi
    tar -C "$ROOT" -czf "$tarball" \
        --exclude=./third_party --exclude=./models --exclude=./.git --exclude=./results/raw \
        --exclude=./profiling/reports --exclude='*.pdf' --exclude='__pycache__' --exclude=./.venv .
    "${C[@]}" upload -s "$SESSION" "$tarball" /content/paa_code.tar.gz
    rm -f "$tarball"
    REMOTE_DIR=/content remote "mkdir -p $REMOTE_DIR && tar -xzf /content/paa_code.tar.gz -C $REMOTE_DIR && echo pushed" 300
}

case "${1:-}" in
    up)
        "${C[@]}" new -s "$SESSION" --gpu "$GPU"
        push
        remote "nvidia-smi && pip install -q -r requirements.txt && python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)' && bash scripts/download_model.sh" 1800
        ;;
    push) push ;;
    run) remote "$2" ;;
    pull)
        rel="${2%/}"
        excl=""; if [[ -n "${PULL_EXCLUDE:-}" ]]; then excl="--exclude=$PULL_EXCLUDE"; fi
        remote "tar -czf /content/pull.tar.gz $excl -C $REMOTE_DIR $rel && ls -la /content/pull.tar.gz" 1200
        "${C[@]}" download -s "$SESSION" /content/pull.tar.gz "$ROOT/.pull.tar.gz"
        tar -xzf "$ROOT/.pull.tar.gz" -C "$ROOT" && rm -f "$ROOT/.pull.tar.gz"
        echo "pulled $rel"
        ;;
    down) "${C[@]}" stop -s "$SESSION" ;;
    *) sed -n '2,12p' "$0"; exit 1 ;;
esac
