#!/usr/bin/env bash
# Live check of the demo video on the course L40S: runs demo/live.py on node13 (SLURM reservation cs3406) and streams
# its output to this terminal. From a laptop it re-runs itself on the login node over `ssh turing` (key login).
#
#   bash demo/live_l40.sh                    # all 8 video cases, 7 timed runs each + energy (~4-5 min)
#   bash demo/live_l40.sh --quick            # 3 timed runs, no energy (~2 min)
#   bash demo/live_l40.sh --only cats,graph
#   bash demo/live_l40.sh --image ~/photo.jpg --prompt "What is in this picture?"   # image path on the cluster
set -euo pipefail
if ! command -v srun >/dev/null; then
    exec ssh -t turing "bash ~/PAA_Project/demo/live_l40.sh${*:+ $(printf '%q ' "$@")}"  # no args -> no '' arg
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[live] requesting the L40S on node13 (waits if our other jobs hold it) ..."
exec srun --job-name=demo_live --partition=u22 --reservation=cs3406 --account=cs3406 --nodelist=node13 \
    --gres=gpu:1 -c 15 --mem-per-cpu=2G --time=0:45:00 --chdir="$ROOT" --pty \
    bash -c 'source /etc/profile.d/modules.sh 2>/dev/null; module load u22/cuda/13.2 2>/dev/null || true
             export HF_HOME=$HOME/.cache/huggingface PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
             exec .venv-ref/bin/python demo/live.py "$@" 2> >(grep -v -E "Warning|warn\(|causal_conv1d|clean_up_tokenization" >&2)' \
    _ "$@"
