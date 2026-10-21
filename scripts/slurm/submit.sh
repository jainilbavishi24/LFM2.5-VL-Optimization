#!/usr/bin/env bash
# Submit a command as a SLURM batch job on the course's L40 node (turing.iiit.ac.in, reservation cs3406, node13).
# Run on the cluster's login node from the project directory:
#
#   bash scripts/slurm/submit.sh <job-name> <hours> "<command>"
#   bash scripts/slurm/submit.sh l40_tests 1 ".venv-ref/bin/python tests/test_kernels.py"
#
# Log: jobs/<job-name>.out. The job runs in the project dir with CUDA 13.2 (nvcc for our JIT extensions) loaded.
# Batch jobs need no open terminal (interactive srun is limited to 6 h and dies with the connection).
set -euo pipefail
NAME="$1"; HOURS="$2"; CMD="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/jobs"
sbatch --job-name="$NAME" --partition=u22 --reservation=cs3406 --account=cs3406 --nodelist=node13 \
    --gres=gpu:1 -c 15 --mem-per-cpu=2G --time="$HOURS:00:00" \
    --output="$ROOT/jobs/$NAME.out" --chdir="$ROOT" \
    --wrap="source /etc/profile.d/modules.sh 2>/dev/null; module load u22/cuda/13.2; \
export HF_HOME=\$HOME/.cache/huggingface PATH=\$HOME/.local/bin:\$PATH; \
echo \"[job] \$(hostname) \$(date) GPU: \$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)\"; \
$CMD; echo JOB_EXIT=\$?"
