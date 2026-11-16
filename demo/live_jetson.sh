#!/usr/bin/env bash
# Live check of the Jetson demo video on the Jetson itself: same cases, same measuring code (demo/record.py Runner),
# every number printed next to the one in the video (demo/video_jetson.json). From a laptop it runs over `ssh jetson`.
#
#   bash demo/live_jetson.sh                 # the video's cases, 7 timed runs each + energy
#   bash demo/live_jetson.sh --quick         # 3 timed runs, no energy
#   bash demo/live_jetson.sh --only cats,graph
#   bash demo/live_jetson.sh --image ~/photo.jpg --prompt "What is in this picture?"   # image path on the Jetson
set -euo pipefail
if [[ ! -f /etc/nv_tegra_release ]]; then
    exec ssh -t jetson "bash ~/PAA_Project/demo/live_jetson.sh${*:+ $(printf '%q ' "$@")}"
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$PWD/.venv-ref/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error
exec .venv-ref/bin/python demo/live.py --video demo/video_jetson.json "$@" \
    2> >(grep -v -E "Warning|warn\(|causal_conv1d|clean_up_tokenization|_warn_unsupported" >&2)
