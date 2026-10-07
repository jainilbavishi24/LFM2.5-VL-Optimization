#!/usr/bin/env bash
# Download model weights into models/ (git-ignored).
#
#   bash scripts/download_model.sh                         # LFM2.5-VL-1.6B (default)
#   bash scripts/download_model.sh LiquidAI/LFM2.5-VL-3B   # generalization check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${1:-LiquidAI/LFM2.5-VL-1.6B}"
# Pin the revision we read the code against, so results are reproducible.
REVISION="${REVISION:-$([[ "$MODEL_ID" == "LiquidAI/LFM2.5-VL-1.6B" ]] && echo 919fde3d022e3f90a4716006f993938ee8c2eb97 || echo main)}"

hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$ROOT/models/${MODEL_ID#*/}"
echo "Downloaded $MODEL_ID@$REVISION to models/${MODEL_ID#*/}"
