#!/usr/bin/env bash
# Create the Python environment for the project (run on the GPU machine).
#
#   bash scripts/setup_env.sh
#
# Env overrides:
#   VENV_DIR=.venv                    where to create the virtualenv
#   TORCH_INDEX_URL=...               PyTorch wheel index (must be CUDA >= 12.8 for sm_120)
#   INSTALL_CAUSAL_CONV1D=1           also build the causal-conv1d CUDA package
#                                     (changes which conv kernel the HF baseline uses!)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install -r "$ROOT/requirements.txt"

if [[ "${INSTALL_CAUSAL_CONV1D:-0}" == "1" ]]; then
    python -m pip install causal-conv1d --no-build-isolation
fi

python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}  capability: {torch.cuda.get_device_capability()}")
    print(f"arch list: {torch.cuda.get_arch_list()}")
else:
    print("WARNING: CUDA not available")
PY
