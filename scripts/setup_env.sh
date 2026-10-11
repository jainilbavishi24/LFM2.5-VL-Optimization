#!/usr/bin/env bash
# Create (or reuse) the Python environment on a GPU machine. Idempotent: a second run only verifies.
#
#   bash scripts/setup_env.sh
#
# Picks the PyTorch wheel index that matches the CUDA version the installed driver supports
# (driver CUDA >= 13.0 -> cu130, >= 12.8 -> cu128, >= 12.6 -> cu126), and pins the same torch
# version as our Colab T4 runs so results stay comparable.
#
# Env overrides:
#   VENV_DIR=.venv                    where to create the virtualenv
#   TORCH_VERSION=2.11.0              torch version to pin ("" = latest on the index)
#   TORCH_INDEX_URL=...               force a wheel index
#   INSTALL_CAUSAL_CONV1D=1           also build the causal-conv1d CUDA package
#                                     (changes which conv kernel the HF baseline uses!)
#   FORCE=1                           reinstall even if the venv already works
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
TORCH_VERSION="${TORCH_VERSION-2.11.0}"

verify() {
    python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}")
assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU: {torch.cuda.get_device_name()}  capability: {torch.cuda.get_device_capability()}  "
      f"bf16: {torch.cuda.is_bf16_supported()}")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); print("matmul ok:", float((x @ x).float().abs().mean()) > 0)
PY
}

if [[ -x "$VENV_DIR/bin/python" && "${FORCE:-0}" != "1" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    if verify 2>/dev/null; then
        echo "[setup_env] existing environment OK ($VENV_DIR)"; exit 0
    fi
    echo "[setup_env] existing environment broken, reinstalling"
fi

if [[ -z "${TORCH_INDEX_URL:-}" ]]; then
    DRV_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)"
    if   (( DRV_CUDA >= 130 )); then TAG=cu130
    elif (( DRV_CUDA >= 128 )); then TAG=cu128
    elif (( DRV_CUDA >= 126 )); then TAG=cu126
    else TAG=cu124; echo "[setup_env] WARNING: old driver (CUDA ${DRV_CUDA}); newest torch may not be available"; fi
    TORCH_INDEX_URL="https://download.pytorch.org/whl/$TAG"
fi
echo "[setup_env] torch index: $TORCH_INDEX_URL"

if command -v uv >/dev/null; then
    uv venv -q --allow-existing "$VENV_DIR"
    PIP=(uv pip install -q --python "$VENV_DIR/bin/python")
else
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install -q --upgrade pip wheel
    PIP=("$VENV_DIR/bin/python" -m pip install -q)
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ -n "$TORCH_VERSION" ]] && "${PIP[@]}" "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX_URL"; then
    :
else
    echo "[setup_env] torch==$TORCH_VERSION not available on this index; installing the latest instead (note it in the log!)"
    "${PIP[@]}" torch torchvision --index-url "$TORCH_INDEX_URL"
fi
"${PIP[@]}" -r "$ROOT/requirements.txt"

if [[ "${INSTALL_CAUSAL_CONV1D:-0}" == "1" ]]; then
    "${PIP[@]}" causal-conv1d --no-build-isolation
fi

verify
