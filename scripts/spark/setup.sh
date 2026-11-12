#!/usr/bin/env bash
# One-time setup on the DGX Spark (GB10, aarch64, sm_121), entirely inside ~/PAA_Team21. Idempotent.
#
#   bash ~/PAA_Team21/PAA_Project/scripts/spark/setup.sh            # venv + torch + requirements + weights + checks
#
# The Spark has no IPv6 route and a slow pypi.org (~20 s per index page): long pip timeouts, IPv4 for downloads.
# The venv is called .venv-ref like on the course machines (system Python 3.12 with headers, so our CUDA
# extensions build); scripts that say ".venv-ref/bin/python" work unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/spark/env.sh"
cd "$ROOT"
PY="$ROOT/.venv-ref/bin/python"

if [[ ! -x "$PY" ]]; then python3 -m venv "$ROOT/.venv-ref"; fi
"$PY" -m pip install -q --upgrade pip wheel
if ! "$PY" -c "import torch, torchvision" 2>/dev/null; then
    "$PY" -m pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130
fi
"$PY" -m pip install -r requirements.txt

if [[ ! -f models/LFM2.5-VL-1.6B/model.safetensors ]] || (( $(stat -c %s models/LFM2.5-VL-1.6B/model.safetensors) < 1000000000 )); then
    "$ROOT/.venv-ref/bin/hf" download LiquidAI/LFM2.5-VL-1.6B --revision 919fde3d022e3f90a4716006f993938ee8c2eb97 \
        --local-dir models/LFM2.5-VL-1.6B
fi

"$PY" - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}")
print(f"GPU {torch.cuda.get_device_name()} capability {torch.cuda.get_device_capability()} "
      f"native bf16 {torch.cuda.is_bf16_supported(including_emulation=False)}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print("bf16 matmul ok:", bool((x @ x).isfinite().all()))
try:
    import pynvml as nv
    nv.nvmlInit(); h = nv.nvmlDeviceGetHandleByIndex(0)
    print("NVML power W:", nv.nvmlDeviceGetPowerUsage(h) / 1000)
    print("NVML energy counter mJ:", nv.nvmlDeviceGetTotalEnergyConsumption(h))
except Exception as e:
    print("NVML:", repr(e))
PY
echo "[spark setup] done"
