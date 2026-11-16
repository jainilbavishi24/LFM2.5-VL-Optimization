#!/usr/bin/env bash
# Setup on the Jetson (Orin NX 16GB, JetPack 7.2, CUDA 13.2, sm_87). Idempotent.
#
#   bash scripts/jetson/setup.sh
#
# The system Python has headers but no pip/ensurepip, and pypi.org is unreachable from the board: uv comes from
# GitHub, torch from download.pytorch.org (aarch64 cu130 wheels; their sm_80 code runs on sm_87), everything else
# through the Jetson AI Lab pypi mirror. Everything stays inside the project folder (.tools/, .venv-ref/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/.tools/uv-cache" UV_HTTP_TIMEOUT=600
UV="$ROOT/.tools/uv"
if [[ ! -x $UV ]]; then
    mkdir -p .tools
    curl -sSfL https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu.tar.gz \
        | tar xz -C .tools --strip-components=1
fi
[[ -x .venv-ref/bin/python ]] || "$UV" venv -q --python /usr/bin/python3.12 .venv-ref
PIP=("$UV" pip install --python .venv-ref/bin/python)
if ! .venv-ref/bin/python -c "import torch, torchvision" 2>/dev/null; then
    "${PIP[@]}" torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130
fi
"${PIP[@]}" -r requirements.txt pyarrow nltk --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple \
    --index-strategy unsafe-best-match

.venv-ref/bin/python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, archs {torch.cuda.get_arch_list()}")
print(f"GPU {torch.cuda.get_device_name()} capability {torch.cuda.get_device_capability()} "
      f"native bf16 {torch.cuda.is_bf16_supported(including_emulation=False)}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); print("bf16 matmul ok:", bool((x @ x).isfinite().all()))
q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
print("sdpa ok:", bool(torch.nn.functional.scaled_dot_product_attention(q, q, q).isfinite().all()))
PY
echo "[jetson setup] done"
