#!/usr/bin/env bash
# Preflight for a GPU session (~1 min): what machine are we on, and can we do everything the session needs?
# Saves results/raw/preflight_<host>_<time>.txt. Run first in every GPU slot (scripts/remote/remote.sh setup).
#
# Checks: GPU + driver + max CUDA, other processes on the GPU (timing would be invalid), nvcc, python,
# sudo/root, Nsight Compute counter permission (else ncu steps are skipped), internet (pip / HF),
# disk, CPU, RAM, tmux.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/results/raw"
OUT="$ROOT/results/raw/preflight_$(hostname -s)_$(date +%Y%m%d-%H%M%S).txt"

{
echo "== host: $(hostname)  user: $(whoami)  date: $(date -Is)"
echo "== OS: $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")  kernel: $(uname -r)"
echo "== CPU: $(lscpu | sed -n 's/^Model name: *//p') x $(nproc) threads   RAM: $(free -g | awk '/Mem:/{print $2" GB total, "$7" GB available"}')"
echo "== disk (home): $(df -h ~ | awk 'NR==2{print $4" free of "$2}')"
echo "== GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem,persistence_mode,compute_mode --format=csv
echo "== max CUDA supported by driver: $(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p')"
echo "== other processes on the GPU (must be empty for valid timing/energy):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
echo "== nvcc: $(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1 || echo MISSING) $(nvcc --version 2>/dev/null | tail -1)"
echo "== python: $(command -v python3) $(python3 --version 2>&1)   conda: $(command -v conda || echo none)   uv: $(command -v uv || echo none)"
echo "== tmux: $(command -v tmux || echo MISSING)   git: $(command -v git || echo MISSING)"
echo "== root: $([[ $EUID -eq 0 ]] && echo yes || echo no)   passwordless sudo: $(sudo -n true 2>/dev/null && echo yes || echo no)"
echo "== profiling restricted to admin (RmProfilingAdminOnly): $(grep -o 'RmProfilingAdminOnly: [0-9]' /proc/driver/nvidia/params 2>/dev/null || echo unknown)"
echo "== internet: pypi $(curl -s -o /dev/null -m 10 -w '%{http_code}' https://pypi.org/simple/ || echo fail), huggingface $(curl -s -o /dev/null -m 10 -w '%{http_code}' https://huggingface.co/api/models/LiquidAI/LFM2.5-VL-1.6B || echo fail), github $(curl -s -o /dev/null -m 10 -w '%{http_code}' https://github.com || echo fail)"

# Nsight Compute counter access: profile one tiny kernel. ERR_NVGPUCTRPERM => no ncu this session.
source "$ROOT/scripts/setup_profilers.sh" > /dev/null 2>&1 || true
if command -v ncu > /dev/null && python3 -c "import torch" 2>/dev/null; then
    if ncu --launch-count 1 python3 -c "import torch; torch.ones(1, device='cuda').add_(1); torch.cuda.synchronize()" 2>&1 | grep -q ERR_NVGPUCTRPERM; then
        echo "== ncu counters: DENIED (ERR_NVGPUCTRPERM) -> run ncu steps only with sudo, or skip them"
    else
        echo "== ncu counters: OK"
    fi
else
    echo "== ncu counters: not tested yet (ncu or torch missing; re-run preflight after setup_env.sh)"
fi
echo "== nsys: $(command -v nsys || echo MISSING)  ncu: $(command -v ncu || echo MISSING)"
} 2>&1 | tee "$OUT"
echo "[preflight] saved $OUT"
