#!/usr/bin/env bash
# Make sure Nsight Systems (nsys) and Nsight Compute (ncu) are available, and check that
# ncu can read GPU performance counters. Safe to re-run.
#
#   source scripts/setup_profilers.sh     # also exports PATH for the current shell
#
# Installs nsys from NVIDIA's CUDA apt repo if missing and we are root (Colab, our GPU box with sudo).
CUDA_BIN="$(ls -d /usr/local/cuda/bin 2>/dev/null || true)"
NCU_DIR="$(ls -d /opt/nvidia/nsight-compute/*/ 2>/dev/null | sort -V | tail -1 || true)"
NSYS_DIR="$(ls -d /opt/nvidia/nsight-systems*/*/bin /opt/nvidia/nsight-systems/*/bin 2>/dev/null | sort -V | tail -1 || true)"
for d in "$CUDA_BIN" "$NCU_DIR" "$NSYS_DIR"; do
    if [[ -n "$d" && ":$PATH:" != *":$d:"* ]]; then export PATH="$d:$PATH"; fi
done

if ! command -v nsys >/dev/null; then
    CUDA_VER="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p' || true)"
    SUDO=""; if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi
    echo "[profilers] nsys not found; installing cuda-nsight-systems-${CUDA_VER} via apt"
    $SUDO apt-get update -qq >/dev/null 2>&1 || true
    $SUDO apt-get install -y -qq "cuda-nsight-systems-${CUDA_VER}" >/dev/null 2>&1 \
        || $SUDO apt-get install -y -qq "$(apt-cache search --names-only '^nsight-systems-20' | sort -V | tail -1 | cut -d' ' -f1)" >/dev/null 2>&1 \
        || echo "[profilers] WARNING: could not install nsys"
    NSYS_DIR="$(ls -d /opt/nvidia/nsight-systems*/*/bin /opt/nvidia/nsight-systems/*/bin /usr/local/cuda/nsight-systems*/bin 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "$NSYS_DIR" ]]; then export PATH="$NSYS_DIR:$PATH"; fi
fi

echo "[profilers] nsys: $(command -v nsys || echo MISSING) $(nsys --version 2>/dev/null | head -1)"
echo "[profilers] ncu : $(command -v ncu || echo MISSING) $(ncu --version 2>/dev/null | tail -1)"

# Performance-counter permission check (ERR_NVGPUCTRPERM => ncu unusable without root / driver setting)
if [[ -r /proc/driver/nvidia/params ]]; then
    echo "[profilers] driver $(grep -E 'RmProfilingAdminOnly' /proc/driver/nvidia/params || echo 'RmProfilingAdminOnly: n/a') (1 = only root may profile)"
fi
echo "[profilers] running as uid $EUID"
