# Source on the DGX Spark before anything else:  source ~/PAA_Team21/PAA_Project/scripts/spark/env.sh
# The account is not ours: everything we write (venv, caches, temp files, weights, results) stays in ~/PAA_Team21.
export PAA_HOME="$HOME/PAA_Team21"
C="$PAA_HOME/cache"
export HF_HOME="$C/huggingface" PIP_CACHE_DIR="$C/pip" XDG_CACHE_HOME="$C" XDG_CONFIG_HOME="$C/config" \
    TORCH_HOME="$C/torch" TORCH_EXTENSIONS_DIR="$C/torch_extensions" TRITON_CACHE_DIR="$C/triton" TRITON_HOME="$C" \
    CUDA_CACHE_PATH="$C/nv" MPLCONFIGDIR="$C/matplotlib" TMPDIR="$PAA_HOME/tmp" \
    PYTHONUSERBASE="$PAA_HOME/pyuser" PIP_DEFAULT_TIMEOUT=180 PIP_DISABLE_PIP_VERSION_CHECK=1
export CUDA_HOME=/usr/local/cuda-13.0 PATH="/usr/local/cuda-13.0/bin:$PATH"
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error
mkdir -p "$C" "$TMPDIR" "$PAA_HOME/logs"
unset C
