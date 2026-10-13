#!/usr/bin/env bash
# O0d reference baseline: llama.cpp (CUDA) on the same images / prompts as benchmarks/bench.py.
#
#   LLAMA=/path/to/llama.cpp GGUF_DIR=/path/to/gguf bash benchmarks/ref_llamacpp.sh [build]
#
# build : cmake build with CUDA for this GPU's arch only (no root needed)
# then  : llama-bench (text: prefill 256/1024, decode 128/512 tokens) and llama-mtmd-cli on each image
#         workload (128 new tokens, greedy, EOS ignored), for each GGUF in $QUANTS. Raw logs + timings are kept.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA="${LLAMA:-$ROOT/../ext/llama.cpp}"
GGUF_DIR="${GGUF_DIR:-$ROOT/../ext/gguf}"
QUANTS="${QUANTS:-BF16 Q8_0}"
MMPROJ="${MMPROJ:-$GGUF_DIR/mmproj-LFM2.5-VL-1.6b-BF16.gguf}"
REPEATS="${REPEATS:-3}"
BIN="$LLAMA/build/bin"

if [[ "${1:-}" == "build" ]]; then
    arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)"
    export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
    cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" -DLLAMA_CURL=OFF \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_IS_DEV=OFF > "$LLAMA/cmake_configure.log" 2>&1 && \
    cmake --build "$LLAMA/build" -j "${JOBS:-8}" --target llama-bench llama-mtmd-cli 2>&1 | tail -3
    ls -la "$BIN"/llama-bench "$BIN"/llama-mtmd-cli
    exit $?
fi

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA //; s/ //g')"
OUT="$ROOT/results/raw/$(date +%Y%m%d-%H%M%S)_${gpu}_ref_llamacpp"
mkdir -p "$OUT/images"
python - "$OUT/images" <<'EOF'
import sys; sys.path.insert(0, "src")
from lfm2opt.workloads import WORKLOADS
for name, wl in WORKLOADS.items():
    if wl.image_size: wl.image().save(f"{sys.argv[1]}/{name}.png")
EOF
echo "llama.cpp $(git -C "$LLAMA" log -1 --format='%h %cd' 2>/dev/null || cat "$LLAMA/LLAMA_COMMIT.txt")" | tee "$OUT/version.txt"

for q in $QUANTS; do
    m="$GGUF_DIR/LFM2.5-VL-1.6B-$q.gguf"
    [[ -f "$m" ]] || { echo "missing $m"; continue; }
    echo "== $q text (llama-bench)"
    "$BIN/llama-bench" -m "$m" -ngl 99 -fa 1 -p 256,1024 -n 128,512 -r "$REPEATS" -o json > "$OUT/bench_$q.json" 2> "$OUT/bench_$q.log"
    python -c "import json,sys; [print(f\"  pp{r['n_prompt']:<5} tg{r['n_gen']:<4} {r['avg_ts']:8.1f} tok/s\") for r in json.load(open(sys.argv[1]))]" "$OUT/bench_$q.json"
    for img in img_small img_hd img_fhd; do
        for r in $(seq 1 "$REPEATS"); do
            "$BIN/llama-mtmd-cli" -m "$m" --mmproj "$MMPROJ" --image "$OUT/images/$img.png" -p "Describe this image in detail." \
                -n 128 --temp 0 --ignore-eos -ngl 99 -fa on --seed 0 > "$OUT/mtmd_${q}_${img}_$r.out" 2> "$OUT/mtmd_${q}_${img}_$r.log"
        done
        echo "== $q $img (last run):"
        grep -hE "encoded in|decoded in|prompt eval time|^\s*eval time|load time" "$OUT/mtmd_${q}_${img}_${REPEATS}.log" | sed 's/^/  /'
    done
done
echo "done -> $OUT"
