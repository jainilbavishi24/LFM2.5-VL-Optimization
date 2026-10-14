#!/usr/bin/env bash
# Download the VLM quality benchmarks (test parquet files from the Hugging Face Hub) into data/eval/.
# The server's internet is throttled per connection, so each file is fetched with parallel range requests (8 MB chunks).
#   MMStar (1500 multiple-choice), OCRBench (1000), POPE (3 x 3000 yes/no), ChartQA test (2500), TextVQA val shard 0 (~1670)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/data/eval}"
PAR="${PAR:-24}"
CH=$((8 * 1024 * 1024))
HF=https://huggingface.co/datasets

pget() {  # pget URL OUT SIZE
    local u="$1" out="$2" size="$3" d="$2.parts" n
    [[ -f $out && $(stat -c %s "$out") -eq $size ]] && { echo "have $out"; return 0; }
    mkdir -p "$d" "$(dirname "$out")"
    n=$(( (size + CH - 1) / CH ))
    export U="$u" SIZE="$size" D="$d"
    seq 0 $((n - 1)) | xargs -P "$PAR" -I{} bash -c '
        i={}; a=$((i*CH)); b=$((a+CH-1)); (( b >= SIZE )) && b=$((SIZE-1)); want=$((b-a+1)); f="$D/$(printf %05d $i)"
        for t in 1 2 3 4 5 6 7 8; do
            [[ -f $f && $(stat -c %s $f) -eq $want ]] && exit 0
            curl -sfL -m 600 -r $a-$b -o $f "$U"
        done; echo "chunk $i FAILED"'
    cat "$d"/* > "$out" && [[ $(stat -c %s "$out") -eq $size ]] && rm -rf "$d" && echo "DONE $out ($size B)"
}
export CH

# repo  path-in-repo  size  local-name
while read -r repo path size name; do
    pget "$HF/$repo/resolve/main/$path" "$OUT/$name" "$size"
done <<'EOF'
Lin-Chen/MMStar mmstar.parquet 41798712 mmstar.parquet
echo840/OCRBench data/test-00000-of-00001.parquet 67576988 ocrbench.parquet
lmms-lab-encoder/ChartQA data/test-00000-of-00001.parquet 72610993 chartqa_test.parquet
lmms-lab-encoder/POPE Full/adversarial-00000-of-00001.parquet 85007167 pope_adversarial.parquet
lmms-lab-encoder/POPE Full/popular-00000-of-00001.parquet 85003430 pope_popular.parquet
lmms-lab-encoder/POPE Full/random-00000-of-00001.parquet 85012317 pope_random.parquet
lmms-lab-encoder/textvqa data/validation-00000-of-00003.parquet 310398494 textvqa_val0.parquet
EOF
ls -la "$OUT"
