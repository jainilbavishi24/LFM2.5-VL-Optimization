#!/usr/bin/env bash
# Everything on the Jetson, step by step (one GPU process at a time; 15 GB shared memory):
#
#   STEPS="tests bench" bash scripts/jetson/run_all.sh > jobs/run.log 2>&1
#
# tests    kernel tests, GPU preprocessing, GPTQ, runtime numerics (BF16 / INT8 / GPTQ INT4, also builds the GPTQ cache)
# bench    speed + energy (INA3221 VDD_IN) for HF, our BF16 stack, INT8, INT8 everywhere, GPTQ INT4
# demo     demo/record.py on the 8 video cases (HF / ours BF16 / ours INT8 everywhere)
# eval     5 benchmarks, 200 questions each
# gemv     GEMV microbenchmark, old vs new int8/int4 conversion
# eval_quant  INT8 everywhere / INT4 GPTQ on the same subset, one process per variant
# bench2   quantized variants again after the conversion change (tag jetson_ours_bp)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv-ref/bin/python
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
export PATH="$PWD/.venv-ref/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda   # ninja + nvcc for the JIT extensions
STEPS="${STEPS:-tests bench demo eval}"
BENCH="$PY benchmarks/bench.py --workloads text,img_small,img_hd,img_fhd --gen-lens 1,128 --batch-sizes 1 --warmup 2 --repeats 5 --no-phases --idle-seconds 3"
step() { local name="$1"; shift; echo "[jetson] $(date +%T) $name start"; "$@"; echo "[jetson] $(date +%T) $name rc=$?"; }

for s in $STEPS; do
    case "$s" in
        tests)
            step test_kernels $PY tests/test_kernels.py
            step test_preprocess $PY tests/test_preprocess.py
            step test_int8_gemm $PY tests/test_int8_gemm.py
            step test_gptq $PY tests/test_gptq.py
            step numerics_bf16 $PY tests/check_runtime_numerics.py --fused --gemv bf16 --fused-prefill
            step numerics_w8 $PY tests/check_runtime_numerics.py --fused --gemv int8 --fused-prefill
            step numerics_w4gptq $PY tests/check_runtime_numerics.py --fused --gemv int4 --fused-prefill --gptq --w4-group 128 ;;
        bench)
            step bench_base $BENCH --variants baseline --tag jetson_base
            step bench_ours $BENCH --variants fused_gemv_pf_vfused,fused_w8_pf_vfused,fused_w8all_pf_vfused,fused_w4gptq_pf_vfused \
                --preprocess gpu --tag jetson_ours
            for d in results/raw/*_jetson_*/; do $PY benchmarks/summarize.py "$d" > /dev/null; done ;;
        gemv)  # GEMV microbenchmark before / after the byte_perm conversion (e5eb2b4), old kernel from a worktree
            rm -rf /tmp/paa_old && git worktree add -f /tmp/paa_old e5eb2b4~1 > /dev/null 2>&1
            step gemv_old env TORCH_EXTENSIONS_DIR=/tmp/paa_old_ext $PY /tmp/paa_old/profiling/bench_gemv.py
            step gemv_new $PY profiling/bench_gemv.py
            git worktree remove --force /tmp/paa_old; rm -rf /tmp/paa_old_ext ;;
        bench2)  # quantized variants again with the new GEMV conversion
            step bench_quant $BENCH --variants fused_w8_pf_vfused,fused_w8all_pf_vfused,fused_w4gptq_pf_vfused \
                --preprocess gpu --tag jetson_ours_bp
            for d in results/raw/*_jetson_ours_bp/; do $PY benchmarks/summarize.py "$d" > /dev/null; done ;;
        demo)
            step demo $PY demo/record.py --only handwriting,chart_bar,cats,math_latex,letter,graph,receipt_35mp,describe_city \
                --tag jetson_demo ;;
        eval)
            step eval $PY benchmarks/eval_quality.py --benches mmstar,pope,ocrbench,chartqa,textvqa --limit 200 \
                --variants baseline,fused_gemv_pf_vfused+gpuprep,fused_w8all_pf_vfused+gpuprep,fused_w4gptq_pf_vfused+gpuprep \
                --tag jetson_subset
            step eval_compare $PY benchmarks/eval_compare.py "$(ls -d results/eval/*_jetson_subset | tail -1)" --md ;;
        eval_quant)  # INT8 everywhere and INT4 on the same 1000 questions, one process each (15 GB shared memory)
            for v in fused_w8all_pf_vfused+gpuprep fused_w4gptq_pf_vfused+gpuprep; do
                step "eval $v" $PY benchmarks/eval_quality.py --benches mmstar,pope,ocrbench,chartqa,textvqa --limit 200 \
                    --variants "$v" --tag jetson_subset_q
            done
            step eval_compare $PY benchmarks/eval_compare.py "$(ls -d results/eval/*_jetson_subset | tail -1)" \
                results/eval/*_jetson_subset_q --md ;;
        *) echo "[jetson] unknown step $s" ;;
    esac
done
echo "[jetson] done $(date +%T)"
