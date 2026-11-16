#!/usr/bin/env bash
# Everything on the Jetson, step by step (one GPU process at a time; 15 GB shared memory):
#
#   STEPS="tests bench" bash scripts/jetson/run_all.sh > jobs/run.log 2>&1
#
# tests    kernel tests, GPU preprocessing, GPTQ, runtime numerics (BF16 / INT8 / GPTQ INT4, also builds the GPTQ cache)
# bench    speed + energy (INA3221 VDD_IN) for HF, our BF16 stack, INT8, INT8 everywhere, GPTQ INT4
# demo     demo/record.py on the 8 video cases (HF / ours BF16 / ours INT8 everywhere)
# eval     5 benchmarks, 200 questions each
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv-ref/bin/python
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
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
        demo)
            step demo $PY demo/record.py --only handwriting,chart_bar,cats,math_latex,letter,graph,receipt_35mp,describe_city \
                --tag jetson_demo ;;
        eval)
            step eval $PY benchmarks/eval_quality.py --benches mmstar,pope,ocrbench,chartqa,textvqa --limit 200 \
                --variants baseline,fused_gemv_pf_vfused+gpuprep,fused_w8all_pf_vfused+gpuprep,fused_w4gptq_pf_vfused+gpuprep \
                --tag jetson_subset
            step eval_compare $PY benchmarks/eval_compare.py "$(ls -d results/eval/*_jetson_subset | tail -1)" --md ;;
        *) echo "[jetson] unknown step $s" ;;
    esac
done
echo "[jetson] done $(date +%T)"
