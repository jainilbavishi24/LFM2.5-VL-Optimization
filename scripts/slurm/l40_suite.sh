#!/usr/bin/env bash
# Everything we measured on the RTX PRO 5000 Blackwell, repeated on the course L40 (Ada, sm_89): correctness, speed /
# energy grid, batching, profiling probes, full quality evaluation, long answers. One SLURM job:
#
#   bash scripts/slurm/submit.sh l40_suite 10 "bash scripts/slurm/l40_suite.sh"      # STEPS="tests bench ..." to subset
#
# Steps keep going after a failure; each prints "[suite] <step> rc=<code>". Outputs: results/raw/*_L40*, results/eval/*.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv-ref/bin/python
STEPS="${STEPS:-tests bench batch probes eval longform vlmeval}"
BENCH="$PY benchmarks/bench.py --workloads text,img_small,img_hd,img_fhd --gen-lens 1,128 --batch-sizes 1 --warmup 2 --repeats 10 --no-phases --idle-seconds 2"

# models and benchmark data are downloaded on the login node (compute nodes may have no internet): wait for that
while [[ -f jobs/downloads.out ]] && ! grep -q DL_EXIT jobs/downloads.out; do sleep 30; done

step() { local name="$1"; shift; echo "[suite] $(date +%T) $name start"; "$@"; echo "[suite] $(date +%T) $name rc=$?"; }

for s in $STEPS; do
    case "$s" in
        tests)
            step test_kernels $PY tests/test_kernels.py
            step test_preprocess $PY tests/test_preprocess.py
            step numerics $PY tests/check_runtime_numerics.py --fused --gemv bf16 --fused-prefill
            step vision $PY tests/check_vision_fused.py ;;
        bench)
            for r in 1 2; do
                step bench_base_r$r $BENCH --variants baseline --tag l40_base_r$r
                step bench_stack_r$r $BENCH --variants fused_gemv_pf_vfused --preprocess gpu --tag l40_stack_r$r
            done
            step bench_w8 $BENCH --variants fused_w8_pf_vfused --preprocess gpu --tag l40_w8_stack
            step bench_long $PY benchmarks/bench.py --variants baseline,fused_gemv_pf_vfused --workloads text,img_fhd \
                --gen-lens 1,1024 --batch-sizes 1 --warmup 1 --repeats 3 --no-phases --idle-seconds 2 --tag l40_long ;;
        batch)
            step batch $PY benchmarks/bench.py --variants baseline,fused_gemv_pf_vfused --workloads img_small \
                --gen-lens 1,128 --batch-sizes 1,2,4,8,16 --warmup 2 --repeats 5 --no-phases --idle-seconds 2 --tag l40_batch ;;
        probes)
            step gemm_efficiency $PY profiling/gemm_efficiency.py
            step ttft_probe $PY profiling/ttft_probe.py
            step decode_step env LFM2_PDL=0 $PY profiling/profile_decode_step.py --fused --gemv bf16 --steps 20 ;;
        eval)
            step eval_full $PY benchmarks/eval_quality.py --benches mmstar,pope,ocrbench,chartqa,textvqa_full \
                --variants baseline,fused_gemv_pf_vfused+gpuprep,fused_w8_pf_vfused+gpuprep --tag l40_fullsize ;;
        longform)
            step longform $PY benchmarks/eval_quality.py --benches longform --limit 500 \
                --variants baseline@fp32,baseline,fused_gemv_pf_vfused+gpuprep,fused_w8_pf_vfused+gpuprep --tag l40_longform ;;
        vlmeval)  # MMStar through VLMEvalKit (needs ~/ext/VLMEvalKit and data/LMUData/MMStar_authors.tsv, see README)
            step mmstar_tsv $PY scripts/prepare_mmstar_vlmeval.py data/LMUData/MMStar_authors.tsv data/LMUData
            step vlmeval env VLMEVAL_DIR="$HOME/ext/VLMEvalKit" LMUData="$PWD/data/LMUData" $PY benchmarks/vlmeval_mmstar.py \
                --local-tsv --variants baseline,fused_gemv_pf_vfused+gpuprep,fused_w8_pf_vfused+gpuprep ;;
        3b)  # generalization: LFM2.5-VL-3B speed (baseline vs our stack) and a quality subset
            step bench_3b_base $PY benchmarks/bench.py --model LiquidAI/LFM2.5-VL-3B --variants baseline \
                --workloads text,img_small,img_fhd --gen-lens 1,128 --batch-sizes 1 --warmup 2 --repeats 5 --no-phases \
                --idle-seconds 2 --tag l40_3b_base
            step bench_3b_stack $PY benchmarks/bench.py --model LiquidAI/LFM2.5-VL-3B --variants fused_gemv_pf_vfused \
                --workloads text,img_small,img_fhd --gen-lens 1,128 --batch-sizes 1 --warmup 2 --repeats 5 --no-phases \
                --idle-seconds 2 --preprocess gpu --tag l40_3b_stack
            step eval_3b $PY benchmarks/eval_quality.py --model LiquidAI/LFM2.5-VL-3B \
                --variants baseline,fused_gemv_pf_vfused+gpuprep \
                --limit mmstar=500,pope=1500,ocrbench=500,chartqa=500,textvqa=500 --tag l40_3b_subset ;;
        ncu)  # Nsight Compute on our kernels (1080p request of the full stack)
            step ncu_ours env NCU_SET=basic LAUNCHES=900 bash profiling/ncu/profile_ours.sh l40_stack_fhd \
                $PY profiling/run_once.py --variant fused_gemv_pf_vfused --workload img_fhd --gen-len 2 --warmup 0 ;;
        *) echo "[suite] unknown step $s" ;;
    esac
done
echo "[suite] done $(date +%T)"
