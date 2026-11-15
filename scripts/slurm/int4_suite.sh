#!/usr/bin/env bash
# INT4 revisited on the L40S: is GPTQ (no training) enough to make 4-bit decode weights keep the answers?
#
#   bash scripts/slurm/submit.sh int4_suite 6 "bash scripts/slurm/int4_suite.sh"      # STEPS="probe bench" to subset
#
# Steps: tests (GPTQ unit tests + kernel tests), probe (4-bit schemes simulated: RTN / GPTQ / NVFP4), numerics
# (teacher-forced runtime check, also builds the GPTQ weight caches in models/gptq/), bench (speed/energy vs the BF16
# stack and INT8 in the same session), longform / coco / full (quality; baselines are the earlier L40S runs:
# results/eval/20261022-100808_L40S_l40_longform, 20261027-151452_L40S_l40_coco_captions, 20261022-091626_L40S_l40_fullsize).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv-ref/bin/python
STEPS="${STEPS:-tests probe numerics bench longform coco full}"
W4="fused_w4_pf_vfused+gpuprep,fused_w4gptq_pf_vfused+gpuprep,fused_w4gptq64_pf_vfused+gpuprep"
BENCH="$PY benchmarks/bench.py --workloads text,img_small,img_hd,img_fhd --gen-lens 1,128 --batch-sizes 1 --warmup 2 --repeats 10 --no-phases --idle-seconds 2"

# GPTQ weight caches (~0.65 GB each) on node-local scratch: the team home is at its 50 GB quota
export LFM2_GPTQ_CACHE="${LFM2_GPTQ_CACHE:-/tmp/paa_team21_gptq_$SLURM_JOB_ID}"
mkdir -p "$LFM2_GPTQ_CACHE" && df -h "$LFM2_GPTQ_CACHE" | tail -1
trap 'rm -rf "$LFM2_GPTQ_CACHE"' EXIT

step() { local name="$1"; shift; echo "[int4] $(date +%T) $name start"; "$@"; echo "[int4] $(date +%T) $name rc=$?"; }
latest() { ls -d results/eval/*_"$1" | tail -1; }

for s in $STEPS; do
    case "$s" in
        tests)
            step test_gptq $PY tests/test_gptq.py
            step test_kernels $PY tests/test_kernels.py ;;
        probe)
            step probe $PY profiling/int4_quality_probe.py --per-bench 20 ;;
        numerics)
            for g in "" "--gptq --w4-group 128" "--gptq --w4-group 64"; do
                step "numerics_w4 $g" $PY tests/check_runtime_numerics.py --fused --gemv int4 --fused-prefill $g
            done
            step numerics_w8 $PY tests/check_runtime_numerics.py --fused --gemv int8 --fused-prefill ;;
        bench)
            step bench_base $BENCH --variants baseline --tag l40_int4day_base
            step bench_ours $BENCH --variants fused_gemv_pf_vfused,fused_w8_pf_vfused,fused_w4_pf_vfused,fused_w4gptq_pf_vfused,fused_w4gptq64_pf_vfused \
                --preprocess gpu --tag l40_int4day_ours
            for d in results/raw/*_l40_int4day_*/; do $PY benchmarks/summarize.py "$d" > /dev/null; done ;;
        longform)
            step longform $PY benchmarks/eval_quality.py --benches longform --limit 500 --variants "$W4" --tag l40_w4_longform
            new=$(latest l40_w4_longform)
            cp results/eval/20261022-100808_L40S_l40_longform/{baseline,baseline@fp32,fused_gemv_pf_vfused+gpuprep,fused_w8_pf_vfused+gpuprep}.jsonl "$new"/
            step longform_compare $PY benchmarks/longform_compare.py "$new" --md ;;
        coco)
            step coco $PY benchmarks/eval_quality.py --benches coco_cap,coco_desc --limit coco_cap=1000,coco_desc=500 \
                --variants "$W4" --tag l40_w4_coco
            step caption_compare $PY benchmarks/caption_compare.py results/eval/20261027-151452_L40S_l40_coco_captions \
                "$(latest l40_w4_coco)" --md ;;
        full)
            step full $PY benchmarks/eval_quality.py --benches mmstar,pope,ocrbench,chartqa,textvqa_full --variants "$W4" \
                --tag l40_w4_fullsize
            step eval_compare $PY benchmarks/eval_compare.py results/eval/20261022-091626_L40S_l40_fullsize \
                "$(latest l40_w4_fullsize)" --md ;;
        *) echo "[int4] unknown step $s" ;;
    esac
done
echo "[int4] done $(date +%T)"
