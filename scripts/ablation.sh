#!/usr/bin/env bash
# Ablation: one optimization at a time, same grid on any GPU (batch 1, text / VGA / 1080p, 128 tokens).
#
#   TAG=l40 bash scripts/ablation.sh            # results/raw/*_<TAG>_ablation*
#
# HF -> O1 CUDA graph -> + fused kernels (cuBLAS GEMV) -> + our GEMV -> + fused prefill -> + fused vision encoder,
# all with HF's CPU preprocessing; then the full stack with GPU preprocessing, without and with the vision CUDA graph.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-.venv-ref/bin/python}"
B="$PY benchmarks/bench.py --workloads text,img_small,img_fhd --gen-lens 1,128 --batch-sizes 1 --warmup 2 --repeats 5 --no-phases --idle-seconds 2"
T="${TAG:-gpu}"
LFM2_VISION_GRAPH=0 $B --variants baseline,cudagraph,fused,fused_gemv,fused_gemv_pf,fused_gemv_pf_vfused --tag "${T}_ablation"
LFM2_VISION_GRAPH=0 $B --variants fused_gemv_pf_vfused --preprocess gpu --tag "${T}_ablation_prep"
LFM2_VISION_GRAPH=1 $B --variants fused_gemv_pf_vfused --preprocess gpu --tag "${T}_ablation_vgraph"
for d in results/raw/*_"${T}"_ablation*/; do $PY benchmarks/summarize.py "$d" > /dev/null; done
echo ABLATION_DONE
