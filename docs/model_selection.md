# Model selection: LFM2.5-VL-1.6B

_Decided: 2026-10-07_

## Decision

We work on **[LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)** in BF16. **[LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)** is used only at the end, to check that our optimizations carry over to a larger model.

## Why not Qwen2-VL (the model in our proposal)

Qwen2-VL is a standard transformer decoder. Its kernel-level bottlenecks (GEMM, attention, RMSNorm) are already heavily optimized by FlashAttention, vLLM, TensorRT-LLM and others, so it would be hard to contribute anything new. LFM2.5-VL's language model is a **hybrid of gated short convolutions and attention**. Its conv block (`B·x → depthwise causal conv1d → C·y`) is not fused into a single kernel in any framework we checked (transformers, vLLM, llama.cpp), which gives us a real, measurable target for custom CUDA work. It is also newer (2026), edge-focused, and supported by transformers, vLLM and llama.cpp, so we have strong reference points.

## Variants compared

Both variants share the **same vision encoder** (SigLIP2 NaFlex so400m, 27 layers) and the **same projector**. Only the language model differs.

| | **LFM2.5-VL-1.6B** | LFM2.5-VL-3B |
|---|---|---|
| Released | Jan 2026 | Aug 2026 |
| Language model | LFM2.5-1.2B | LFM2.5-2.6B |
| LM layers | 16 (10 conv + 6 attention) | 30 (22 conv + 8 attention) |
| Hidden size / FFN | 2048 / 8192 | 2048 / 10752 |
| Vocabulary | 65,536 | 128,000 |
| Parameters (BF16 size) | 1.60B (3.2 GB) | 3.12B (6.2 GB) |
| Weights read per decoded token | ~2.34 GB | ~5.3 GB |
| Min. transformers version | 5.1 | 5.x (config written by 5.8.1) |

## Reasons for choosing the 1.6B

1. **More room for kernel-level gains.** At batch 1, decoding reads every weight once per token and runs about 300+ small kernels. In the smaller model, launch overhead and the elementwise/normalization kernels are a larger share of time, which is exactly what fusion, CUDA graphs and custom kernels address. In the larger model the share is diluted by the big GEMVs, which cuBLAS already handles well.
2. **Same kernel types in both.** The 3B adds only depth and width, with no new operator types. Everything built for the 1.6B runs on the 3B unchanged, which gives us a ready-made generalization experiment for the final report.
3. **Vision encoder is a larger share of time.** The ~400M-parameter vision tower is about a quarter of the 1.6B model, so VLM-specific optimizations (vision encoder, time to first token) matter more than in the 3B.
4. **Edge story.** At 3.2 GB in BF16 (less when quantized) it fits Jetson Orin-class devices, matching our Applications & Edge AI track.
5. **Faster iteration.** Profiling, Nsight Compute replays and quality evaluation are about 2× cheaper.

## Risks

- **Smaller absolute numbers:** gains in ms/token are small in absolute terms, so measurements must be careful (warm-up, clock locking, repeated runs).
- **Quality evaluation:** the 1.6B is weaker on knowledge-heavy tasks (per the model card), so we evaluate quality as *agreement with baseline outputs* plus a VQA/OCR subset, not absolute accuracy.
