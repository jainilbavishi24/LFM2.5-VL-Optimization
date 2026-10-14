# Experiment log

A dated record of everything we tried, measured and decided, including failures. Newest entries go at the top.

Entry template:

```
## YYYY-MM-DD: <short title>  (<who>)
**Goal:**
**Setup:** commit, GPU, driver/CUDA/PyTorch versions, model, input, generation settings
**What we did:**
**Result:** numbers + link to results/raw/<file>
**Takeaway / next step:**
```

---

## 2026-10-14 (slot 2, 16:00–18:00): GPU image preprocessing (P1), quality evaluation on 5 VLM benchmarks (Q1), fused vision encoder (O7d), fused LM prefill (O3b)  (Claude)

**Direction (decided with the team this morning):** the main results are lossless BF16 GPU/CUDA optimizations. Quantization (INT8) is only an extension, evaluated separately; INT4 is dropped. Every variant is graded on standard VLM benchmarks (accuracy) next to latency and energy.

**Where the CPU preprocessing time goes** ([profiling/preprocess_probe.py](../profiling/preprocess_probe.py), Core Ultra 9, 24 torch threads): the HF image processor (torchvision resize with antialias in uint8 → normalize → tile → patchify → pad) takes 22 ms at 720p and 69 ms at 1080p inside the processor call. It is also very noisy run to run (1080p: 25–69 ms, 720p: 4–31 ms across runs), probably from contention in the CPU thread pool. Tokenizing the expanded prompt (2327 tokens at 1080p) costs about 2–3 ms. HF's own `device="cuda"` option runs the same torchvision ops on the GPU in 2.05 ms (1080p).

**P1: fused GPU preprocessing** ([kernels/image_prep.cu](../kernels/image_prep.cu), [src/lfm2opt/preprocess.py](../src/lfm2opt/preprocess.py), `GpuImageProcessor.wrap(processor)`, `bench.py --preprocess gpu`):
- How it works:
  - The uint8 image goes to the GPU once, through a reused pinned buffer.
  - Kernel 1 is the horizontal antialiased bilinear pass into a uint8 buffer.
  - Kernel 2 is the vertical pass fused with rescale/normalize, the 512×512 tile split, 16×16 patchify and the write into the padded `pixel_values` layout. The thumbnail is a second resize job into the last tile slot.
  - The int16 fixed-point weights are computed on the host with ATen's formula for the uint8 path (Pillow-style: per-pass precision, `1 << (prec-1)` rounding, saturate). The output is therefore **bit-identical to the HF CPU path**: [tests/test_preprocess.py](../tests/test_preprocess.py) passes 28/28 (14 sizes from 64×64 to 4032×3024, including upscaling, portrait and odd sizes; photos and noise; same `input_ids`, masks, spatial shapes, `pixel_values` max diff 0).
- Timing at 1080p:
  - The kernels take 25 µs, H2D 0.22 ms, and PIL → numpy (host repack) 0.75 ms. The image processor drops from 22–69 ms to **1.46 ms** (HF on CUDA: 2.05).
  - Full processor call with tokenization: 0.7 / 1.6 / 3.9 ms (VGA / 720p / 1080p) instead of 1.4 / 5–31 / 25–63 ms.
- **End to end at 1080p** (preprocess + generate, `results/raw/20261014-1627*_prep_cpu`, `…162815…_prep_gpu`):

  | Configuration | First token | 128 tokens |
  | --- | --- | --- |
  | HF baseline | 136.5 ms | 608 ms |
  | `fused_gemv_vcudnn` + GPU preprocessing | 101.6 ms | 368 ms |

  At 720p: 69.6 → 34.5 ms to the first token.

**Runtime change:** `StaticDecoder.generate(stop_at_eos=True)`, used when `min_new_tokens < max_new_tokens` (HF semantics).
- The EOS mask is now a device-side bias that the captured graphs read at replay, so the same graph serves fixed-length benchmarking and normal generation.
- Replays stop once every row has emitted EOS (checked after 1, 2, 4, … steps, then every 16).
- Teacher-forced check unchanged (127/127 on text and img_small; 124/127 at 1080p with flips only at exact or near ties).
- New variant `fused_gemv_vcudnn`: BF16 decode + cuDNN vision attention, the best configuration without quantization.

**Q1: quality-evaluation harness** ([benchmarks/eval_quality.py](../benchmarks/eval_quality.py), [benchmarks/eval_compare.py](../benchmarks/eval_compare.py), data via [scripts/download_eval_data.sh](../scripts/download_eval_data.sh); the server needed pyarrow, fetched with parallel range requests because uv's single connection crawled):
- Benchmarks and scoring:
  - MMStar (1500; option letter)
  - POPE (3 × 3000; yes/no, accuracy + F1)
  - OCRBench (1000; substring match as in the official scorer)
  - ChartQA test (2500; relaxed accuracy, 5 %)
  - TextVQA val shard 0 (1667; VQA accuracy, leave-one-out over the 10 human answers)
- Protocol: same prompts for every variant, greedy, stop at EOS, batch 1. Each variant can carry `+gpuprep`.
- Recorded: per-question prediction, token ids, score, preprocessing and generation time; energy per benchmark from NVML.
- `eval_compare.py` reports, against the baseline: score delta, identical-answer rate, discordant pairs and the exact McNemar p-value.
- Smoke test (20 questions per benchmark): baseline and `fused_gemv_vcudnn` score the same on all five; 2 of 100 answers differ in text and are equally right.

**Q1 result: our optimizations do not change output quality** (`results/eval/20261014-163454_RTXPRO5000Blackwell_full`; POPE subsampled to 1000 per split; `+gpuprep` = our GPU preprocessing).

| Variant | Benchmark | n | Score % | Δ (pp) | Same answer % | Ref ✓ var ✗ | Ref ✗ var ✓ | McNemar p | ms/question | J/question |
|---|---|---|---|---|---|---|---|---|---|---|
| HF baseline | MMStar | 1500 | 49.07 | | | | | | 60.2 | 10.04 |
| | POPE (acc; F1 88.95) | 3000 | 89.73 | | | | | | 51.0 | 8.94 |
| | OCRBench | 1000 | 79.70 | | | | | | 117.1 | 25.88 |
| | ChartQA | 2500 | 80.20 | | | | | | 74.3 | 15.56 |
| | TextVQA | 1667 | 82.07 | | | | | | 137.3 | 34.78 |
| **`fused_gemv_vcudnn+gpuprep`** (BF16) | MMStar | 1500 | 48.87 | −0.20 | 97.6 | 9 | 6 | 0.61 | 26.9 | 8.22 |
| | POPE | 3000 | 89.80 | +0.07 | 99.8 | 0 | 2 | 0.50 | 19.8 | 6.34 |
| | OCRBench | 1000 | 79.70 | 0.00 | 94.5 | 5 | 5 | 1.00 | 66.9 | 20.53 |
| | ChartQA | 2500 | 80.04 | −0.16 | 96.6 | 8 | 4 | 0.39 | 36.2 | 11.57 |
| | TextVQA | 1667 | 82.20 | +0.13 | 98.5 | 5 | 7 | 0.77 | 84.4 | 26.13 |
| `fused_w8_vcudnn+gpuprep` (extension, INT8 weights) | MMStar | 1500 | 49.07 | 0.00 | 97.4 | 8 | 8 | 1.00 | 24.6 | 7.56 |
| | POPE | 3000 | 89.80 | +0.07 | 99.9 | 0 | 2 | 0.50 | 18.9 | 6.07 |
| | OCRBench | 1000 | 79.40 | −0.30 | 93.3 | 7 | 4 | 0.55 | 53.9 | 16.65 |
| | ChartQA | 2500 | 80.04 | −0.16 | 96.1 | 8 | 4 | 0.39 | 33.3 | 10.74 |
| | TextVQA | 1667 | 82.13 | +0.06 | 98.3 | 5 | 6 | 1.00 | 81.3 | 25.17 |

- Baseline scores are in the published range for LFM2.5-VL-1.6B, so the prompts and scorers are sane.
- **Every difference is within ±0.3 pp and none is significant (all McNemar p ≥ 0.39).** 94–99.9 % of answers are textually identical. The rest come from BF16 near-ties (different summation order in our kernels and in cuDNN attention); they flip in both directions about equally often.
- A whole question (CPU or GPU preprocessing + generation) is **1.6–2.6× faster** and uses **18–29 % less GPU energy** with our BF16 stack. GPU energy per question includes the time the GPU waits for the CPU preprocessing, which is why the baseline loses so much here.
- The INT8 extension is also quality-neutral on these benchmarks. It is a little faster per question, because short answers are dominated by prefill, not decode.

**O7d: fused SigLIP2 vision encoder** ([src/lfm2opt/vision.py](../src/lfm2opt/vision.py), [kernels/vision_ops.cu](../kernels/vision_ops.cu); variants `vision_fused` and `fused_gemv_vfused`):
- What it does:
  - q|k|v merged into one GEMM.
  - Padding-free attention: tiles grouped into runs of equal valid length, flash kernel per run without a mask. At 1080p that is 8 full tiles plus the 1008-patch thumbnail.
  - GELU (tanh) in the fc1 GEMM epilogue (cuBLASLt `GELU_BIAS` via `torch._addmm_activation`).
  - Residual add + LayerNorm fused into one warp-per-row kernel.
- Accuracy ([tests/check_vision_fused.py](../tests/check_vision_fused.py), image features at the valid patches vs an FP32 copy of the tower): relative error 5.0 / 3.7 / 4.6 % (VGA / 720p / 1080p) vs HF BF16's own 4.9 / 3.7 / 4.9 %. So it is as accurate as HF; it only rounds in a different order.
- Time ([profiling/ttft_probe.py](../profiling/ttft_probe.py), CUDA events): vision 1080p HF 77.3 → cuDNN attention 61.7 → **fused 53.9 ms**; 720p 25.6 → 20.8 → 18.8; VGA 12.0 → 10.0 → 8.1.
- Ablation at 1080p (all on: 53.4 ms):

  | Removed | Cost |
  | --- | --- |
  | Flash-varlen (cuDNN with mask instead) | +2.5 ms |
  | GELU epilogue | +2.8 ms |
  | Fused add+LayerNorm | +1.7 ms |
  | All three (only the merged QKV left) | 59.9 ms |

**O3b: fused LM prefill** (`StaticDecoder(fused_prefill=True)`; new kernels `qk_norm_rope_kv_prefill` and `shortconv_prefill` in [kernels/lfm2_decode.cu](../kernels/lfm2_decode.cu), plus the decode `add_rmsnorm` / `silu_mul` on all T rows):
- The prompt goes through cuBLAS GEMMs on the merged q|k|v and w1|w3 weights and our kernels, writing k/v and the conv state straight into the static decode buffers (no HF cache, no copy).
- Teacher-forced check: first token matches HF on text / VGA / 1080p (max |Δlogit| 0.13–0.33), and decode agreement is 126–127/127 (flips only at exact ties).
- LM prefill: text 3.9 → 2.5 ms, VGA 5.3 → 4.3, 720p 13.3 → 12.0, **1080p 37.0 → 31.8 ms**.

**Full BF16 stack** (`fused_gemv_pf_vfused`, GPU preprocessing; `results/raw/20261014-170328_*_stack_bf16`):

| Workload | First token, baseline | First token, ours |
| --- | --- | --- |
| Text | 6.6 ms | **2.6 ms** |
| VGA | 20.0 ms | **12.3 ms** |
| 720p | 39.0 ms | **29.4 ms** |
| 1080p | 111–114 ms | **85.0 ms** |

- At 1080p, 128 tokens take 582 → **352 ms**.
- With preprocessing included, 1080p is **136.5 → 89 ms to the first token (1.53×)** and **608 → 356 ms for 128 tokens (1.71×)**.

 of the night's best variants, 3B, batching, FP8 vision (rejected)  (Claude)

**Full grids** (`bench.py --preset full --no-phases`, 10 repeats; `results/raw/20261014-022*`; TPOT at 128 new tokens, J/token dynamic in parentheses):

| Variant | text | img_small | img_hd | img_fhd | J/token | TTFT 1080p |
| --- | --- | --- | --- | --- | --- | --- |
| HF eager (baseline, slot 1) | 3.83 | 3.83 | 3.84 | 3.81 ms | 0.74–0.89 | 116 ms |
| `fused` (O1–O4, cuBLAS GEMV) | 2.23 | 2.24 | 2.27 | 2.32 | 0.65–0.70 | 116 |
| `fused_gemv` (+ our BF16 GEMV) | **2.08** | 2.09 | 2.10 | 2.14 | 0.61–0.63 | 118 |
| `fused_w8` (+ INT8 weights) | **1.11** | 1.13 | 1.15 | 1.19 | 0.32–0.35 | 118 |
| `fused_w4` (+ INT4 g128, lossy) | 0.76 | 0.78 | 0.80 | 0.84 | 0.22–0.26 | 118 |
| `fused_w8_vcudnn` (+ cuDNN vision attention) | 1.11 | 1.13 | 1.15 | 1.19 | 0.32–0.35 | **102 ms** (vision 73.5 → 62.9) |

- With our GEMV everywhere, the BF16 step beats cuBLAS inside the graph (2.23 → 2.08 ms). The GPU now runs at its 300 W limit during decode (it is busy streaming weights rather than waiting), so energy per token falls in proportion to the bytes: **INT8 = 2.3× less energy per token than HF eager**.
- **LFM2.5-VL-3B (generalization, `3b_fused`):** TPOT 6.50 (HF) → 4.95 (`fused`, cuBLAS) → **2.50 ms (`fused_w8`)**, J/token 1.81 → 0.74. Same kernels, unchanged (30 layers, 10752-wide MLP).
- **Batching (`fused_batch`, img_small):** the fused runtime also runs batched through the cuBLAS path: batch 4: 2.60 ms, 1538 tok/s (HF 1021); batch 16: 2.72 ms, **5878 tok/s** (HF 3972).
- **FP8 vision GEMMs (O7b, `vision_fp8`): rejected.** W8A8 E4M3 (per-channel weights, dynamic per-token activations, `torch._scaled_mm`) made the vision tower *slower* (1080p TTFT 97 → 136 ms: the eager activation-quantization kernels cost more than the FP8 GEMMs save) and inaccurate (image-feature cosine as low as 0.50, outputs diverge at token 7). Would need fused quantization and finer (per-block) activation scales.

**Batched decode with our GEMV (02:40, tag `batch_sweep_final`, img_small, tok/s at 128 tokens):** the direct GEMV now takes x [B, K] and reads each weight chunk once for up to 8 sequences (105/105 tests incl. B = 2, 3, 4, 7, 8, 13).

| batch | HF eager | `fused_gemv` (BF16) | `fused_w8` (INT8) |
| --- | --- | --- | --- |
| 1 | 272 | 480 | **895** |
| 4 | 1039 | 1819 | **2929** |
| 8 | 2051 | 3066 | **3696** |
| 16 | 3987 | **5937** | 3835 → 5900 with the hybrid rule |

From batch 8–16 the GEMV becomes compute-bound (INT8→FP32 conversion and FMA per weight per sequence) while cuBLAS uses tensor cores. Dispatch rule now: our GEMV for BF16 up to batch 4 and INT8 up to batch 8, otherwise cuBLAS on BF16 weights (the INT8 runtime keeps the merged BF16 copies, +1.1 GB). A W8A8 tensor-core GEMM would be the proper large-batch INT8 path.

**INT4, two more attempts (02:43):** (1) AWQ-style activation-aware scales (calibrated on 4 prompts *different* from the benchmark prompts, `fused_w4_awq`, `quant.quantize_int4_awq`): g64 + clip 92 % argmax agreement (91 % without AWQ), g128 89 %; median |Δlogit| still ~2. (2) Sensitivity, INT8 everywhere except one weight kind in INT4 (g64 + clip; img_small + img_fhd, 254 steps): qkv 96.5 %, o 98.4 %, in 97.6 %, out 97.6 %, w1‖w3 96.1 %, w2 96.5 %, lm_head 96.5 % (INT8 alone: 98.4 % on these two inputs). Every kind costs something and the errors add up; no cheap subset. **Conclusion: INT8 is the deployable setting; good INT4 needs GPTQ-class methods (future work).**

**`torch.compile` on the vision tower (O7c, `fused_w8_vcompile`): rejected.** 1080p TTFT 97.8 → 118 ms (img_small 15.8 → 18.4), outputs identical: inside the compiled graph Inductor's GEMMs and its SDPA choice lose to cuBLAS + cuDNN.

**Error bars (02:46–03:46, `plan_refs.sh` filler cycling the three headline variants, 13–14 full grids each, same code; `python benchmarks/aggregate_runs.py --after 20261014-0225`), all in `.venv-ref`:**

| variant | TPOT text | TPOT img_fhd | J/token text | TTFT img_fhd |
| --- | --- | --- | --- | --- |
| HF eager | 3.716 ± 0.039 ms | 3.694 ± 0.046 | 0.882 ± 0.029 | 118.9 ± 0.2 ms |
| `fused_gemv` (BF16) | **2.078 ± 0.001** (1.79×) | 2.141 ± 0.001 | 0.594 ± 0.010 | 118.7 ± 0.2 |
| `fused_w8` (INT8) | **1.110 ± 0.001** (3.35×) | 1.188 ± 0.001 | **0.334 ± 0.005** (2.6× less) | 118.2 ± 0.2 |

Our CUDA-graph runtime is also ~40× more repeatable than eager HF (σ 0.001 vs 0.04 ms): no Python/launch jitter in the decode loop.

**Night summary:** batch-1 decode on the Blackwell went from 3.83 ms (HF eager) to 2.08 ms in BF16 (1.84×, beats vLLM 2.36 and llama.cpp BF16 2.26), and to 1.11 ms with INT8 weights (3.45×, beats llama.cpp Q8_0 1.45; 98.7 % teacher-forced agreement). 1080p TTFT 116 → 102 ms. All of it is CUDA graphs plus our own kernels (fused elementwise/norm/conv/RoPE/attention, BF16/INT8/INT4 GEMV, PDL).

---

## 2026-10-14 (night, 02:00–02:25): GEMV tuning, PDL, INT4 quality sweep, vision attention, vLLM TTFT  (Claude)

**GEMV ([kernels/lfm2_decode.cu](../kernels/lfm2_decode.cu), microbenchmark [profiling/bench_gemv.py](../profiling/bench_gemv.py): real decode shapes, > 300 MB of distinct weights per shape so L2 can't help):**
- Several rows per warp (x slice reused): small gain (INT8 1.40 → 1.33 ms TPOT).
- Transposed (bank-conflict-free) x in shared memory: ~5 % *slower*. Bank conflicts were not the problem.
- **The problem was the prologue:** every block first copied x (up to 16 KB) into shared memory and synchronized before any math. Each lane loading its own x slice straight from L1/L2 (`gemv_direct_kernel`, 1 row/warp, 4 chunks in flight) brings every shape to 1.1–1.23 TB/s (INT8 qkv 8.7 → 5.4 µs, o 8.4 → 3.7 µs, w2 21.7 → 15.2 µs). **INT8 TPOT 1.33 → 1.11 ms** (3.45× HF eager); INT4 0.74 ms.
- Programmatic Dependent Launch (`griddepcontrol`, weights prefetched before the wait): correct, but only ~1 % inside the CUDA graph. Kept on (`LFM2_PDL=0` to disable); with PDL on, profiler kernel times overlap, so profile with it off.

**INT4 quality sweep** (teacher-forced argmax agreement vs HF BF16, 381 steps): g128 RTN 86 %; g128 + clip search 91 %; g64 + clip 91 %; g32 + clip 93 %; g64 + clip with lm_head and w2 in INT8 93 %; g32 + clip with lm_head, w2, o, out in INT8 93 % (median |Δlogit| 1.1–1.5). INT8: 98.7 %, median 0.25. **Round-to-nearest INT4 is too lossy for this 1.6B model; calibrated methods (AWQ/GPTQ) would be needed.** INT8 is the quality-safe configuration.

**Vision attention (O7a, [profiling/vision_attn_probe.py](../profiling/vision_attn_probe.py), 9 tiles × 1024 patches × 16 heads × 72):** with HF's padding mask PyTorch picks the sm80 memory-efficient kernel (1038 µs/layer, 42 TFLOPS); **cuDNN: 419 µs (2.5×)**, same accuracy; flash only without a mask (287 µs). Variant `vision_cudnn` → **1080p TTFT 113 → 98 ms**. Next: pack only valid patches and use flash (varlen), FP8 for the vision GEMMs (35.7 ms at ~200 TFLOPS BF16).

**vLLM, honest TTFT** (prefix cache and multimodal cache off; the first run served repeated requests from cache, "TTFT 6 ms at 1080p"): text 5.5, img_small 19.3, img_hd 49.3, **img_fhd 127.7 ms** (HF: 6.9 / 20.2 / 39.8 / 116); TPOT 2.36 ms. vLLM without CUDA graphs (`enforce_eager`): TPOT 3.8 ms, same as HF eager.

---

## 2026-10-14 (night): O2–O4 fused CUDA kernels + O6 INT8 weight-only GEMV  (Claude, overnight slots)

**Goal:** after O1 the step is GPU-bound at ~2.9 ms: ~700 kernels, of which ~600 tiny elementwise/norm/cat ops. Fuse them, then attack the weight bytes.

**Setup:** `.venv-ref` (uv's Python 3.12.15 with headers; the system Python can't build extensions), torch 2.11.0+cu130, nvcc 13.3, sm_120. Extension = [kernels/lfm2_decode.cu](../kernels/lfm2_decode.cu) (kernels + raw-pointer launchers) + [kernels/lfm2_decode.cpp](../kernels/lfm2_decode.cpp) (PyTorch bindings; nvcc 13 cannot parse ATen's `List_inl.h`, so torch headers stay out of the .cu), JIT-built by [src/lfm2opt/kernels.py](../src/lfm2opt/kernels.py). Runtime: `StaticDecoder(fused=True, gemv=...)`, variants `fused` / `fused_gemv` / `fused_w8`.

**Kernels (each replaces a chain of PyTorch ops, keeping HF's BF16 rounding points):**

| Kernel | Replaces | Per step |
| --- | --- | --- |
| `add_rmsnorm` | residual add + RMSNorm (pow, mean, rsqrt, mul, cast, mul) | 33 |
| `shortconv_step` | B·x, cat, state copy, depthwise conv, C·y, transposes | 10 |
| `qk_norm_rope_kv` | q/k head RMSNorm, RoPE (cos/sin from the device-side position), K/V cache write | 6 |
| `decode_attention_split` + combine | SDPA over the static cache: flash-decoding, grid (8 KV heads × 16 splits), the 4 query heads of a KV head share K/V loads | 12 |
| `silu_mul` | SiLU + mul on the merged w1‖w3 output | 16 |
| `gemv` (BF16 / INT8) | `nn.Linear` at batch 1: one warp per row, 16-byte `ld.global.nc.L1::no_allocate` loads, 4 in flight per lane, x in shared memory | 65 |

Plus merged projections (q‖k‖v: 1 GEMV instead of 3; w1‖w3: 1 instead of 2). Step: **~700 → 149 kernels**.

**Correctness:** [tests/test_kernels.py](../tests/test_kernels.py) 27/27 at the real shapes: norm, conv, QK-norm/RoPE/KV and SwiGLU are **bit-exact** vs the PyTorch reference; attention max error 5e-4 (one BF16 ulp); GEMVs within accumulation-order noise. End to end, teacher-forced vs HF ([tests/check_runtime_numerics.py](../tests/check_runtime_numerics.py) `--fused`): argmax agrees on 376/381 steps, max |Δlogit| 0.28, every flip at an HF top-2 margin ≤ 0.06.

**Results (quick tests, batch 1, 128 tokens; full grids are in the overnight plan):**

| Variant | TPOT text | TPOT img_fhd (ctx 2.3k) | J per 128-token request (text) |
| --- | --- | --- | --- |
| baseline (HF eager) | 3.83 ms | 3.81 ms | ~100 J |
| O1 `cudagraph` | 2.91 | 2.94 | ~88 |
| + O2–O4 `fused`, first attention kernel | 2.23 | 2.65 | 84 |
| + split-K attention | 2.23 | **2.27** | 84 |
| + our BF16 GEMV (`fused_gemv`) | 2.24 | 2.28 | 70 |
| **+ O6 INT8 weights (`fused_w8`)** | **1.40** | **1.43** | **45** |
| llama.cpp BF16 / Q8_0 (reference) | 2.26 / 1.45 | – | – |

- After fusion, `profiling/profile_decode_step.py` shows the step is **94 % GEMV** (2.10 of 2.24 ms). The practical DRAM read roof measured on this card is 1.23 TB/s (a 2 GB `sum`; the 1.34 TB/s spec is not reachable), and cuBLAS's GEMVs already reach ~1.11 TB/s (~90 %). So our own BF16 GEMV only matches cuBLAS, and the BF16 floor is ~1.9 ms.
- **O6 halves the bytes:** INT8 weights with a symmetric per-output-row scale (computed at load, decode only; prefill keeps the BF16 weights). TPOT 2.24 → 1.40 ms; energy per request roughly halves. Quality, teacher-forced vs BF16 HF: argmax 376/381, max |Δlogit| 0.52, flips only at margins ≤ 0.125. Needs a task-level check (captions / VQA accuracy) before the report claims it.
- The INT8 GEMV runs at ~75 % of the roof (1.17 GB in ~1.26 ms), so it has more headroom than the BF16 one. Next: more rows per warp so each x load is reused, `dp4a`-style integer dot or packed conversions.
- Pitfalls hit tonight: (1) `torch.compile`/Triton and extension builds fail with the system Python (no `Python.h`); (2) a 16-lane-group loop must keep the same trip count on all lanes before `__shfl_xor_sync`; (3) `bench.py` kept earlier variants' models alive in multi-variant runs (the instrumentor held a reference), inflating "peak memory". Fixed.

**Takeaway / next step:** batch-1 decode is now 2.7× faster than HF eager (3.83 → 1.40 ms) with INT8 weights, or 1.7× with BF16, and beats llama.cpp at both precisions. Next: a task-level INT8 quality eval, INT8 GEMV tuning, prefill/vision (TTFT is still HF: 116 ms at 1080p), and full-grid + energy numbers.

---

## 2026-10-13/14: O1 first cut: static decode runtime + CUDA graph  (Claude, overnight slots)

**Goal:** remove the host/launch overhead that bounds decode (703 kernels/step, GPU busy 45 %).

**Setup:** e707a4c+, Blackwell server, `.venv` (system Python 3.12.3), torch 2.11.0+cu130, BF16, variant `cudagraph` = [src/lfm2opt/runtime.py](../src/lfm2opt/runtime.py).

**What we did:** prefill stays in Hugging Face; its KV cache (6 attention layers) and conv states (10 short-conv layers) are copied into static buffers (`[B, 8, max_len, 64]` per attention layer, `max_len` rounded up to 256; `[B, 2048, 3]` per conv layer). One decode step is our own function over those buffers: K/V written in place with `index_copy_` at a device-side position, attention = matmul → softmax (FP32) → matmul over all `max_len` slots with a length mask, conv state shifted in place, greedy argmax and the output-token write inside the step. The step is captured once per (batch, max_len) as a `torch.cuda.CUDAGraph` and replayed per token; the host only reads the tokens at the end. EOS logits are masked like HF's `min_new_tokens` (first attempt didn't, and diverged by picking EOS).

**Result (`results/raw/20261013-235*_o1_cudagraph_full`, batch 1, 10 repeats):**

| Input | TPOT baseline → O1 | tok/s | TTFT |
| --- | --- | --- | --- |
| text | 3.83 → 2.91 ms | 261 → 344 | 6.9 → 4.5 ms |
| img_small | 3.83 → 2.92 ms | 261 → 343 | 20.2 → 17.8 ms |
| img_hd | 3.84 → 2.93 ms | 260 → 341 | 39.8 → 38.1 ms |
| img_fhd | 3.81 → 2.93 ms | 262 → 342 | 116 → 116 ms |

(TTFT for text/img_small drops because the first token no longer goes through `generate()`'s setup.)
- **Correctness** ([tests/check_runtime_numerics.py](../tests/check_runtime_numerics.py), teacher-forced against HF, 128 tokens × 3 inputs): argmax agrees on 377/381 steps; max |Δlogit| 0.35 (median 0.13); every disagreement is at an HF top-1/top-2 margin of 0.0–0.06, i.e. a BF16 near-tie. Free-running greedy outputs are identical for 4/6 configs, the other two branch at such a tie into equally fluent text.
- For comparison, `torch.compile(mode="reduce-overhead")` + HF static cache gives ~2.75 ms (smoke test): Inductor also fuses the elementwise ops, which our eager-op graph doesn't.
- What is left: the graph removes launch gaps, so TPOT ≈ GPU kernel time (~2.9 ms, matching the profiler's 2.98 ms/step). Against the ~1.7 ms weight-read floor, the remaining ~1.2 ms are the ~600 small elementwise/norm/cat kernels and GEMVs at ~70 % of peak bandwidth.

**Takeaway / next step:** O1 works (+31 % decode throughput, numerically equivalent). Next: O2–O4 fusions inside the graph (residual+RMSNorm, gated short-conv, QK-norm+RoPE, merged projections), then a GEMV kernel tuned for the decode shapes. Also: energy per token falls less than time, because the GPU stays at ~240 W; to investigate with clocks/power traces.

---

## 2026-10-13: Slot 1 on the server: RTX PRO 5000 Blackwell baseline + full profiling  (Jainil + Claude)

**Setup:** commit 98f6b98 → e707a4c, RTX PRO 5000 Blackwell 48 GB (sm_120, 110 SMs, 96 MB L2, 300 W), driver 595.84 (CUDA 13.2), torch 2.11.0+cu130, transformers 5.19.0, BF16 (native), Ubuntu 24.04, Core Ultra 9 285K. LFM2.5-VL-1.6B @ 919fde3 (sha256 verified), no `causal-conv1d` (PyTorch fallback, same as T4).

**What happened in the slot (lessons for the next ones):**
- Setup took 47 min instead of 20: the server's internet is capped at ~4.5 MB/s in total and **~300 KB/s per connection**, so uv timed out on big CUDA wheels (default 30 s timeout) and a single-stream model download would have taken 2.5 h. Fix: `UV_HTTP_TIMEOUT=900`, and the weights fetched with 24 parallel HTTP range requests (`pget.sh`, ~4.5 MB/s → 3 GB in 13 min). Our laptop → server LAN is even slower (~0.2 MB/s): never push big files.
- The laptop changed network mid-slot and the SSH master died, but **processes on the server survived** (only the slot end kills them). The duplicate setup job we launched then fought over uv's cache lock.
- **ncu works without root** (`RmProfilingAdminOnly=0`, counters OK as uid 1022), thanks to the TA. System Nsight Systems 2026.1 / Compute 2026.2 support sm_120.
- The system Python has no headers (`Python.h`), so anything that compiles C at runtime (Triton / `torch.compile`, `causal-conv1d`) fails in `.venv`. Fix: `.venv-ref` on uv's standalone Python 3.12.15 (same torch/transformers).
- Everything is 3–6× faster than the T4 estimates: the whole slot-1 preset + backlog (22 steps incl. 9 ncu runs) took **44 min**.

**Result (baseline, batch 1, `results/raw/20261013-2246*_RTXPRO5000Blackwell_*`):**

| Input | TTFT | vision | LM prefill | TPOT | J/token |
| --- | --- | --- | --- | --- | --- |
| text | 6.9 ms | – | 4.0 ms | 3.83 ms | 0.74 |
| img_small (1 tile) | 20.2 ms | 11.9 | 5.2 | 3.83 ms | 0.77 |
| img_hd (3 tiles) | 39.8 ms | 24.4 | 12.1 | 3.84 ms | 0.83 |
| img_fhd (9 tiles) | 116 ms | 73.5 | 33.9 | 3.81 ms | 0.89 |

- **Decode is overhead-bound, even more clearly than on the T4:** TPOT is 3.8 ms for every input and stays 3.86 ms at 1024 tokens; batch 1 → 16 only moves TPOT 3.85 → 4.05 ms (260 → 3947 tok/s, J/token 0.85 → 0.06). The weights-read floor is ~1.7 ms. Under the PyTorch profiler: 703 kernels/step, kernel time 2.98 ms/step (72 % GEMV), GPU busy 45 %.
- The GPU draws ~200–250 W at batch 1 even though it waits on the CPU most of the time, so overhead costs energy too.
- **Attention:** on Blackwell, SDPA picks the flash kernel with native GQA in the model (no mask), unlike the T4's FP32 math path. `sdpa_repeat_kv` is therefore no help here (TPOT 3.95 → 4.21 ms, slightly worse): O5 was a T4-specific problem.
- ncu (decode step, img_small): GEMVs reach ~70 % of peak DRAM bandwidth, `lm_head` 94 %; the ~600 small elementwise/reduce/cat kernels are <2 % of peak each. (The DRAM-bytes metric we collect reads 0 on sm_120: fix the metric name in `profile_ncu.sh`.)
- Reproducibility: `bench_repeat` 40 min after `bench_full` matches within ~1 %.

**First reference baseline (O0b, `torch.compile` + static cache, smoke test):** compiles to one CUDA graph without graph breaks once our hooks are fully detached (`--no-phases` used to leave them attached; fixed). TPOT 3.76 → ~2.75 ms. The remaining gap to the ~1.7 ms floor is `generate()`'s own per-token Python work (logits processors, stopping check with a host sync, `cat` of `input_ids`): the case for O1 (our own static decode loop + CUDA graph) is now measured.

**Takeaway / next step:** Eval-1 baseline and profiling on the real GPU are done. Overnight (extra slots): O0a `causal-conv1d`, O0b/O0a+b in full, vLLM and llama.cpp references, 3B generalization run. Then O1.

---

## 2026-10-13: Dress rehearsals for the first server slot (Colab T4) + backlog results

**Goal:** run everything tonight's slot will run, end to end, before touching the server.

**Rehearsal 1, `SUITE=slot1 SLOT_MINUTES=100`:** all 11 steps DONE, none failed or skipped, 98 of 100 min. Step times (T4): timing block 22 min, PyTorch profiler 2.5, Nsight Systems 1.2, attention probe 0.6, ncu decode 28.5, ncu vision 15.5, ncu prefill 27.7 (its timeout was correctly capped to the 29 min left).

**Rehearsal 2:** (a) the no-root Nsight 2026.3 install (`dpkg -x` into a user folder) profiles kernels with both `ncu` and `nsys`; (b) a too-short deadline correctly SKIPs `bench_full` and continues; (c) an epoch `DEADLINE` (what `remote.sh slot` passes) works; (d) all backlog steps run: `sdpa_rkv` 2.5 min, `bench_long` 6.3, `bench_batch_big` 4.2, `model_3b` 16.8 (incl. 6.2 GB download). Raw: `results/raw/20261013-*_T4_*`.

**New results from the backlog runs (T4, FP16; this VM was ~10–15 % slower than the 7 Oct one, so compare only within a run):**

| Experiment | Result |
| --- | --- |
| `sdpa_repeat_kv` vs baseline, img_fhd | LM prefill 512 → 293 ms (−43 %), TTFT 1070 → 868 ms (−19 %); TPOT unchanged (decode is overhead-bound); **outputs token-identical** (4/4) |
| `sdpa_repeat_kv`, img_small | prefill 34 → 31 ms; TPOT 24.4 → 22.2 ms; outputs identical |
| 1024-token answers | TPOT averages 25.4 ms over 1024 tokens vs ~21–23 ms over 128 → decode slows as the context grows |
| Batch 1 / 16 / 32 / 64 | 43 / 525 / 831 / 1195 tok/s; no OOM at 64 |
| LFM2.5-VL-3B baseline | TPOT 42–47 ms (≈ 2× the 1.6B: 30 layers → more launches per token); TTFT img_fhd 1835 ms (LM prefill 1149 ms) |

**Takeaway:** the slot-1 job and its backlog are rehearsed end to end. The O5 quick win is real for long prompts (−43 % prefill at 2300 tokens) and doesn't change outputs.

---

## 2026-10-13: Attention-path probe on the T4 corrects our explanation of the FP32 math path

**Setup:** `profiling/sdpa_probe.py`, first run (inside the Colab dress rehearsal of `slot1`), T4, FP16. Raw: `results/raw/20261013-*_T4_sdpa_probe.json` (on Colab; pulled with the rehearsal).

**Results:**

- **Inside generate() at batch 1, transformers passes no attention mask** and therefore uses `enable_gqa=True` (prefill 6/6 calls, decode 42/42 calls).
- **On the T4, `enable_gqa=True` itself forces the FP32 math backend:** neither flash (not on sm_75) nor memory-efficient attention supports GQA there.
- Expanding K/V (`repeat_kv`) instead lets SDPA use the memory-efficient kernel:

  | Case | `enable_gqa` (today) | `repeat_kv` + mem-efficient | Speed-up |
  | --- | --- | --- | --- |
  | Prefill 2327 tokens, causal | 45.6 ms | 1.9 ms | ~23× |
  | Prefill 252 tokens, causal | 0.60 ms | 0.04 ms | ~15× |
  | Decode, context 2304 | 0.63 ms | 0.18 ms | 3.6× |
  | Decode, context 256 | 0.68 ms | 0.05 ms | ~15× |

**Correction:** the 2026-10-08 entry said the math path comes from an attention mask + `repeat_kv`. That was wrong: no mask is passed at batch 1; the cause is `enable_gqa=True` on a GPU without a GQA-capable fused kernel. The large per-step K/V copies are the math path's FP16→FP32 casts, not `repeat_kv`.

**Takeaway:** on the T4, O5 has an almost free first step (force `repeat_kv` + memory-efficient SDPA); on sm_80+ the flash backend may support GQA natively, which tonight's server probe will show. A proper GQA-aware kernel (no K/V expansion) remains the real O5.

---

## 2026-10-13: Measurement GPU changed again: course "RTX 5000" server, 3 × 2-hour slots/week

**Change:** instead of the L40 (6 h/week) we use the course "RTX 5000" server: team account over SSH, **three 2-hour slots per week**. The exact card (Quadro RTX 5000 / RTX 5000 Ada / RTX PRO 5000 Blackwell) is not confirmed yet; `scripts/preflight.sh` will tell. The team password is never stored in the repo; access will be via SSH key (`ssh-copy-id`).

**What changed for 2-hour slots:**

- `run_suite.sh` rewritten: fine-grained steps run in **priority order**; `SLOT_MINUTES` / `DEADLINE` → a step starts only if its estimate fits, every step has a timeout (2× estimate, capped by time left); the log ends with `DONE / FAILED / SKIPPED` so skipped steps carry over to the next slot; `EST_SCALE` rescales the T4-based estimates once the server is calibrated. New presets `slot1`, `slot2`; the attention probe is a suite step (`sdpa`).
- `remote.sh`: generic `gpu` alias, `slot <preset> <min>` (one command per slot, in tmux), `watch <job>` (syncs results every 10 min until the job ends), `stop`, one-time key setup and a ControlMaster fallback if the server refuses keys.
- Session guide renamed `notes/l40_sessions.md` → [notes/gpu_sessions.md](notes/gpu_sessions.md) and rewritten for 2-hour slots (slot-1 minute plan, following slots, failure playbook). Docs made hardware-neutral; card-specific numbers live in its §1 table.
- **Dress rehearsal:** `SUITE=slot1 SLOT_MINUTES=100` started on the Colab T4 to test the new suite, deadline logic and `sdpa_probe.py` before any real slot.

---

## 2026-10-11: T4 Nsight Compute at long context (decode step 512, 1080p input)

**Goal:** close the two profiling gaps of 2026-10-07: decode cost late in a generation / at long context, and vision + prefill on the 9-tile 1080p input.

**Setup:** same as 2026-10-07 (T4, FP16, torch 2.11.0+cu130, transformers 5.19.0, baseline). New option `run_once.py --start-at-decode-step N`. Reports: `profiling/reports/20261011-*_T4_ncu_*` (CSVs in git, `.ncu-rep` local only).

**Results (Nsight Compute; clocks locked to base, so use the ratios):**

| Decode step | Context | GPU time / step | DRAM / step | GEMVs (105) | Elementwise (578) | Attention FP32 bmm (6) | KV `cat` copies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1, img_small | ~250 | 17.3 ms | 3.70 GB | 11.5 ms (80 % BW) | 3.4 ms | 0.11 ms | 0.09 ms |
| 512, img_small | ~760 | 18.2 ms | 3.92 GB | 11.5 ms | 4.0 ms | 0.27 ms | 0.12 ms |
| 1, img_fhd | ~2330 | **21.2 ms** | **4.63 GB** | 11.9 ms | **6.0 ms** | 0.73 ms | 0.24 ms |

- Decode GPU time grows **+23 %** from 250 to 2330 tokens of context; the weight GEMVs stay flat, so all of the growth is attention-related.
- DRAM traffic grows by **0.93 GB per step**, while the actual KV cache at 2330 tokens is only ~28 MB (12 KB/token): **~33× more traffic than the cache size**, from `repeat_kv` expanding K/V to 32 heads, FP16→FP32 casts of the math path, the FP32 score tensors and the `torch.cat` re-allocation. A GQA-aware attention over a static cache (O1 + O5) should remove almost all of it.
- **Vision, 9 tiles:** same pattern as 1 tile, scaled ~linearly (8.5× time for 9 tiles): attention (mem-efficient FMHA) **43 % of vision time** at SM 23 %, DRAM 6 %, occupancy 25 %; GEMMs 49 % at SM 72 %.
- **Prefill, complete captures (796 kernels = all of prefill + lm_head; `20261011-1*_T4_ncu_full_prefill_*`):**

  | LM prefill | GEMMs (FP16 tensor cores) | FP32 math-path attention (softmax + FP32 bmm) | Elementwise / copy / reduce |
  | --- | --- | --- | --- |
  | img_small, 252 tokens | 67 % | ~4 % | ~23 % |
  | img_fhd, 2327 tokens | 37 % | **30 %** (softmax 13 %, sgemm 17 %) | ~30 % (incl. mask add / where / isneginf) |

  So at long prompts the attention math path is the second-largest prefill cost (O5); at short prompts prefill is GEMM-bound.

**Lesson:** the `LAUNCHES` caps were too low for prefill: the PyTorch profiler counts 795–845 prefill kernels and 430–460 vision + 7–63 projector kernels, but prefill was capped at 500 (vision at 500 was enough for 1 tile, missed the last projector call for 9 tiles). Caps raised in `run_suite.sh` (prefill 1000, vision 600) and prefill re-captured completely for both inputs (the partial 500-kernel captures are kept but superseded).

**Status:** T4 baseline profiling is complete.

---

## 2026-10-11: Measurement GPU changed to L40 (6 h/week over SSH)

**Change:** we get an **NVIDIA L40 48 GB** (Ada, sm_89, 864 GB/s, 300 W), reached over SSH, **6 hours per team per week**, instead of the RTX PRO 5000 Blackwell (8 h/week) assumed so far. Earlier log entries mentioning the RTX PRO 5000 are kept as written.

**Consequences:**

- Decode floor 2.7 ms/token (was 1.7 ms assumed); baseline precision BF16.
- FP8 weights still possible (Ada FP8 tensor cores); NVFP4 dropped (Blackwell-only).
- The SDPA flash backend exists on sm_89, so whether the T4's FP32 math attention path also happens there is an explicit session-1 question (`profiling/sdpa_probe.py`).
- 6 h is tight, so slots are planned to the minute with priorities, and nothing is debugged on the L40: [notes/l40_sessions.md](notes/l40_sessions.md).

**Tooling added:** `scripts/remote/remote.sh` (push / setup / tmux `bg` / log / incremental `sync`), `scripts/preflight.sh` (GPU, other users, driver, nvcc, sudo, ncu counter access, internet, disk), driver-aware and idempotent `setup_env.sh` (uses `uv` if present, pins torch 2.11.0), `run_suite.sh` presets (`SUITE=timing|profile|ncu|variant`) and automatic ncu skip without counter access, `profiling/sdpa_probe.py`.
None of the new L40 tooling has run against the real server yet (no access details); `sdpa_probe.py` will be rehearsed on Colab first.

**Lesson (Colab):** the `colab.sh bg` launcher used `mkdir … && setsid … &`, which backgrounds the whole `&&` list in a subshell that holds the output pipe, so the launching call never returned and blocked the Colab kernel for the job's whole run. Fixed with `;`. A kernel restart freed it without killing the `setsid` job. The VM of 2026-10-08 expired before its `ncu_ctx` reports were pulled; they are being re-run.

---

## 2026-10-08: Optimization plan written

**What we did:** turned the T4 profiling into a step-by-step plan with exact code locations, evidence, design, checks and open questions: [notes/optimization_plan.md](notes/optimization_plan.md).

**New details found while writing it:**

- Per-op count of one decode step (791 kernels): 93 GEMV, ~360 RMSNorm pieces, ~130 attention non-GEMM (RoPE, KV `cat`, `repeat_kv`, math-path SDPA), ~70 short-conv non-GEMM, 32 SwiGLU, 32 residual adds, ~30 generate() glue.
- ~~Cause of the FP32 math attention path: mask → `repeat_kv`~~ **Wrong, corrected 2026-10-13:** at batch 1 no mask is passed and `enable_gqa=True` is what forces the math backend on the T4 (see the 2026-10-13 probe entry).

**Next step:** O1 (static cache + CUDA graphs) and O2 (fused RMSNorm) on the T4; O0 reference baselines in the same Colab run.

---

## 2026-10-07: Baseline characterization on Colab T4

**Goal:** Build the benchmark/profiling harness, debug it on the free Colab T4, and get a first baseline + bottleneck picture before our limited RTX PRO 5000 hours.

**Setup:** Tesla T4 (sm_75, 70 W cap, 320 GB/s), driver 580.82.07, CUDA 13.0, torch 2.11.0+cu130, transformers 5.19.0, LFM2.5-VL-1.6B @ `919fde3`, **FP16** (T4 has no BF16), SDPA attention, `causal-conv1d` **not** installed (torch fallback for the conv). Greedy decoding, fixed output length. Commit `23f4db7-dirty` / `18d519f`.
Raw data: `results/raw/20261007-*_T4_baseline_*` (summary.md in each), profiles in `profiling/reports/20261007-14*_T4_*`.

**What we did:** `scripts/run_suite.sh` = benchmark grid (4 workloads × gen {1,32,128,512} × 10 repeats), batch sweep (1–16), FP32-vs-FP16 output check, hook-overhead check, PyTorch-profiler kernel attribution (3 workloads), Nsight Systems (2 workloads), Nsight Compute (decode step, vision, prefill).

**Results (medians, batch 1):**

| Workload | Tiles | Image tokens | TTFT ms | Vision ms | LM prefill ms | TPOT ms | Decode tok/s | Decode J/token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 0 | 0 | 32 | – | 20 | 20.0 | 50 | 1.43 |
| img_small (640×480) | 1 | 234 | 121 | 54 | 44 | 21.3 | 47 | 1.47 |
| img_hd (1280×720) | 3 | 764 | 402 | 226 | 158 | 20.1 | 50 | 1.42 |
| img_fhd (1920×1080) | 9 | 2300 | 1319 | 663 | 634 | 20.7 | 48 | 1.47 |

(TPOT/J from gen=128 vs gen=1. Idle power 14.4 W; average power during inference ~65 W.)

Batch sweep (img_small, 128 tokens): TPOT 20.3 → 25.5 ms from bs 1 → 16, i.e. 49 → 627 tok/s and 1.43 → 0.11 J/token.

**Findings:**

1. **Decode is launch/CPU-overhead bound, not GPU bound.** 788 kernels per decode step; GPU kernel time ≈ 11 ms of a ~20 ms step (profiler: GPU busy 32–34 %). TPOT is the same (~20–22 ms) whether the SM clock is ~600 or ~1560 MHz, and batching 16× costs only +25 % step time.
2. **The weight GEMVs are already efficient:** 92 GEMVs/step (exactly the 92 linear layers) are 85 % of decode kernel time and run at ~80 % of T4 peak DRAM bandwidth (lm_head 96 %, Nsight Compute). Not the place to win.
3. **~690 small kernels per step use only 1–5 % of peak bandwidth.** Main sources: RMSNorm decomposed into 8 kernels × 45 norms/step = ~360 kernels (fp32 cast, pow, mean, add eps, rsqrt, mul, cast back, mul weight; corrected 2026-10-08 from the per-op counts, first estimate was ~315), elementwise gates in the short-conv blocks, dtype copies, KV-cache `torch.cat`.
4. **SDPA falls back to the *math* backend for the LM attention** (GQA + mask, no flash on Turing): FP32 bmm + softmax + `isneginf/where/all` safe-softmax kernels. Cheap in decode for short contexts, but for the 1080p image it materializes 32×2327×2327 FP32 scores in prefill: ~230 ms of the 634 ms LM prefill (softmax 90 ms, mask add 50 ms, where 37 ms, isneginf 19 ms, casts ~30 ms), and copies the whole KV cache to FP32 every decode step. Likely T4-specific (flash backend supports GQA on sm_80+) → verify on RTX PRO 5000.
5. **TTFT for high-res images is vision + prefill compute:** ~70 ms per tile in SigLIP2 (GEMMs 321 ms + mem-efficient attention 274 ms for 9 tiles) and LM prefill grows with image tokens.
6. **Energy:** the T4 sits at its 70 W power cap (avg ~65 W) during inference and throttles (SM clock 465–1590 MHz). Decode ≈ 1.45 J/token (≈ 1.1 J/token above idle). Since time is overhead-bound, removing overhead should cut energy/token roughly proportionally.
7. **FP16 is numerically safe on T4:** greedy outputs identical to FP32 for 64 tokens (text and image).
8. **Instrumentation overhead:** phase hooks cost ≤ ~7 % TPOT (20.75 vs 22.14 ms, within run-to-run noise); headline TPOT should come from `--no-phases` runs.

**Nsight Compute (img_small; clocks locked to base, so times are inflated; use the ratios):**
Reports: `profiling/reports/20261007-1*_T4_ncu_*` (`.ncu-rep` local only, `*_metrics.csv` in git).

| Phase / kernel | Count | Share of phase time | DRAM % of peak | SM % of peak | Occupancy % | Grid |
| --- | --- | --- | --- | --- | --- | --- |
| Decode: cuBLAS GEMV (weights) | 105 | 67 % | 89 (median) | 55 | 56 | 1536 blocks |
| Decode: lm_head GEMV | 1 | 8 % | 96 | – | – | – |
| Decode: elementwise kernels | ~580 | ~20 % | 1–4 | 0.1 | 13 | **1 block** |
| Decode: reduce (RMSNorm mean) | 54 | 3 % | 1.7 | 0.3 | 46 | **1 block** |
| Vision: SigLIP2 attention (mem-efficient FMHA, sm75) | 27 | **43 %** | 5.5 | 21.6 | 24 | 512 blocks |
| Vision: GEMMs (tensor cores) | 165 | 48 % | 17 | 69 | 25 | 36 blocks |
| Vision: LayerNorm | 55 | 3 % | 31 | 52 | 86 | 1024 blocks |
| LM prefill (252 tokens): GEMMs | 59 | 70 % | 31–39 | – | – | – |
| LM prefill: elementwise/copy/reduce | ~400 | 22 % | 30–45 | – | – | – |

- The decode elementwise/reduce kernels are launched with a **single thread block** (1 of 40 SMs): they are pure launch latency, which confirms the fusion + CUDA-graph direction.
- **Vision attention is the biggest single vision cost and is badly utilized** (neither compute- nor memory-bound, 24 % occupancy). Head dim 72 (1152 / 16 heads) is an awkward size for attention kernels; each 512×512 tile is a 1024-token sequence (936 real + padding). A new optimization candidate for TTFT.
- One decode step under ncu: 800 kernels, 3.7 GB DRAM traffic (≈ the 2.34 GB of weights + lm_head + activations/copies).

**Tooling lessons:**

- Nsight Compute `--set full` over a whole decode step (~800 kernels) needed >9 GB host RAM and >75 min on Colab and was OOM-killed (exit 137). Now: `basic` set + explicit DRAM/SM metrics over whole phases, `full` set only on the first ~150 decode kernels (`ncu_deep` step).
- `ncu` must be pointed at NVTX ranges with `--profile-from-start off` + `cudaProfilerStart`, else it profiles model loading. Prefill and decode `lm_head` are separate ranges so decode captures stay clean.
- Nsight Compute locks clocks to base and serializes kernels: absolute times are ~1.5× real; use it for bandwidth/efficiency ratios, not latency.
- `nsys` is not preinstalled on Colab (`cuda-nsight-systems-13-0` via apt, handled by `scripts/setup_profilers.sh`); `ncu` is, but not on PATH.
- `pkill -f 'ncu '` also killed the remote shell running it (its own command line matched). Kill by PID.
- Colab runs vary: occasional repeats are 1.5× slower (shared VM); always report medians.

**Next step:** optimization candidates by expected payoff: (a) remove launch overhead (CUDA graphs + static cache), (b) fused RMSNorm/residual and fused gated short-conv to cut kernel count, (c) fix the attention path (no FP32 math fallback), (d) vision attention (head dim 72, 24 % occupancy). Prepare the RTX PRO 5000 session.

---

## 2026-10-07: Model switch and code reading

**Goal:** Pick the model variant and understand the inference pipeline before setting up the baseline.

**What we did:**

- Switched from Qwen2-VL (in the proposal) to LFM2.5-VL. Compared the 1.6B and 3B variants and chose **1.6B**; the 3B is kept as a generalization check. Reasoning in [model_selection.md](model_selection.md).
- Cloned reference code into `third_party/`: transformers v5.19.0, vLLM, llama.cpp, and the HF model repo (weights still to be downloaded on the GPU machine).
- Traced the full inference path. Walkthrough in [inference_pipeline.md](inference_pipeline.md).

**Findings (from reading code, not measured yet):**

- Language model = 10 gated short-conv layers + 6 GQA layers. The KV cache is only about 12 KB/token, so KV-cache compression is not a useful target.
- Batch-1 decode reads about 2.34 GB/token, giving a bandwidth floor of about 1.7 ms/token on the RTX PRO 5000.
- The transformers baseline grows the KV cache with `torch.cat` every step, runs eagerly with no CUDA graphs, pads every vision tile to 1024 patches, and loops over tiles in Python.
- The short conv silently falls back to `F.conv1d` if `causal-conv1d` is not installed. The baseline must pin this.
- No framework (transformers, vLLM, llama.cpp) fuses `B·x → conv1d → C·y` into one kernel. This is our candidate custom kernel.

**Next step:** Set up the environment on the Blackwell GPU, download the weights, and write the baseline benchmark (TTFT, TPOT, tokens/s, J/token).
