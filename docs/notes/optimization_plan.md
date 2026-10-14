# Optimization plan and TODO

_Living plan: what we will optimize, exactly where, why, how, how we check it, and the open questions per step. Update the checkboxes and answers as we go; results go in [../experiment_log.md](../experiment_log.md)._
_Created 2026-10-08 from the T4 baseline profiling of 2026-10-07._

Code links point into `third_party/transformers` (v5.19.0), the baseline we optimize.

---

## 0. Profiling status

| Item | Status | Evidence |
|---|---|---|
| T4 benchmark grid (4 inputs × 4 output lengths × 10 repeats), batch sweep, FP16 vs FP32, hook overhead | ✅ | `results/raw/20261007-1*_T4_baseline_*/summary.md` |
| T4 PyTorch profiler: every kernel → phase + launching op (text, img_small, img_fhd) | ✅ | `profiling/reports/20261007-14*_T4_torchprof_*/{kernels,breakdown}.csv`, `summary.json` |
| T4 Nsight Systems timelines (img_small, img_fhd) | ✅ | `profiling/reports/20261007-14*_T4_nsys_*` |
| T4 Nsight Compute: one decode step (800 kernels), vision (433), prefill (500), full-set on 150 decode kernels | ✅ | `profiling/reports/20261007-1*_T4_ncu_*/ *_metrics.csv` |
| T4 Nsight Compute on a **late** decode step (step 512) and at long context (img_fhd) | ✅ 2026-10-11 | decode +23 % GPU time and +0.93 GB DRAM/step at 2330 context, all attention-related (O1/O5) |
| T4 Nsight Compute on **img_fhd** (9 tiles) vision + prefill, complete prefill for both inputs | ✅ 2026-10-11 | 1080p prefill: GEMMs 37 %, FP32 math-path attention 30 % (O5) |
| **Course GPU server (RTX 5000): same suite**, slots 1–2 of [gpu_sessions.md](gpu_sessions.md) | ⬜ waiting for the server address | needed for Eval 1 |
| Reference baselines: `torch.compile`, `causal-conv1d` package, vLLM, llama.cpp | ⬜ | see O0 |

---

## 1. Where the time goes (T4, FP16, batch 1)

| Phase | Time | What dominates |
|---|---|---|
| Decode, per token | ~20 ms wall, of which ~11.4 ms GPU kernels | **GPU idle ~2/3 of the step** waiting for CPU launches; 791 kernels per token |
| TTFT, 1 tile (640×480) | 121 ms = vision 54 + LM prefill 44 + rest | vision attention, GEMMs |
| TTFT, 9 tiles (1920×1080) | 1319 ms = vision 663 + LM prefill 634 | vision GEMMs + attention; LM prefill attention on FP32 math path (~230 ms) |
| Energy | ~1.45 J per decoded token at ~65 W (cap 70 W) | time-proportional: less time → less energy |

**Decode-step anatomy** (measured, PyTorch profiler, img_small, kernels launched per decode step):

| Source in the model | Kernels / step | GPU µs / step | Code |
|---|---|---|---|
| 92 linear layers + lm_head (cuBLAS GEMV) | 93 | ~9,600 | `nn.Linear` everywhere |
| **RMSNorm ×45** (16 operator_norm, 16 ffn_norm, 6 q_norm, 6 k_norm, 1 final): cast→pow→mean→add eps→rsqrt→mul→cast→mul weight | **~360** | ~1,000 | [modeling_lfm2.py#L53-L58](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L53-L58) |
| Attention layers ×6, non-GEMM: RoPE (mul, neg, cat, add), KV `torch.cat`, `repeat_kv` copies, SDPA **math path** (FP32 bmm, softmax, isneginf, where, all, fill) | ~130 | ~450 | [L220-L257](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L220-L257), [sdpa_attention.py#L108-L114](../../third_party/transformers/src/transformers/integrations/sdpa_attention.py#L108-L114) |
| Short-conv layers ×10, non-GEMM: padding-mask mul, B·x, state `cat`, conv (cuDNN), state copy, C·y, transpose copy | ~70 | ~250 | [L342-L381](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L342-L381), [L273-L289](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L273-L289) |
| SwiGLU ×16, non-GEMM: silu, mul | 32 | ~60 | [L127-L128](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L127-L128) |
| Residual adds | 32 | ~45 | [L423-L424](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L423-L424) |
| generate() glue: sampling, embedding, masks, stop checks | ~30 | ~60 | `generation/utils.py` `_sample` (L3097) |

Nsight Compute: GEMVs run at **~89 % of peak DRAM bandwidth** (lm_head 96 %), so they are already near the roof. The ~690 other kernels run at 1–4 % of bandwidth and are mostly launched as a **single thread block** (1 of 40 SMs). They are pure launch latency.

**Bounds for decode on the T4:** weights 2.34 GB/token ÷ 320 GB/s = **7.3 ms** floor. Current GEMVs ≈ 8.6–9.6 ms; total 20 ms. So:
remove overhead (O1–O4) → ~9–10 ms/token (≈2×); then fewer bytes (O6, INT8 weights) → ~5 ms/token (≈4× total).
Server GPU (RTX PRO 5000 Blackwell, ~1.34 TB/s): floor ≈ 1.7 ms/token, so the overhead share is far larger than on the T4 and O1 matters even more.

---

## 2. Optimizations

Order = expected payoff ÷ effort, based on the evidence above. Each one is implemented as a **variant** in [src/lfm2opt/variants.py](../../src/lfm2opt/variants.py) and measured with the same suite (`VARIANT=<name> bash scripts/run_suite.sh`) on top of the previous ones (cumulative ablation).

Common rules for every step:
- **Correctness first:** unit test in `tests/` vs the PyTorch reference (max abs/rel error, FP16 and BF16), then `benchmarks/compare_outputs.py` against the baseline run (greedy tokens should match; if not, report first divergent token and the logit difference).
- **Measure:** TPOT, TTFT, J/token (bench), kernels/step and GPU-busy % (torch_profile), achieved bandwidth (ncu).
- **Log:** every attempt, including failures, in the experiment log.

---

### O0. Reference baselines (no new code, needed for an honest comparison)

**Why:** our gains must be shown against "what you get by turning on existing switches", not only against eager HF.
**How:** run the suite with (a) `causal-conv1d` installed (real CUDA conv kernel instead of the PyTorch fallback), (b) HF static cache + `torch.compile(mode="reduce-overhead")` (CUDA graphs via the compiler), (c) vLLM, (d) llama.cpp CUDA (GGUF F16 and Q8_0).

- [x] O0a: suite with `causal-conv1d` (built into `.venv-ref`, PYTHONPATH-only): TPOT −1–3 %, 2026-10-14
- [x] O0b: `compile_static` variant: works (one CUDA graph, no graph breaks once hooks are detached); TPOT 2.72 ms text, 3.26 ms at 1080p
- [x] O0c: vLLM 0.31 (`benchmarks/ref_vllm.py`, caches off): TPOT 2.36 ms, TTFT 1080p 128 ms
- [x] O0d: llama.cpp b9acf13 (`benchmarks/ref_llamacpp.sh`): BF16 2.26 ms/token, Q8_0 1.45 ms/token (llama-bench); per-image timings still to parse

**Open questions**
- Q0.1 Does `cache_implementation="static"` work for LFM2's conv + attention hybrid cache in transformers 5.19, and does `generate()` then auto-compile? (`_valid_auto_compile_criteria`, generation/utils.py L2418)
- Q0.2 Does `causal-conv1d` build for sm_75 / sm_89 with torch 2.11 and the server's CUDA version? Prebuilt wheel or source build (time)?
- Q0.3 vLLM / llama.cpp: do they support LFM2.5-VL vision on T4 and on the server GPU? Which versions?
- Q0.4 How do we make the comparison fair (same tokens, same image preprocessing, same dtype)?

---

### O1. Static decode runtime + CUDA graphs  ⭐ first

**Where:** the whole decode step: `generate()` loop ([generation/utils.py `_sample`, L3097](../../third_party/transformers/src/transformers/generation/utils.py#L3097)), `DynamicLayer.update` → `torch.cat` every step ([cache_utils.py#L145](../../third_party/transformers/src/transformers/cache_utils.py#L145)), conv state `cat` + copy ([modeling_lfm2.py#L283-L284](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L283-L284)).

**Evidence:** 791 launches/token; GPU busy ~33 %; cudaLaunchKernel ≈ 10 µs CPU each (nsys); TPOT unchanged between ~600 MHz and ~1560 MHz SM clock; batch 16 costs only +25 % step time.

**Why:** a CUDA graph replays the whole step with one launch, so CPU/Python overhead disappears and the GPU runs back-to-back. It is the single biggest lever and makes every later fusion visible.

**How:**
1. Own decode loop in `src/lfm2opt/runtime.py`: prefill with HF (unchanged), then decode with our step function that calls the HF modules' weights directly.
2. Static buffers: KV cache preallocated `[B, 8, max_len, 64]` per attention layer, written in place at `pos` (no `cat`); conv state `[B, 2048, 3]` updated in place (roll by 1); fixed-size input token / position tensors.
3. Attention over the static cache with a length mask (or a decode kernel that takes `seq_len`, see O5) so shapes never change → capturable.
4. `torch.cuda.CUDAGraph` capture of one decode step (after warm-up on a side stream); replay per token; greedy argmax inside the graph; host reads one token per step (or every k steps).
5. Variant `cudagraph`.

**Check:** token-identical to baseline (greedy); TPOT, GPU-busy % ≈ 100 % inside the step (nsys), J/token.
**Expected:** T4 TPOT 20 → ~11–12 ms; bigger relative gain on the server GPU.

- [x] O1a: static KV + conv-state cache, in-place updates (`src/lfm2opt/runtime.py`); teacher-forced check vs HF
- [x] O1b: graph capture of decode step, replay loop: TPOT 3.83 → 2.91 ms
- [ ] O1c: benchmark + nsys timeline (before/after picture for the report)
- [ ] O1d: measure graph capture cost and memory; decide max context per graph

**Open questions**
- Q1.1 Reuse HF modules inside our loop, or re-implement the decoder forward in our own code (cleaner for later fusion, more work)?
- Q1.2 One graph per batch size? Per max context length bucket? What is the memory cost of the static KV cache at 32k context (≈ 12 KB/token → 400 MB)?
- Q1.3 How to handle the growing attention length inside a fixed graph: mask over max_len (wasted work) vs. a kernel that reads `seq_len` from device memory (O5)?
- Q1.4 Sampling: greedy only for benchmarks; do we need temperature/min-p inside the graph for the demo?
- Q1.5 Does capture work with the cuDNN conv and cuBLAS GEMV workspaces? (set `CUBLAS_WORKSPACE_CONFIG`, avoid allocations during capture)
- Q1.6 How do we get per-phase timing once everything is one graph? (nsys graph-node tracing `--cuda-graph-trace=node`)

---

### O2. Fused residual-add + RMSNorm kernel

**Where:** `Lfm2RMSNorm.forward` ([modeling_lfm2.py#L53-L58](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L53-L58)), used 45×/step: `operator_norm`, `ffn_norm` per layer ([L394-L395](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L394-L395)), `q_layernorm`/`k_layernorm` ([L217-L218](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L217-L218)), `embedding_norm` (L461); residual adds at [L423-L424](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L423-L424).

**Evidence:** ~360 of 791 kernels/step; each is a 1-block launch at <2 % of bandwidth.

**Why:** 8 kernels → 1 per norm; residual add folded in (`h = x + y; out = rmsnorm(h)` in one pass, writing both `h` and `out`).

**How (CUDA):** one block per row (hidden 2048 → 256 threads × 8 elements, 128-bit vectorized loads), FP32 accumulation, warp-shuffle + shared-memory reduction for the sum of squares, write `x * rsqrt(mean+eps) * w` in FP16/BF16. Variants: `rmsnorm(x)`, `add_rmsnorm(x, residual) → (sum, normed)`, and per-head `qk_rmsnorm` (64-wide rows, many heads per block). Python binding via `torch.utils.cpp_extension`, sources in `kernels/`.

**Check:** max error vs PyTorch reference (FP32 math) < 1e-3 relative; token-identical generation; kernels/step drops by ~315.
**Expected:** −~1 ms GPU time/step on T4 without graphs, plus fewer nodes inside the graph with O1.

- [x] O2a: `add_rmsnorm` kernel + binding + unit test (decode shapes; bit-exact). Prefill shapes still open
- [ ] O2b: patch modules via a variant (`fused_rmsnorm`), verify outputs
- [ ] O2c: ncu: achieved bandwidth of the fused kernel for prefill-size rows

**Open questions**
- Q2.1 Exact numerics of HF: it casts to FP32, normalizes, casts back to FP16/BF16, **then** multiplies by the weight. Our kernel must match this order for token-identical outputs. Do we match exactly or accept tiny differences?
- Q2.2 Decode rows are tiny (1×2048): is one block per row enough, or should several norms run in one launch? (with CUDA graphs, launch count matters less; latency per kernel still does)
- Q2.3 Fuse further into the next GEMV's input (norm → GEMV prologue)? Only if we write our own GEMV (O6).

---

### O3. Fused gated short-convolution kernel (LFM2-specific)

**Where:** `Lfm2ShortConv.forward` ([modeling_lfm2.py#L342-L381](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L342-L381)); decode path `causal_conv1d_update` ([L273-L289](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L273-L289)), prefill path `causal_conv1d_fn` ([L293-L312](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L293-L312)). 10 of 16 layers.

**Evidence:** ~7 non-GEMM kernels per conv layer per step (~70/step); conv itself is a cuDNN grouped conv at 2.4 % of bandwidth; currently using the **PyTorch fallback** (no `causal-conv1d`). No framework fuses the gates with the conv (checked vLLM `short_conv.py`, llama.cpp `ssm-conv.cu`).

**Why:** `B·x → depthwise conv (k=3, with state) → C·y` is elementwise per channel plus a 3-tap window. One kernel can read `in_proj` output once, update the conv state in place, and write `C·y` ready for `out_proj`. That gives fewer launches and no intermediate tensors or transposes.

**How (CUDA):**
- Decode kernel: one thread per channel (2048 channels), reads B, C, x for the token, state[2] (last two inputs), weights[3]; computes `u = B*x`, `y = C * (w0*s0 + w1*s1 + w2*u)`, shifts the state in place.
- Prefill kernel: tile over (channels × time); each thread block loads a time tile plus a 2-token halo; same math; writes final state.
- Layout: consume `in_proj` output in its natural `[B, T, 3*2048]` layout (avoid the `.transpose(-1,-2)` + `.contiguous()` in HF).

**Check:** vs PyTorch reference incl. state continuity (prefill then N decode steps == full-sequence conv); token-identical generation.
**Expected:** −~60 launches/step, −prefill memory traffic; compare against O0a (`causal-conv1d` package) as the "existing best" kernel.

- [x] O3a: `shortconv_step` decode kernel + test (bit-exact incl. state)
- [ ] O3b: prefill kernel + test (halo handling, padding mask)
- [ ] O3c: variant `fused_shortconv`; ablation vs `causal-conv1d`

**Open questions**
- Q3.1 The padding mask multiply (`apply_mask_to_padding_states`, L260-L269): needed for batch > 1 with left padding. Fold it into the kernel?
- Q3.2 Conv weight is `[2048, 1, 3]`, bias off (`conv_bias: false`). Confirm no activation (HF passes none). ✅ from config, re-check in test.
- Q3.3 Should the fused kernel also do `out_proj`'s input cast/layout so `out_proj` is a plain GEMV?

---

### O4. Fused QK-norm + RoPE (+ merged QKV and gate/up projections)

**Where:** `Lfm2Attention.forward` ([L231-L236](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L231-L236)): q_proj/k_proj/v_proj (3 GEMVs), `q_layernorm`/`k_layernorm`, `apply_rotary_pos_emb` ([L139-L161](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L139-L161)) with `rotate_half` (neg + cat) and 4 muls + 2 adds; MLP `w1`/`w3` (2 GEMVs, [L123-L128](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L123-L128)); RoPE cos/sin recomputed every step ([L102-L108](../../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L102-L108)).

**Evidence:** RoPE ≈ 10 kernels per attention layer; 3 + 2 separate GEMVs per layer where 1 + 1 would do.

**How:** concatenate weights once at load (`Wqkv` 2048→3072, `W13` 2048→16384); one kernel per attention layer does per-head RMSNorm on q and k + RoPE (cos/sin table precomputed once) + writes k, v directly into the static KV cache slot (with O1). SwiGLU `silu(a)*b` fused as one elementwise kernel (or GEMV epilogue later).

- [x] O4a: merged q|k|v and w1|w3 weights in the fused runtime
- [x] O4b: `qk_norm_rope_kv` kernel + test (bit-exact)
- [x] O4c: `silu_mul` kernel + test (bit-exact)

**Open questions**
- Q4.1 RoPE convention: HF `rotate_half` (split halves, not interleaved), θ = 1e6, head_dim 64. Verify against llama.cpp's `rope_type` for LFM2.
- Q4.2 Does merging GEMVs actually help on the T4 (they're already ~89 % of bandwidth), or only reduce launches?

---

### O5. Attention without the FP32 math fallback

**Where:** [sdpa_attention.py#L108-L114](../../third_party/transformers/src/transformers/integrations/sdpa_attention.py#L108-L114). At batch 1 transformers passes **no mask**, so `use_gqa_in_sdpa` ([L30-L37](../../third_party/transformers/src/transformers/integrations/sdpa_attention.py#L30-L37)) returns True and SDPA is called with `enable_gqa=True`. On the T4 no fused kernel supports GQA → the **math** backend (FP32 casts of Q/K/V, FP32 scores, safe-softmax `isneginf/where/all`). Measured with `profiling/sdpa_probe.py` (2026-10-13); an earlier mask/`repeat_kv` explanation was wrong.

**Evidence:** decode: 12 large copies + ~10 math-path kernels per step; copies grow with context (img_fhd decode: copies 16 % of decode GPU time). Prefill img_fhd: ~230 of 634 ms (FP32 softmax over 32×2327×2327 = 90 ms, mask add 50 ms, where 37 ms, isneginf 19 ms, casts ~30 ms).

**How:**
- Decode: GQA-aware decode attention kernel (one block per KV head, 4 query heads share the loaded K/V, online softmax, split over sequence "flash-decoding" when context is long), reading the static cache with `seq_len` from device memory (CUDA-graph friendly, O1).
- Prefill: causal attention without materialized mask (FlashAttention-style tiling, FP16 in / FP32 accumulate); on sm_80+ first check whether PyTorch's flash backend already does this when `is_causal=True` and no mask.

- [x] O5a (T4): `enable_gqa` → math path; `repeat_kv` + mem-efficient is 3.6–23× faster (decode 2304: 0.63 → 0.18 ms; prefill 2327: 45.6 → 1.9 ms)
- [x] Quick-win variant `sdpa_repeat_kv` (variants.py): T4 img_fhd LM prefill 512 → 293 ms (−43 %), TTFT −19 %, outputs token-identical
- [x] O5a (server): Blackwell already uses flash with native GQA in the model; `sdpa_repeat_kv` is no help there
- [x] O5b: decode attention kernels (single-block and split-K flash-decoding) + tests at len 1 and 2301
- [ ] O5c: prefill path (own kernel on T4, or mask-free flash SDPA where available)

**Open questions**
- Q5.1 ✅ (T4): rejected because of GQA (`enable_gqa=True` unsupported by mem-efficient on sm_75). Server: answered by the slot-1 probe.
- Q5.2 ✅ (T4): at batch 1 HF already drops the mask; that is exactly what selects `enable_gqa=True`.
- Q5.3 Context lengths that matter for the demo: ~250 (1 tile) to ~2.3k (9 tiles) + output. Is split-K needed below 4k?

---

### O6. Low-precision weights for decode GEMV

**Where:** all 92 decode GEMVs + lm_head (`nn.Linear`), 2.34 GB read per token.

**Evidence:** GEMVs at ~89 % of peak bandwidth: the only way to make them faster is to read fewer bytes. Matters most on bandwidth-starved devices (T4 320 GB/s, Jetson Orin 68–205 GB/s).

**How:** weight-only quantization, activations stay FP16/BF16:
- INT8 per-channel (2× fewer bytes), INT4 group-wise (g=64/128, ~3.6× fewer bytes) with dequant in the GEMV inner loop (T4, Jetson);
- FP8 (E4M3) and NVFP4 on the RTX PRO 5000 Blackwell (native Blackwell tensor-core formats).
Own GEMV kernel (vectorized loads, warp-per-row reduction), compared against cuBLAS FP16 and against existing kernels (e.g. torchao / llama.cpp `mmvq`).

- [x] O6a: INT8 weight-only GEMV kernel + test + microbenchmark (`profiling/bench_gemv.py`): TPOT 1.33 ms
- [ ] O6b: quality: greedy-token agreement + a small VQA/OCR set (needs eval set, see G3)
- [x] O6c: INT4 group-wise GEMV (RTN, clip search, mixed INT8): 0.99 ms but 86–93 % argmax agreement → needs calibration (AWQ/GPTQ)
- [ ] O6d: FP8 and NVFP4 weights on the RTX PRO 5000 Blackwell

**Open questions**
- Q6.1 Which layers tolerate quantization? (lm_head and the vision tower often hurt quality most; keep FP16 first)
- Q6.2 Calibration-free (RTN) enough, or AWQ/GPTQ-style needed? Effort vs. course scope.
- Q6.3 On sm_89, is an FP8 GEMV path for M=1 available via `torch._scaled_mm` / CUTLASS, and is it faster than INT8 weight-only for a bandwidth-bound GEMV? Or do we write the kernel?

---

### O7. Vision encoder (SigLIP2) attention and fusions

**Where:** `Siglip2Attention` ([modeling_siglip2.py#L279](../../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py#L279)), `Siglip2MLP` (L338), `Siglip2EncoderLayer` (L353): 27 layers, hidden 1152, 16 heads × **head dim 72**, sequence 1024 per tile (padded); position-embedding resize loop per image (L131); per-tile projector loop ([modeling_lfm2_vl.py#L187-L201](../../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L187-L201)).

**Evidence:** ncu (1 tile): attention = **43 % of vision time** at SM 21.6 %, DRAM 5.5 %, occupancy 24 % (`fmha_cutlassF_f16_aligned_32x128_rf_sm75`); GEMMs 48 % at SM 69 %; ~70 ms per tile; img_fhd vision 663 ms of 1319 ms TTFT.

**How:** attention kernel specialized for head dim 72 (pad to 80/96 in registers, or flash-style tiling with dim 72), packed/unpadded tiles (varlen, like vLLM `lfm2_siglip2.py`), fused bias+GELU and LayerNorm; batch the projector over tiles.

- [ ] O7a: understand why the FMHA kernel is at 24 % occupancy (registers? shared memory? head dim 72 alignment?) from the full ncu sections
- [ ] O7b: try the PyTorch flash backend on the server GPU for head dim 72
- [ ] O7c: own attention kernel or tuned config; varlen packing
- [ ] O7d: fused GELU/LayerNorm, batched projector

**Open questions**
- Q7.1 Is TTFT a priority for our edge use case (camera frame → caption), or is decode speed what users feel? (decides O7 vs O6 order)
- Q7.2 Fewer image tokens (`max_image_tokens`, `max_tiles`) is a model-level knob, not GPU work. Do we report it as a trade-off curve only?

---

### O8. (Stretch) Persistent decode megakernel

All layers of one decode step in one kernel launch with grid-wide sync, weights streamed by SM. Only if O1–O6 are done and time remains. Course catalogue explicitly lists "Kernel Fusion & MegaKernels".

**Open questions:** Q8.1 grid sync cost on T4 vs the server GPU; Q8.2 is CUDA-graph + fused kernels already within ~10 % of the GEMV floor (then a megakernel is not worth it)?

---

## 3. Cross-cutting work

### G1. GPU-server slots (3 × 2 h/week over SSH)
Full plan, rules, timings and failure playbook: [gpu_sessions.md](gpu_sessions.md).
- [x] Server address, `gpu` SSH alias, key login, `remote.sh check`
- [x] Deadline-aware slot presets (`SUITE=slot1|slot2`, `SLOT_MINUTES`), `remote.sh slot` / `watch` / `stop`
- [x] Tooling: `scripts/remote/remote.sh` (push / setup / bg in tmux / log / sync), `scripts/preflight.sh`, driver-aware `setup_env.sh`, suite presets (`SUITE=timing|profile|ncu|variant`), ncu auto-skip without counter access
- [x] Session 1 (+ two extra night slots): baseline suite (BF16), `sdpa_probe.py`, O0 reference baselines, O1–O6 first versions
- [ ] Sessions 2–5: optimization variants, ablation stack, FP8 (O6d), 3B, final numbers

### G2. Jetson (final weeks)
- [ ] Which board (Orin Nano / NX / AGX)? JetPack version → CUDA/torch versions
- [ ] Energy measurement without NVML: `tegrastats` / INA3221 rails → add a Jetson backend to `src/lfm2opt/energy.py`
- [ ] Build kernels for sm_87; run suite subset

### G3. Quality evaluation
- [ ] Greedy-token agreement vs baseline (already in the harness)
- [ ] Small fixed eval set: e.g. 100 VQA + 50 OCR images (which dataset? licence?), scored by exact match / ANLS
- [ ] Logit/perplexity difference for quantized variants

### G4. Generalization: LFM2.5-VL-3B
- [ ] Run the final variant stack on the 3B (same kernels, more layers, 128k vocab) on the server GPU

### G5. Report and evaluations
- Eval 1 (week 5): baseline + profiling + bottleneck evidence (T4 ✅, server ⬜) + this plan
- Eval 2 (week 7): O1–O3 working with before/after numbers
- Final (week 9): O4–O7, quantization, cross-GPU (T4 / server RTX 5000 / Jetson) + 3B, quality, demo

---

## 4. Global open questions (for the team / TAs)

1. GPU-server access: see G1 and gpu_sessions.md §8.
2. Is a custom decode runtime (O1) that reuses HF weights acceptable as "our system", with HF transformers as the baseline? (Course CO6 is about building exactly such a runtime.)
3. Batch size focus: batch 1 (edge, interactive) as the main metric, batch sweep as a secondary result. Agree?
4. Precision for the headline numbers: BF16 on the RTX PRO 5000 and Jetson, FP16 on the T4. Agree?
5. Which input is the "headline" workload for the report: 1 tile (fast camera frames) or 9 tiles (documents/high-res)?
6. Team split proposal: (A) runtime + CUDA graphs: O1, O4; (B) custom kernels: O2, O3; (C) attention + vision + quantization: O5, O6, O7. Who takes which?

---

## 5. Direction after the first optimization night (decided 2026-10-14)

**Quality first.** The main result is GPU/CUDA-level optimizations that keep the model's BF16 math (CUDA graphs, fused kernels, our GEMV, attention, preprocessing). Quantization is **not** counted as an optimization; INT8 is shown only as a separate extension with measured benchmark accuracy (fine-tuning only if it ever drops, mindful of GPU time). INT4 is dropped.

**Main benchmark = quality + speed + energy** on standard VLM benchmarks, run by our own eval script (official scoring, identical prompts, greedy) for baseline vs our BF16 (vs INT8): MMStar (1500), POPE (~1000 subset), OCRBench (1000), ChartQA or TextVQA (~500–1000 subset). Report accuracy, number of changed answers + McNemar test, TTFT/TPOT/energy over the dataset. Every new optimization must pass it.

**Next GPU work, in order:**
- [x] Q1: eval harness + datasets (downloaded on the server with parallel range requests). 14 Oct: BF16 stack and INT8 within ±0.3 pp on all 5 benchmarks, McNemar p ≥ 0.39
- [x] P1 (14 Oct, bit-exact, 1080p 25–69 → 1.5 ms): GPU image preprocessing kernel (resize + normalize + tile/patchify); CPU preprocessing is 58 ms at 1080p and 32 ms at 720p, outside our TTFT numbers (end-to-end 1080p ≈ 160 ms)
- [x] O7d (14 Oct, 1080p vision 61.7 → 53.9 ms vs cuDNN): vision encoder: padding-free attention (flash 287 vs cuDNN 419 µs/layer), fused LayerNorm and bias+GELU
- [x] O3b (14 Oct, 1080p LM prefill 37.0 → 31.8 ms): LM prefill fusion: multi-row add_rmsnorm / silu_mul, prefill short-conv kernel
- [ ] O2d: decode GEMV epilogue fusion (w1|w3 + silu_mul, in_proj + short-conv): ~149 → ~120 kernels
- [ ] ncu of our own kernels (bandwidth / occupancy table for the report)

Target: 1080p end-to-end latency ≈ 160 → 95–100 ms, with unchanged benchmark accuracy.
