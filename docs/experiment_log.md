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
- Cause of the FP32 math attention path: with an attention mask present, transformers' `use_gqa_in_sdpa` (sdpa_attention.py L30-L37) returns False, so K/V are expanded to 32 heads with `repeat_kv` every step (the large per-step copies) and SDPA gets an explicit mask, which on the T4 lands on the math backend. Whether the mask can be dropped at batch 1 is open question Q5.2.

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
