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
3. **~680 small kernels per step use only 2–5 % of peak bandwidth.** Main sources: RMSNorm decomposed into ~7 kernels × 45 norms/step (fp32 cast, pow, mean, add, rsqrt, mul, cast), elementwise gates in the short-conv blocks, dtype copies, KV-cache `torch.cat`.
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
