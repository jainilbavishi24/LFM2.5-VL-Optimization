# Improving Inference Speed and Energy Efficiency of a Vision-Language Model

**Course:** CS3.406 Programming AI Accelerators (Monsoon 2026) \
**Team:** Jainil Bavishi (2023102045) · Gautam Gandhi (2023102059) · Bibek Singh Dhody (2023101054) \
**Model:** [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) \
**Hardware:** course GPU server with an **NVIDIA RTX PRO 5000 Blackwell 48 GB** (sm_120) over SSH, 3 × 2 h slots/week, for measurements · Google Colab T4 for development · Jetson Orin at the end

> _Last updated: 2026-10-13 (measurement GPU: course RTX 5000 server, 2-hour slots)_

We profile the inference of the LFM2.5-VL-1.6B vision-language model on NVIDIA GPUs (RTX PRO 5000 Blackwell, T4, later Jetson), find where time and energy actually go, and remove those bottlenecks with **GPU/CUDA-level optimizations**: custom and fused kernels, CUDA graphs, and low-precision GEMV/GEMM. Every optimization is measured against a reproducible baseline for latency, throughput, energy per token, and output quality.

---

## Project status

| Phase | Course week | Status |
|---|---|---|
| Topic selection & proposal | 1–2 | ✅ Done (proposal submitted with Qwen2-VL) |
| Model change: Qwen2-VL → LFM2.5-VL-1.6B | 2 | ✅ Done — see [docs/model_selection.md](docs/model_selection.md) |
| Code reading: inference pipeline | 2–3 | ✅ Done — code map in [docs/inference_pipeline.md](docs/inference_pipeline.md); stage-by-stage explainer with shapes and team split in [docs/pipeline_explained.md](docs/pipeline_explained.md) |
| Benchmark + profiling harness (`benchmarks/`, `profiling/`, `scripts/run_suite.sh`) | 3 | ✅ Done, debugged on Colab T4 |
| Baseline benchmarking + profiling on **T4** | 3 | ✅ Complete (incl. long-context and full-prefill Nsight Compute, 2026-10-11) — see Results and the log |
| Baseline benchmarking + profiling on the **course server, RTX PRO 5000 Blackwell** (3 × 2 h/week, SSH) | 4–5 | ✅ Done in slot 1 (2026-10-13): full grid, batch sweep, torch profiler, nsys, 9 Nsight Compute runs — see Results |
| Reference baselines O0a–O0d (`causal-conv1d`, `torch.compile`, vLLM, llama.cpp) | 5 | ✅ Done (night of 13–14 Oct) — see Results |
| O1 CUDA graph runtime, O2–O4 fused CUDA kernels, O7a vision attention, P1 GPU image preprocessing | 5–6 | ✅ Working (all BF16): decode 3.83 → 2.08 ms; 1080p first token incl. preprocessing 136 → 102 ms — see Results |
| Quality evaluation on standard VLM benchmarks (MMStar, POPE, OCRBench, ChartQA, TextVQA) | 6 | ✅ 14 Oct: our BF16 stack and INT8 are within ±0.3 pp of the baseline, no significant difference (McNemar) |
| O7d fused vision encoder, O3b fused LM prefill, O2d GEMV epilogues | 6 | ✅ 1080p vision 77 → 54 ms, LM prefill 37 → 31 ms; first token incl. preprocessing 164 → 90 ms, TPOT 3.70 → 2.07–2.12 ms (3 repeats) |
| Extension: INT8 weight-only GEMV (O6) | 6 | ✅ 1.11 ms/token; evaluated separately (it changes the model) |
| Extension: INT8 everywhere (prefill + vision on our INT8 tensor-core GEMMs, no BF16 weight copies) | 7 | ✅ 27 Oct (L40S): peak memory 4.6 → **2.1 GB** (HF 3.3), 1080p first token 84.5 → 72.4 ms; 5 benchmarks within −0.36…+0.08 pp (p ≥ 0.18); long answers drift more (ROUGE-L −0.056 vs HF BF16) |
| Bottleneck analysis | 4 | ✅ T4 and Blackwell: decode is launch/host-overhead bound (TPOT 3.8 ms vs ~1.7 ms weight-read floor; batch 16 costs +5 %); plan in [docs/notes/optimization_plan.md](docs/notes/optimization_plan.md) |
| **Evaluation 1:** baseline & problem understanding | 5 | 📅 **Thu 22 Oct** — server baseline needed; timeline in [docs/notes/gpu_sessions.md](docs/notes/gpu_sessions.md) §0 |
| Main optimization work | 6 | — |
| **Evaluation 2:** working prototype | 7 | — |
| Final experiments, ablations, 3B generalization check | 8 | — |
| **Final evaluation:** demo, presentation & report | 9 | — |

The day-to-day record of what we tried and measured is in [docs/experiment_log.md](docs/experiment_log.md).

---

## Why LFM2.5-VL-1.6B

Our proposal named Qwen2-VL. We switched to Liquid AI's **LFM2.5-VL-1.6B** because it is a recent, edge-oriented VLM whose language model is a **hybrid of short gated convolutions and attention**. That opens up kernel work existing frameworks have not done (for example, a fused gated short-conv kernel), rather than re-tuning a standard transformer.

We chose the 1.6B over the 3B because:

- at batch size 1, decoding is limited by memory bandwidth and kernel-launch overhead, and that overhead is a larger share of time in the smaller model, so there is more to gain;
- both use the same vision tower and the same set of operations, so everything we build for the 1.6B also runs on the 3B, which we will use as a generalization experiment;
- at 3.2 GB in BF16 it fits edge devices such as the Jetson Orin family.

Full reasoning is in [docs/model_selection.md](docs/model_selection.md).

### Architecture at a glance

| Component | Details |
|---|---|
| Vision encoder | SigLIP2 NaFlex so400m: 27 layers, hidden size 1152, patch size 16 (~400M parameters) |
| Projector | 2×2 pixel-unshuffle → Linear(4608→2048) → GELU → Linear(2048→2048) |
| Language model | LFM2.5-1.2B: 16 layers = **10 gated short-conv** + **6 GQA attention** (32 query / 8 KV heads, head size 64), SwiGLU MLP (feed-forward width 8192), RMSNorm, RoPE |
| Vocabulary / context | 65,536 / 32,768 tokens |
| Image tokens | 64–256 per 512×512 tile; large images use 2–10 tiles plus a thumbnail |

---

## Methodology

We follow the course methodology: **measure first, understand the bottleneck, optimize second.**

```
Baseline → Benchmark → Profile → Identify bottleneck → Optimize → Benchmark again → Analyze
```

**Metrics**

- Latency: time to first token (TTFT), split into vision encoding and LLM prefill; time per output token (TPOT); end-to-end latency
- Throughput: decode tokens/s
- Energy: joules per request and per generated token, from NVML power sampling; average power
- Memory: peak GPU memory
- Kernel level: kernel time breakdown and achieved DRAM bandwidth vs. peak, from Nsight Systems and Nsight Compute
- Quality: exact-match of greedy outputs vs. baseline, logit difference, and a VQA/OCR benchmark subset

**Workloads**

- Single-tile image (≤512×512) vs. multi-tile high-resolution image
- Short vs. long generations
- Batch size 1 (edge / interactive), plus a small batch sweep

**Baselines**

1. Hugging Face `transformers` v5.19.0, eager PyTorch, BF16. This is the primary baseline.
2. The same with `torch.compile` / the `causal-conv1d` package. This is the "easy wins" reference, so our gains are not just from switching on existing flags.
3. vLLM and llama.cpp, as external reference points.

---

## Optimization roadmap (ordered by evidence so far)

The detailed plan (exact code locations, evidence, design, checks and open questions for every step, with TODO checkboxes) is in [docs/notes/optimization_plan.md](docs/notes/optimization_plan.md).

| # | Candidate | Targets | Expected effect |
|---|---|---|---|
| 1 | CUDA graphs / persistent decode kernel, with a static KV cache instead of the `torch.cat`-grown cache | Decode loop | Remove CPU and launch overhead (T4: GPU busy only ~33 % of a decode step) |
| 2 | Fused residual-add + RMSNorm; fused QK-RMSNorm + RoPE | All 16 layers | RMSNorm alone is 8 kernels × 45 calls ≈ 360 of 791 kernels per step |
| 3 | Fused gated short-conv CUDA kernel: `B·x → causal conv1d (k=3) → C·y` plus in-place conv-state update | 10 conv layers, decode & prefill | Fewer launches, no intermediate memory traffic |
| 4 | Attention without the SDPA FP32 math fallback (GQA-aware fused attention) | LM prefill & decode | T4 1080p prefill: ~230 of 634 ms is FP32 math-path attention; check whether it also happens on the server GPU (`profiling/sdpa_probe.py`) |
| 5 | Merged QKV and gate/up projections | Linear layers | Fewer, larger GEMVs (GEMVs already run at ~80 % of peak bandwidth) |
| 6 | Low-precision weights (INT8 / INT4; FP8 and NVFP4 on Blackwell) GEMV for decode | Decode once overhead is gone | Fewer bytes per token → speed and energy |
| 7 | Better vision-encoder attention (head dim 72, unpadded/varlen tiles), fused LayerNorm/GELU | SigLIP2, every image | Vision attention is 43 % of vision time at 24 % occupancy (T4); ~70 ms per tile |

**Back-of-envelope decode bound:** about 2.34 GB of weights are read per token. That gives a floor of ~1.7 ms/token on the RTX PRO 5000 Blackwell (~1.34 TB/s) and 7.3 ms/token on the T4 (320 GB/s). The attention layers' KV cache is only about 12 KB per token, so KV-cache compression is **not** a priority for this model.

---

## Repository layout

```
PAA_Project/
├── README.md                 ← you are here (kept up to date)
├── requirements.txt          ← Python dependencies (PyTorch installed separately; see setup)
├── docs/
│   ├── course/               ← course description, project guidelines, our proposal
│   ├── model_selection.md    ← why LFM2.5-VL-1.6B (vs 3B, vs Qwen2-VL)
│   ├── inference_pipeline.md ← code walkthrough of the inference path, with file/line refs
│   ├── pipeline_explained.md ← every stage explained with tensor shapes; who studies what
│   ├── experiment_log.md     ← dated log of experiments, measurements, decisions
│   └── notes/                ← optimization_plan.md (plan + TODO + open questions), gpu_sessions.md (how we use the 2 h GPU slots)
├── src/lfm2opt/              ← our Python package: model loading, runners, metrics, kernel bindings
├── kernels/                  ← custom CUDA/C++ kernels (fused short-conv, RMSNorm, GEMV, ...)
├── benchmarks/               ← benchmark scripts (latency / throughput / energy)
├── profiling/
│   ├── nsys/                 ← Nsight Systems scripts (reports are git-ignored)
│   └── ncu/                  ← Nsight Compute scripts (reports are git-ignored)
├── tests/                    ← correctness tests: our kernels vs PyTorch reference
├── results/
│   ├── raw/                  ← raw CSV/JSON results (one file per run, timestamped)
│   └── plots/                ← generated figures
├── report/                   ← LaTeX technical report (main.tex + sections/ + figures/)
├── presentations/            ← slides for Eval 1, Eval 2, Final
├── scripts/                  ← setup_env / download_model / preflight / run_suite; colab/ (T4 dev) and remote/ (GPU server over SSH)
├── assets/images/            ← fixed test images used for all benchmarks
├── models/                   ← downloaded weights (git-ignored)
└── third_party/              ← upstream repos for reading/reference (git-ignored)
```

### Third-party code (pinned)

These are cloned locally for reading and as reference implementations. They are not part of our git history.

| Repo | Commit / tag | Why we need it |
|---|---|---|
| [huggingface/transformers](https://github.com/huggingface/transformers) | `v5.19.0` (`c4c4605`) | Baseline implementation: `models/lfm2_vl`, `models/lfm2`, `models/siglip2` |
| [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) | `919fde3` | Config, processor config, chat template (weights via `scripts/download_model.sh`) |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | `3ca00a8` | Optimized reference: Triton short-conv, packed SigLIP2 encoder |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `b9acf13` | Optimized reference: hand-written CUDA kernels (`ssm-conv.cu`, quantized GEMV, CUDA graphs) |

---

## Setup

> These steps run on the GPU machine. For the GPU server, `scripts/remote/remote.sh setup` does all of them from our own machine over SSH (see [docs/notes/gpu_sessions.md](docs/notes/gpu_sessions.md)). `setup_env.sh` picks the PyTorch CUDA build that matches the server's driver.

```bash
# 1. Create the environment and install PyTorch + dependencies
bash scripts/setup_env.sh

# 2. Download the model weights (~3.2 GB) into models/
bash scripts/download_model.sh

# 3. Check that the GPU and toolchain are visible
nvidia-smi && nvcc --version && python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

## Running benchmarks and profiling

Every script works the same on any NVIDIA GPU (T4, the course GPU server, later Jetson), and writes self-describing results (GPU, versions, git commit, config) next to the numbers.

| What | Command | Output |
|---|---|---|
| **Whole suite** (bench + profiling) | `bash scripts/run_suite.sh` (pick steps with `STEPS="bench_full nsys"`) | `results/raw/`, `profiling/reports/`, log in `results/raw/suite_*.log` |
| Latency / throughput / energy grid | `python benchmarks/bench.py --preset quick\|full\|batch` | `results/raw/<run>/runs.jsonl`, `outputs.jsonl` |
| Derived metrics table (TTFT, TPOT, tok/s, J/token) | `python benchmarks/summarize.py results/raw/<run>` | `summary.md`, `summary.csv` |
| Do two runs produce the same tokens? | `python benchmarks/compare_outputs.py <ref_run> <test_run>` | match / first divergent token |
| Kernel breakdown by phase and op category | `python profiling/torch_profile.py --workload img_small --gen-len 32` | `kernels.csv`, `breakdown.csv`, `summary.json`, `trace.json` |
| Nsight Systems timeline | `bash profiling/nsys/profile_nsys.sh img_small 32` | `.nsys-rep` + CSV summaries |
| Nsight Compute, one phase | `bash profiling/ncu/profile_ncu.sh decode\|prefill\|vision img_small` | `.ncu-rep` + `*_metrics.csv` |
| Bandwidth per kernel from ncu | `python profiling/ncu/analyze_ncu.py <..._metrics.csv> --steps 1` | table: time, MB, GB/s, % of peak |

Workloads (fixed inputs, defined in `src/lfm2opt/workloads.py`): `text` (no image), `img_small` (640×480, 1 tile), `img_hd` (1280×720), `img_fhd` (1920×1080, multi-tile). Decoding is greedy with a fixed number of new tokens, so runs are comparable across GPUs and optimizations. Optimizations are registered as _variants_ in `src/lfm2opt/variants.py` and selected with `--variants` / `--variant`.

### Developing on Google Colab (T4)

We have unlimited Colab T4 access, so all scripts are developed and debugged there before using our limited GPU-server slots (3 × 2 h/week). From a machine with the authenticated `colab` CLI:

```bash
bash scripts/colab/colab.sh up                       # T4 session + code + deps + model weights
bash scripts/colab/colab.sh bg suite "bash scripts/run_suite.sh"   # long jobs: detached on the VM
bash scripts/colab/colab.sh log suite                              # check progress
bash scripts/colab/colab.sh pull results/raw         # copy results back (PULL_EXCLUDE='*.ncu-rep' for big files)
bash scripts/colab/colab.sh down                     # always stop the session
```

T4 caveats: no BF16 (runs in FP16), 70 W power cap with throttling, shared VM. T4 numbers are a separate data point in our cross-GPU study, never a substitute for GPU-server numbers.

---

## Results

### Baseline, Colab Tesla T4 (FP16, HF transformers 5.19.0 eager, SDPA, batch 1)

Medians of 10 repeats; TPOT and J/token from 128-token vs 1-token runs. Raw data and per-run configs: `results/raw/20261007-134718_T4_baseline_full/`. Analysis in the [2026-10-07 log entry](docs/experiment_log.md).

| Workload | Image tokens | TTFT (ms) | of which vision / LM prefill (ms) | TPOT (ms) | Decode (tok/s) | Energy (J/token) | Peak mem (GB) |
|---|---|---|---|---|---|---|---|
| text | 0 | 32 | – / 20 | 20.0 | 50 | 1.43 | 3.0 |
| img_small 640×480 (1 tile) | 234 | 121 | 54 / 44 | 21.3 | 47 | 1.47 | 3.0 |
| img_hd 1280×720 (3 tiles) | 764 | 402 | 226 / 158 | 20.1 | 50 | 1.42 | 3.2 |
| img_fhd 1920×1080 (9 tiles) | 2300 | 1319 | 663 / 634 | 20.7 | 48 | 1.47 | 4.7 |

Batch 16 (img_small): 627 tok/s, 0.11 J/token. FP16 outputs are token-identical to FP32.

### Baseline, RTX PRO 5000 Blackwell (course server; BF16, HF transformers 5.19.0 eager, SDPA, batch 1)

Medians of 10 repeats, `results/raw/20261013-224658_RTXPRO5000Blackwell_baseline_full/`; analysis in the [2026-10-13 slot-1 log entry](docs/experiment_log.md).

| Workload | Image tokens | TTFT (ms) | of which vision / LM prefill (ms) | TPOT (ms) | Decode (tok/s) | Energy (J/token) | Peak mem (GB) |
|---|---|---|---|---|---|---|---|
| text | 0 | 6.9 | – / 4.0 | 3.83 | 261 | 0.74 | 3.0 |
| img_small 640×480 (1 tile) | 234 | 20.2 | 11.9 / 5.2 | 3.83 | 261 | 0.77 | 3.0 |
| img_hd 1280×720 (3 tiles) | 764 | 39.8 | 24.4 / 12.1 | 3.84 | 260 | 0.83 | 3.1 |
| img_fhd 1920×1080 (9 tiles) | 2300 | 116 | 73.5 / 33.9 | 3.81 | 262 | 0.89 | 3.3 |

Batch 16 (img_small): 3947 tok/s at TPOT 4.05 ms, 0.06 J/token. TPOT is flat in context length (3.86 ms at 1024 tokens). Weight-read floor ≈ 2.34 GB / 1.34 TB/s ≈ 1.7 ms.

### Decode optimizations so far (RTX PRO 5000 Blackwell, batch 1, TPOT at 128 new tokens)

Full grids from the night of 13–14 Oct; details, correctness checks and pitfalls in the [experiment log](docs/experiment_log.md). "Agreement" = teacher-forced greedy argmax agreement with HF BF16 over 381 steps (BF16 variants differ only at exact/near ties).

| Variant | What it adds | TPOT text / 1080p | Speed-up | J/token | Agreement |
|---|---|---|---|---|---|
| HF transformers eager (baseline) | — | 3.83 / 3.81 ms | 1.0× | 0.74–0.89 | — |
| `causal-conv1d` (O0a) | NVIDIA-style conv kernels | 3.73 / 3.70 | 1.03× | | — |
| HF static cache + `torch.compile` (O0b) | PyTorch compiler + CUDA graphs | 2.72 / 3.26 | 1.2–1.4× | | — |
| vLLM 0.31 (O0c) | CUDA graphs, own kernels | 2.36 / 2.36 | 1.6× | | — |
| llama.cpp BF16 / Q8_0 (O0d) | GGUF, CUDA backend | 2.26 ; 1.45 | 1.7× ; 2.6× | | — |
| **O1** `cudagraph` | our static decode loop, one CUDA graph per token | 2.91 / 2.94 | 1.3× | | 99 % |
| **O1–O4** `fused_gemv` | + fused kernels (norm, conv, RoPE/KV, split-K attention, SwiGLU), merged projections, our BF16 GEMV — 149 kernels/step instead of ~700 | **2.08 / 2.14** | **1.84×** | 0.61–0.63 | 99 % |
| **ours: full BF16 stack** `fused_gemv_pf_vfused` | + GEMV with fused SwiGLU / short-conv epilogues (O2d, 123 kernels/step), fused prefill (O3b) and vision (O7d) | **2.07 / 2.12** (3 repeats, baseline 3.70 / 3.69) | **1.75–1.78×** | **0.58–0.63** | 99 %; benchmarks ±0.3 pp |

**Extension: weight quantization.** It changes the model, so it is not counted as one of our optimizations. It is evaluated separately on the quality benchmarks.

| Variant | What it adds | TPOT text / 1080p | Speed-up | J/token | Agreement |
|---|---|---|---|---|---|
| O1–O6 `fused_w8` | + INT8 weight-only GEMV (per-row scales) | 1.11 / 1.19 | 3.45× | 0.32–0.35 | 98.7 % |
| O1–O6b `fused_w4` | + INT4 group-128 GEMV (round-to-nearest) | 0.76 / 0.84 | 5.0× | 0.22–0.26 | 86 % (dropped: too lossy) |

**INT8 everywhere** (`fused_w8all_pf_vfused`, L40S, 27 Oct; [kernels/int8_gemm.cu](kernels/int8_gemm.cu), [src/lfm2opt/int8.py](src/lfm2opt/int8.py)): the prefill and vision GEMMs also run on the INT8 weights, on our own W8A8 tensor-core GEMM (`mma.sync` s8, scales + bias + GELU in the epilogue, activation scales per token and group of 64 channels; the norm / SwiGLU / fc1 kernels write the next GEMM's INT8 input directly), and no BF16 copy of a quantized weight is kept:

| L40S, batch 1 | HF baseline | ours BF16 | INT8 decode only | INT8 everywhere |
|---|---|---|---|---|
| peak GPU memory | 3.0–3.3 GB | 4.3–4.6 GB | 5.4–5.7 GB | **1.8–2.1 GB** |
| TPOT | 9.03 ms | 3.30 ms | 1.79 ms | 1.79 ms |
| first token 1080p / text (GPU preprocessing) | 114.9 / 18.0 ms | 84.5 / 4.2 ms | 85.2 / 4.2 ms | **72.4 / 3.6 ms** |
| J/token | 1.43 | 0.76 | 0.45 | 0.45 |
| 5 benchmarks vs baseline | | ±0.2 pp | ±0.2 pp | −0.36…+0.08 pp, p ≥ 0.18 |
| long answers, ROUGE-L vs FP32 (HF BF16 0.813) | | 0.826 | 0.802 | **0.757** (−0.056, CI [−0.077, −0.034]) |

Short answers are unchanged, but W8A8 activations make long answers drift measurably more than BF16 or weight-only INT8; whether that drift costs accuracy needs a reference-scored long-answer test.

Prefill/vision: cuDNN attention for SigLIP2 (`vision_cudnn`, O7a) cuts 1080p TTFT 116 → 102 ms (vision 73.5 → 62.9 ms). Generalization: on LFM2.5-VL-3B, TPOT 6.50 → 4.95 ms with the fused BF16 runtime (2.50 ms with INT8). Batched decode (img_small): batch 4 2929 tok/s (HF 1039), batch 16 ~5900 tok/s (HF 3987).

### Image preprocessing and end to end (RTX PRO 5000 Blackwell, batch 1)

HF's processor resizes, tiles and patchifies the image on the CPU: 22–69 ms at 1080p, and very noisy. Our two CUDA kernels ([kernels/image_prep.cu](kernels/image_prep.cu)) produce **bit-identical** `pixel_values` on the GPU, verified on 28 image sizes and contents by [tests/test_preprocess.py](tests/test_preprocess.py):

| | VGA | 720p | 1080p |
|---|---|---|---|
| HF processor, CPU | 1.4 ms | 5–31 ms | 25–63 ms |
| HF processor, torchvision on CUDA (`device="cuda"`) | 0.45 | 0.98 | 2.05 (image part only) |
| **ours** (`--preprocess gpu`), incl. tokenization | **0.7** | **1.6** | **3.9** (kernels 25 µs; the rest is PIL → numpy, H2D and the tokenizer) |

Further prefill-side work (all BF16, CUDA events, 1080p):
- **Vision encoder (O7d):** merged q|k|v, padding-free flash attention, GELU in the fc1 GEMM epilogue, fused add+LayerNorm kernel. 77.3 → 53.9 ms, as accurate as HF's BF16 path vs FP32.
- **LM prefill (O3b):** our norm / short-conv / qk-norm+RoPE+KV / SwiGLU kernels over all prompt tokens, writing straight into the decode buffers. 37.0 → 31.8 ms.

**Final end-to-end comparison** (mean ± std over 3 full repeats, 10 runs each; preprocessing + generation; `results/raw/*_final_{base,stack}_r*`, `benchmarks/aggregate_runs.py --pattern 'final_*'`):

| Workload | First token, HF baseline (CPU preprocessing) | First token, ours (all BF16) | Speed-up | TPOT baseline → ours | J/token baseline → ours |
|---|---|---|---|---|---|
| Text | 6.7 ms | **2.8 ms** | 2.4× | 3.70 → **2.07 ms** | 0.84 → **0.58** |
| VGA (1 tile) | 21.4 ms | **13.0 ms** | 1.6× | 3.70 → **2.08** | 0.87 → **0.59** |
| 720p (3 tiles) | 57 ± 14 ms | **31.6 ms** | 1.8× | 3.71 → **2.09** | 0.87 → **0.63** |
| 1080p (9 tiles) | 164 ± 15 ms | **89.9 ± 0.8 ms** | 1.8× | 3.69 → **2.12** | 0.90 → **0.63** |

**Second GPU (NVIDIA L40S, Ada; course SLURM cluster):** everything reproduces.
- Decode: TPOT 9.0 → **3.3 ms (2.7×)**, energy per token −48 %.
- First token including preprocessing: text 21.6 → 4.8 ms, 1080p 146 → 93 ms.
- Batch 16: 1,760 → 4,146 tok/s.
- Quality on all five full benchmarks within ±0.2 pp (all McNemar p ≥ 0.23); VLMEvalKit MMStar 50.07 for baseline and ours alike.
- LFM2.5-VL-3B on the L40S: TPOT 15.1 → 7.5 ms (2.0×), quality within ±0.2 pp.
- Our GEMV kernels reach 81–97 % of the L40S's DRAM bandwidth (ncu).
- Details in the [experiment log](docs/experiment_log.md).

How close to the limit is this? At 1080p, 64.5 of the 87 ms first token are cuBLAS GEMMs running at 192 TFLOPS. That is 88 % of the best BF16 rate we measured on this GPU (219 TFLOPS; [profiling/gemm_efficiency.py](profiling/gemm_efficiency.py)). The decode step is at 1.09× its weight-read floor. Larger BF16 gains would need fewer bytes or FLOPs, i.e. quantization, which is our separately evaluated extension.

"Ours" is `fused_gemv_pf_vfused` with `--preprocess gpu`. It combines the CUDA-graph decode with fused kernels and our GEMV (O1–O4, O2d), the fused LM prefill (O3b), the fused vision encoder (O7d) and GPU preprocessing (P1). It uses no quantization. The baseline's large spread comes from its CPU preprocessing.

**Output quality on standard VLM benchmarks** (full test/val sets; [benchmarks/eval_quality.py](benchmarks/eval_quality.py); same prompts, greedy, exact McNemar test vs the baseline; `results/eval/*_fullsize`). "Ours" is the full BF16 stack `fused_gemv_pf_vfused` + GPU preprocessing:

| Benchmark | n | HF baseline | Ours (BF16) | Δ (pp) | McNemar p | Time per question | GPU energy per question |
|---|---|---|---|---|---|---|---|
| MMStar | 1500 | 49.07 | 48.80 | −0.27 | 0.48 | 65 → 22 ms | 10.0 → 6.9 J |
| POPE (all 3 splits; accuracy) | 9000 | 89.14 | 89.11 | −0.03 | 0.65 | 53 → 16 ms | 9.1 → 5.3 J |
| OCRBench | 1000 | 79.70 | 80.00 | +0.30 | 0.51 | 119 → 61 ms | 26.2 → 18.7 J |
| ChartQA (test) | 2500 | 80.20 | 80.20 | 0.00 | 1.00 | 76 → 32 ms | 15.7 → 10.2 J |
| TextVQA (val) | 5000 | 81.11 | 81.12 | +0.01 | 1.00 | 137 → 74 ms | 34.9 → 23.1 J |

**Cross-check with the official toolkit:** run through VLMEvalKit with LiquidAI's own model integration and VLMEvalKit's scorer, the baseline reproduces the model card's MMStar score (50.20 vs 50.67 published), and our BF16 stack scores 49.87 (McNemar p = 0.33, 97.5 % identical predictions), 3.3× faster per question ([benchmarks/vlmeval_mmstar.py](benchmarks/vlmeval_mmstar.py)).

No difference is significant, and 94.7–99.7 % of the answers are word-for-word identical. The rest are BF16 near-ties that flip both ways equally often. The INT8 extension is also within ±0.4 pp (McNemar p ≥ 0.26). On LFM2.5-VL-3B (subset of 500 per benchmark) the same stack is also within ±0.4 pp, with 96.6–99.9 % identical answers. On long answers (500 detailed image descriptions of about 81 tokens, FP32 as reference), our BF16 stack stays as close to FP32 as HF's own BF16 does (ROUGE-L 0.814 vs 0.823, difference not significant). INT8 drifts slightly more (0.801, significant but small).

The fused BF16 step is 94 % GEMV at ~90 % of the measured DRAM read roof (1.23 TB/s), so BF16 is close to its floor (~1.9 ms); the remaining wins come from fewer weight bytes (INT8/INT4) and from prefill/vision (TTFT is still HF's: 116 ms at 1080p).

---

## Team conventions

- **Never report a number without its config:** every results file records the git commit, GPU, driver/CUDA/PyTorch versions, model, image, prompt, and generation settings.
- **Warm-up and repeat:** discard warm-up runs and report the median and spread over at least 10 runs. Lock clocks where possible.
- **Correctness before speed:** every custom kernel gets a test in `tests/` against the PyTorch reference before it is benchmarked.
- **Log as you go:** add an entry to [docs/experiment_log.md](docs/experiment_log.md) for every experiment, including failed ones.
- **Keep this README current:** update the status table, results table, and roadmap whenever something changes.

## References

- Liquid AI, *LFM2 Technical Report*, arXiv:2511.23404 (2025)
- [LFM2.5-VL-1.6B model card](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) · [LFM2.5-VL-3B blog post](https://huggingface.co/blog/LiquidAI/lfm2-5-vl-3b)
- [Course project guidelines](docs/course/course_project_guidelines.pdf) · [Our proposal](docs/course/project_proposal.pdf)
