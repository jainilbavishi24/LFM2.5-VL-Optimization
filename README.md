# Improving Inference Speed and Energy Efficiency of a Vision-Language Model

**Course:** CS3.406 Programming AI Accelerators (Monsoon 2026)
**Team:** Jainil Bavishi (2023102045) · Gautam Gandhi (2023102059) · Bibek Singh Dhody (2023101054)
**Model:** [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)
**Hardware:** NVIDIA RTX PRO 5000 Blackwell (single GPU)

> _Last updated: 2026-10-08 (optimization plan)_

We profile the inference of the LFM2.5-VL-1.6B vision-language model on a Blackwell GPU, find where time and energy actually go, and remove those bottlenecks with **GPU/CUDA-level optimizations**: custom and fused kernels, CUDA graphs, and low-precision GEMV/GEMM. Every optimization is measured against a reproducible baseline for latency, throughput, energy per token, and output quality.

---

## Project status

| Phase | Course week | Status |
|---|---|---|
| Topic selection & proposal | 1–2 | ✅ Done (proposal submitted with Qwen2-VL) |
| Model change: Qwen2-VL → LFM2.5-VL-1.6B | 2 | ✅ Done — see [docs/model_selection.md](docs/model_selection.md) |
| Code reading: inference pipeline | 2–3 | ✅ Done — code map in [docs/inference_pipeline.md](docs/inference_pipeline.md); stage-by-stage explainer with shapes and team split in [docs/pipeline_explained.md](docs/pipeline_explained.md) |
| Benchmark + profiling harness (`benchmarks/`, `profiling/`, `scripts/run_suite.sh`) | 3 | ✅ Done, debugged on Colab T4 |
| Baseline benchmarking + profiling on **T4** | 3 | ✅ Done — see Results and the 2026-10-07 log entry |
| Baseline benchmarking + profiling on **RTX PRO 5000** | 3–4 | ⏳ Next — waiting for access details |
| Bottleneck analysis | 4 | ✅ on T4: decode is launch-overhead bound; plan in [docs/notes/optimization_plan.md](docs/notes/optimization_plan.md) |
| **Evaluation 1:** baseline & problem understanding | 5 | — |
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
| 4 | Attention without the SDPA FP32 math fallback (GQA-aware fused attention) | LM prefill & decode | T4 1080p prefill: ~230 of 634 ms is FP32 math-path attention; check whether it also happens on Blackwell |
| 5 | Merged QKV and gate/up projections | Linear layers | Fewer, larger GEMVs (GEMVs already run at ~80 % of peak bandwidth) |
| 6 | Low-precision weights (INT8 / FP8 / NVFP4) GEMV for decode | Decode once overhead is gone | Fewer bytes per token → speed and energy |
| 7 | Better vision-encoder attention (head dim 72, unpadded/varlen tiles), fused LayerNorm/GELU | SigLIP2, every image | Vision attention is 43 % of vision time at 24 % occupancy (T4); ~70 ms per tile |

**Back-of-envelope decode bound:** about 2.34 GB of weights are read per token. At about 1.34 TB/s, that gives roughly 1.7 ms/token, or about 575 tokens/s at batch 1. The attention layers' KV cache is only about 12 KB per token, so KV-cache compression is **not** a priority for this model.

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
│   └── notes/                ← working notes; optimization_plan.md = plan + TODO + open questions
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
├── scripts/                  ← environment setup, model download, utilities
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

> These steps are run on the GPU machine. The RTX PRO 5000 Blackwell is compute capability 12.0 (`sm_120`), so PyTorch must be a CUDA 12.8+ build.

```bash
# 1. Create the environment and install PyTorch + dependencies
bash scripts/setup_env.sh

# 2. Download the model weights (~3.2 GB) into models/
bash scripts/download_model.sh

# 3. Check that the GPU and toolchain are visible
nvidia-smi && nvcc --version && python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

## Running benchmarks and profiling

Every script works the same on any NVIDIA GPU (T4, RTX PRO 5000, later Jetson), and writes self-describing results (GPU, versions, git commit, config) next to the numbers.

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

We have unlimited Colab T4 access, so all scripts are developed and debugged there before using our limited RTX PRO 5000 hours. From a machine with the authenticated `colab` CLI:

```bash
bash scripts/colab/colab.sh up                       # T4 session + code + deps + model weights
bash scripts/colab/colab.sh bg suite "bash scripts/run_suite.sh"   # long jobs: detached on the VM
bash scripts/colab/colab.sh log suite                              # check progress
bash scripts/colab/colab.sh pull results/raw         # copy results back (PULL_EXCLUDE='*.ncu-rep' for big files)
bash scripts/colab/colab.sh down                     # always stop the session
```

T4 caveats: no BF16 (runs in FP16), 70 W power cap with throttling, shared VM. T4 numbers are a separate data point in our cross-GPU study, never a substitute for RTX PRO 5000 numbers.

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

### Baseline, RTX PRO 5000 Blackwell

_Pending GPU access._

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
