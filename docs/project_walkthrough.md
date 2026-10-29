# The project from zero: what we did, why, and how to run and read it yourself

This document tells the whole project as a story, in the order it happened:

1. how we picked the model;
2. how we read the official inference code;
3. how we wrote the baseline;
4. how we profiled it;
5. how every optimization followed from a measurement;
6. how we proved the answers did not change.

Every step names the files involved, explains what the code does, and gives the commands to run it and the way to read its output.

It does not repeat everything. Three existing documents go deeper on specific parts:

| Document | What it has |
|---|---|
| [inference_pipeline.md](inference_pipeline.md) | A code map of the Hugging Face (HF) implementation, with line numbers |
| [pipeline_explained.md](pipeline_explained.md) | Every stage of the model with its tensor shapes |
| [experiment_log.md](experiment_log.md) | Every experiment with its raw numbers and the results folder it came from |

Links into `third_party/` work only on a machine where the reference code was cloned (it is git-ignored). The pinned commits are listed in the [README](../README.md#third-party-code-pinned).

---

## Contents

0. [Words used throughout](#0-words-used-throughout)
1. [Choosing the model](#1-choosing-the-model)
2. [Reading the official inference pipeline](#2-reading-the-official-inference-pipeline)
3. [Writing the baseline](#3-writing-the-baseline)
4. [First measurements on a free T4](#4-first-measurements-on-a-free-t4)
5. [The profiling toolbox: how we got GPU-level information](#5-the-profiling-toolbox-how-we-got-gpu-level-information)
6. [Moving to the real GPU](#6-moving-to-the-real-gpu)
7. [The plan, and the reference baselines](#7-the-plan-and-the-reference-baselines)
8. [The optimizations, one by one](#8-the-optimizations-one-by-one)
9. [Proving the answers did not change](#9-proving-the-answers-did-not-change)
10. [The extension: INT8 quantization](#10-the-extension-int8-quantization)
11. [Reproducing on other GPUs](#11-reproducing-on-other-gpus)
12. [The demo video and its live proof](#12-the-demo-video-and-its-live-proof)
13. [Where we are, and what is next](#13-where-we-are-and-what-is-next)
- [Appendix A: command cheat-sheet](#appendix-a-command-cheat-sheet)
- [Appendix B: result file formats](#appendix-b-result-file-formats)

---

## 0. Words used throughout

| Term | Meaning |
|---|---|
| **Prefill** | The first forward pass of a request: the whole prompt (text tokens plus image tokens) goes through the language model at once. Large matrix-matrix products (GEMMs), so it is compute-bound. |
| **Decode** | Every later step: one new token goes through the model, which reads its own KV cache. Matrix-vector products (GEMVs), so it is memory-bandwidth-bound at batch 1. |
| **TTFT** | Time to first token = preprocessing (sometimes excluded, always stated) + vision encoder + prefill + the first token. |
| **TPOT** | Time per output token during decode. |
| **Kernel** | One function launched on the GPU. A PyTorch op such as `x * y` launches one or more kernels. |
| **Launch overhead** | The CPU time to tell the GPU to run a kernel (a few µs each). With hundreds of tiny kernels per token, the GPU waits on the CPU. |
| **CUDA graph** | A recording of a whole sequence of kernel launches, replayed with one call. It removes per-kernel launch overhead. |
| **Weight-read floor** | At batch 1 every weight is read once per token, so TPOT can never beat `bytes of weights / memory bandwidth`. |
| **NVTX range** | A named label pushed and popped around code, which profilers show on their timelines and can filter on. |
| **nsys / ncu** | NVIDIA Nsight Systems (a timeline of everything) and Nsight Compute (hardware counters per kernel). |
| **NVML** | NVIDIA's management library. We read power and the GPU's energy counter through it. |
| **Variant** | One configuration of the model (the baseline, or the baseline plus some of our optimizations), registered by name in [variants.py](../src/lfm2opt/variants.py). |

---

## 1. Choosing the model

Our proposal named **Qwen2-VL**. Before writing any code we switched to **Liquid AI's LFM2.5-VL-1.6B** ([model_selection.md](model_selection.md)).

- **Why not Qwen2-VL.** It is a standard transformer. Its kernel-level bottlenecks (GEMM, attention, RMSNorm) are already heavily optimized by FlashAttention, vLLM and TensorRT-LLM, so there would be little left for us to contribute.
- **Why LFM2.5-VL.** Its language model is a **hybrid**: 10 *gated short-convolution* layers and 6 attention layers. No framework we checked (transformers, vLLM, llama.cpp) fuses the conv block into one kernel, which gave us a real CUDA target. It is also edge-oriented, which matches our track.
- **Why the 1.6B and not the 3B.**
  - At batch 1, launch overhead is a bigger share of the smaller model's time, so there is more for us to remove.
  - It fits a Jetson.
  - Both models share the same vision tower and the same operators, so the 3B became a ready-made generalization test at the end.

**The model at a glance:**

| Part | What it is |
|---|---|
| Vision encoder | SigLIP2 NaFlex, 27 layers, hidden size 1152, 16×16 patches |
| Projector | 2×2 pixel-unshuffle → Linear 4608→2048 → GELU → Linear 2048→2048 |
| Language model | 16 layers in the pattern `C C A C C A C C A C A C A C A C` (C = short conv, A = attention). GQA with 32 query / 8 KV heads of size 64. SwiGLU MLP 2048→8192→2048. RMSNorm. RoPE. |
| Vocabulary | 65,536 tokens. `lm_head` is tied to the embedding. |
| Images | 64–256 tokens per 512×512 tile. Large images become 2–10 tiles plus a thumbnail. |

---

## 2. Reading the official inference pipeline

Before measuring anything we traced exactly what happens when someone runs the model the official way. We cloned the pinned reference code into `third_party/`: HF transformers v5.19.0, the model repo itself, vLLM and llama.cpp.

### 2.1 The official entry point

The model card ([third_party/LFM2.5-VL-1.6B/README.md](../third_party/LFM2.5-VL-1.6B/README.md), lines 110–148) shows the entire user-facing API:

```python
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", dtype="bfloat16")
processor = AutoProcessor.from_pretrained(model_id)
conversation = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": "What is in this image?"}]}]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True, tokenize=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

| Line | What it does |
|---|---|
| `from_pretrained` | Reads `config.json`, builds the Python module tree (vision tower, projector, language model, `lm_head`), loads `model.safetensors` (3.2 GB in BF16) and places it on the GPU. |
| `AutoProcessor` | Bundles three things: the **tokenizer**, the **chat template** (`chat_template.jinja`) and the **image processor** (resize, tile, normalize, patchify). |
| `apply_chat_template(... tokenize=True)` | Turns the conversation into the model's text format, replaces the image placeholder with the right number of `<image>` tokens, tokenizes, and runs the image processor. The result is a dict of tensors (next table). |
| `model.generate(**inputs, ...)` | The whole inference: prefill, then one decode step per token, until EOS or `max_new_tokens`. |

**What goes into `generate`.** These are the "input arguments" that eventually reach the model's `forward()`:

| Key | Shape (one 1080p image) | Meaning |
|---|---|---|
| `input_ids` | `[1, 2327]` | Prompt token ids. 2300 of them are the `<image>` placeholder token, the rest are text. |
| `attention_mask` | `[1, 2327]` | 1 for real tokens (padding only matters in batches). |
| `pixel_values` | `[9, 1024, 768]` | 9 tiles (8 tiles plus a thumbnail) × 1024 patch slots × (16·16·3) pixel values. |
| `pixel_attention_mask` | `[9, 1024]` | Which patch slots are real. Tiles smaller than 1024 patches are padded. |
| `spatial_shapes` | `[9, 2]` | Patch-grid height and width of each tile. |

The model card also recommends sampling (`temperature=0.1, min_p=0.15, repetition_penalty=1.05`). Its `generation_config.json` sets none of these, so a plain `generate()` is greedy.

### 2.2 What `generate()` does, file by file

We followed the call chain down through the HF code. Each row is one stop.

| Step | File : line | What happens |
|---|---|---|
| 1 | [generation/utils.py:2563](../third_party/transformers/src/transformers/generation/utils.py#L2563) `generate` | Validates arguments, builds the logits processors and stopping criteria, then calls `_sample`. |
| 2 | [generation/utils.py:3097](../third_party/transformers/src/transformers/generation/utils.py#L3097) `_sample` | The generation loop. It calls `_prefill` ([L4244](../third_party/transformers/src/transformers/generation/utils.py#L4244)) once for the whole prompt. |
| 3 | [generation/utils.py:3197–3264](../third_party/transformers/src/transformers/generation/utils.py#L3197) | One **Python loop iteration per token**: `prepare_inputs_for_generation`, then `outputs = model_forward(**model_inputs)` (**this is where `model.forward()` is called**, L3210), logits to FP32 (L3222), logits processors, `argmax` (L3253), `input_ids = torch.cat([...])` (L3260), stopping check (L3262). |
| 4 | [generation/utils.py:745–755](../third_party/transformers/src/transformers/generation/utils.py#L745) | After the first iteration, `pixel_values` and the other image inputs are **dropped**. The vision tower runs only in prefill. |
| 5 | [lfm2_vl/modeling_lfm2_vl.py:341](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L341) `Lfm2VlForConditionalGeneration.forward` | Calls `self.model(...)` (L411), then `lm_head` on the last position only (`logits_to_keep`, L428). |
| 6 | [modeling_lfm2_vl.py:232](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L232) `Lfm2VlModel.forward` | Embeds the tokens. In prefill only, calls `get_image_features` (L264), then `masked_scatter` (L280) writes the image features over the `<image>` token embeddings. Calls the language model. |
| 7 | [modeling_lfm2_vl.py:160](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L160) `get_image_features` | Runs the SigLIP2 vision tower on all tiles at once. Then, **in a Python loop over tiles**, removes the padding, reshapes each tile to its grid and runs the projector (`Lfm2VlMultiModalProjector`, L37). |
| 8 | [lfm2/modeling_lfm2.py:469](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L469) `Lfm2Model.forward` | Creates a `DynamicCache` on the first call (L486), the causal mask and RoPE cos/sin (L507), runs the 16 decoder layers, then the final RMSNorm (L520). |
| 9 | [modeling_lfm2.py:384–427](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L384) `Lfm2DecoderLayer` | `h = x + Operator(RMSNorm(x))`, then `y = h + MLP(RMSNorm(h))`. The operator is either a short conv or attention. |
| 10 | [modeling_lfm2.py:316–381](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L316) `Lfm2ShortConv` | `in_proj` (2048→6144), split into B, C and x, `B*x`, a depthwise causal conv with 3 taps (`causal_conv1d_update` in decode, `causal_conv1d_fn` in prefill), `C*y`, `out_proj`. |
| 11 | [modeling_lfm2.py:202–271](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L202) `Lfm2Attention` | Separate q, k, v projections, RMSNorm on q and k per head, RoPE, cache update, SDPA with GQA, `out_proj`. |
| 12 | [modeling_lfm2.py:111–128](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L111) `Lfm2MLP` | `w2(silu(w1 x) * w3 x)`. |
| 13 | [cache_utils.py:145](../third_party/transformers/src/transformers/cache_utils.py#L145) | Grows the KV cache with `torch.cat` **every step**, reallocating and copying it. |
| 14 | [image_processing_lfm2_vl.py:382](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L382) `resize_and_split` | Small images get `smart_resize` (L331). Large ones are cut into 512×512 tiles plus a thumbnail (`crop_image_to_patches`, L283), patchified (L134) and padded to 1024 patches. All of this runs **on the CPU**. |

The RMSNorm is a good example of why the code reading mattered ([modeling_lfm2.py:53–58](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L53)):

```python
hidden_states = hidden_states.to(torch.float32)                          # kernel 1: cast
variance = hidden_states.pow(2).mean(-1, keepdim=True)                   # kernels 2, 3
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)   # kernels 4, 5, 6
return self.weight * hidden_states.to(input_dtype)                       # kernels 7, 8
```

These are four lines of Python, but eager PyTorch runs them as **about 8 separate GPU kernels**, and there are 33 such norms per decoded token. Each kernel reads and writes a 2048-element vector (4 KB), which takes far less time than launching the kernel. The profiler later confirmed exactly this.

### 2.3 What we concluded before measuring anything

This is from the code reading alone ([experiment_log.md, "Model switch and code reading"](experiment_log.md)).

1. **Decode must read every weight once per token.** That is about 1.17 B parameters, or **2.34 GB per token** (the vision tower is not used in decode). On the RTX PRO 5000 Blackwell (~1.34 TB/s) this gives a floor of about 1.7 ms/token. On the T4 (320 GB/s) it is about 7.3 ms. Any time above the floor is overhead we can remove.
2. **The KV cache is tiny:** only 6 attention layers × 8 KV heads × 64 × 2 (K and V) × 2 bytes = 12 KB per token. KV-cache compression, a popular optimization, is useless here.
3. **The HF baseline has obvious inefficiencies:**
   - an eager Python loop, with no CUDA graphs;
   - `torch.cat` cache growth;
   - every vision tile padded to 1024 patches;
   - a Python loop over tiles in the projector;
   - CPU image preprocessing.
4. **The conv silently changes implementation.** If the `causal-conv1d` package is installed, HF uses its CUDA kernels; otherwise it falls back to `F.conv1d`. The baseline therefore records which one ran ([env.py](../src/lfm2opt/env.py), `causal_conv1d_installed`).
5. **Nobody fuses the conv block** `B·x → conv → C·y` into one kernel. That became our first custom-kernel candidate.

At the end of section 5 of [inference_pipeline.md](inference_pipeline.md) we wrote down open questions that only profiling could answer. Sections 4–6 below answer them.

---

## 3. Writing the baseline

**The rule we set:** the baseline must be *the HF model exactly as shipped*, with no change to its math. Everything we add is only for **loading it reproducibly, feeding it fixed inputs and measuring it**. All of it is in the small package [src/lfm2opt/](../src/lfm2opt/) and the script [benchmarks/bench.py](../benchmarks/bench.py).

### 3.1 Loading the model: [src/lfm2opt/model.py](../src/lfm2opt/model.py)

```python
def load(model=DEFAULT_MODEL, dtype="auto", attn_impl="sdpa", device="cuda"):      # model.py:48
    path, revision = resolve_model_path(model)          # local models/<name> or the Hub at a pinned commit
    torch_dtype = resolve_dtype(dtype)                  # bf16 if the GPU has native BF16, else fp16 (T4)
    processor = AutoProcessor.from_pretrained(path, revision=revision)
    net = AutoModelForImageTextToText.from_pretrained(
        path, revision=revision, dtype=torch_dtype, attn_implementation=attn_impl).to(device)
    net.eval()
    return net, processor
```

| Detail | Why |
|---|---|
| `resolve_model_path` (L38) | Uses the downloaded copy in `models/` (from [scripts/download_model.sh](../scripts/download_model.sh)), otherwise the Hub at the **pinned revision** `919fde3` (L17), the same commit whose code we read. |
| `resolve_dtype` (L22) | BF16 where the GPU supports it natively. The T4 has no BF16, so the baseline runs FP16 there. FP16 outputs were checked to be token-identical to FP32. |
| `attn_implementation="sdpa"` | Stated explicitly so it can't change silently between versions. |
| `.to(device)` instead of `device_map="auto"` | One GPU, so placement is explicit. |
| `net.eval()` | Inference mode for the modules (no dropout). |

```python
def build_inputs(processor, image, prompt, batch_size=1, device="cuda"):           # model.py:59
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         add_generation_prompt=True, tokenize=False)
    kwargs = {"text": [text] * batch_size, "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = [[image]] * batch_size
    return processor(**kwargs).to(device)
```

This produces exactly the same tensors as the model card's one-call `apply_chat_template(..., tokenize=True)`, in two steps:

1. The chat template gives the prompt **string**.
2. `processor(text=..., images=...)` tokenizes it and runs the image processor.

Two steps let us **time preprocessing on its own** and later **swap in our GPU image processor** (section 8.5) without touching anything else. `describe_inputs` (L70) records the token, tile and patch counts that define the workload's size.

### 3.2 Fixed inputs: [src/lfm2opt/workloads.py](../src/lfm2opt/workloads.py)

All four workloads use **one real COCO photo**, `assets/images/coco_000000039769.jpg`, resized, so only the *size* changes between them:

| Workload | Image | Tiles | Image tokens | Prompt |
|---|---|---|---|---|
| `text` | none | 0 | 0 | "Explain in detail how a bicycle works." |
| `img_small` | 640×480 | 1 | 234 | "Describe this image in detail." |
| `img_hd` | 1280×720 | 3 | 764 | same |
| `img_fhd` | 1920×1080 | 9 | 2300 | same |

### 3.3 Where inference is actually called: [benchmarks/bench.py](../benchmarks/bench.py)

The baseline never calls `forward()` by hand. Like a real user, it calls HF's `generate()`, and HF calls `forward()` inside its loop (section 2.2, step 3):

```python
def run_generate(net, inputs, n_new, pad_id):                                      # bench.py:81
    return net.generate(**inputs, max_new_tokens=n_new, min_new_tokens=n_new,
                        do_sample=False, pad_token_id=pad_id)
```

`**inputs` unpacks the dict from `build_inputs` (`input_ids`, `attention_mask`, `pixel_values`, `pixel_attention_mask`, `spatial_shapes`) as keyword arguments.

So for one benchmark request the call chain is:

```
bench.py:165   out = run_generate(net, inputs, n_new, pad_id)
  └ HF generate()  →  _sample()  →  _prefill()                          (generation/utils.py)
       └ model_forward(**model_inputs)  = Lfm2VlForConditionalGeneration.forward      ← prefill, with pixel_values
            └ Lfm2VlModel.forward → get_image_features (SigLIP2 + projector) → masked_scatter
                 → Lfm2Model.forward (16 layers) → lm_head
     loop, once per new token:
       └ model_forward(input_ids=[last token], past_key_values=cache)                 ← decode, no pixel_values
       └ argmax → torch.cat → stopping check
```

**The generation settings, and why:**

| Setting | Value | Why |
|---|---|---|
| `do_sample` | `False` (greedy) | Deterministic, so any token difference between two runs comes from numerics, never from random sampling. |
| `min_new_tokens` | `= max_new_tokens` | **Fixed length**: EOS is suppressed, so every repeat and every variant does exactly the same amount of work. The quality evaluation and the demo use normal EOS stopping instead. |
| `pad_token_id` | the tokenizer's | Silences a warning; no effect at batch 1. |

**The timed loop** (bench.py:150–178), simplified:

```python
with torch.inference_mode():
    for _ in range(args.warmup):                 # warm-up: cuBLAS handles, allocator, caches
        run_generate(net, inputs, n_new, pad_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for rep in range(args.repeats):
        instr.reset(); meter.start()             # phase hooks + NVML energy
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = run_generate(net, inputs, n_new, pad_id)
        torch.cuda.synchronize(); wall_ms = (time.perf_counter() - t0) * 1e3
        energy = meter.stop(); phases = instr.summary()
        ...write one row to runs.jsonl...
```

- `synchronize()` before and after: GPU work is asynchronous, so without it the clock would stop before the GPU finished.
- Warm-up runs are discarded. The median over `--repeats` runs is reported.
- Preprocessing is timed separately, before the loop (L138–144), so TTFT can be quoted with or without it.
- The first repeat's generated token ids and text go to `outputs.jsonl`, so the outputs of two runs can be compared later.

**How TTFT, TPOT and energy are derived:**

```
TTFT   = wall time of generate(max_new_tokens=1)
TPOT   = (wall(N) − wall(1)) / (N − 1)          e.g. N = 128
J/tok  = (E(N) − E(1)) / (N − 1) / batch
```

The difference trick removes everything except the decode steps.

### 3.4 The measurement helpers

**[src/lfm2opt/variants.py](../src/lfm2opt/variants.py): the registry.** Every configuration is a function registered under a name. The baseline is literally the identity:

```python
@register("baseline")
def _baseline(net, processor):
    """Hugging Face transformers as shipped (eager PyTorch, SDPA attention)."""
    return net
```

Every later optimization registers here too, for example `fused_gemv_pf_vfused`. That means every script can measure every variant on identical inputs with `--variants baseline,fused_gemv_pf_vfused`.

**[src/lfm2opt/instrument.py](../src/lfm2opt/instrument.py): phase timing without touching the model.** `PhaseInstrumentor` registers PyTorch **forward hooks** (a pre-hook and a post-hook) on four modules: `vision_tower`, `multi_modal_projector`, `language_model` and `lm_head`.

- The first language-model call after `reset()` is labelled `lm_prefill`, and every later one `lm_decode` (`_phase`, L64).
- The same hooks serve three tools, selected by `modes`, so all of them agree on what "prefill" means:
  - `"events"`: a `torch.cuda.Event` is recorded at the start and end of each phase. GPU time comes from `start.elapsed_time(end)` with no extra synchronization. bench.py uses this.
  - `"nvtx"`: `torch.cuda.nvtx.range_push(phase)` / `range_pop()`, so Nsight shows and filters by phase. [profiling/run_once.py](../profiling/run_once.py) uses this.
  - `"record_function"`: PyTorch-profiler annotations, so every kernel can be attributed to a phase. [profiling/torch_profile.py](../profiling/torch_profile.py) uses this.
- **Hooks cost a little host time.** We measured it (≤ 7 % of TPOT on the T4), so headline TPOT comes from runs with `--no-phases`.

**[src/lfm2opt/energy.py](../src/lfm2opt/energy.py): energy.** `EnergyMeter.start()` reads NVML's **cumulative energy counter** (millijoules since driver load) and starts a 10 ms sampling thread for power, SM clock and temperature. `stop()` returns the difference in energy and the power/clock statistics.

- Before each run, bench.py measures **idle power** for 5 s, so "dynamic" energy above idle can be reported too.
- The counter updates too coarsely for a single short request, so per-run energy is taken over the whole repeat loop and divided by the repeat count.
- Low SM clocks in the samples reveal throttling (the T4 at its 70 W cap).

**[src/lfm2opt/env.py](../src/lfm2opt/env.py): reproducibility.** Every results folder gets an `env.json` recording:

- the GPU, compute capability, driver, CUDA, cuDNN, torch and transformers versions;
- the git commit (with `-dirty` if there were uncommitted changes);
- clocks, power limit, temperature;
- whether `causal_conv1d`, `flash_attn` or `kernels` are installed.

We never report a number without its config.

### 3.5 Our baseline vs the official HF example: the differences

| Aspect | Official model card | Our baseline ([model.py](../src/lfm2opt/model.py), [bench.py](../benchmarks/bench.py)) | Changes the math? |
|---|---|---|---|
| Model code | transformers `Lfm2VlForConditionalGeneration` | **The same**, untouched (`variants.py` `baseline` returns `net`) | no |
| Weights | Hub, latest | Hub or local copy at **pinned** revision `919fde3` | no (same weights) |
| dtype | `bfloat16` | `auto`: BF16 on Ampere and newer, FP16 on the T4 | only on the T4 (checked: identical to FP32) |
| Placement | `device_map="auto"` | `.to("cuda")`, `attn_implementation="sdpa"` stated | no |
| Input building | one `apply_chat_template(tokenize=True)` | template → string, then `processor(text, images)` | no (same tensors), but lets us time and replace preprocessing |
| Decoding | `generate(max_new_tokens=64)`, greedy by default (card recommends sampling) | greedy, `min_new_tokens = max_new_tokens` (fixed length) in benchmarks; stop at EOS in quality/demo runs | no; fixed length only removes EOS as a choice |
| Execution | one call | `torch.inference_mode()`, warm-ups, `synchronize()`, repeats, medians | no |
| Measurement | none | CUDA events per phase (hooks), NVML energy, peak memory, `env.json` | no (hooks cost ≤ 7 % host time; headline numbers without them) |
| Inputs | any image | 4 fixed workloads from one photo | — |
| Batching | 1 | `--batch-sizes` replicates the same prompt | — |

---

## 4. First measurements on a free T4

Course GPU hours were limited, so we built and debugged everything on a **free Google Colab T4** first.

- [scripts/colab/colab.sh](../scripts/colab/colab.sh) drives a Colab VM from the laptop: `up` (create the session, push code, install, download weights), `run`, `bg` (detached long jobs), `log`, `pull`, `down`.
- [scripts/run_suite.sh](../scripts/run_suite.sh) runs the whole measurement suite as named steps: the benchmark grid, batch sweep, FP32 check, hook overhead, PyTorch profiler, nsys, ncu…

**Baseline on the T4 (FP16, batch 1):**

| Workload | TTFT | Vision / LM prefill | TPOT | J/token |
|---|---|---|---|---|
| text | 32 ms | – / 20 | 20.0 ms | 1.43 |
| img_small | 121 | 54 / 44 | 21.3 | 1.47 |
| img_fhd (9 tiles) | 1319 | 663 / 634 | 20.7 | 1.47 |

**What the profilers showed** (how to read each tool is in section 5):

1. **Decode is CPU/launch-overhead bound, not GPU bound.** Each step was 788 kernels, but the GPU was busy only 32–34 % of the time. TPOT did not change with the SM clock, and batching 16× cost only +25 % time per step.
2. **The weight GEMVs were already efficient.** 92 GEMVs per step (exactly the 92 linear layers) took 85 % of kernel time at ~80 % of peak bandwidth. They were not the place to win.
3. **~690 small kernels per step used 1–5 % of the bandwidth**, each launched as a **single thread block**. RMSNorm alone accounted for ~360 of them (8 kernels × 45 norms).
4. **The attention fell back to the slow FP32 "math" path** on the T4, which cost ~230 of the 634 ms of 1080p prefill. A later probe found the real cause: `enable_gqa=True` has no fused kernel on the T4.
5. **Vision attention ran at 24 % occupancy** (head size 72, padded tiles) and took 43 % of vision time.

These findings became the optimization plan ([notes/optimization_plan.md](notes/optimization_plan.md)): **first remove the launch overhead, then fuse the tiny kernels, then fix attention and the vision tower.**

---

## 5. The profiling toolbox: how we got GPU-level information

We use the tools in layers, from coarse to fine. Each answers a different question.

| Layer | Tool / our script | Question it answers | Output |
|---|---|---|---|
| 1 | Wall clock + CUDA events, [bench.py](../benchmarks/bench.py) | How long, end to end and per phase? | `runs.jsonl` → `summary.md` |
| 2 | NVML, [energy.py](../src/lfm2opt/energy.py) | How much energy, at what power and clock? | columns in `summary.md` |
| 3 | PyTorch profiler, [torch_profile.py](../profiling/torch_profile.py) | Which kernels run, how many, in which phase, launched by which PyTorch op? Is the GPU idle? | `summary.json`, `breakdown.csv`, `kernels.csv`, `trace.json` |
| 4 | Nsight Systems, [profile_nsys.sh](../profiling/nsys/profile_nsys.sh) | What does the timeline look like (CPU launches vs GPU kernels, gaps, CUDA graphs)? | `.nsys-rep` (GUI) + CSV summaries |
| 5 | Nsight Compute, [profile_ncu.sh](../profiling/ncu/profile_ncu.sh), [profile_ours.sh](../profiling/ncu/profile_ours.sh) | For each kernel: how close to the hardware limit (DRAM %, SM %, occupancy)? | `.ncu-rep` (GUI) + `*_metrics.csv` |
| 6 | Targeted probes in [profiling/](../profiling/) | One specific question (which attention backend? how fast could this GEMM be?) | printed table or JSON |

### 5.1 Isolating the request to profile: [profiling/run_once.py](../profiling/run_once.py)

Profilers record everything by default, including 30 s of model loading. `run_once.py` makes sure they see **one warm request**:

```python
for _ in range(a.warmup):
    net.generate(**inputs, **gen)            # warm-up, not profiled
with PhaseInstrumentor(net, modes=("nvtx",)).attached():
    torch.cuda.profiler.start()              # = cudaProfilerStart()
    net.generate(**inputs, **gen)            # the profiled request, phases as NVTX ranges
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
```

- nsys runs with `--capture-range=cudaProfilerApi` and ncu with `--profile-from-start off`, so both record only between `start()` and `stop()`.
- The NVTX ranges (`vision_tower`, `projector`, `lm_prefill`, `lm_decode`, `lm_head`, `lm_head_prefill`) let ncu profile *only* one phase.
- `--start-at-decode-step N` starts the capture at the N-th decode step instead, using a pre-hook on the language model (L61–70). That lets us profile a late, long-context step without recording everything before it.

### 5.2 PyTorch profiler: counting and attributing kernels

**Run it:**

```bash
python profiling/torch_profile.py --workload img_small --gen-len 32 [--variant baseline]
```

Output goes to `profiling/reports/<time>_<GPU>_torchprof_<variant>_<workload>_g<gen>/`.

**How it works.** It runs one `generate()` under `torch.profiler.profile(activities=[CPU, CUDA])` with the phase hooks in `record_function` mode, exports the Chrome trace, then analyzes it (`analyze_trace`, L59):

1. Every GPU kernel in the trace has a *correlation id* linking it to the CPU-side `cudaLaunchKernel` call that launched it. That gives each kernel its **launch timestamp**.
2. The launch timestamp falls inside one phase annotation (`phase_of`), which tells us **which phase** the kernel belongs to. It also falls inside the innermost PyTorch op (`op_of`), which tells us **which aten op** launched it (`aten::mm`, `aten::mul`…).
3. The kernel's name is matched against regexes (`CATEGORIES`, L42) to give **a category**: gemm, attention, conv, norm_reduce, copy_index, elementwise.
4. It sums kernel durations per (phase, category), counts kernels per decode step, and compares the summed kernel time to the wall time. That gives the **GPU-busy fraction**.

**How to read it.** A real example, the Blackwell baseline on img_small, 32 tokens ([summary.json](../profiling/reports/20261013-225052_RTXPRO5000Blackwell_torchprof_baseline_img_small_g32_baseline/summary.json)):

```json
"decode_wall_ms_per_step": 6.66,      ← wall time per token under the profiler (the profiler itself adds overhead)
"decode_kernel_ms_per_step": 2.98,    ← sum of kernel durations per token: what the GPU actually worked
"decode_kernels_per_step": 703.1,     ← kernels launched per token
"decode_gpu_busy_frac": 0.448,        ← 2.98 / 6.66: the GPU is idle 55 % of the time
"lm_decode_kernel_share_by_category": {"gemm": 0.724, "elementwise": 0.15, "copy_index": 0.074, ...}
```

**Interpretation:** 703 kernels per token with the GPU idle most of the time means launch overhead dominates. The GEMMs (the useful weight reads) are 72 % of kernel time. Everything else is small elementwise and copy work that fusion can remove.

- **`breakdown.csv`** has one row per (phase, category) with a count and total µs. For example, `lm_decode,elementwise,11222,12527.45` means 11,222 elementwise kernels over 31 decode steps (~362 per step), totalling 12.5 ms.
- **`kernels.csv`** has one row per (phase, category, aten op, kernel name), sorted by total time. The first row is the cuBLAS `gemvx` kernel launched by `aten::mm`: 2,480 launches, 23.9 µs each.
- **`ops_table.txt`** is PyTorch's own top-ops table.
- **`trace.json`** (git-ignored, large) opens in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing`. It shows a CPU row with Python and aten ops, and a GPU stream row with the kernels. The gaps between kernels on the GPU row are the launch overhead, visible directly.

For our own runtime, [profiling/profile_decode_step.py](../profiling/profile_decode_step.py) does the same for one decode step of the fused runtime. It runs the step eagerly, because inside a CUDA graph the profiler cannot attribute kernels to ops.

```bash
python profiling/profile_decode_step.py --fused --gemv bf16 --workload img_small --steps 20
```

### 5.3 Nsight Systems: the timeline

**Run it:**

```bash
bash profiling/nsys/profile_nsys.sh img_small 32                    # baseline
PYTHON=.venv-ref/bin/python bash profiling/nsys/profile_nsys.sh img_fhd 32 --variant fused_gemv_pf_vfused   # ours
```

Our variants need `.venv-ref`, a Python with headers, to build the CUDA extension. [scripts/setup_profilers.sh](../scripts/setup_profilers.sh) puts `nsys`/`ncu` on the PATH and installs nsys where it is missing (e.g. Colab).

**The command inside the script** (profile_nsys.sh:19), flag by flag:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \             # record CUDA API calls + kernels, our NVTX phase ranges, OS runtime calls
    --cuda-graph-trace=node \            # show each kernel inside a replayed CUDA graph (default shows the graph as one block)
    --sample=none --cpuctxsw=none \      # no CPU sampling (smaller report, less overhead)
    --capture-range=cudaProfilerApi --capture-range-end=stop \   # record only between profiler.start() and stop()
    -o <report> python profiling/run_once.py --workload img_small --gen-len 32
nsys stats --format csv --report cuda_gpu_kern_sum --report nvtx_sum --report cuda_api_sum <report>.nsys-rep
```

**The three CSVs** (example: Blackwell baseline, img_small, 32 tokens, in `profiling/reports/20261013-225112_RTXPRO5000Blackwell_nsys_img_small_g32/`):

| File | One row per | Example and how to read it |
|---|---|---|
| `*_cuda_gpu_kern_sum.csv` | kernel name | `63.3 %, 2512 instances, avg 26 µs, …gemvx::kernel…`: cuBLAS GEMV is 63 % of all GPU time, 2512 launches. Sort by `Time (%)` to see where GPU time goes. |
| `*_nvtx_sum.csv` | NVTX range | `:lm_decode 31 instances, avg 5.15 ms`: 31 decode steps of 5.15 ms wall each. `:vision_tower 1 instance, 5.2 ms`. This gives the time per phase. |
| `*_cuda_api_sum.csv` | CUDA API call | `cudaLaunchKernel 22,387 calls, 53.6 ms total, avg 2.4 µs`: the CPU spent 54 ms just launching kernels for one 32-token request. That is the launch overhead in one number. |

**In the GUI.** Open the `.nsys-rep` (git-ignored, but in `profiling/reports/` on the laptop after a sync) in the Nsight Systems desktop app; no GPU is needed.

- Expand the **CUDA HW** row to see the kernels.
- The **NVTX** row shows our phases.
- The **CUDA API** row shows the CPU's launch calls.
- In the baseline the decode phase is a comb of tiny kernels with gaps between them, and the API row is saturated with launches.
- With our runtime the decode steps are dense blocks of back-to-back graph nodes.

**Caveat.** Our kernels use Programmatic Dependent Launch (PDL, section 8.3): a kernel may start before the previous one finishes and wait inside. This inflates their per-kernel durations in nsys. For per-kernel numbers, run with `LFM2_PDL=0`; use the timeline for the overall picture.

### 5.4 Nsight Compute: hardware counters per kernel

nsys tells you *when* each kernel ran. ncu tells you *how well* it used the hardware. It replays every kernel several times to read different hardware counters, so it is slow and serializes kernels.

**Run it:**

```bash
bash profiling/ncu/profile_ncu.sh decode  img_small        # kernels of the decode steps
bash profiling/ncu/profile_ncu.sh prefill img_fhd           # LM prefill kernels
bash profiling/ncu/profile_ncu.sh vision  img_fhd           # vision tower + projector
GEN=520 NCU_LABEL=step512_ bash profiling/ncu/profile_ncu.sh decode img_small --start-at-decode-step 512
```

Environment knobs: `NCU_SET=basic|full`, `LAUNCHES=<max kernels>`, `NCU_LABEL`, `GEN`.

**The command inside** (profile_ncu.sh:36), flag by flag:

```bash
ncu --target-processes all \
    --profile-from-start off \                         # wait for profiler.start() in run_once.py
    --nvtx --nvtx-include "lm_decode/" --nvtx-include "lm_head/" \   # only kernels inside these NVTX ranges
    --launch-count 600 \                               # stop after N kernels (≈ 1 decode step is ~700–800)
    --set full \                                       # which metric sections ("basic" is ~5x faster)
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__grid_size,launch__block_size,launch__registers_per_thread \
    -o ncu_decode python profiling/run_once.py --workload img_small --gen-len 3
ncu --import ncu_decode.ncu-rep --csv --page raw --metrics <same list> > ncu_decode_metrics.csv
```

**What the metrics mean:**

| Metric | Meaning | How we use it |
|---|---|---|
| `gpu__time_duration.sum` | kernel duration | where the time goes |
| `dram__bytes_read.sum` / `_write.sum` | bytes moved to/from GPU memory | bytes ÷ time = achieved bandwidth |
| `dram__throughput…pct_of_peak` | memory bandwidth used, % of peak | **the key number for decode** (memory-bound): GEMVs should be near 90–100 % |
| `sm__throughput…pct_of_peak` | compute pipelines used, % of peak | the key number for compute-bound GEMMs |
| `sm__warps_active…pct_of_peak` | achieved occupancy | low occupancy plus low throughput = latency-bound kernel |
| `launch__grid_size`, `block_size`, `registers_per_thread` | launch shape | a grid of 1 block uses 1 of ~100 SMs |

**Reading the CSV.** The first data row holds the units (`%`, `us`, `Mbyte`). After that there is one row per kernel launch. The NVTX column shows which range the kernel was in. A real row from the Blackwell baseline decode:

```
Kernel: at::unrolled_elementwise_kernel<direct_copy_kernel_cuda…>   Block (128,1,1)   Grid (1,1,1)
dram__throughput 1.04 %    gpu__time_duration 2.4 us
```

This is a dtype-copy kernel: one block of 128 threads that finishes in 2.4 µs while using 1 % of the memory bandwidth. Its launch alone costs about as much as its work. Hundreds of these per token is the case for fusion.

**Aggregating the CSV:** [profiling/ncu/analyze_ncu.py](../profiling/ncu/analyze_ncu.py) groups the rows by (phase, short kernel name) and prints count, time, MB moved, achieved GB/s and % of peak bandwidth. On the T4 baseline decode (`--steps 1`, run on the committed CSV, no GPU needed):

```bash
python profiling/ncu/analyze_ncu.py profiling/reports/20261007-152958_T4_ncu_decode_img_small/ncu_decode_metrics.csv --steps 1
```
```
800 kernels, 17.266 ms, 3695.2 MB DRAM per step; overall 214.0 GB/s
phase       kernel                          n   time ms  %time      MB   GB/s  %peakBW
lm_decode   internal::kernel (GEMV)       105    11.494   66.6  3205.75  278.9    79.7
lm_decode   vectorized_elementwise_kernel 369     1.654    9.6    26.73   16.2     2.7
lm_head     internal::kernel (GEMV)         1     1.309    7.6   401.50  306.8    95.8
lm_decode   elementwise_kernel            140     1.076    6.2    27.99   26.0     4.3
lm_decode   reduce_kernel                  54     0.595    3.4     3.11    5.2     1.7
```

**How to read this:**

- The GEMVs move 3.2 GB at 80 % of peak: fine.
- `lm_head` is at 96 %: excellent.
- The 369 + 140 + 54 elementwise and reduce kernels move almost no data (27 MB) at 2–4 % of peak, yet take ~20 % of the time. That is pure launch latency.

**Profiling only our kernels:** [profile_ours.sh](../profiling/ncu/profile_ours.sh) runs any command under ncu with `--kernel-name regex:gemv|add_rmsnorm|shortconv|…` and `LFM2_PDL=0`, then prints a per-kernel table (µs, DRAM %, SM %, occupancy):

```bash
bash profiling/ncu/profile_ours.sh stack_fhd .venv-ref/bin/python profiling/run_once.py --workload img_fhd --gen-len 3 --variant fused_gemv_pf_vfused
```

**In the GUI.** Open a `.ncu-rep` in the Nsight Compute app and pick a kernel.

- **GPU Speed Of Light** shows memory % and compute % (which roof the kernel is near).
- **Memory Workload Analysis** shows bytes per memory level (DRAM vs L2 vs L1).
- **Launch Statistics** and **Occupancy** show the grid, blocks per SM, and register or shared-memory limits.
- With `--set full`, **Source Counters** shows stall reasons.

**Lessons we paid for:**

- `--set full` over a whole decode step (~800 kernels) needed > 9 GB of host RAM and was OOM-killed on Colab. We now use `basic` plus explicit metrics over whole phases, and `full` only on the first 150 decode kernels (the `ncu_deep` suite step).
- ncu **locks clocks to base** and serializes kernels, so absolute times are ~1.5× real. Use it for ratios (bandwidth %, share of time), never for latency.
- ncu needs access to the GPU performance counters. Without it, it fails with `ERR_NVGPUCTRPERM`; the suite detects this and skips the ncu steps. On the course server the TA enabled access for non-root users.
- `LAUNCHES` must cover the whole phase: the PyTorch profiler counted 795–845 prefill kernels, and an early cap of 500 missed the tail.
- On Blackwell (sm_120) the DRAM-bytes metric read 0 in our first capture. The % of peak metric still worked.

### 5.5 Energy, and power as a diagnostic

The `J/run`, `J/tok (dyn)`, `avg W` and `min SM MHz` columns of `summary.md` come from NVML (section 3.4). Two things we learned from them:

- **On the T4 the GPU sat at its 70 W cap and throttled**, with SM clocks between 465 and 1590 MHz. That is why the T4 numbers are a separate data point, never a substitute for server numbers.
- **On the Blackwell the GPU drew ~200–250 W while mostly waiting on the CPU.** Overhead therefore costs energy as well as time.

After our optimizations, decode runs at the 300 W limit, busy streaming weights instead of waiting. Energy per token then falls in proportion to the bytes read.

### 5.6 Targeted probes

Probes are small scripts that answer one question each, usually at the model's real tensor shapes, timed with CUDA events.

| Probe | Question | What it found |
|---|---|---|
| [sdpa_probe.py](../profiling/sdpa_probe.py) | Which attention backend does the LM really use, and what do the alternatives cost? | T4: `enable_gqa=True` forces the FP32 math path; `repeat_kv` would be 4–23× faster. Blackwell: flash with native GQA, no problem. |
| [vision_attn_probe.py](../profiling/vision_attn_probe.py) | Vision attention at 9 tiles × 1024 patches, per backend, with and without the padding mask | HF's mask selects the sm80 memory-efficient kernel (1038 µs/layer); cuDNN 419 µs; flash without a mask 287 µs |
| [preprocess_probe.py](../profiling/preprocess_probe.py) | Where does CPU preprocessing time go? | 22–69 ms at 1080p, very noisy; tokenization 2–3 ms |
| [ttft_probe.py](../profiling/ttft_probe.py) | Vision and LM prefill times, HF vs ours | used for every first-token optimization |
| [ttft_kernels.py](../profiling/ttft_kernels.py) | Kernel breakdown of our final 1080p first token | GEMMs 78 %, flash attention 13 %, our kernels 7 %, glue 1.6 % |
| [gemm_efficiency.py](../profiling/gemm_efficiency.py) | Are the first-token GEMMs near the tensor-core peak? | 88 % of the best achievable BF16 rate on Blackwell |
| [bench_gemv.py](../profiling/bench_gemv.py) | GEMV microbenchmark, cuBLAS vs ours, with > 400 MB of distinct weights so L2 can't help | drove the GEMV tuning (section 8.3) |
| [int8_gemm_probe.py](../profiling/int8_gemm_probe.py), [int8_quality_probe.py](../profiling/int8_quality_probe.py) | Would INT8 help prefill and vision, and which scheme keeps the answers? | section 10 |

### 5.7 The reasoning that ties it together

Every decision used one of two simple models:

- **Memory-bound work (decode).** Compare achieved bandwidth (bytes ÷ time) with the peak. A GEMV at 90 % of peak is done. A step that takes 2× its weight-read floor has overhead to remove.
- **Compute-bound work (first token).** Compare achieved TFLOPS with the best measured GEMM. At 88 % of peak, only fewer FLOPs or lower precision can help.

The *GPU-busy fraction* separates the two cases. If the GPU is idle half the time, no faster kernel helps until the launches are gone.

---

## 6. Moving to the real GPU

Our measurement GPU changed twice before we got it:

1. The plan was an RTX PRO 5000 at 8 h/week.
2. Then an L40 at 6 h/week.
3. Finally, the **course server with an RTX PRO 5000 Blackwell 48 GB**, in **3 × 2-hour slots per week**, over SSH.

The server's rules shaped the tooling: we could log in only during our slot, and every process was killed at its end.

| Tool | What it does |
|---|---|
| [scripts/remote/remote.sh](../scripts/remote/remote.sh) | Drives the server from the laptop through one SSH master connection: `push` (rsync code), `setup`, `slot <preset> <end time>`, `watch`, `sync` (pull results), `stop`. Login uses the `gpu` SSH alias; no password is stored anywhere. |
| [scripts/preflight.sh](../scripts/preflight.sh) | Checks the GPU model, other users, driver, nvcc, ncu counter access, internet and disk. |
| [scripts/setup_env.sh](../scripts/setup_env.sh) | Idempotent environment setup that picks the PyTorch build matching the driver. |
| [scripts/run_suite.sh](../scripts/run_suite.sh) | Runs steps **in priority order under a deadline** (`SLOT_MINUTES` / `SLOT_END`). A step starts only if its estimate fits, runs under a timeout, and the log ends with DONE / FAILED / SKIPPED so leftovers carry over to the next slot. A backlog keeps the GPU busy if the essentials finish early. |

**We rehearsed slot 1 end to end on Colab twice before the real slot.** In the real slot:

- Setup took 47 min because the server's internet was capped at about 300 KB/s per connection. We fixed it with long `uv` timeouts and 24 parallel HTTP range requests for the weights.
- The system Python had no headers, so nothing that compiles code at runtime worked (`torch.compile`, CUDA extensions). We fixed it with `.venv-ref`, built on a standalone Python. **This is why our variants need `.venv-ref/bin/python`.**
- The whole slot-1 suite (22 steps including 9 ncu runs) finished in 44 min.

**Blackwell baseline (BF16, batch 1):**

| Workload | TTFT | Vision / LM prefill | TPOT | J/token |
|---|---|---|---|---|
| text | 6.9 ms | – / 4.0 | 3.83 ms | 0.74 |
| img_small | 20.2 | 11.9 / 5.2 | 3.83 | 0.77 |
| img_fhd | 116 | 73.5 / 33.9 | 3.81 | 0.89 |

- TPOT is the same 3.8 ms for every input, and still 3.86 ms at 1024 tokens of context.
- Batch 16 costs only +5 % per step.
- The profiler shows 703 kernels per step with the GPU 45 % busy.
- **Decode is overhead-bound here even more clearly than on the T4**: 3.8 ms against a ~1.7 ms floor.
- One T4 finding did *not* carry over: on Blackwell, SDPA uses flash attention with native GQA, so the attention fix (O5) was T4-only. **Re-checking each finding on the target GPU before acting on it saved us real work.**

---

## 7. The plan, and the reference baselines

[notes/optimization_plan.md](notes/optimization_plan.md) turned the profiles into numbered steps. Each step has its code location, evidence, design, correctness check and open questions:

| # | Optimization | Evidence for it |
|---|---|---|
| O1 | Static decode runtime + **CUDA graph** | GPU idle ~55 % of each step; ~700 launches per token |
| O2 | Fused residual add + RMSNorm | ~360 RMSNorm kernels per step |
| O3 | Fused gated short conv (decode and prefill) | 6–8 kernels per conv block; no framework fuses it |
| O4 | Fused QK-norm + RoPE + KV write; merged q\|k\|v and w1\|w3 projections | ~130 attention non-GEMM kernels per step |
| O5 | Attention without the FP32 math fallback | T4 only (see above) |
| O6 | Low-precision weight GEMV (INT8 / INT4) | decode is bytes-bound once overhead is gone |
| O7 | Vision encoder: attention backend, padding-free tiles, fusions | vision attention at 24 % occupancy; padded tiles |
| P1 | GPU image preprocessing | 22–69 ms of CPU work at 1080p |

**Reference baselines (O0).** So that our gains are honest and not just "switch on an existing flag", we also measured what existing tools give on the same workloads:

| Reference | How | Decode TPOT |
|---|---|---|
| O0a `causal-conv1d` installed | HF picks up NVIDIA-style conv kernels | 3.73 ms (1.03×) |
| O0b HF static cache + `torch.compile(mode="reduce-overhead")` | variant `compile_static` in [variants.py](../src/lfm2opt/variants.py) | 2.72 ms (text) |
| O0c vLLM | [benchmarks/ref_vllm.py](../benchmarks/ref_vllm.py), its own venv; prefix and multimodal caches **off** for an honest TTFT | 2.36 ms |
| O0d llama.cpp BF16 / Q8_0 | [benchmarks/ref_llamacpp.sh](../benchmarks/ref_llamacpp.sh) | 2.26 / 1.45 ms |

**Direction set by the team** after the first optimization night: the **main result is lossless BF16 GPU work**. Quantization (INT8) is only a separately evaluated extension, and every variant is graded on standard VLM benchmarks next to its speed.

---

## 8. The optimizations, one by one

Each optimization follows the same loop: **evidence → build → check correctness → measure → log**.

### 8.1 O1: our own decode loop, replayed as a CUDA graph ([src/lfm2opt/runtime.py](../src/lfm2opt/runtime.py))

**Problem.** HF's decode loop is Python. It launches ~700 kernels one by one, grows the cache with `torch.cat` (new shapes every step), and synchronizes with the host every token to check stopping. A CUDA graph can't be captured over changing shapes or host syncs.

**What we built.** The `StaticDecoder` class.

1. **Static buffers** (`_alloc`, L68):
   - a KV cache `[B, 8, max_len, 64]` per attention layer, with `max_len` rounded up to a bucket of 256;
   - a conv state `[B, 2048, 3]` per conv layer;
   - the current token, position and step counter, and an output-token buffer.

   All of these **live on the GPU**, so nothing changes shape and the host never needs to read anything.
2. **One decode step as our own function** (`_step`, L86). It runs the same math as HF's modules, reusing their weights (`layer.operator_norm(h)`, `a.q_proj(x)`, …). The cache handling differs:
   - K and V are written in place with `index_copy_` at the device-side position;
   - attention is matmul → softmax → matmul over all `max_len` slots, with a mask for the empty ones;
   - the conv state is shifted in place;
   - greedy `argmax` and the write of the token into the output buffer happen *inside* the step.
3. **Capture once, replay per token** (`_get`, L293):

   ```python
   side = torch.cuda.Stream()                      # warm up on a side stream (cuBLAS handles, allocator)
   with torch.cuda.stream(side):
       for _ in range(3): step(s)
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):                       # record every kernel launch of one step
       step(s)
   ```

   `generate` (L458) then simply calls `s["graph"].replay()` once per token: **one launch instead of ~700**.
4. **Prefill is still HF's.** `prefill` (L417) calls the model's forward directly:

   ```python
   out = self.net(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1, **model_inputs)
   ```

   It then copies HF's KV and conv caches into the static buffers. This line (L434) is the one place where *our* code calls `forward()` itself.
5. **The variant** `cudagraph` ([variants.py:62](../src/lfm2opt/variants.py#L62)) replaces `net.generate` with this runtime. Every script keeps calling `net.generate(...)` with the same arguments, so nothing else changes.

**EOS.** Benchmarks need fixed length (EOS suppressed, like HF's `min_new_tokens`), while real use stops at EOS.

- The EOS mask is a **device-side bias** that the captured graph reads at replay. The same graph serves both modes.
- With `stop_at_eos`, the host checks for EOS after 1, 2, 4… then every 16 steps. Each check is one host sync.

**Correctness: [tests/check_runtime_numerics.py](../tests/check_runtime_numerics.py).** A *teacher-forced* comparison. HF generates 128 tokens. Our runtime then decodes **the same token sequence**: each step's input is forced to HF's token, so both see identical contexts and their logits can be compared directly. If the argmax differs where HF's top-2 margin is tiny, it is a BF16 near-tie; a flip at a large margin would be a bug. Result: argmax agrees on 377/381 steps, and every flip is at a margin ≤ 0.06.

```bash
python tests/check_runtime_numerics.py --workloads text,img_small,img_fhd --gen 128 [--fused]
```

**Result.** TPOT 3.83 → **2.91 ms**. Now the GPU is never idle in decode, so TPOT ≈ the kernel time. The remaining ~1.2 ms above the floor are the ~600 tiny kernels and not-quite-peak GEMVs.

### 8.2 O2–O4: fused CUDA kernels ([kernels/lfm2_decode.cu](../kernels/lfm2_decode.cu))

**Problem.** After O1 the step was GPU-bound at ~2.9 ms over ~700 kernels, and ~600 of them were tiny elementwise, norm and copy ops.

**How our CUDA code is organized:**

- **`kernels/<name>.cu`** holds the kernels plus plain C++ launchers that take raw pointers. nvcc 13 could not parse some PyTorch headers, so torch headers stay out of the `.cu` file.
- **`kernels/<name>.cpp`** holds the PyTorch bindings (`m.def("add_rmsnorm", …)`). They check tensors and call the launchers.
- **[src/lfm2opt/kernels.py](../src/lfm2opt/kernels.py)** **JIT-compiles** them with `torch.utils.cpp_extension.load` for the current GPU's architecture on first use, caches the build, and returns a Python module.

There are four extensions: `lfm2_decode` (decode and prefill), `image_prep`, `vision_ops` and `int8_gemm`.

**The decode kernels.** Each replaces a chain of PyTorch ops and keeps HF's BF16 rounding points:

| Kernel | Replaces | Per step |
|---|---|---|
| `add_rmsnorm` | residual add + RMSNorm (cast, pow, mean, rsqrt, mul, cast, mul) | 33 |
| `shortconv_step` | B·x, `cat`, state copy, depthwise conv, C·y, transposes | 10 |
| `qk_norm_rope_kv` | q/k head RMSNorm, RoPE from the device-side position, K/V cache write | 6 |
| `decode_attention_split` + `attn_combine` | attention over the static cache, flash-decoding style: 8 KV heads × 16 splits; the 4 query heads of a KV head share its K/V loads | 12 |
| `silu_mul` | SiLU × gate on the merged w1\|w3 output | 16 |
| `gemv` (BF16 / INT8) | `nn.Linear` at batch 1 | 65 |

The q, k and v weights are concatenated once into one matrix, and w1 and w3 into another (`_prepare_fused`, runtime.py:140). Then one GEMV replaces three, and one replaces two.

`_step_fused` (runtime.py:244) is the whole fused step. It is short enough to read in one sitting, and it maps one-to-one onto the HF layer structure of section 2.2. **~700 → 149 kernels per step.**

**Correctness: [tests/test_kernels.py](../tests/test_kernels.py).** Each kernel is compared with a PyTorch reference written with the same ops as `modeling_lfm2.py`, on random BF16 inputs at the real shapes.

- norm, conv, QK-norm/RoPE/KV and SwiGLU are **bit-exact**;
- attention is within one BF16 ulp;
- GEMVs are within the noise of accumulation order.

```bash
.venv-ref/bin/python tests/test_kernels.py        # 109/109 at the end of the project
```

**Result.** TPOT 2.91 → 2.23 ms. The split-K attention kept TPOT flat at long context: 1080p 2.65 → 2.27 ms.

### 8.3 Making the GEMVs and launches as fast as the hardware allows

After fusion, `profile_decode_step.py` showed the step was **94 % GEMV**. So the GEMVs were the remaining target.

- **Measuring the real roof.** A 2 GB read reaches 1.23 TB/s on this card; the 1.34 TB/s on the spec sheet is not reachable. cuBLAS's GEMVs were already at ~90 % of that, so the BF16 floor is ~1.9 ms.
- **Our GEMV** (`gemv_direct_kernel`): one warp per output row, 16-byte non-caching loads, 4 loads in flight per lane.
  - The breakthrough came from ncu plus the microbenchmark: our first version copied the input vector into shared memory and synchronized before doing any math. Removing that prologue brought every decode shape to 1.1–1.23 TB/s.
  - Variant `fused_gemv`: TPOT **2.08 ms** in BF16, which beats vLLM (2.36) and llama.cpp (2.26).
- **Programmatic Dependent Launch (PDL).** Each GEMV starts prefetching its weights while the previous kernel finishes (`griddepcontrol`). It is about 1 % faster inside the graph, and only available on sm_90+. `LFM2_PDL=0` turns it off for profiling.
- **O2d, GEMV epilogues.** The w1|w3 GEMV applies SwiGLU itself, and the in_proj GEMV does the conv step itself (`gemv_epi_kernel`). That brings the step down to **123 kernels**, bit-identical.
  - The first version was *slower*, because it halved the number of warps streaming weights. The second version gives each row its own warp and meets in shared memory.
  - Gain: ~10 µs per token. Small, as expected, because the step was already at 1.09× its floor.

### 8.4 Repeatability and error bars

[benchmarks/aggregate_runs.py](../benchmarks/aggregate_runs.py) collects every full-grid run of each variant (13–14 repeats in the end) and prints mean ± std:

| Variant | TPOT text | J/token text |
|---|---|---|
| HF eager | 3.716 ± 0.039 ms | 0.882 ± 0.029 |
| `fused_gemv` (BF16) | **2.078 ± 0.001** (1.79×) | 0.594 ± 0.010 |

Our CUDA-graph runtime is also about 40× more repeatable than eager HF, because there is no Python or launch jitter.

### 8.5 P1: image preprocessing on the GPU ([kernels/image_prep.cu](../kernels/image_prep.cu), [src/lfm2opt/preprocess.py](../src/lfm2opt/preprocess.py))

**Problem.** HF resizes, tiles, normalizes and patchifies on the CPU: 22–69 ms at 1080p, more than the whole vision tower, and very noisy.

**What we built.** `GpuImageProcessor.wrap(processor)` returns a processor with the same call signature, so `build_inputs` is unchanged. Inside it:

1. The uint8 image is copied to the GPU once, through a reused pinned buffer.
2. Kernel 1 does the horizontal antialiased resize.
3. Kernel 2 does the vertical resize **fused with** normalize, the tile split, 16×16 patchify and the padded `pixel_values` layout.
4. The resize weights are computed on the host with exactly ATen's integer formula (`aa_weights`), so the output is **bit-identical** to HF's CPU path. Bicubic was added for the 3B.

**Check:** [tests/test_preprocess.py](../tests/test_preprocess.py) runs 28 image sizes and contents and requires `pixel_values` max difference = 0. Use it with `bench.py --preprocess gpu`, or by appending `+gpuprep` to a variant name in the eval and demo scripts.

**Result** (1080p): preprocessing 25–63 ms → 3.9 ms including tokenization; the kernels themselves take 25 µs.

### 8.6 The first token: vision encoder (O7) and LM prefill (O3b)

- **O7a**, variant `vision_cudnn`: run SigLIP2's attention on cuDNN instead of the kernel HF's padding mask selects. 1080p vision 73.5 → 62.9 ms.
- **O7d, fused vision encoder** ([src/lfm2opt/vision.py](../src/lfm2opt/vision.py), [kernels/vision_ops.cu](../kernels/vision_ops.cu)), variant `vision_fused`:
  - one merged q|k|v GEMM;
  - **padding-free attention**: tiles are grouped by valid length and each group runs the flash kernel *without* a mask on just its real patches;
  - GELU fused into the fc1 GEMM's epilogue;
  - a fused residual-add + LayerNorm kernel.

  1080p vision 77.3 → **53.9 ms**. [tests/check_vision_fused.py](../tests/check_vision_fused.py) shows its features are exactly as close to an FP32 reference as HF's own BF16 path is.
- **O3b, fused prefill** (`StaticDecoder(fused_prefill=True)`, `_prefill_fused`, runtime.py:332): the prompt goes through cuBLAS GEMMs on the merged weights plus our norm, conv, RoPE/KV and SwiGLU kernels, writing **straight into the static decode buffers**. No HF cache, no copy. At this point HF's forward is no longer called at all; we call `get_image_features` ourselves (`_embed`, L316). 1080p LM prefill 37.0 → **31.2 ms**.

**The full BF16 stack** is variant `fused_gemv_pf_vfused` plus GPU preprocessing. It combines O1–O4, O2d, O3b, O7d and P1, with **no quantization**.

**How close to the limit is it?** [gemm_efficiency.py](../profiling/gemm_efficiency.py) found that 64.5 of the 87 ms of the 1080p first token are cuBLAS GEMMs at **88 % of the best measured BF16 rate**. [ttft_kernels.py](../profiling/ttft_kernels.py) found that what is left is flash attention (13 %), our kernels (7 %) and glue (1.6 %). Decode is at 1.09× its weight-read floor. **Nothing large is left that a BF16 kernel could remove.** That is why the next step had to be quantization (section 10).

### 8.7 What we tried and dropped (all logged)

| Attempt | Why it was dropped |
|---|---|
| `sdpa_repeat_kv` on Blackwell | It fixed a T4-only problem; on Blackwell it was slightly slower. |
| FP8 vision GEMMs (`vision_fp8`) | Slower (separate quantization kernels) and inaccurate (outputs diverged at token 7). |
| `torch.compile` on the vision tower | Slower than cuBLAS + cuDNN (118 vs 98 ms). |
| cuDNN instead of flash for the padding-free vision attention | Slower. |
| O2d first version | Slower; redesigned. |
| INT4 GEMV (round-to-nearest, AWQ-style, group sizes, mixed precision) | 0.76 ms/token, but only 86–93 % argmax agreement. Too lossy without GPTQ-class methods. |

---

## 9. Proving the answers did not change

Speed means nothing if the answers change. We checked quality at four levels, from cheapest to strongest.

| Level | Tool | What it shows |
|---|---|---|
| Kernel | `tests/test_kernels.py`, `test_preprocess.py`, `test_int8_gemm.py` | each kernel matches its reference (mostly bit-exact) |
| Logits | `tests/check_runtime_numerics.py`, `tests/check_vision_fused.py` | same next-token decisions as HF except at near-ties |
| Benchmarks | [eval_quality.py](../benchmarks/eval_quality.py) + [eval_compare.py](../benchmarks/eval_compare.py) | same accuracy on 5 standard VLM benchmarks, with a significance test |
| Long answers | [longform_compare.py](../benchmarks/longform_compare.py), [caption_compare.py](../benchmarks/caption_compare.py) | long descriptions stay as close to FP32 as HF's BF16, and score the same against human references |

### 9.1 Benchmarks

**Run:**

```bash
bash scripts/download_eval_data.sh                       # once: parquet files into data/eval/
python benchmarks/eval_quality.py --variants baseline,fused_gemv_pf_vfused+gpuprep [--limit pope=3000] [--tag x]
python benchmarks/eval_compare.py results/eval/<run> --md
```

**The benchmarks and their scoring:**

| Benchmark | Questions | Score |
|---|---|---|
| MMStar | 1500 multiple choice | option letter |
| POPE | 3 × 3000 yes/no | object hallucination |
| OCRBench | 1000 | substring match, like the official scorer |
| ChartQA | 2500 | relaxed accuracy, 5 % |
| TextVQA | 5000 | VQA accuracy |

The protocol is the same for every variant: identical prompts, greedy decoding, stop at EOS, batch 1. Image decoding is outside the timed region.

**Output** in `results/eval/<run>/`:

- one `<variant>.jsonl` per variant, with one line per question: prediction, token ids, score, timings;
- `summary.jsonl` with the score per benchmark and the energy;
- `config.json`, `env.json`.

**How to read `eval_compare`.** It pairs every question across the two variants and reports:

| Column | Meaning |
|---|---|
| score, Δ | each variant's accuracy and the difference |
| same answer % | identical text |
| discordant pairs | baseline right / ours wrong, and the reverse |
| **exact McNemar p-value** | the right test for paired yes/no outcomes; it uses only the discordant pairs |

p ≥ 0.05 means the difference is indistinguishable from noise.

**Result** (full sets, Blackwell, BF16 stack):

| Benchmark | Baseline | Ours | p |
|---|---|---|---|
| MMStar | 49.07 | 48.80 | 0.48 |
| POPE | 89.14 | 89.11 | 0.65 |
| OCRBench | 79.70 | 80.00 | 0.51 |
| ChartQA | 80.20 | 80.20 | 1.00 |
| TextVQA | 81.11 | 81.12 | 1.00 |

- 94.7–99.7 % of the answers are word-for-word identical.
- The rest are BF16 near-tie flips: 35 in our favour and 30 against.
- Per question, ours is 1.9–3.3× faster and uses 29–42 % less GPU energy.

**Cross-check with the official toolkit.** [benchmarks/vlmeval_mmstar.py](../benchmarks/vlmeval_mmstar.py) runs MMStar through VLMEvalKit with LiquidAI's own model integration and scorer. The baseline reproduces the model card: 50.20 vs the published 50.67. Ours scores 49.87 (p = 0.33).

### 9.2 Long answers

The benchmark answers are 1–11 tokens long, so they barely exercise decode.

- **`longform` bench.** "Describe this image in detail." on 500 distinct COCO images. The reference is **HF in FP32**, because greedy BF16 already drifts from FP32 at near-ties. The question is whether ours drifts *more* than HF's own BF16 does.
  - It does not: token ROUGE-L vs FP32 is 0.814 for ours vs 0.823 for HF BF16, not significant.
- **`coco_cap` / `coco_desc`**, scored against human references:
  - **CIDEr** for one-line captions vs 5 human captions.
  - **CHAIR** for long descriptions: the share of mentioned COCO objects that are not in the image, i.e. hallucinations.
  - Differences come with a paired bootstrap 95 % CI.

  ```bash
  python benchmarks/eval_quality.py --benches coco_cap,coco_desc --variants baseline,fused_gemv_pf_vfused+gpuprep
  python benchmarks/caption_compare.py results/eval/<run> --md
  ```

---

## 10. The extension: INT8 quantization

BF16 was at its limits (section 8.6), so the remaining gains needed fewer bytes. Quantization changes the model, so it is **reported separately** and graded on the same quality tests.

### 10.1 INT8 weight-only decode (O6)

Every decode weight is stored as INT8 with one FP scale per output row (`quantize_int8` in [src/lfm2opt/quant.py](../src/lfm2opt/quant.py)). The same GEMV kernel converts it to FP32 in registers. Half the bytes means roughly half the time:

- TPOT **1.11 ms** on Blackwell (3.35× HF); llama.cpp Q8_0 gets 1.45.
- J/token 0.33 (2.6× less).
- Benchmarks within ±0.4 pp, all p ≥ 0.26.

Variants: `fused_w8`, and `fused_w8_pf_vfused` on the full stack.

### 10.2 INT8 everywhere: prefill and vision too, for edge memory

The weight-only variant kept *both* BF16 and INT8 copies, so it used **more** memory (5.7 GB). For a Jetson we wanted INT8 weights only.

**Probes first** ([int8_gemm_probe.py](../profiling/int8_gemm_probe.py), [int8_quality_probe.py](../profiling/int8_quality_probe.py)):

- INT8 tensor cores make the first-token GEMMs −31 % faster, **but only if the scales are applied inside the GEMM's epilogue**. cuBLAS writes int32, and the extra pass to rescale it eats the gain.
- W8A8 (INT8 activations as well) cost quality mainly at two places: the inputs of the vision fc2 and the LM w2 (outlier activations after GELU and SwiGLU). One scale **per token and per group of 64 channels** fixed this; it was at the BF16 noise floor.

**Kernels** ([kernels/int8_gemm.cu](../kernels/int8_gemm.cu), [src/lfm2opt/int8.py](../src/lfm2opt/int8.py), tests in [tests/test_int8_gemm.py](../tests/test_int8_gemm.py)):

- `gemm_i8`: an INT8 `mma.sync` tensor-core GEMM with a 3-stage `cp.async` pipeline. It applies the scales, bias and GELU in registers and stores BF16 once. It is bit-exact vs an fp64 reference and as fast as cuBLAS's own INT8 GEMM.
- The per-64-channel activation scales are folded in per K stage.
- The norm and SwiGLU kernels (`add_rmsnorm_q`, `silu_mul_q`, `add_layernorm_q`) write the next GEMM's INT8 input directly.

The variant is `fused_w8all_pf_vfused`.

**Result (L40S):**

| | HF | ours BF16 | INT8 everywhere |
|---|---|---|---|
| Peak memory | 3.0–3.3 GB | 4.3–4.6 GB | **1.8–2.1 GB** |
| TPOT | 9.03 ms | 3.30 ms | 1.79 ms |
| First token, 1080p | 114.9 ms | 84.5 ms | 72.4 ms |
| 5 benchmarks | | ±0.2 pp | −0.36…+0.08 pp, all p ≥ 0.18 |
| COCO CIDEr / CHAIR_i | 122.8 / 5.78 % | 123.3 / 5.69 % | 123.3 / 5.86 % (n.s.) |

Long answers are *worded* differently from FP32 more often (ROUGE-L −0.056; the vision encoder is the main source). But scored against human references they are just as correct and hallucinate no more, so **no retraining was needed**.

---

## 11. Reproducing on other GPUs

**T4 (Colab).** To make the code portable:

- PDL is compiled only on sm_90+;
- the padding-free vision attention falls back to the memory-efficient kernel;
- `LFM2_DTYPE=bf16` forces BF16 (our kernels are BF16-only).

Results:

- All correctness tests pass.
- Decode is **2.3×** faster (INT8 4.2×) and uses −58 % J/token.
- The first token does **not** reproduce: the T4 has no BF16 tensor cores. Our kernels would need an FP16 version for that.

**L40S (course SLURM cluster).** Jobs are submitted from the login node (`ssh turing`, key login):

```bash
bash scripts/slurm/submit.sh <job-name> <hours> "<command>"     # sbatch on node13, reservation cs3406; log in jobs/<name>.out
bash scripts/slurm/submit.sh l40_suite 10 "bash scripts/slurm/l40_suite.sh"   # everything: tests, grid, batching, probes, quality
```

[l40_suite.sh](../scripts/slurm/l40_suite.sh) repeats the whole Blackwell study in one job:

- TPOT 9.0 → **3.3 ms (2.7×)**, energy −48 %;
- first token 1080p 146 → 93 ms;
- quality within ±0.2 pp on all five full benchmarks;
- our GEMVs at 81–97 % of the L40S's DRAM bandwidth;
- the 3B: TPOT 15.1 → 7.5 ms with unchanged quality, using the same kernels unchanged.

---

## 12. The demo video and its live proof

**Recording** ([demo/record.py](../demo/record.py), [demo/cases.json](../demo/cases.json)). Real images from 148×60 px to 35 MP and prompts as a user would type them. `Runner.measure` runs 3 warm-ups, 7 timed runs, then an untraced loop of ≥ 3 s for energy.

Every token gets a GPU timestamp:

- **HF:** a forward hook on `lm_head`, which runs once per token.
- **Ours:** the opt-in `StaticDecoder.trace` list, one CUDA event after the prefill and after each graph replay.

Image prep and the vision encoder are marked through a wrapper around `get_image_features`. Traced and untraced times agree within 1 %. Output: `results/demo/<run>/traces.json`.

**Rendering** ([demo/render.py](../demo/render.py), no GPU). It replays the median run of each case: tokens appear at their measured times, once at real speed and once slowed by a stated factor.

**Live proof** ([demo/live.py](../demo/live.py)). Re-runs the video's cases on the L40S with the *same* `Runner` and prints every number next to the video's.

```bash
python demo/record.py --tag l40_demo                                  # on the GPU
python demo/render.py results/demo/<run> --cases handwriting,cats,... --out presentations/demo   # anywhere
bash demo/live_l40.sh            # from the laptop: ssh → srun on node13 → live vs video table (--quick, --only cats,graph)
```

**Result:**

- 2.2× (BF16) and 3.1× (INT8) faster than HF to the complete answer (geometric mean over 8 cases), at 66 % / 49 % of HF's energy.
- The live run matched the video: 24/24 answers identical, and our decode time per token identical to 0.01 ms.
- Three totals differed by more than 4 %. All three are CPU-bound cases and are flagged by the script.

---

## 13. Where we are, and what is next

**The main result is lossless BF16** (`fused_gemv_pf_vfused` + GPU preprocessing):

| | Blackwell (HF → ours) | L40S (HF → ours) |
|---|---|---|
| Decode TPOT | 3.70 → 2.07 ms (1.8×) | 9.0 → 3.3 ms (2.7×) |
| First token 1080p, incl. preprocessing | 164 → 90 ms | 146 → 93 ms |
| Energy per token | −30 % | −48 % |
| Quality (5 benchmarks) | ±0.3 pp, n.s. | ±0.2 pp, n.s. |

**The INT8 extension:** 1.8 ms/token and **2.1 GB** peak memory on the L40S (HF 3.3 GB), with the same quality.

**How close to the limits:**

- decode is at ~1.1–1.2× its weight-read floor;
- the first-token GEMMs are at 88 % of the achievable tensor-core rate on Blackwell.

**Next steps, in order:**

1. A CUDA graph for the vision encoder. At one tile it is launch-bound, for BF16 and INT8 alike.
2. FP16 versions of our kernels, so the first-token gains reproduce on GPUs without BF16 (T4, some Jetsons).
3. The Jetson Orin: energy there comes from `tegrastats` rails, not NVML (see the TODO in [energy.py](../src/lfm2opt/energy.py)).

---

## Appendix A: command cheat-sheet

The GPU column says where each command can run:

- **any** means any CUDA GPU with `.venv` or `.venv-ref`;
- **ours** means our variants, which need `.venv-ref/bin/python` on the course machines (it has Python headers for the JIT build);
- **none** means a laptop is enough.

| Task | Command | GPU |
|---|---|---|
| Set up env, download weights | `bash scripts/setup_env.sh && bash scripts/download_model.sh` | any |
| Benchmark grid | `python benchmarks/bench.py --preset full --variants baseline,fused_gemv_pf_vfused [--preprocess gpu] [--no-phases]` | ours |
| Derived metrics table | `python benchmarks/summarize.py results/raw/<run>` | none |
| Same tokens in two runs? | `python benchmarks/compare_outputs.py results/raw/<a> results/raw/<b>` | none |
| Mean ± std over repeat runs | `python benchmarks/aggregate_runs.py --pattern 'final_*'` | none |
| Whole suite under a deadline | `SUITE=slot1 SLOT_MINUTES=100 bash scripts/run_suite.sh` | any |
| PyTorch-profiler breakdown | `python profiling/torch_profile.py --workload img_small --gen-len 32 [--variant …]` | any |
| One decode step of our runtime | `python profiling/profile_decode_step.py --fused --gemv bf16 --steps 20` | ours |
| Nsight Systems timeline | `bash profiling/nsys/profile_nsys.sh img_fhd 32 [--variant …]` (`PYTHON=.venv-ref/bin/python` for ours) | any |
| Nsight Compute, one phase | `NCU_SET=basic LAUNCHES=800 bash profiling/ncu/profile_ncu.sh decode img_small` | any + counter access |
| Nsight Compute, our kernels only | `bash profiling/ncu/profile_ours.sh <label> .venv-ref/bin/python <script> …` | ours + counter access |
| Bandwidth table from an ncu CSV | `python profiling/ncu/analyze_ncu.py <…_metrics.csv> --steps 1` | none |
| First-token components | `python profiling/ttft_probe.py --workloads img_fhd` | ours |
| Kernel tests | `python tests/test_kernels.py`, `tests/test_preprocess.py`, `tests/test_int8_gemm.py` | ours |
| Teacher-forced numerics | `python tests/check_runtime_numerics.py --gen 128 --fused` | ours |
| Quality eval | `python benchmarks/eval_quality.py --variants baseline,fused_gemv_pf_vfused+gpuprep` | ours |
| Compare quality | `python benchmarks/eval_compare.py results/eval/<run> --md` | none |
| Report figures | `python benchmarks/plot_results.py` → `report/figures/` | none |
| L40S job | `bash scripts/slurm/submit.sh <name> <hours> "<cmd>"` on the login node | — |
| Demo live check | `bash demo/live_l40.sh [--quick]` | — |

## Appendix B: result file formats

| Where | File | Content |
|---|---|---|
| `results/raw/<time>_<GPU>_<tag>/` (bench.py) | `env.json`, `config.json` | machine and versions; the exact arguments |
| | `idle.json` | idle power before the run |
| | `runs.jsonl` | one line per timed run (`kind: "run"`): `wall_ms`, energy fields, phase times (`vision_ms`, `lm_prefill_ms`, `lm_decode_ms` [list per step]…); one line per configuration (`kind: "config"`): median, min and max wall time, loop energy, peak memory, preprocessing time |
| | `outputs.jsonl` | generated token ids and text of the first repeat of each configuration |
| | `summary.md` / `summary.csv` | from summarize.py: TTFT, vision, prefill, TPOT, tok/s, host vs GPU step time, J/run, J/token (and above idle), average W, minimum SM clock, peak memory |
| `profiling/reports/<…>_torchprof_…/` | `summary.json`, `breakdown.csv`, `kernels.csv`, `ops_table.txt`, `trace.json` | section 5.2 |
| `profiling/reports/<…>_nsys_…/` | `*.nsys-rep`, `*_cuda_gpu_kern_sum.csv`, `*_nvtx_sum.csv`, `*_cuda_api_sum.csv` | section 5.3 |
| `profiling/reports/<…>_ncu_…/` | `*.ncu-rep`, `*_metrics.csv` | section 5.4 |
| `results/eval/<run>/` | `<variant>.jsonl`, `summary.jsonl`, `config.json`, `env.json` | section 9 |
| `results/demo/<run>/` | `traces.json` (per variant and case: answer ids, text pieces, per-run `token_ms` timestamps, median run, peak memory, energy) | section 12 |

`.nsys-rep`, `.ncu-rep` and `trace.json` are git-ignored because they are large. They exist only on the machine that produced them, or on the laptop after `remote.sh sync` / `colab.sh pull`. The CSV summaries are committed.
