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
