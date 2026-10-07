# LFM2.5-VL-1.6B inference pipeline

_Code walkthrough of the baseline (Hugging Face transformers v5.19.0) with pointers to the optimized reference implementations. All links point into `third_party/`. Last updated: 2026-10-07._

## 1. End-to-end flow

```
                       ┌──────────────────── PREFILL (once per request) ─────────────────────┐
image ─► preprocess ─► SigLIP2 vision encoder ─► projector ─► merge into token embeddings ─┐
(CPU)    (resize/tile,  (27 ViT layers, per tile)  (pixel-unshuffle                          │
          patchify,                                  + 2-layer MLP)                          ▼
          pad to 1024)                                                         LFM2 language model ─► logits ─► token
text ─► chat template ─► tokenizer ─► token embeddings ───────────────────────►  (16 layers)
                       └─────────────────────────────────────────────────────────────────────┘
                       ┌──────────────────── DECODE (once per output token) ─────────────────┐
                       │ last token ─► embedding ─► 16 LM layers (uses conv state + KV cache)│
                       │            ─► final RMSNorm ─► lm_head (65,536 × 2048) ─► sampling  │
                       └─────────────────────────────────────────────────────────────────────┘
```

Entry point (from the model card): `processor.apply_chat_template(...)` → `model.generate(...)`. The vision tower runs **only during prefill**. On later steps `generate` drops `pixel_values` ([generation/utils.py:751](../third_party/transformers/src/transformers/generation/utils.py#L751)).

## 2. Stage by stage

### 2.1 Image preprocessing (CPU, torchvision)
File: [image_processing_lfm2_vl.py](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py)

- `resize_and_split` ([L382](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L382)) chooses between two paths:
  - **Small image** (≲ 724×724, i.e. ≤ 2 × 256 tokens' worth of pixels): `smart_resize` ([L331](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L331)) makes both sides multiples of 32, giving 64–256 image tokens.
  - **Large image:** `crop_image_to_patches` ([L283](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L283)) cuts it into 2–10 tiles of 512×512 plus a thumbnail. Each tile gives 32×32 = 1024 patches, which become 256 tokens.
- Each tile is rescaled and normalized (mean = std = 0.5), split into 16×16 patches (`convert_image_to_patches`, [L134](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L134)), and **padded to `max_num_patches = 1024`** with a pixel mask ([L535](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L535)).
- Output: `pixel_values [N_tiles, 1024, 768]`, `pixel_attention_mask [N_tiles, 1024]`, `spatial_shapes [N_tiles, 2]`.

### 2.2 Vision encoder: SigLIP2 NaFlex so400m
File: [modeling_siglip2.py](../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py)

- Patch embedding is a `Linear(768 → 1152)` (not a conv).
- Position embeddings: a learned 16×16 grid, **bilinearly resized per image in a Python loop** to each tile's patch grid ([L131–L190](../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py#L131)).
- 27 × `Siglip2EncoderLayer`: LayerNorm → MHA (16 heads × 72) → residual → LayerNorm → MLP (1152 → 4304 → 1152, GELU-tanh) → residual.
- Attention runs over the **padded** 1024-patch sequences with a bidirectional mask. A dense mask stops SDPA from using its flash backend.
- Cost: about 0.84 TFLOP per tile in the linear layers alone, so roughly 9 TFLOP for a 10-tile + thumbnail image. **This dominates time to first token for high-resolution inputs.**

### 2.3 Projector and merge
File: [modeling_lfm2_vl.py](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py)

- `get_image_features` ([L160–L204](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L160-L204)) unpads each tile, reshapes it to (h, w, 1152), and calls the projector **in a Python loop over tiles**.
- `Lfm2VlMultiModalProjector` ([L37–L73](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L37-L73)): pixel-unshuffle 2×2 (1152 → 4608 channels, 4× fewer tokens) → Linear(4608 → 2048) → GELU → Linear(2048 → 2048).
- The image embeddings replace the `<image>` token embeddings via `masked_scatter` ([L280](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L280)).

### 2.4 Language model: LFM2.5-1.2B (hybrid conv + attention)
File: [modeling_lfm2.py](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py)

Layer pattern (`C` = conv, `A` = attention): `C C A C C A C C A C A C A C A C`

Each decoder layer ([L397–L426](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L397-L426)):
```
h = x + Operator(RMSNorm(x))        # Operator = ShortConv or Attention
y = h + SwiGLU_MLP(RMSNorm(h))      # w2( silu(w1 h) * w3 h ), 2048 → 8192 → 2048
```

**Gated short conv** (`Lfm2ShortConv`, [L342–L381](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L342-L381)):
```
B, C, x = split(in_proj(h))          # Linear 2048 → 6144
u = B * x                            # elementwise gate
u = causal_depthwise_conv1d(u, k=3)  # per-channel, 3 taps, state = last 2 inputs (cache holds 3)
y = out_proj(C * u)                  # elementwise gate, Linear 2048 → 2048
```
- Decode uses `causal_conv1d_update` ([L273](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L273)); prefill uses `causal_conv1d_fn` ([L293](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L293)).
- ⚠️ These use the `causal-conv1d` CUDA package **if it is installed**, and otherwise silently fall back to `F.conv1d` ([hub_kernels.py:984](../third_party/transformers/src/transformers/integrations/hub_kernels.py#L984)). The baseline must record which one ran.
- In eager mode a conv block is about 6–8 kernel launches (GEMM, transpose/chunk, mul, conv, state copy, mul, transpose, GEMM).

**Attention** (`Lfm2Attention`, [L220–L257](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L220-L257)):
- Separate `q_proj` (2048 → 2048), `k_proj` and `v_proj` (2048 → 512), so three GEMMs.
- RMSNorm on Q and K per head (head size 64), then RoPE (θ = 1e6), then GQA attention (32 query heads / 8 KV heads) via SDPA or eager, then `out_proj`.

**Final:** `embedding_norm` (RMSNorm) → `lm_head` (tied to the embeddings, 65,536 × 2048 = 134M parameters, the largest single GEMV in decode).

### 2.5 Cache
File: [cache_utils.py](../third_party/transformers/src/transformers/cache_utils.py)

| Layer type | Cache | Size (BF16, batch 1) | Update |
|---|---|---|---|
| Conv (×10) | `LinearAttentionLayer` ([L1066](../third_party/transformers/src/transformers/cache_utils.py#L1066)), conv state `[B, 2048, 3]` | 12 KB per layer, **constant** | in-place copy |
| Attention (×6) | `DynamicLayer` ([L113](../third_party/transformers/src/transformers/cache_utils.py#L113)), K/V `[B, 8, T, 64]` | **12 KB per token** in total | `torch.cat` **every step** ([L145](../third_party/transformers/src/transformers/cache_utils.py#L145)): reallocates and copies the whole cache |

Even at 4k context the KV cache is under 50 MB, so **KV-cache compression is not worth targeting**. Removing the reallocate-and-copy (a static cache) is worth doing, mainly because it enables CUDA graphs.

### 2.6 Generation loop
`GenerationMixin.generate` → `_sample` ([generation/utils.py:3097](../third_party/transformers/src/transformers/generation/utils.py#L3097)): an eager Python loop that runs one forward pass per token, applies logits processors (temperature, min-p, repetition penalty), samples, and checks stopping criteria. There are no CUDA graphs unless `torch.compile` is used with a static cache.

## 3. Parameter and byte budget (1.6B)

| Block | Parameters | Notes |
|---|---|---|
| SwiGLU MLPs (16 × 3 × 2048 × 8192) | 805M | Largest share of decode bytes |
| Conv blocks (10 × [in 6144×2048 + out 2048×2048]) | 168M | |
| Attention blocks (6 × [2048² × 2 + 512×2048 × 2]) | 63M | |
| Embedding / lm_head (tied) | 134M | Read once per token as the lm_head GEMV |
| Vision tower (SigLIP2 so400m) | ~411M | Prefill only |
| Projector | ~14M | Prefill only |

**Decode (batch 1):** about 1.17B parameters, or **~2.34 GB read per token**. At about 1.34 TB/s that is a floor of about 1.7 ms/token (≈ 575 tokens/s). Any gap between this floor and what we measure is overhead to remove: kernel launches, Python, extra memory traffic from unfused elementwise ops, and cache reallocation.

## 4. Optimized reference implementations

| | vLLM | llama.cpp |
|---|---|---|
| LM model code | [models/lfm2.py](../third_party/vllm/vllm/model_executor/models/lfm2.py) | [src/models/lfm2.cpp](../third_party/llama.cpp/src/models/lfm2.cpp) |
| Short conv | [short_conv.py `forward_cuda`](../third_party/vllm/vllm/model_executor/layers/mamba/short_conv.py#L215): Triton `causal_conv1d_fn/update` ([ops/causal_conv1d.py](../third_party/vllm/vllm/model_executor/layers/mamba/ops/causal_conv1d.py)); `B*x` and `C*y` remain **separate kernels** | `build_shortconv_block` ([L159](../third_party/llama.cpp/src/models/lfm2.cpp#L159)) → `ggml_ssm_conv` ([ssm-conv.cu](../third_party/llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu), FP32 only); gating kept separate |
| Linear layers | Merged QKV, merged gate/up (`MergedColumnParallelLinear`) | Quantized GEMV (`mmvq.cu`), float GEMV (`mmvf.cu`) |
| Vision | [lfm2_siglip2.py](../third_party/vllm/vllm/model_executor/models/lfm2_siglip2.py): **packed/unpadded** tiles with varlen attention (`cu_seqlens`) | [tools/mtmd/clip.cpp](../third_party/llama.cpp/tools/mtmd/clip.cpp) (`PROJECTOR_TYPE_LFM2`) |
| Launch overhead | CUDA graphs + `torch.compile` | CUDA graphs ([ggml-cuda.cu](../third_party/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)) |

**Gap we can target:** no framework fuses the whole gated short-conv (`B·x → conv → C·y`, plus the state update) into one kernel. Doing so is a self-contained CUDA project, and it applies to 10 of the 16 layers in both decode and prefill.

## 5. Open questions to answer by profiling

1. Baseline decode: what fraction of time per token is GPU kernel time versus CPU/launch gaps? (Nsight Systems timeline)
2. Of the GPU time, how much goes to GEMV vs. elementwise/norm vs. conv vs. attention? (Nsight Compute / PyTorch profiler)
3. What DRAM bandwidth do the GEMVs and the conv kernel actually achieve compared with peak?
4. Time to first token: how does it split between preprocessing, vision encoder, projector and LLM prefill, for 1 tile vs. 11 tiles?
5. Power: does decode draw well below TDP (memory-bound) while the vision encoder runs near TDP (compute-bound)? What are the energy implications?
