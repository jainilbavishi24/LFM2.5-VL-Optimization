# LFM2.5-VL-1.6B: the inference pipeline, explained stage by stage

_For the team. Assumes basic ML knowledge (linear layers, attention, softmax, embeddings). Code references are to the baseline Hugging Face transformers v5.19.0 in `third_party/`. Last updated: 2026-10-07._

> **Every dimension in this doc was checked against the code and config.** As a cross-check, adding up all the parameter shapes listed below gives **1,596,625,904**, exactly the parameter count Hugging Face reports for the checkpoint.

---

## 0. How to use this doc (work split)

The pipeline is cut into **three parts with clean hand-off points**. Each person owns one part. The input and output of each part are fully specified below, so you can study your part without waiting on the others.

| Owner | Part | Stages | Takes in | Hands over |
|---|---|---|---|---|
| **Jainil** | **A. From pixels to vision features** | 1 Image preprocessing · 2 Text and tokenization · 3 SigLIP2 vision encoder | Raw image + text prompt | `input_ids [1, L]`, `vision_features [N_t, 1024, 1152]`, `pixel_attention_mask`, `spatial_shapes` |
| **Gautam** | **B. From vision features into the language model** | 4 Projector · 5 Merge into embeddings · 6 Decoder-layer skeleton (RMSNorm, residuals) · 7 Gated short-conv block · 8 SwiGLU MLP | Part A's outputs | Hidden states `[1, L, 2048]` flowing through the layers |
| **Bibek** | **C. Attention, caches and generation** | 9 Attention block (QK-norm, RoPE, GQA) · 10 Caches (conv state + KV) · 11 Final norm, lm_head and sampling · 12 Prefill vs. decode loop | Hidden states `[1, L, 2048]` | Generated text |

Each part is about one third of the conceptual load: A has the most "data plumbing", B has the custom LFM2 block (our main kernel target), and C has the generation dynamics (our main overhead target).

**For each stage, every owner should be able to explain:** (1) the input tensor shapes and what they mean, (2) the exact computation, (3) the output shapes, (4) which code lines do it, and (5) where the time and memory traffic go (this is what we will optimize).

**Suggested independent-study method:** in one GPU session, run the model once on the worked example below and save every hand-off tensor with forward hooks (`torch.save`). Each person can then load their stage's real input on Colab and check that their own re-implementation reproduces the real output.

---

## 1. The big picture

```
                         PREFILL (runs once per request)
 image ──► [1] preprocess ──► [3] SigLIP2 encoder ──► [4] projector ──┐
          (tile, patchify,     (27 ViT layers)        (pixel-unshuffle │
           pad to 1024)                                + MLP)          ▼
 text  ──► [2] chat template ──► tokenizer ──► token embeddings ──► [5] merge
                                                                       │
                                                                       ▼
                     [6–9] LFM2 language model, 16 layers ──► [11] final norm ──► lm_head ──► first token
                     (10 gated short-conv + 6 attention;          writes [10] caches)

                         DECODE (runs once per generated token)
 last token ──► embedding ──► 16 layers (read + update caches) ──► final norm ──► lm_head ──► sample ──► repeat
```

Key facts:
- The vision encoder and projector run **only in prefill**. During decode, `generate` stops passing `pixel_values` ([generation/utils.py:751](../third_party/transformers/src/transformers/generation/utils.py#L751)).
- The language model is a **hybrid**. Layer types by index 0–15: `C C A C C A C C A C A C A C A C` (C = gated short-conv, A = attention). Only the 6 attention layers need a growing KV cache.
- All weights are BF16. Hidden size of the language model = **2048**; of the vision encoder = **1152**.

### Notation

| Symbol | Meaning | Worked-example value |
|---|---|---|
| `N_t` | number of image "tiles" fed to the vision encoder | 1 |
| `h_p, w_p` | patch grid of a tile (rows, cols of 16×16 patches) | 26, 36 |
| `P = h_p·w_p` | real (non-padding) patches in a tile | 936 |
| `I` | image tokens after the projector (= Σ over tiles of `h_p/2 · w_p/2`) | 234 |
| `L` | prompt length in tokens (text + image tokens) | ≈ 250 (print `input_ids.shape` to get the exact value) |
| `T` | tokens already in the cache (during decode) | grows from L |

### Worked examples (use these to check your understanding)

- **Example 1 (main): small image.** `assets/images/coco_000000039769.jpg`, 640×480 (W×H), prompt "What is in this image?". Gives **1 tile, 26×36 patches, 234 image tokens**.
- **Example 2: large image.** A 1920×1080 photo. Gives **8 tiles (4 wide × 2 high) of 512×512 plus a 384×672 thumbnail = 9 encoder sequences, 8·256 + 252 = 2300 image tokens**.

Both are derived step by step in Stage 1.

---

# PART A: From pixels to vision features (Jainil)

## Stage 1: Image preprocessing (CPU)

**Code:** `Lfm2VlImageProcessor` in [image_processing_lfm2_vl.py](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py), settings from [processor_config.json](../third_party/LFM2.5-VL-1.6B/processor_config.json).

**Input:** one RGB image, `uint8 [3, H, W]`.
**Output:**

| Tensor | Shape | Meaning |
|---|---|---|
| `pixel_values` | `[N_t, 1024, 768]` float | each tile as a sequence of 1024 flattened 16×16×3 patches (padded) |
| `pixel_attention_mask` | `[N_t, 1024]` int | 1 = real patch, 0 = padding |
| `spatial_shapes` | `[N_t, 2]` long | `(h_p, w_p)` of each tile |
| (internal) `image_rows, image_cols, image_sizes` | | used by Stage 2 to decide how many `<image>` tokens to insert |

### 1a. Decide: one tile, or many? ([`resize_and_split`, L382](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L382))

Key settings: patch size 16, downsample factor 2 (so **32 px = 1 final token** per side), `min_image_tokens = 64`, `max_image_tokens = 256`, tile size 512, `max_pixels_tolerance = 2.0`, 2–10 tiles.

- `_is_image_too_large` ([L366](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L366)): round H and W to multiples of 32. If `H·W > 256 tokens × 32² px × 2.0 = 524,288 px` (about 724×724), the image is "large".
- **Small image:** `smart_resize` ([L331](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L331)) rescales it, keeping the aspect ratio, so that both sides are multiples of 32 and the area is between 64·32² and 256·32² = 262,144 px (i.e. 64–256 tokens).
- **Large image:** `crop_image_to_patches` ([L283](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L283)) picks a grid `(cols, rows)` with 2 ≤ cols·rows ≤ 10 whose aspect ratio is closest to the image's. It resizes the image to `512·cols × 512·rows` and cuts it into 512×512 tiles (row-major order). It then appends a **thumbnail**: the whole image `smart_resize`d as if it were small.

**Example 1, step by step (640×480):** rounded area 640·480 = 307,200 ≤ 524,288, so it is **small**. Since 307,200 > 262,144, it must shrink: β = √(307,200/262,144) = 1.0825. H → ⌊480/1.0825/32⌋·32 = **416**, W → ⌊640/1.0825/32⌋·32 = **576**. Patch grid 26×36 = **936 patches**, giving 13×18 = **234 tokens**.

**Example 2, step by step (1920×1080):** rounded area 1920·1088 ≫ 524,288, so it is **large**. Aspect 1.78. The closest allowed grid ratios are 2/1 and 4/2 (both 2.0); the tie is broken toward the larger grid because the image area exceeds half of 4·2·512². So the grid is **4×2 = 8 tiles**, from a 2048×1024 resize. Thumbnail: β = √(2,073,600/262,144) = 2.8125, giving 384×672 → 24×42 = 1008 patches → **252 tokens**.

### 1b. Normalize and patchify ([`_preprocess`, L438](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L438))

1. **Rescale and normalize:** `x = (x/255 − 0.5)/0.5`, so pixel values lie in [−1, 1].
2. **Patchify** (`convert_image_to_patches`, [L134](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L134)): `[N, 3, H, W] → reshape [N, 3, h_p, 16, w_p, 16] → permute → [N, h_p, w_p, 16, 16, 3] → [N, h_p·w_p, 768]`. Each row is one patch flattened in (row, col, channel) order; patches are in row-major order over the grid.
3. **Pad** (`pad_along_first_dim`, [L150](../third_party/transformers/src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py#L150)): every tile is padded with zeros to **1024 patches**, and a mask marks the real ones.

Example 1: `pixel_values [1, 1024, 768]` (936 real + 88 padding), `spatial_shapes [[26, 36]]`.
Example 2: `pixel_values [9, 1024, 768]`. The 8 full tiles have 1024 real patches each (32×32); the thumbnail has 1008 real + 16 padding.

**Performance notes:** this runs on the CPU (torchvision). Padding means the encoder computes on patches that are thrown away: 9% wasted in Example 1, and up to 75% for a tiny 64-token image.

---

## Stage 2: Text, chat template and tokenization

**Code:** [processing_lfm2_vl.py](../third_party/transformers/src/transformers/models/lfm2_vl/processing_lfm2_vl.py), [chat_template.jinja](../third_party/LFM2.5-VL-1.6B/chat_template.jinja).

**Input:** the conversation (list of messages with image and text parts).
**Output:** `input_ids [1, L]` (int64), `attention_mask [1, L]` (all ones for a single prompt).

1. **Chat template** turns the conversation into a string:
   ```
   <|startoftext|><|im_start|>user
   <image>What is in this image?<|im_end|>
   <|im_start|>assistant
   ```
2. **Image placeholder expansion** (`_build_image_tokens`, [L191](../third_party/transformers/src/transformers/models/lfm2_vl/processing_lfm2_vl.py#L191)): the single `<image>` is replaced by exactly as many `<image>` tokens as the projector will output, wrapped in special tokens.
   - Single tile: `<|image_start|>` + 234 × `<image>` + `<|image_end|>`.
   - Multi-tile: `<|image_start|>`, then for each tile `<|img_row_r_col_c|>` + 256 × `<image>`, then `<|img_thumbnail|>` + 252 × `<image>`, then `<|image_end|>`.
   - Token count per tile/thumbnail = `⌈h_p/2⌉ · ⌈w_p/2⌉`. **This must match Stage 4's output exactly**, or Stage 5 raises an error.
3. **Tokenizer** (Hugging Face fast tokenizer, vocabulary 65,536) gives the ids. Useful ids: `<|pad|>` = 0, `<|startoftext|>` = 1, `<|im_end|>` = 7 (also the stop token), `<image>` = 396.

Example 1: L ≈ 15 text tokens + 236 image-related tokens ≈ **250**. Example 2: L ≈ 15 + 2311 ≈ **2330**.

**Performance notes:** negligible cost, but it **sets L**, which determines prefill cost. More tiles means more tokens means a slower first token.

---

## Stage 3: SigLIP2 vision encoder

**Code:** [modeling_siglip2.py](../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py) (`Siglip2VisionModel`, L499). Config: 27 layers, hidden 1152, 16 heads × 72, MLP 4304, LayerNorm eps 1e-6, no pooling head.

**Input:** `pixel_values [N_t, 1024, 768]`, `pixel_attention_mask [N_t, 1024]`, `spatial_shapes [N_t, 2]`.
**Output:** `last_hidden_state [N_t, 1024, 1152]` (rows past P for each tile are garbage from padding and get dropped in Stage 4).

### 3a. Patch embedding ([`Siglip2VisionEmbeddings`, L114](../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py#L114))

- `patch_embedding`: **Linear(768 → 1152, with bias)**, applied to every patch. This is the same as the usual Conv2d(16×16, stride 16), since patches were flattened beforehand. `[N_t, 1024, 768] → [N_t, 1024, 1152]`.
- **Position embeddings:** a learned table of 256 vectors, viewed as a **16×16×1152 grid**. For each tile, the grid is **bilinearly resized** (with antialiasing) to `h_p × w_p` (e.g. 16×16 → 26×36), flattened to `[P, 1152]`, and written into a `[1024, 1152]` buffer. Padding rows are filled with a copy of row 0. This happens in a **Python loop over tiles** ([L167](../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py#L167)). This is the "NaFlex" trick: one position table serves any aspect ratio.
- `embeddings = patch_embeds + pos_embeds` → `[N_t, 1024, 1152]`.

### 3b. Attention mask

`create_bidirectional_mask` turns `pixel_attention_mask [N_t, 1024]` into a 4-D boolean mask (broadcastable to `[N_t, 1, 1024, 1024]`) that stops every query from attending **to** padding keys. Attention is bidirectional (not causal): every real patch sees every other real patch in the **same tile only**. Tiles never see each other.

### 3c. Encoder layer, ×27 ([`Siglip2EncoderLayer`, L353](../third_party/transformers/src/transformers/models/siglip2/modeling_siglip2.py#L353)), pre-norm transformer

```
x: [N_t, 1024, 1152]
r = x
x = LayerNorm1(x)                                   # mean/var over 1152, learned γ, β
q, k, v = Linear_q(x), Linear_k(x), Linear_v(x)     # each 1152→1152 with bias → [N_t, 1024, 1152]
q, k, v → view [N_t, 1024, 16, 72] → transpose [N_t, 16, 1024, 72]
a = softmax(q·kᵀ / √72 + mask) · v                  # [N_t, 16, 1024, 72], via PyTorch SDPA
a → [N_t, 1024, 1152] → out_proj (1152→1152, bias)
x = r + a
r = x
x = LayerNorm2(x)
x = fc2( gelu_tanh( fc1(x) ) )                      # 1152 → 4304 → 1152, with biases
x = r + x
```

### 3d. Post-LayerNorm

`post_layernorm` (LayerNorm over 1152) → `last_hidden_state [N_t, 1024, 1152]`. There is no pooling head (`vision_use_head = false`); all patch features are kept.

**Parameters:** ≈ 412.6M (15.24M per layer × 27, plus embeddings).
**Compute:** ≈ 0.97 TFLOP per 1024-patch tile (≈ 0.84 in linear layers, ≈ 0.13 in attention), so ≈ 8.7 TFLOP for Example 2.

**Performance notes:** this is **compute-bound** (big matrix multiplies, 1024 tokens per tile). It dominates time to first token for large images. Waste comes from padding, the explicit mask (which can stop SDPA from using its fastest flash kernel), and the per-tile Python loop for position embeddings.

### ➜ Hand-off from Part A to Part B

| Tensor | Shape | Example 1 |
|---|---|---|
| `input_ids` | `[1, L]` | `[1, ≈250]` |
| `last_hidden_state` | `[N_t, 1024, 1152]` | `[1, 1024, 1152]` |
| `pixel_attention_mask` | `[N_t, 1024]` | 936 ones, 88 zeros |
| `spatial_shapes` | `[N_t, 2]` | `[[26, 36]]` |

---

# PART B: From vision features into the language model (Gautam)

## Stage 4: Multimodal projector

**Code:** `get_image_features` and `Lfm2VlMultiModalProjector` in [modeling_lfm2_vl.py:37-73, 160-204](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L37-L73).

**Input:** Part A's vision features, mask and spatial shapes.
**Output:** `image_features [I, 2048]`, one row per `<image>` token, in tile order (tiles row-major, thumbnail last).

**Loop over tiles** (Python `for`, [L187](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L187)). For tile i:
1. **Unpad:** keep the first `P = mask.sum()` rows → `[1, P, 1152]` (Example 1: `[1, 936, 1152]`).
2. **Back to 2-D:** reshape to `[1, h_p, w_p, 1152]` = `[1, 26, 36, 1152]`.
3. **Pixel-unshuffle (2×2 → 1):** every 2×2 block of neighbouring patches is concatenated along channels: `[1, 26, 36, 1152] → [1, 13, 18, 4608]`. This cuts the token count by 4×, and no information is lost. (Implemented with two reshape+permute steps, [L65](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L65).)
4. **MLP:** `Linear(4608 → 2048, bias) → GELU (exact/erf) → Linear(2048 → 2048, bias)` → `[1, 13, 18, 2048]`. No LayerNorm in this checkpoint (`projector_use_layernorm = false`).
5. **Flatten:** `[234, 2048]`.

All tiles are concatenated: Example 1 gives `[234, 2048]`; Example 2 gives `[8·256 + 252, 2048] = [2300, 2048]`.

**Parameters:** 13.6M. The cost is small, but it is a Python loop with several small kernels per tile.

## Stage 5: Merging image features into the token sequence

**Code:** `Lfm2VlModel.forward` ([L259-280](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L259-L280)).

**Input:** `input_ids [1, L]`, `image_features [I, 2048]`.
**Output:** `inputs_embeds [1, L, 2048]` (BF16).

1. **Token embedding lookup:** `embed_tokens` is a table `[65,536 × 2048]`; row `id` is that token's vector. Gives `[1, L, 2048]`. The `<image>` positions get a placeholder vector for now.
2. **Placeholder mask:** `input_ids == 396` → boolean `[1, L, 1]`. The code checks that exactly I positions are marked.
3. **`masked_scatter`:** overwrites the I placeholder rows, in order, with the I rows of `image_features`.

From here on the language model **does not know which tokens came from the image**. They are just vectors in the sequence.

## Stage 6: Language-model skeleton (shared by all 16 layers)

**Code:** `Lfm2Model.forward` ([modeling_lfm2.py:469-525](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L469-L525)), `Lfm2DecoderLayer` ([L384-426](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L384-L426)).

**Input:** `inputs_embeds [1, L, 2048]`. **Output (after 16 layers):** `[1, L, 2048]`.

Before the layers, the model computes once per forward pass:
- `position_ids = arange(L)` (in decode: `[T]`, the next position), shape `[1, L]`.
- RoPE `cos, sin` tables `[1, L, 64]` (used only by attention layers; see Stage 9).
- Masks: a causal mask for attention layers and a padding mask for conv layers. With one unpadded prompt these are typically `None` (no padding to hide).

**Each decoder layer** (the same structure whether the layer is conv or attention):
```
r = x                                      # [1, L, 2048]
x = Operator( RMSNorm_op(x) )              # Operator = ShortConv (Stage 7) or Attention (Stage 9)
x = x + r
x = x + MLP( RMSNorm_ffn(x) )              # SwiGLU (Stage 8)
```

**RMSNorm** ([L44](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L44)), with learned weight `w [2048]` and eps = 1e-5:
```
y = w ⊙ x / sqrt( mean(x², over last dim) + eps )      # computed in FP32, cast back to BF16
```
Unlike LayerNorm there is no mean subtraction and no bias.

**Performance notes:** each RMSNorm and each residual add is a separate small, memory-bound kernel (and RMSNorm alone launches several kernels in eager mode: cast, pow, mean, rsqrt, mul, cast, mul). These are fusion targets: residual-add + RMSNorm can be one kernel.

## Stage 7: Gated short-convolution block (10 of the 16 layers; our main kernel target)

**Code:** `Lfm2ShortConv` ([L316-381](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L316-L381)). Used in layers 0, 1, 3, 4, 6, 7, 9, 11, 13, 15.

**Idea:** a cheap replacement for attention. Each channel looks back at only the **previous 2 tokens** (a causal depthwise 1-D convolution of width 3), surrounded by two multiplicative **gates** (B and C) computed from the input itself. Long-range mixing is left to the 6 attention layers.

**Weights:** `in_proj` Linear(2048 → 6144, no bias); `conv.weight [2048, 1, 3]` (one 3-tap filter per channel, no bias); `out_proj` Linear(2048 → 2048, no bias).

**Math, per channel c and position t:**
```
[B, C, x] = split(in_proj(h), 3)            # three 2048-dim vectors per token
u[c, t]   = B[c, t] · x[c, t]               # input gate
v[c, t]   = w[c,0]·u[c,t−2] + w[c,1]·u[c,t−1] + w[c,2]·u[c,t]   # causal conv (u at negative t = 0, or from cache)
y[c, t]   = C[c, t] · v[c, t]               # output gate
out       = out_proj(y)
```

**Prefill, shape by shape** (input `h [1, L, 2048]` = RMSNorm output):

| Step | Code | Shape |
|---|---|---|
| in_proj | `self.in_proj(h)` | `[1, L, 6144]` |
| to channels-first | `.transpose(-1, -2)` | `[1, 6144, L]` |
| split | `.chunk(3, dim=-2)` | B, C, x: each `[1, 2048, L]` |
| input gate | `B * x` | `[1, 2048, L]` |
| save conv state | `update_conv_state`: copy the **last 3 columns** of `B*x` into the cache | cache `[1, 2048, 3]` |
| causal conv | `F.conv1d(groups=2048, padding=2)` then keep the first L outputs (or the `causal-conv1d` CUDA kernel if installed) | `[1, 2048, L]` |
| output gate | `C * conv_out` | `[1, 2048, L]` |
| back to tokens-first | `.transpose(-1, -2).contiguous()` | `[1, L, 2048]` |
| out_proj | `self.out_proj(y)` | `[1, L, 2048]` |

**Decode** (L = 1; input `[1, 1, 2048]`), via `causal_conv1d_update` ([L273](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L273)):
1. `in_proj` → `[1, 6144, 1]` → B, C, x `[1, 2048, 1]` → `u = B*x`.
2. `cat(conv_state [1,2048,3], u) → [1, 2048, 4]`. The last 3 columns are copied back into `conv_state` (in place).
3. `conv1d` without padding over the 4 columns gives 2 outputs; keep the last one → `[1, 2048, 1]`.
4. `C *`, transpose, `out_proj` → `[1, 1, 2048]`.

**Parameters per block:** 16.78M (in 12.58M, out 4.19M, conv 6K).
**Kernels per block in eager decode:** about 8–10 (GEMV, transpose/chunk views, mul, cat, copy, conv, slice, mul, contiguous, GEMV). The work between the two GEMVs is tiny (a few multiply-adds per channel), so **launch overhead and extra memory round-trips dominate**. A single fused kernel doing gate → conv → state update → gate is the target (see `docs/inference_pipeline.md` §4: no framework does this yet).

> ⚠️ Without the `causal-conv1d` package, the conv uses PyTorch `F.conv1d`. With it, a CUDA kernel is used. Always record which one ran.

## Stage 8: SwiGLU feed-forward (all 16 layers)

**Code:** `Lfm2MLP` ([L111-128](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L111-L128)).

**Input/output:** `[1, L, 2048] → [1, L, 2048]`.
```
g = w1(x)            # Linear 2048 → 8192 (no bias)   "gate"
u = w3(x)            # Linear 2048 → 8192 (no bias)   "up"
y = w2( silu(g) ⊙ u ) # Linear 8192 → 2048 (no bias)   "down"; silu(z) = z·sigmoid(z)
```
The config says `intermediate_size = 12288`, but `block_auto_adjust_ff_dim` shrinks it to ⌊2·12288/3⌋ = **8192** ([L115](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L115)).

**Parameters:** 50.3M per layer × 16 = **805M**, about 69% of the language model. **In decode the MLPs are most of the bytes read per token**, so their GEMVs set the bandwidth floor. Fusion targets: merge w1 and w3 into one GEMV, and fuse `silu(g)⊙u`.

### ➜ Hand-off from Part B to Part C

Part C receives `x [1, L, 2048]` (RMSNorm already applied) at the input of each attention layer (indices 2, 5, 8, 10, 12, 14), and the output `[1, L, 2048]` of layer 15.

---

# PART C: Attention, caches and generation (Bibek)

## Stage 9: Attention block (6 of the 16 layers)

**Code:** `Lfm2Attention` ([L202-257](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L202-L257)), SDPA wrapper [sdpa_attention.py](../third_party/transformers/src/transformers/integrations/sdpa_attention.py).

**Config:** 32 query heads, 8 key/value heads (**GQA**: each KV head is shared by 4 query heads), head size 64, RoPE θ = 1,000,000.
**Weights:** `q_proj` 2048→2048, `k_proj` 2048→512, `v_proj` 2048→512, `out_proj` 2048→2048 (no biases); `q_layernorm`, `k_layernorm` are RMSNorms of size 64.

**Prefill, step by step** (input `x [1, L, 2048]`; past length T = 0):

| Step | Computation | Shape |
|---|---|---|
| Q | `q_proj(x).view(1, L, 32, 64)` | `[1, L, 32, 64]` |
| QK-norm on Q | RMSNorm over the 64 dims of **each head** | same |
| to heads-first | `.transpose(1, 2)` | Q `[1, 32, L, 64]` |
| K | `k_proj` → view → RMSNorm per head → transpose | K `[1, 8, L, 64]` |
| V | `v_proj` → view → transpose (no norm) | V `[1, 8, L, 64]` |
| RoPE | rotate Q and K by position (below) | unchanged shapes |
| cache update | append K, V to this layer's cache, get all keys/values so far | K, V `[1, 8, T+L, 64]` |
| attention | `softmax(Q Kᵀ / √64 + causal) V`, 4 query heads per KV head | `[1, 32, L, 64]` |
| merge heads | transpose → `[1, L, 32, 64]` → reshape | `[1, L, 2048]` |
| out_proj | Linear 2048 → 2048 | `[1, L, 2048]` |

**QK-norm:** normalizing Q and K per head (before RoPE) keeps the attention logits in a stable range. It is a small, memory-bound op, and a candidate to fuse with RoPE.

**RoPE** ([L64-161](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L64-L161)) encodes position by rotating pairs of dimensions:
```
inv_freq[i] = 1 / 1e6^(2i/64),  i = 0..31                    # 32 frequencies
angle[p, i] = p · inv_freq[i]                                 # p = token position
cos, sin    = cos/sin of [angle, angle]                       # [1, L, 64]
rotate_half(q) = [ −q[32:64], q[0:32] ]
q' = q ⊙ cos + rotate_half(q) ⊙ sin       (same for k)
```
Dimension i is paired with dimension i+32. The dot product q'·k' then depends only on the **relative** distance between positions.

**Attention kernel:** PyTorch `scaled_dot_product_attention` ([L170](../third_party/transformers/src/transformers/integrations/sdpa_attention.py#L170)). In prefill with no mask it uses `is_causal=True` and native GQA (`enable_gqa=True`), so a fast flash/efficient kernel can be used. If a mask is present, K and V are first **physically repeated** 4× to 32 heads (`repeat_kv`).

**Decode** (L = 1): Q is `[1, 32, 1, 64]`; K, V from the cache are `[1, 8, T+1, 64]`; the output is `[1, 1, 2048]`. Here `is_causal` is false (a single query may attend to everything before it).

**Parameters per block:** 10.5M (×6 = 63M). Attention FLOPs in prefill grow as L², which matters for Example 2 (L ≈ 2330) but is small for Example 1.

## Stage 10: Caches (what persists between forward passes)

**Code:** [cache_utils.py](../third_party/transformers/src/transformers/cache_utils.py), a hybrid `DynamicCache` with one entry per layer.

| Layer type | Cache object | What it stores | Shape (batch 1) | Size | How it is updated |
|---|---|---|---|---|---|
| Conv (×10) | `LinearAttentionLayer` ([L1066](../third_party/transformers/src/transformers/cache_utils.py#L1066)) | last 3 values of `u = B·x` per channel | `[1, 2048, 3]` | 12 KB/layer, **constant** | in-place `copy_` (static address, CUDA-graph friendly) |
| Attention (×6) | `DynamicLayer` ([L113](../third_party/transformers/src/transformers/cache_utils.py#L113)) | all past K and V | `[1, 8, T, 64]` each | 2 KB per token per layer; **12 KB/token total** | `torch.cat([old, new])` **every step** ([L145](../third_party/transformers/src/transformers/cache_utils.py#L145)): allocates a new tensor and copies the whole cache |

Why this matters:
- A 4k-token context uses less than 50 MB of KV cache, so **compressing the KV cache is not a priority** for this model.
- The `torch.cat` growth costs O(T) copying per step and **changes the tensor address every step**, which prevents CUDA-graph capture. A static (preallocated) cache fixes both problems.
- `get_seq_length()` (the number of tokens seen so far) comes from the attention caches and is used to compute the next position id.

## Stage 11: Final norm, lm_head, and choosing the next token

**Code:** [modeling_lfm2.py:520](../third_party/transformers/src/transformers/models/lfm2/modeling_lfm2.py#L520), [modeling_lfm2_vl.py:425-428](../third_party/transformers/src/transformers/models/lfm2_vl/modeling_lfm2_vl.py#L425-L428), [generation/utils.py:3222-3260](../third_party/transformers/src/transformers/generation/utils.py#L3222-L3260).

1. **Final RMSNorm.** Watch out: in the code it is called `embedding_norm`, but it is applied **after the last layer**, not after the embedding. `[1, L, 2048] → [1, L, 2048]`.
2. **lm_head:** Linear(2048 → 65,536, no bias), **sharing its weight with `embed_tokens`** (tied weights, 134M parameters). Output logits `[1, L', 65536]`, where `L'` is controlled by `logits_to_keep`. `generate` only uses the **last** position, `logits[:, -1]` → `[1, 65536]`, upcast to FP32. (Checking whether prefill computes logits for all L positions or only the last is a profiling item.)
3. **Logits processors:** e.g. repetition penalty, temperature, min-p. The model card recommends `temperature=0.1, min_p=0.15, repetition_penalty=1.05`. The checkpoint's `generation_config.json` sets none of these, so **by default `generate` is greedy** (`argmax`). For benchmarks we fix greedy decoding so outputs are deterministic and comparable.
4. **Pick the token:** `argmax` (greedy) or `softmax` + `multinomial` (sampling) → `next_token [1]`.
5. **Stop check:** stop at `<|im_end|>` (id 7) or `max_new_tokens`.

**Performance notes:** the lm_head GEMV is the **single largest weight read in decode** (268 MB per token in BF16). The logits processors are a chain of small kernels on a 65,536-vector, and in eager mode the stop check can force a GPU→CPU sync every step.

## Stage 12: Prefill vs. decode (the generation loop)

**Code:** `GenerationMixin._sample` ([generation/utils.py:3097](../third_party/transformers/src/transformers/generation/utils.py#L3097)).

```
outputs = prefill(input_ids [1, L], pixel_values, ...)    # Stages 1–11 once; fills all caches
loop:
    token = select(outputs.logits[:, -1])                  # Stage 11
    input_ids = cat(input_ids, token)                      # [1, L + n]
    if stop: break
    outputs = forward(token [1, 1], caches)                # Stages 5–11 with L = 1, NO vision encoder
```

| | Prefill | Decode (per token) |
|---|---|---|
| Tokens processed | L (≈250 or ≈2330) | 1 |
| Vision encoder | yes | no |
| Main ops | GEMMs (matrix × matrix) | GEMVs (matrix × vector) |
| Bottleneck type | **compute-bound** | **memory-bandwidth- and launch-bound** |
| Weights read | 3.2 GB (all, once) | ≈ 2.34 GB (language model incl. lm_head, every token) |
| Metric | time to first token (TTFT) | time per output token (TPOT), tokens/s |

**The decode floor:** reading 2.34 GB per token at ≈ 1.34 TB/s (RTX PRO 5000) takes ≈ 1.7 ms, so at most ≈ 575 tokens/s at batch 1. Eager PyTorch launches about 300+ kernels per token from Python. If each launch costs a few µs of CPU time, overhead alone can exceed the floor. **Measuring this gap is the first job of profiling.**

---

## Appendix: complete parameter budget (sums to the checkpoint's 1,596,625,904)

| Component | Count | Each | Total |
|---|---|---|---|
| Vision: patch embedding (768×1152 + bias) | 1 | 885,888 | 885,888 |
| Vision: position table (256×1152) | 1 | 294,912 | 294,912 |
| Vision: encoder layer (4 attention linears + MLP + 2 LayerNorms) | 27 | 15,239,504 | 411,466,608 |
| Vision: post-LayerNorm | 1 | 2,304 | 2,304 |
| Projector (4608×2048 + 2048×2048, with biases) | 1 | 13,635,584 | 13,635,584 |
| LM: token embedding = lm_head (tied, 65,536×2048) | 1 | 134,217,728 | 134,217,728 |
| LM: SwiGLU MLP (3 × 2048×8192) | 16 | 50,331,648 | 805,306,368 |
| LM: RMSNorms (operator + ffn) | 16 | 4,096 | 65,536 |
| LM: short-conv block (6144×2048 + 2048×2048 + 2048×3) | 10 | 16,783,360 | 167,833,600 |
| LM: attention block (2048² ×2 + 512×2048 ×2 + 2×64) | 6 | 10,485,888 | 62,915,328 |
| LM: final norm | 1 | 2,048 | 2,048 |
| **Total** | | | **1,596,625,904** |

## Questions each owner should be able to answer (self-check)

**Part A (Jainil):** For a 1000×300 image, single tile or multi-tile? Give `spatial_shapes` and the number of image tokens. Why are positional embeddings resized instead of looked up? What fraction of encoder FLOPs is wasted on padding in Example 1? Why can't patches from different tiles attend to each other?

**Part B (Gautam):** Write the pixel-unshuffle as index arithmetic. Why does decode only need 3 cached values per channel for the conv? Count the bytes read and written by each eager kernel in one decode-step conv block, and compare with an ideal fused kernel. Why is the MLP width 8192 and not 12288?

**Part C (Bibek):** Show that RoPE makes q·k depend only on relative position. Why does GQA reduce KV-cache size by 4×? Why does `torch.cat` in the KV cache prevent CUDA graphs? Derive the 2.34 GB/token decode figure and the ≈ 1.7 ms floor.
