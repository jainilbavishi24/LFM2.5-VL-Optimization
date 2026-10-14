"""O1: static decode runtime + CUDA graph for the LFM2 language model.

Prefill (vision tower, projector, LM over the prompt) still runs through Hugging Face. Decode then runs our
own step function over **static** buffers, so one step can be captured as a CUDA graph and replayed per token:

* KV cache per attention layer: [B, n_kv, max_len, head_dim], written in place at `pos` (index_copy_, no cat);
  attention reads all `max_len` slots with a length mask (shapes never change -> capturable).
* conv state per short-conv layer: [B, hidden, L] (L = 3), shifted in place.
* token / position / step counters live on the GPU; greedy argmax happens inside the graph; generated tokens are
  written into a device buffer, so the host never synchronizes until the end of the request.

The math is the same as transformers' modeling_lfm2.py (eager path); only the cache handling and the attention
kernel (plain matmul-softmax-matmul over the static cache instead of SDPA) differ.
"""

import torch
import torch.nn.functional as F
from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb


class StaticDecoder:
    def __init__(self, net, bucket: int = 256, keep_logits: bool = False, fused: bool = False, attn_splits: int = 16,
                 gemv: str = "cublas", w4_group: int = 128, w4_clip: bool = False, w4_keep_int8: tuple = (),
                 calib: dict | None = None, fused_prefill: bool = False):
        self.net = net
        self.fused_prefill = fused_prefill  # O3b: prompt through our kernels straight into the static buffers
        self.lm = net.model.language_model
        self.lm_head = net.lm_head
        cfg = self.lm.config
        self.cfg = cfg
        self.layers = list(self.lm.layers[: cfg.num_hidden_layers])
        self.is_attn = [t == "full_attention" for t in cfg.layer_types[: cfg.num_hidden_layers]]
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.conv_len = cfg.conv_L_cache
        self.bucket = bucket
        self.keep_logits = keep_logits  # copy each step's raw logits into s["logits"] (numerics checks only)
        self.dtype = next(net.parameters()).dtype
        self.device = next(net.parameters()).device
        self.graphs = {}  # (B, max_len) -> state dict incl. the captured graph
        # Fixed-length generation = HF's min_new_tokens: EOS can never be chosen (its logits get -inf added).
        # The bias lives on the device, so captured graphs read it at replay: generate(stop_at_eos=True) zeroes it.
        eos = net.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.eos = torch.tensor(eos, dtype=torch.long, device=self.device)
        self.eos_bias = torch.full((1, len(eos)), float("-inf"), dtype=self.dtype, device=self.device)
        self.fused = fused
        self.attn_splits = attn_splits  # fused path: flash-decoding splits over positions (0 = one block per head)
        self.w4_group, self.w4_clip, self.w4_keep_int8 = w4_group, w4_clip, set(w4_keep_int8)  # int4 options
        self.calib = calib or {}  # int4 AWQ: {weight key: calibration inputs [M, K]} (collect_calibration)
        self.lin_inv_s = {}
        self.gemv = gemv  # fused path, batch 1: "cublas" (F.linear), "bf16" (our GEMV), "int8" / "int4" (O6: weight-only)
        if fused:
            self._prepare_fused()

    # ---------------------------------------------------------------- buffers + one decode step (graph body)
    def _alloc(self, B: int, max_len: int, max_new: int):
        d, t = self.device, self.dtype
        s = dict(B=B, max_len=max_len, max_new=max_new)
        s["k"] = {i: torch.zeros(B, self.n_kv, max_len, self.head_dim, device=d, dtype=t)
                  for i, a in enumerate(self.is_attn) if a}
        s["v"] = {i: torch.zeros_like(s["k"][i]) for i in s["k"]}
        s["conv"] = {i: torch.zeros(B, self.cfg.hidden_size, self.conv_len, device=d, dtype=t)
                     for i, a in enumerate(self.is_attn) if not a}
        s["tok"] = torch.zeros(B, 1, dtype=torch.long, device=d)        # current input token
        s["pos"] = torch.zeros(1, dtype=torch.long, device=d)           # its position (= cache slot)
        s["step"] = torch.zeros(1, dtype=torch.long, device=d)          # index into the output buffer
        s["out"] = torch.zeros(B, max_new, dtype=torch.long, device=d)  # generated tokens
        if self.keep_logits:
            s["logits"] = torch.zeros(B, self.lm_head.out_features, device=d, dtype=torch.float32)
        s["slots"] = torch.arange(max_len, device=d)
        return s

    @torch.no_grad()
    def _step(self, s):
        B = s["B"]
        pos = s["pos"]
        h = self.lm.embed_tokens(s["tok"])                                       # [B, 1, H]
        cos, sin = self.lm.rotary_emb(h, position_ids=pos.view(1, 1).expand(B, 1))
        # additive mask over the static cache: slots > pos are empty
        mask = torch.where(s["slots"] <= pos, 0.0, float("-inf")).to(torch.float32).view(1, 1, 1, -1)
        for i, layer in enumerate(self.layers):
            residual = h
            x = layer.operator_norm(h)
            if self.is_attn[i]:
                a = layer.self_attn
                q = a.q_layernorm(a.q_proj(x).view(B, 1, self.n_heads, self.head_dim)).transpose(1, 2)
                k = a.k_layernorm(a.k_proj(x).view(B, 1, self.n_kv, self.head_dim)).transpose(1, 2)
                v = a.v_proj(x).view(B, 1, self.n_kv, self.head_dim).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kc, vc = s["k"][i], s["v"][i]
                kc.index_copy_(2, pos, k)
                vc.index_copy_(2, pos, v)
                g = self.n_heads // self.n_kv                                    # GQA: 4 query heads per KV head
                qg = q.view(B, self.n_kv, g, self.head_dim)                      # [B, kv, g, D]
                scores = torch.matmul(qg, kc.transpose(2, 3)).float() * a.scaling + mask   # [B, kv, g, L]
                p = torch.softmax(scores, dim=-1).to(self.dtype)
                o = torch.matmul(p, vc)                                          # [B, kv, g, D]
                o = o.reshape(B, 1, self.n_heads * self.head_dim)
                y = a.out_proj(o)
            else:
                c = layer.conv
                bcx = c.in_proj(x).transpose(-1, -2)                             # [B, 3H, 1]
                b_, c_, xx = bcx.chunk(3, dim=-2)
                bx = b_ * xx                                                     # [B, H, 1]
                st = s["conv"][i]
                st.copy_(torch.cat([st[:, :, 1:], bx.to(st.dtype)], dim=-1))     # shift in the new column
                w = c.conv.weight.squeeze(1)                                     # [H, L]
                conv_out = (st * w).sum(-1, keepdim=True)                        # depthwise causal conv, last output
                if c.conv.bias is not None:
                    conv_out = conv_out + c.conv.bias.view(1, -1, 1)
                y = (c_ * conv_out.to(c_.dtype)).transpose(-1, -2).contiguous()
                y = c.out_proj(y)
            h = y + residual
            h = h + layer.feed_forward(layer.ffn_norm(h))
        h = self.lm.embedding_norm(h)
        logits = self.lm_head(h[:, -1])                                          # [B, V]
        if self.keep_logits:
            s["logits"].copy_(logits)
        if self.eos.numel():
            logits.index_add_(1, self.eos, self.eos_bias.to(logits.dtype).expand(B, -1))
        nxt = torch.argmax(logits, dim=-1)                                       # greedy
        s["out"].index_copy_(1, s["step"], nxt.view(B, 1))
        s["tok"].copy_(nxt.view(B, 1))
        pos.add_(1)
        s["step"].add_(1)

    # ---------------------------------------------------------------- O2-O4: fused-kernel decode step
    def _prepare_fused(self):
        """Load the CUDA extension and build merged weights (q|k|v and w1|w3: one GEMV instead of 3 / 2)."""
        from .kernels import ext

        self.K = ext()
        self.eps = float(self.cfg.norm_eps)
        rot = self.lm.rotary_emb
        self.inv_freq = rot.inv_freq.to(self.device, torch.float32).contiguous()
        self.rope_scaling = float(getattr(rot, "attention_scaling", 1.0))
        self.w_qkv, self.w_13, self.conv_w = {}, {}, {}
        for i, layer in enumerate(self.layers):
            if not self.is_attn[i]:
                self.conv_w[i] = layer.conv.conv.weight.squeeze(1).contiguous()  # [H, 3]
            if self.is_attn[i]:
                a = layer.self_attn
                self.w_qkv[i] = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
            f = layer.feed_forward
            self.w_13[i] = torch.cat([f.w1.weight, f.w3.weight], 0).contiguous()
        self.norm_w = [l.operator_norm.weight for l in self.layers] + [self.lm.embedding_norm.weight]
        # every decode GEMV weight, by key; O6 replaces them by INT8 + per-row scale
        self.lin = {}
        for i, layer in enumerate(self.layers):
            if self.is_attn[i]:
                self.lin[("qkv", i)] = self.w_qkv[i]
                self.lin[("o", i)] = layer.self_attn.out_proj.weight
            else:
                self.lin[("in", i)] = layer.conv.in_proj.weight
                self.lin[("out", i)] = layer.conv.out_proj.weight
            self.lin[("w13", i)] = self.w_13[i]
            self.lin[("w2", i)] = layer.feed_forward.w2.weight
        self.lin[("head", 0)] = self.lm_head.weight
        self.lin_scale = {}
        if self.gemv in ("int8", "int4"):
            from .quant import quantize_int4, quantize_int4_awq, quantize_int8

            self.lin_bf16 = dict(self.lin)  # kept for large batches, where cuBLAS' tensor-core GEMM beats our GEMV
            for key in list(self.lin):
                if self.gemv == "int8" or key[0] in self.w4_keep_int8:     # mixed precision: sensitive ones in INT8
                    self.lin[key], self.lin_scale[key] = quantize_int8(self.lin[key])
                elif key in self.calib:
                    self.lin[key], self.lin_scale[key], self.lin_inv_s[key] = quantize_int4_awq(
                        self.lin[key], self.calib[key].to(self.device), self.w4_group, clip_search=self.w4_clip)
                else:
                    self.lin[key], self.lin_scale[key] = quantize_int4(self.lin[key], self.w4_group, self.w4_clip)
            torch.cuda.empty_cache()

    def _linear(self, x, key, bias=None):
        """Decode GEMV for weight `key`: cuBLAS (F.linear) or our BF16 / INT8 kernel (batch 1 only)."""
        w = self.lin[key]
        B = x.shape[0]
        # our GEMV (measured, bench tag "batch_sweep_final"): BF16 up to batch 4 (cuBLAS' tensor-core GEMM wins
        # from 8 on), INT8 up to 8 (chunks of 8 rows share each weight read), INT4 batch 1 only; else BF16 cuBLAS
        if (self.gemv == "cublas" or B > 8 or (B > 1 and w.dtype == torch.uint8)
                or (B > 4 and w.dtype == torch.bfloat16)):
            if w.dtype != torch.bfloat16:  # quantized decode weights: large batches use the kept BF16 copy
                w = self.lin_bf16[key]
            return F.linear(x, w, bias)
        y = torch.empty(B, w.shape[0], device=x.device, dtype=x.dtype)
        if w.dtype == torch.uint8:
            if key in self.lin_inv_s:  # AWQ: the weight columns were scaled by s, so x is scaled by 1/s
                x = x * self.lin_inv_s[key].to(x.dtype)
            self.K.gemv_w4(w, self.lin_scale[key], x, y)
        else:
            self.K.gemv(w, self.lin_scale.get(key), x, y)
        return y if bias is None else y + bias

    @torch.no_grad()
    def _step_fused(self, s):
        K, B, pos, H = self.K, s["B"], s["pos"], self.cfg.hidden_size
        h = F.embedding(s["tok"].view(B), self.lm.embed_tokens.weight)          # [B, H] residual stream
        x = torch.empty_like(h)
        K.add_rmsnorm(h, None, self.norm_w[0], x, self.eps)
        for i, layer in enumerate(self.layers):
            if self.is_attn[i]:
                a = layer.self_attn
                qkv = self._linear(x, ("qkv", i))                                 # [B, (32+8+8)*64]
                q = torch.empty(B, self.n_heads, self.head_dim, device=h.device, dtype=h.dtype)
                K.qk_norm_rope_kv(qkv, a.q_layernorm.weight, a.k_layernorm.weight, self.inv_freq,
                                  self.rope_scaling, pos, q, s["k"][i], s["v"][i], self.eps)
                o = torch.empty_like(q)
                if self.attn_splits:
                    ws = torch.empty(B, self.n_heads, self.attn_splits, 66, device=h.device, dtype=torch.float32)
                    K.decode_attention_split(q, s["k"][i], s["v"][i], pos, o, ws, a.scaling)
                else:
                    K.decode_attention(q, s["k"][i], s["v"][i], pos, o, a.scaling)
                y = self._linear(o.view(B, -1), ("o", i))
            else:
                c = layer.conv
                bcx = self._linear(x, ("in", i), c.in_proj.bias)                  # [B, 3H]
                cy = torch.empty(B, H, device=h.device, dtype=h.dtype)
                K.shortconv_step(bcx, s["conv"][i], self.conv_w[i], c.conv.bias, cy)
                y = self._linear(cy, ("out", i), c.out_proj.bias)
            K.add_rmsnorm(h, y, layer.ffn_norm.weight, x, self.eps)               # h += mixer; x = ffn_norm(h)
            g = self._linear(x, ("w13", i))                                       # [B, 2I]
            act = torch.empty(B, g.shape[-1] // 2, device=h.device, dtype=h.dtype)
            K.silu_mul(g, act)
            y = self._linear(act, ("w2", i))
            K.add_rmsnorm(h, y, self.norm_w[i + 1], x, self.eps)                  # h += mlp; x = next norm(h)
        logits = self._linear(x, ("head", 0))                                     # x = embedding_norm(h)
        if self.keep_logits:
            s["logits"].copy_(logits)
        if self.eos.numel():
            logits.index_add_(1, self.eos, self.eos_bias.to(logits.dtype).expand(B, -1))
        nxt = torch.argmax(logits, dim=-1)
        s["out"].index_copy_(1, s["step"], nxt.view(B, 1))
        s["tok"].copy_(nxt.view(B, 1))
        pos.add_(1)
        s["step"].add_(1)

    def _get(self, B: int, need_len: int, max_new: int):
        max_len = -(-need_len // self.bucket) * self.bucket
        key = (B, max_len)
        if key not in self.graphs or self.graphs[key]["max_new"] < max_new:
            s = self._alloc(B, max_len, max(max_new, 1))
            step = self._step_fused if self.fused else self._step
            # warm up on a side stream (cuBLAS handles, allocator), then capture one step
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    s["pos"].zero_(); s["step"].zero_()
                    step(s)
            torch.cuda.current_stream().wait_stream(side)
            s["pos"].zero_(); s["step"].zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step(s)
            s["graph"] = g
            self.graphs[key] = s
        return self.graphs[key]

    # ---------------------------------------------------------------- prefill through our kernels (O3b)
    def _embed(self, input_ids, pixel_values=None, spatial_shapes=None, pixel_attention_mask=None, **_):
        """Token embeddings with the image features scattered in (= Lfm2VlModel.forward before the LM)."""
        m = self.net.model
        emb = m.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            feats = m.get_image_features(pixel_values=pixel_values, spatial_shapes=spatial_shapes,
                                         pixel_attention_mask=pixel_attention_mask, return_dict=True).pooler_output
            feats = torch.cat(feats, dim=0).to(emb.dtype)
            emb = emb.masked_scatter((input_ids == m.config.image_token_id).unsqueeze(-1), feats)
        return emb

    def _wbf16(self, key):
        return self.lin_bf16[key] if key in self.lin_scale else self.lin[key]

    @torch.no_grad()
    def _prefill_fused(self, input_ids, s, **model_inputs):
        """All prompt tokens at once: cuBLAS GEMMs on the merged BF16 weights + our fused kernels, k/v and conv state
        written straight into the static buffers. Returns the last position's logits [B, V]."""
        K, H = self.K, self.cfg.hidden_size
        B, T = input_ids.shape
        h = self._embed(input_ids, **model_inputs).reshape(B * T, H).contiguous()   # residual stream
        x = torch.empty_like(h)
        K.add_rmsnorm(h, None, self.norm_w[0], x, self.eps)
        for i, layer in enumerate(self.layers):
            if self.is_attn[i]:
                a = layer.self_attn
                qkv = F.linear(x, self._wbf16(("qkv", i)))
                q = torch.empty(B, self.n_heads, T, self.head_dim, device=h.device, dtype=h.dtype)
                K.qk_norm_rope_kv_prefill(qkv, a.q_layernorm.weight, a.k_layernorm.weight, self.inv_freq,
                                          self.rope_scaling, q, s["k"][i], s["v"][i], self.eps)
                o = F.scaled_dot_product_attention(q, s["k"][i][:, :, :T], s["v"][i][:, :, :T], is_causal=True,
                                                   scale=a.scaling, enable_gqa=True)
                y = F.linear(o.transpose(1, 2).reshape(B * T, -1), self._wbf16(("o", i)))
            else:
                c = layer.conv
                bcx = F.linear(x, self._wbf16(("in", i)), c.in_proj.bias)
                cy = torch.empty(B * T, H, device=h.device, dtype=h.dtype)
                K.shortconv_prefill(bcx, s["conv"][i], self.conv_w[i], c.conv.bias, cy, T)
                y = F.linear(cy, self._wbf16(("out", i)), c.out_proj.bias)
            K.add_rmsnorm(h, y, layer.ffn_norm.weight, x, self.eps)
            g = F.linear(x, self._wbf16(("w13", i)))
            act = torch.empty(B * T, g.shape[-1] // 2, device=h.device, dtype=h.dtype)
            K.silu_mul(g, act)
            K.add_rmsnorm(h, F.linear(act, self._wbf16(("w2", i))), self.norm_w[i + 1], x, self.eps)
        return F.linear(x.view(B, T, H)[:, -1], self._wbf16(("head", 0)))                # x = embedding_norm(h)

    # ---------------------------------------------------------------- request
    @torch.no_grad()
    def prefill(self, input_ids, max_new_tokens: int, attention_mask=None, **model_inputs):
        """HF prefill, then load its KV / conv state into our static buffers (or, with fused_prefill, our own prefill
        straight into them). Returns (first token [B], state)."""
        B, T = input_ids.shape
        if self.fused and self.fused_prefill and (attention_mask is None or bool(attention_mask.all())):
            s = self._get(B, T + max(max_new_tokens, 1), max(max_new_tokens - 1, 1))
            logits = self._prefill_fused(input_ids, s, **model_inputs)
            if self.keep_logits:
                self.last_prefill_logits = logits.float()
            if self.eos.numel():
                logits = logits.index_add(1, self.eos, self.eos_bias.to(logits.dtype).expand(B, -1))
            first = logits.argmax(-1)
            s["tok"].copy_(first.view(B, 1))
            s["pos"].fill_(T)
            s["step"].zero_()
            return first, (s if max_new_tokens > 1 else None)
        out = self.net(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1,
                       **model_inputs)
        logits = out.logits[:, -1]
        if self.eos.numel():
            logits = logits.index_add(1, self.eos, self.eos_bias.to(logits.dtype).expand(B, -1))
        first = logits.argmax(-1)                                                # token 1 (from prefill)
        if max_new_tokens <= 1:
            return first, None
        s = self._get(B, T + max_new_tokens, max_new_tokens - 1)
        cache = out.past_key_values
        for i in range(len(self.layers)):
            layer = cache.layers[i]
            if self.is_attn[i]:
                s["k"][i][:, :, :T].copy_(layer.keys); s["k"][i][:, :, T:].zero_()
                s["v"][i][:, :, :T].copy_(layer.values); s["v"][i][:, :, T:].zero_()
            else:
                cs = layer.conv_states[0]
                s["conv"][i].copy_(cs[..., -self.conv_len:])
        s["tok"].copy_(first.view(B, 1))
        s["pos"].fill_(T)
        s["step"].zero_()
        return first, s

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int, attention_mask=None, stop_at_eos: bool = False,
                 check_every: int = 16, **model_inputs):
        """Greedy. Default: fixed length (EOS masked) like the benchmark. stop_at_eos=True: EOS allowed, and the replay
        loop stops once every row has produced one (checked after 1, 2, 4, ... steps, then every `check_every`: each
        check is a host sync; tokens after a row's EOS are junk, cut them like HF does). Returns [B, prompt + new] ids."""
        B = input_ids.shape[0]
        self.eos_bias.fill_(0.0 if stop_at_eos else float("-inf"))
        first, s = self.prefill(input_ids, max_new_tokens, attention_mask=attention_mask, **model_inputs)
        if s is None:
            return torch.cat([input_ids, first.view(B, 1)], dim=1)
        n = max_new_tokens - 1
        if stop_at_eos and bool(torch.isin(first, self.eos).all()):
            n = 0
        done, chunk = 0, 1
        while done < n:
            k = min(chunk, n - done) if stop_at_eos else n
            chunk = min(2 * chunk, check_every)
            for _ in range(k):
                s["graph"].replay()
            done += k
            if stop_at_eos and done < n and bool(
                    (torch.isin(first, self.eos) | torch.isin(s["out"][:, :done], self.eos).any(1)).all()):
                break
        new = torch.cat([first.view(B, 1), s["out"][:, :done]], dim=1)
        return torch.cat([input_ids, new], dim=1)


@torch.no_grad()
def collect_calibration(net, batches, max_rows: int = 2048, seed: int = 0):
    """Inputs of every decode projection (keys as in StaticDecoder.lin) while HF generates on `batches`
    (list of (model inputs, n_new)). Returns {key: [<= max_rows, K] BF16 on CPU} for AWQ."""
    lm = net.model.language_model
    cfg = lm.config
    rows = {}
    hooks = []

    def grab(key):
        def hook(mod, args):
            rows.setdefault(key, []).append(args[0].detach().reshape(-1, args[0].shape[-1]).to("cpu", torch.bfloat16))
        return hook

    for i, layer in enumerate(lm.layers[: cfg.num_hidden_layers]):
        if cfg.layer_types[i] == "full_attention":
            hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(grab(("qkv", i))))
            hooks.append(layer.self_attn.out_proj.register_forward_pre_hook(grab(("o", i))))
        else:
            hooks.append(layer.conv.in_proj.register_forward_pre_hook(grab(("in", i))))
            hooks.append(layer.conv.out_proj.register_forward_pre_hook(grab(("out", i))))
        hooks.append(layer.feed_forward.w1.register_forward_pre_hook(grab(("w13", i))))
        hooks.append(layer.feed_forward.w2.register_forward_pre_hook(grab(("w2", i))))
    hooks.append(net.lm_head.register_forward_pre_hook(grab(("head", 0))))
    try:
        for inputs, n_new in batches:
            net.generate(**inputs, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False)
    finally:
        for h in hooks:
            h.remove()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for key, chunks in rows.items():
        x = torch.cat(chunks, 0)
        if x.shape[0] > max_rows:
            x = x[torch.randperm(x.shape[0], generator=g)[:max_rows]]
        out[key] = x
    return out
