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
    def __init__(self, net, bucket: int = 256, keep_logits: bool = False, fused: bool = False):
        self.net = net
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
        # Fixed-length generation = HF's min_new_tokens: EOS can never be chosen (its logits are masked)
        eos = net.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.eos = torch.tensor(eos, dtype=torch.long, device=self.device)
        self.fused = fused
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
            logits.index_fill_(1, self.eos, float("-inf"))
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

    @torch.no_grad()
    def _step_fused(self, s):
        K, B, pos, H = self.K, s["B"], s["pos"], self.cfg.hidden_size
        h = F.embedding(s["tok"].view(B), self.lm.embed_tokens.weight)          # [B, H] residual stream
        x = torch.empty_like(h)
        K.add_rmsnorm(h, None, self.norm_w[0], x, self.eps)
        for i, layer in enumerate(self.layers):
            if self.is_attn[i]:
                a = layer.self_attn
                qkv = F.linear(x, self.w_qkv[i])                                  # [B, (32+8+8)*64]
                q = torch.empty(B, self.n_heads, self.head_dim, device=h.device, dtype=h.dtype)
                K.qk_norm_rope_kv(qkv, a.q_layernorm.weight, a.k_layernorm.weight, self.inv_freq,
                                  self.rope_scaling, pos, q, s["k"][i], s["v"][i], self.eps)
                o = torch.empty_like(q)
                K.decode_attention(q, s["k"][i], s["v"][i], pos, o, a.scaling)
                y = F.linear(o.view(B, -1), a.out_proj.weight)
            else:
                c = layer.conv
                bcx = F.linear(x, c.in_proj.weight, c.in_proj.bias)               # [B, 3H]
                cy = torch.empty(B, H, device=h.device, dtype=h.dtype)
                K.shortconv_step(bcx, s["conv"][i], self.conv_w[i], c.conv.bias, cy)
                y = F.linear(cy, c.out_proj.weight, c.out_proj.bias)
            K.add_rmsnorm(h, y, layer.ffn_norm.weight, x, self.eps)               # h += mixer; x = ffn_norm(h)
            g = F.linear(x, self.w_13[i])                                         # [B, 2I]
            act = torch.empty(B, g.shape[-1] // 2, device=h.device, dtype=h.dtype)
            K.silu_mul(g, act)
            y = F.linear(act, layer.feed_forward.w2.weight)
            K.add_rmsnorm(h, y, self.norm_w[i + 1], x, self.eps)                  # h += mlp; x = next norm(h)
        logits = F.linear(x, self.lm_head.weight)                                 # x = embedding_norm(h)
        if self.keep_logits:
            s["logits"].copy_(logits)
        if self.eos.numel():
            logits.index_fill_(1, self.eos, float("-inf"))
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

    # ---------------------------------------------------------------- request
    @torch.no_grad()
    def prefill(self, input_ids, max_new_tokens: int, attention_mask=None, **model_inputs):
        """HF prefill, then load its KV / conv state into our static buffers. Returns (first token [B], state)."""
        B, T = input_ids.shape
        out = self.net(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1,
                       **model_inputs)
        logits = out.logits[:, -1]
        if self.eos.numel():
            logits = logits.index_fill(1, self.eos, float("-inf"))
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
    def generate(self, input_ids, max_new_tokens: int, attention_mask=None, **model_inputs):
        """Greedy, fixed length (EOS ignored) like the benchmark. Returns [B, prompt + new] token ids."""
        B = input_ids.shape[0]
        first, s = self.prefill(input_ids, max_new_tokens, attention_mask=attention_mask, **model_inputs)
        if s is None:
            return torch.cat([input_ids, first.view(B, 1)], dim=1)
        for _ in range(max_new_tokens - 1):
            s["graph"].replay()
        new = torch.cat([first.view(B, 1), s["out"][:, : max_new_tokens - 1]], dim=1)
        return torch.cat([input_ids, new], dim=1)
