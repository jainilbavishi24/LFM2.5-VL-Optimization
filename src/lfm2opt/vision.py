"""O7d: fused SigLIP2 vision encoder (BF16, same math as HF up to rounding).

Per encoder layer HF runs: LayerNorm, q/k/v GEMMs, masked SDPA (padding mask -> cuDNN / mem-efficient kernel),
out GEMM, residual add, LayerNorm, fc1 GEMM, GELU, fc2 GEMM, residual add. Here:
  * q|k|v merged into one GEMM (weights concatenated once);
  * padding-free attention: tiles are grouped into runs of equal valid length and each run calls the flash kernel
    *without* a mask on just its valid patches (a 1080p image = 8 full tiles + a 1008-patch thumbnail; padded
    queries/keys are never computed, padded rows are zeroed and dropped by the projector as in HF);
  * GELU (tanh) fused into the fc1 GEMM epilogue (cuBLASLt GELU_BIAS via torch._addmm_activation): the 9216 x 4304
    activation is written once instead of written, read and written again;
  * residual add + next LayerNorm fused into one kernel (kernels/vision_ops.cu, add_layernorm).
  * int8=True (extension, changes the weights): every encoder GEMM on INT8 weights through kernels/int8_gemm.cu,
    W8A8 on INT8 tensor cores except the kinds in LFM2_INT8_A16 (default fc2), W8A16 there; GELU stays in the fc1
    epilogue. The BF16 copies are not kept (drop_bf16 also frees the HF modules' weights).

    from lfm2opt.vision import patch_vision_tower; patch_vision_tower(net)
"""

import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .kernels import vision_ext


class FusedSiglip2Encoder:
    def __init__(self, vt, attn: str = "flash_varlen", gelu_epilogue: bool = True, fuse_ln: bool = True,
                 int8: bool = False, drop_bf16: bool = False):
        """attn: "flash_varlen" (default) / "cudnn_varlen": padding-free runs on the flash / cuDNN kernel without a mask;
        "cudnn_mask": HF semantics (padding mask), for ablation."""
        self.vt = vt
        self.attn, self.gelu_epilogue, self.fuse_ln = attn, gelu_epilogue, fuse_ln
        cfg = vt.config
        self.H = cfg.num_attention_heads
        self.Dh = cfg.hidden_size // self.H
        self.scale = self.Dh**-0.5
        self.layers = []
        for layer in vt.encoder.layers:
            a = layer.self_attn
            self.layers.append(dict(
                wqkv=torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).contiguous(),
                bqkv=torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).contiguous(),
                wo=a.out_proj.weight, bo=a.out_proj.bias,
                ln1=layer.layer_norm1, ln2=layer.layer_norm2,
                w1=layer.mlp.fc1.weight, b1=layer.mlp.fc1.bias, w2=layer.mlp.fc2.weight, b2=layer.mlp.fc2.bias,
            ))
        self.ext = vision_ext() if fuse_ln else None
        self.int8 = int8
        if int8:
            from .int8 import Int8Linear, a16_kinds

            a16 = a16_kinds()
            for p in self.layers:
                for key, w, b, kind in (("qkv", "wqkv", "bqkv", "vqkv"), ("o", "wo", "bo", "vout"),
                                        ("fc1", "w1", "b1", "fc1"), ("fc2", "w2", "b2", "fc2")):
                    p[key] = Int8Linear(p.pop(w), p[b], a8=kind not in a16)
            if drop_bf16:
                for layer in vt.encoder.layers:
                    a, m = layer.self_attn, layer.mlp
                    for lin in (a.q_proj, a.k_proj, a.v_proj, a.out_proj, m.fc1, m.fc2):
                        lin.weight = torch.nn.Parameter(lin.weight.new_empty(0), requires_grad=False)
            torch.cuda.empty_cache()

    def _add_ln(self, h, y, ln, x):
        """h += y (y may be None); x = ln(h)."""
        if self.fuse_ln:
            self.ext.add_layernorm(h, y, ln.weight, ln.bias, x, ln.eps)
        else:
            if y is not None:
                h.add_(y)
            x.copy_(F.layer_norm(h, ln.normalized_shape, ln.weight, ln.bias, ln.eps))

    def _attention(self, qkv, runs, T, L):
        qkv5 = qkv.view(T, L, 3, self.H, self.Dh)
        if self.attn == "cudnn_mask":  # HF semantics (padding mask), cuDNN kernel: for ablation
            q, k, v = (qkv5[:, :, j].transpose(1, 2) for j in range(3))
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask, scale=self.scale)
            return o.transpose(1, 2).reshape(T, L, -1)
        # flash needs sm_80+: older GPUs run the same padding-free runs on the memory-efficient kernel, or (BF16 on a
        # T4, which neither supports) on the math path
        backend = [SDPBackend.CUDNN_ATTENTION if self.attn == "cudnn_varlen" else SDPBackend.FLASH_ATTENTION,
                   SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        if len(runs) == 1 and runs[0][2] == L:  # every tile full: no copy of the output
            q, k, v = (qkv5[:, :, j].transpose(1, 2) for j in range(3))
            with sdpa_kernel(backend):
                o = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            return o.transpose(1, 2).reshape(T, L, -1)
        out = torch.empty(T, L, self.H, self.Dh, dtype=qkv.dtype, device=qkv.device)
        for a, b, n in runs:
            q, k, v = (qkv5[a:b, :n, j].transpose(1, 2) for j in range(3))
            with sdpa_kernel(backend):
                o = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out[a:b, :n] = o.transpose(1, 2)
            if n < L:
                out[a:b, n:] = 0
        return out.view(T, L, -1)

    @torch.no_grad()
    def __call__(self, pixel_values, pixel_attention_mask, spatial_shapes, **kwargs):
        vt = self.vt
        h = vt.embeddings(pixel_values, spatial_shapes).contiguous()
        T, L, D = h.shape
        lens = pixel_attention_mask.sum(1).tolist()  # valid patches per tile (padding is at the end)
        runs, a = [], 0
        for t in range(1, T + 1):
            if t == T or lens[t] != lens[a]:
                runs.append((a, t, int(lens[a])))
                a = t
        if self.attn == "cudnn_mask":
            valid = pixel_attention_mask.bool()
            self._mask = valid[:, None, None, :]
        x = torch.empty_like(h)
        self._add_ln(h, None, self.layers[0]["ln1"], x)
        n_layers = len(self.layers)
        for i, p in enumerate(self.layers):
            if self.int8:
                qkv = p["qkv"](x)
                y = p["o"](self._attention(qkv, runs, T, L))
            else:
                qkv = F.linear(x, p["wqkv"], p["bqkv"])
                y = F.linear(self._attention(qkv, runs, T, L), p["wo"], p["bo"])
            self._add_ln(h, y, p["ln2"], x)
            if self.int8:
                f = p["fc1"](x.view(-1, D), gelu=True)
            elif self.gelu_epilogue:
                f = torch._addmm_activation(p["b1"], x.view(-1, D), p["w1"].t(), use_gelu=True)
            else:
                f = F.gelu(F.linear(x.view(-1, D), p["w1"], p["b1"]), approximate="tanh")
            y = (p["fc2"](f) if self.int8 else F.linear(f, p["w2"], p["b2"])).view(T, L, D)
            if i + 1 < n_layers:
                self._add_ln(h, y, self.layers[i + 1]["ln1"], x)
            else:
                h.add_(y)
        return BaseModelOutputWithPooling(last_hidden_state=vt.post_layernorm(h), pooler_output=None)


def patch_vision_tower(net, **opts):
    """Replace the vision tower's forward with FusedSiglip2Encoder. Options also from LFM2_VISION, e.g.
    LFM2_VISION="attn=cudnn_mask,gelu_epilogue=0,fuse_ln=0" for ablations."""
    for kv in filter(None, os.environ.get("LFM2_VISION", "").split(",")):
        k, v = kv.split("=")
        opts[k] = v if k == "attn" else v not in ("0", "false", "no")
    vt = net.model.vision_tower
    enc = FusedSiglip2Encoder(vt, **opts)
    vt.forward = enc
    vt._fused_encoder = enc
    return net
