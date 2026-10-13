"""SigLIP2 vision attention at the 1080p workload's shape (9 tiles x 1024 patches, 16 heads, head dim 72):
time every SDPA backend, with the padding mask (what HF passes) and without it (all patches valid).

    python profiling/vision_attn_probe.py
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, L, D = 9, 16, 1024, 72
torch.manual_seed(0)
q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
valid = torch.ones(B, L, dtype=torch.bool, device="cuda")
valid[-1, 1008:] = False  # img_fhd: 9200 valid of 9216 patches
masks = {"bool mask": valid[:, None, None, :], "additive mask": torch.where(valid, 0.0, float("-inf")).to(torch.bfloat16)[:, None, None, :],
         "no mask": None}
flops = 4 * B * H * L * L * D

ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=masks["bool mask"]).to(torch.bfloat16)
for mname, m in masks.items():
    for be in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel([be]):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
                for _ in range(3):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=m)
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(20):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=m)
                e.record()
                torch.cuda.synchronize()
            ms = s.elapsed_time(e) / 20
            err = (out.float() - ref.float())[:-1].abs().max().item()  # unpadded tiles
            print(f"{mname:13s} {be.name:20s} {ms * 1e3:8.1f} us  {flops / ms / 1e9:6.1f} TFLOPS  max err (full tiles) {err:.3g}")
        except Exception as ex:  # noqa: BLE001
            print(f"{mname:13s} {be.name:20s} unsupported: {str(ex).splitlines()[0][:90]}")
