# 20261014-172225_RTXPRO5000Blackwell_final_stack_r2

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 31.5 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 87.0 | – | – | – | – | –/– | 25.07 | – (–) | 293.2 | 2055 | 4.62 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 87.0 | – | – | 2.12 | 472.7 | –/– | 105.17 | 0.631 (0.564) | 300.1 | 2407 | 4.62 |
| fused_gemv_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 30.6 | – | – | – | – | –/– | 9.70 | – (–) | 285.4 | 2565 | 4.39 |
| fused_gemv_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 30.6 | – | – | 2.09 | 478.5 | –/– | 88.15 | 0.618 (0.552) | 295.8 | 2190 | 4.39 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.4 | – | – | – | – | –/– | 3.55 | – (–) | 285.9 | 2550 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.4 | – | – | 2.08 | 480.8 | –/– | 79.14 | 0.595 (0.530) | 289.9 | 2565 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 2.7 | – | – | – | – | –/– | 0.80 | – (–) | 57.1 | 2407 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 2.7 | – | – | 2.07 | 482.3 | –/– | 74.72 | 0.582 (0.517) | 285.2 | 2580 | 4.27 |
