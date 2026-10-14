# 20261014-172320_RTXPRO5000Blackwell_final_stack_r3

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 35.7 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 87.4 | – | – | – | – | –/– | 24.74 | – (–) | 294.8 | 2040 | 4.62 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 87.4 | – | – | 2.12 | 470.8 | –/– | 106.24 | 0.642 (0.566) | 300.0 | 2355 | 4.62 |
| fused_gemv_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 30.3 | – | – | – | – | –/– | 7.39 | – (–) | 290.0 | 2565 | 4.39 |
| fused_gemv_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 30.3 | – | – | 2.09 | 477.8 | –/– | 91.69 | 0.664 (0.589) | 298.1 | 2212 | 4.39 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.4 | – | – | – | – | –/– | 3.05 | – (–) | 290.1 | 2550 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.4 | – | – | 2.08 | 480.7 | –/– | 79.89 | 0.605 (0.531) | 293.5 | 2565 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 2.6 | – | – | – | – | –/– | 0.00 | – (–) | 57.7 | 2407 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 2.6 | – | – | 2.07 | 482.2 | –/– | 76.59 | 0.603 (0.529) | 288.1 | 2572 | 4.27 |
