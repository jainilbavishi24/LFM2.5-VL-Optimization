# 20261014-172130_RTXPRO5000Blackwell_final_stack_r1

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 30.6 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 85.8 | – | – | – | – | –/– | 24.79 | – (–) | 287.9 | 2077 | 4.62 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 85.8 | – | – | 2.11 | 474.9 | –/– | 104.31 | 0.626 (0.562) | 296.2 | 2250 | 4.62 |
| fused_gemv_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 29.7 | – | – | – | – | –/– | 6.86 | – (–) | 280.5 | 2572 | 4.39 |
| fused_gemv_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 29.7 | – | – | 2.09 | 477.7 | –/– | 84.50 | 0.611 (0.547) | 287.5 | 2542 | 4.39 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.3 | – | – | – | – | –/– | 3.32 | – (–) | 275.0 | 2580 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.3 | – | – | 2.08 | 480.9 | –/– | 75.59 | 0.569 (0.506) | 280.3 | 2550 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 2.6 | – | – | – | – | –/– | 0.00 | – (–) | 49.7 | 2400 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 2.6 | – | – | 2.07 | 482.3 | –/– | 71.97 | 0.567 (0.503) | 274.9 | 2557 | 4.27 |
