# 20261022-091124_L40S_l40_stack_r2

GPU: NVIDIA L40S (sm_89), driver 610.57.04, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 65.8 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 87.2 | – | – | – | – | –/– | 32.31 | – (–) | 293.3 | 1995 | 4.62 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 87.2 | – | – | 3.34 | 299.7 | –/– | 140.35 | 0.851 (0.631) | 270.1 | 1890 | 4.62 |
| fused_gemv_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 29.7 | – | – | – | – | –/– | 14.47 | – (–) | 229.8 | 2130 | 4.39 |
| fused_gemv_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 29.7 | – | – | 3.30 | 302.6 | –/– | 120.65 | 0.836 (0.618) | 260.4 | 2115 | 4.39 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.5 | – | – | – | – | –/– | 5.09 | – (–) | 246.3 | 2520 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.5 | – | – | 3.30 | 303.5 | –/– | 114.44 | 0.861 (0.644) | 259.5 | 2520 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 4.2 | – | – | – | – | –/– | 2.68 | – (–) | 99.5 | 2520 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 4.2 | – | – | 3.29 | 303.7 | –/– | 111.72 | 0.859 (0.642) | 255.6 | 2520 | 4.27 |
