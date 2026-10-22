# 20261022-090933_L40S_l40_stack_r1

GPU: NVIDIA L40S (sm_89), driver 610.57.04, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 36.4 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 86.4 | – | – | – | – | –/– | 32.08 | – (–) | 288.4 | 1860 | 4.62 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 86.4 | – | – | 3.34 | 299.6 | –/– | 138.60 | 0.839 (0.717) | 266.9 | 1920 | 4.62 |
| fused_gemv_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 29.4 | – | – | – | – | –/– | 14.74 | – (–) | 258.7 | 2055 | 4.39 |
| fused_gemv_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 29.4 | – | – | 3.31 | 302.5 | –/– | 117.83 | 0.812 (0.691) | 257.2 | 2130 | 4.39 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.5 | – | – | – | – | –/– | 5.73 | – (–) | 253.5 | 2520 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.5 | – | – | 3.30 | 303.5 | –/– | 112.57 | 0.841 (0.721) | 257.1 | 2520 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 4.4 | – | – | – | – | –/– | 2.86 | – (–) | 90.8 | 2520 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 4.4 | – | – | 3.29 | 303.8 | –/– | 108.98 | 0.836 (0.716) | 248.9 | 2520 | 4.27 |
