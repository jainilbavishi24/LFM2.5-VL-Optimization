# 20261022-104510_L40S_l40_3b_stack

GPU: NVIDIA L40S (sm_89), driver 610.57.04, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 63.9 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 124.9 | – | – | – | – | –/– | 45.25 | – (–) | 271.3 | 1830 | 9.00 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 124.9 | – | – | 7.54 | 132.7 | –/– | 291.79 | 1.941 (1.460) | 268.6 | 1920 | 9.00 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 19.3 | – | – | – | – | –/– | 11.40 | – (–) | 250.5 | 2490 | 8.64 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 19.3 | – | – | 7.48 | 133.7 | –/– | 248.96 | 1.871 (1.393) | 254.0 | 2505 | 8.64 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 9.1 | – | – | – | – | –/– | 3.60 | – (–) | 91.5 | 2520 | 8.60 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 9.1 | – | – | 7.48 | 133.7 | –/– | 244.76 | 1.899 (1.421) | 249.6 | 2520 | 8.60 |
