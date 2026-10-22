# 20261022-091227_L40S_l40_long

GPU: NVIDIA L40S (sm_89), driver 610.57.04, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 40.9 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 117.9 | – | – | – | – | –/– | 38.99 | – (–) | 194.7 | 2100 | 3.26 |
| baseline | img_fhd | 1 | 1024 | 9 | 2300 | 117.9 | – | – | 9.01 | 110.9 | –/– | 1771.73 | 1.694 (1.325) | 188.9 | 2085 | 3.26 |
| baseline | text | 1 | 1 | 0 | 0 | 21.8 | – | – | – | – | –/– | 3.75 | – (–) | 94.4 | 2520 | 2.99 |
| baseline | text | 1 | 1024 | 0 | 0 | 21.8 | – | – | 9.00 | 111.1 | –/– | 1671.20 | 1.630 (1.262) | 181.4 | 2520 | 3.01 |
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 87.5 | – | – | – | – | –/– | 28.79 | – (–) | 229.2 | 1920 | 4.62 |
| fused_gemv_pf_vfused | img_fhd | 1 | 1024 | 9 | 2300 | 87.5 | – | – | 3.37 | 296.6 | –/– | 930.22 | 0.881 (0.743) | 264.0 | 1875 | 4.66 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 4.2 | – | – | – | – | –/– | 4.63 | – (–) | 97.4 | 2520 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 1024 | 0 | 0 | 4.2 | – | – | 3.32 | 301.5 | –/– | 887.73 | 0.863 (0.727) | 261.3 | 2520 | 4.29 |
