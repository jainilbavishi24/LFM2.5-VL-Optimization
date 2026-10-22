# 20261022-091159_L40S_l40_w8_stack

GPU: NVIDIA L40S (sm_89), driver 610.57.04, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 66.9 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_w8_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 87.1 | – | – | – | – | –/– | 32.02 | – (–) | 307.7 | 1860 | 5.71 |
| fused_w8_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 87.1 | – | – | 1.82 | 548.1 | –/– | 94.26 | 0.490 (0.368) | 288.4 | 1845 | 5.71 |
| fused_w8_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 30.0 | – | – | – | – | –/– | 11.67 | – (–) | 277.3 | 2100 | 5.48 |
| fused_w8_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 30.0 | – | – | 1.79 | 558.5 | –/– | 74.85 | 0.497 (0.378) | 282.3 | 2115 | 5.48 |
| fused_w8_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.5 | – | – | – | – | –/– | 7.34 | – (–) | 266.7 | 2520 | 5.40 |
| fused_w8_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.5 | – | – | 1.78 | 562.6 | –/– | 70.00 | 0.493 (0.374) | 268.7 | 2505 | 5.40 |
| fused_w8_pf_vfused | text | 1 | 1 | 0 | 0 | 4.2 | – | – | – | – | –/– | 2.91 | – (–) | 120.2 | 2520 | 5.36 |
| fused_w8_pf_vfused | text | 1 | 128 | 0 | 0 | 4.2 | – | – | 1.77 | 565.0 | –/– | 65.83 | 0.495 (0.377) | 274.1 | 2520 | 5.36 |
