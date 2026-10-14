# 20261014-160027_T4_t4_stack_bf16

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 12.7 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 6446.7 | – | – | – | – | –/– | 432.61 | – (–) | 66.6 | 705 | 6.09 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 6446.7 | – | – | 9.16 | 109.2 | –/– | 507.43 | 0.589 (0.472) | 66.8 | 735 | 6.09 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 659.0 | – | – | – | – | –/– | 45.81 | – (–) | 66.3 | 900 | 4.43 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 659.0 | – | – | 9.27 | 107.8 | –/– | 122.72 | 0.606 (0.487) | 66.5 | 825 | 4.44 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 51.4 | – | – | – | – | –/– | 3.55 | – (–) | 60.5 | 885 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 51.4 | – | – | 9.03 | 110.7 | –/– | 82.75 | 0.624 (0.509) | 67.6 | 1050 | 4.27 |
