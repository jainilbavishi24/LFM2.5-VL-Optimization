# 20261014-173348_RTXPRO5000Blackwell_batch_stack

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 8.3 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.2 | – | – | – | – | –/– | 3.11 | – (–) | 35.4 | 2407 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.2 | – | – | 2.08 | 480.5 | –/– | 68.71 | 0.517 (0.499) | 239.5 | 2527 | 4.31 |
| fused_gemv_pf_vfused | img_small | 2 | 1 | 1 | 234 | 19.7 | – | – | – | – | –/– | 4.53 | – (–) | 253.6 | 2587 | 4.35 |
| fused_gemv_pf_vfused | img_small | 2 | 128 | 1 | 234 | 19.7 | – | – | 2.13 | 938.0 | –/– | 80.66 | 0.300 (0.291) | 276.4 | 2437 | 4.36 |
| fused_gemv_pf_vfused | img_small | 4 | 1 | 1 | 234 | 36.4 | – | – | – | – | –/– | 8.30 | – (–) | 277.4 | 2565 | 4.44 |
| fused_gemv_pf_vfused | img_small | 4 | 128 | 1 | 234 | 36.4 | – | – | 2.18 | 1832.6 | –/– | 93.14 | 0.167 (0.162) | 300.6 | 2400 | 4.46 |
| fused_gemv_pf_vfused | img_small | 8 | 1 | 1 | 234 | 71.9 | – | – | – | – | –/– | 21.50 | – (–) | 295.0 | 2452 | 4.64 |
| fused_gemv_pf_vfused | img_small | 8 | 128 | 1 | 234 | 71.9 | – | – | 2.60 | 3073.4 | –/– | 105.28 | 0.082 (0.080) | 271.7 | 2505 | 4.70 |
| fused_gemv_pf_vfused | img_small | 16 | 1 | 1 | 234 | 155.4 | – | – | – | – | –/– | 46.39 | – (–) | 283.3 | 2107 | 5.03 |
| fused_gemv_pf_vfused | img_small | 16 | 128 | 1 | 234 | 155.4 | – | – | 2.64 | 6058.3 | –/– | 137.38 | 0.045 (0.043) | 287.6 | 2505 | 5.13 |
