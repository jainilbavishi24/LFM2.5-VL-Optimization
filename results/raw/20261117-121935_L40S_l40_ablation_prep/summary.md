# 20261117-121935_L40S_l40_ablation_prep

GPU: NVIDIA L40S (sm_89), driver 620.70.06, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 47.4 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 83.4 | – | – | – | – | –/– | 31.86 | – (–) | 261.4 | 2010 | 4.61 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 83.4 | – | – | 3.35 | 298.8 | –/– | 126.42 | 0.745 (0.586) | 255.1 | 2025 | 4.61 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.7 | – | – | – | – | –/– | 8.16 | – (–) | 221.7 | 2520 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.7 | – | – | 3.30 | 303.2 | –/– | 106.03 | 0.771 (0.614) | 241.9 | 2520 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 4.3 | – | – | – | – | –/– | 1.93 | – (–) | 93.8 | 2520 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 4.3 | – | – | 3.29 | 303.7 | –/– | 99.50 | 0.768 (0.612) | 234.2 | 2520 | 4.27 |
