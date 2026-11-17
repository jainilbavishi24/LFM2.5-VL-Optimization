# 20261117-121956_L40S_l40_ablation_vgraph

GPU: NVIDIA L40S (sm_89), driver 620.70.06, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 34.7 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 83.1 | – | – | – | – | –/– | 32.13 | – (–) | 266.6 | 2025 | 4.67 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 83.1 | – | – | 3.35 | 298.8 | –/– | 130.70 | 0.776 (0.660) | 249.6 | 1965 | 4.67 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.5 | – | – | – | – | –/– | 3.67 | – (–) | 231.7 | 2520 | 4.32 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.5 | – | – | 3.30 | 303.2 | –/– | 105.69 | 0.803 (0.689) | 241.9 | 2520 | 4.33 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 4.2 | – | – | – | – | –/– | 2.16 | – (–) | 91.0 | 2520 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 4.2 | – | – | 3.29 | 303.6 | –/– | 99.42 | 0.766 (0.652) | 238.0 | 2520 | 4.27 |
