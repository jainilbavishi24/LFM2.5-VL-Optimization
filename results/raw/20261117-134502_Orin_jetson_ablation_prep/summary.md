# 20261117-134502_Orin_jetson_ablation_prep

GPU: Orin (sm_87), driver None, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 7.0 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 1394.6 | – | – | – | – | –/– | 43.55 | – (–) | 31.2 | 1173 | 4.61 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 1394.6 | – | – | 26.82 | 37.3 | –/– | 121.62 | 0.615 (0.428) | 25.3 | 1173 | 4.61 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 160.9 | – | – | – | – | –/– | 4.62 | – (–) | 28.9 | 1173 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 160.9 | – | – | 26.23 | 38.1 | –/– | 81.50 | 0.605 (0.423) | 23.3 | 1173 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 43.2 | – | – | – | – | –/– | 0.58 | – (–) | 12.9 | 1173 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 43.2 | – | – | 26.03 | 38.4 | –/– | 76.86 | 0.601 (0.419) | 22.9 | 1173 | 4.27 |
