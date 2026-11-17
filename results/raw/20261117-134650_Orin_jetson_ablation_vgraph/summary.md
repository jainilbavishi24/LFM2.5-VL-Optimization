# 20261117-134650_Orin_jetson_ablation_vgraph

GPU: Orin (sm_87), driver None, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 6.9 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 1393.1 | – | – | – | – | –/– | 43.60 | – (–) | 31.3 | 1173 | 4.67 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 1393.1 | – | – | 26.88 | 37.2 | –/– | 121.76 | 0.615 (0.429) | 25.4 | 1173 | 4.67 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 159.7 | – | – | – | – | –/– | 4.66 | – (–) | 29.2 | 1173 | 4.32 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 159.7 | – | – | 26.25 | 38.1 | –/– | 81.59 | 0.606 (0.424) | 23.3 | 1173 | 4.33 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 43.3 | – | – | – | – | –/– | 0.58 | – (–) | 13.0 | 1173 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 43.3 | – | – | 25.95 | 38.5 | –/– | 76.82 | 0.600 (0.420) | 23.0 | 1173 | 4.27 |
