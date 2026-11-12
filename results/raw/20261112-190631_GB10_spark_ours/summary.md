# 20261112-190631_GB10_spark_ours

GPU: NVIDIA GB10 (sm_121), driver None, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 5.2 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_gemv_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 247.1 | – | – | – | – | –/– | 14.45 | – (–) | 49.7 | 2158 | 4.61 |
| fused_gemv_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 247.1 | – | – | 10.19 | 98.1 | –/– | 41.95 | 0.217 (0.164) | 27.2 | 2158 | 4.61 |
| fused_gemv_pf_vfused | img_small | 1 | 1 | 1 | 234 | 34.2 | – | – | – | – | –/– | 1.41 | – (–) | 20.9 | 2184 | 4.30 |
| fused_gemv_pf_vfused | img_small | 1 | 128 | 1 | 234 | 34.2 | – | – | 9.89 | 101.1 | –/– | 28.33 | 0.212 (0.161) | 21.8 | 2177 | 4.31 |
| fused_gemv_pf_vfused | text | 1 | 1 | 0 | 0 | 14.6 | – | – | – | – | –/– | 0.31 | – (–) | 10.1 | 2190 | 4.27 |
| fused_gemv_pf_vfused | text | 1 | 128 | 0 | 0 | 14.6 | – | – | 9.88 | 101.3 | –/– | 26.67 | 0.208 (0.156) | 20.9 | 2184 | 4.27 |
| fused_w8all_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 243.0 | – | – | – | – | –/– | 12.83 | – (–) | 51.2 | 2171 | 2.10 |
| fused_w8all_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 243.0 | – | – | 5.26 | 190.0 | –/– | 29.98 | 0.135 (0.108) | 32.8 | 2171 | 2.10 |
| fused_w8all_pf_vfused | img_small | 1 | 1 | 1 | 234 | 30.7 | – | – | – | – | –/– | 1.41 | – (–) | 25.0 | 2177 | 1.81 |
| fused_w8all_pf_vfused | img_small | 1 | 128 | 1 | 234 | 30.7 | – | – | 4.92 | 203.4 | –/– | 17.12 | 0.124 (0.098) | 26.2 | 2177 | 1.82 |
| fused_w8all_pf_vfused | text | 1 | 1 | 0 | 0 | 7.7 | – | – | – | – | –/– | 0.36 | – (–) | 10.9 | 2190 | 1.79 |
| fused_w8all_pf_vfused | text | 1 | 128 | 0 | 0 | 7.7 | – | – | 4.89 | 204.4 | –/– | 15.90 | 0.122 (0.097) | 24.9 | 2177 | 1.79 |
