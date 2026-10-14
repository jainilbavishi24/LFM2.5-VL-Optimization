# 20261014-160247_T4_t4_w8_stack

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 13.2 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_w8_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 6469.7 | – | – | – | – | –/– | 434.73 | – (–) | 66.6 | 705 | 7.18 |
| fused_w8_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 6469.7 | – | – | 4.93 | 202.9 | –/– | 472.93 | 0.301 (0.236) | 66.4 | 750 | 7.18 |
| fused_w8_pf_vfused | img_small | 1 | 1 | 1 | 234 | 662.2 | – | – | – | – | –/– | 45.60 | – (–) | 66.3 | 885 | 5.53 |
| fused_w8_pf_vfused | img_small | 1 | 128 | 1 | 234 | 662.2 | – | – | 5.25 | 190.6 | –/– | 88.60 | 0.339 (0.270) | 66.6 | 870 | 5.53 |
| fused_w8_pf_vfused | text | 1 | 1 | 0 | 0 | 51.2 | – | – | – | – | –/– | 4.08 | – (–) | 62.7 | 1185 | 5.36 |
| fused_w8_pf_vfused | text | 1 | 128 | 0 | 0 | 51.2 | – | – | 5.09 | 196.5 | –/– | 47.38 | 0.341 (0.274) | 68.0 | 960 | 5.36 |
