# 20261014-172344_RTXPRO5000Blackwell_final_w8_stack

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 30.5 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_w8_pf_vfused | img_fhd | 1 | 1 | 9 | 2300 | 87.0 | – | – | – | – | –/– | 24.81 | – (–) | 294.2 | 2062 | 5.71 |
| fused_w8_pf_vfused | img_fhd | 1 | 128 | 9 | 2300 | 87.0 | – | – | 1.19 | 843.4 | –/– | 69.92 | 0.355 (0.319) | 300.3 | 2047 | 5.71 |
| fused_w8_pf_vfused | img_hd | 1 | 1 | 3 | 764 | 30.5 | – | – | – | – | –/– | 6.41 | – (–) | 300.4 | 2377 | 5.48 |
| fused_w8_pf_vfused | img_hd | 1 | 128 | 3 | 764 | 30.5 | – | – | 1.15 | 868.4 | –/– | 51.96 | 0.359 (0.324) | 300.3 | 2377 | 5.48 |
| fused_w8_pf_vfused | img_small | 1 | 1 | 1 | 234 | 12.4 | – | – | – | – | –/– | 1.52 | – (–) | 294.7 | 2542 | 5.40 |
| fused_w8_pf_vfused | img_small | 1 | 128 | 1 | 234 | 12.4 | – | – | 1.13 | 882.5 | –/– | 45.06 | 0.343 (0.308) | 294.6 | 2452 | 5.40 |
| fused_w8_pf_vfused | text | 1 | 1 | 0 | 0 | 2.6 | – | – | – | – | –/– | 0.00 | – (–) | 59.1 | 2392 | 5.36 |
| fused_w8_pf_vfused | text | 1 | 128 | 0 | 0 | 2.6 | – | – | 1.12 | 894.8 | –/– | 42.73 | 0.336 (0.302) | 252.2 | 2392 | 5.36 |
