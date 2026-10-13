# 20261013-230615_RTXPRO5000Blackwell_baseline_sdpa_rkv

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 30.5 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 117.8 | 77.1 | 36.4 | – | – | –/– | 32.83 | – (–) | 194.8 | 2227 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 117.8 | 74.1 | 34.5 | 3.95 | 253.4 | 3.96/3.56 | 152.42 | 0.942 (0.821) | 251.7 | 2445 | 3.26 |
| baseline | img_small | 1 | 1 | 1 | 234 | 20.6 | 12.1 | 5.2 | – | – | –/– | 2.88 | – (–) | 50.7 | 2392 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 20.6 | 12.1 | 5.2 | 4.09 | 244.4 | 4.09/3.67 | 122.35 | 0.941 (0.816) | 221.4 | 2535 | 3.02 |
| sdpa_repeat_kv | img_fhd | 1 | 1 | 9 | 2300 | 118.5 | 77.5 | 36.7 | – | – | –/– | 34.49 | – (–) | 221.2 | 2212 | 3.26 |
| sdpa_repeat_kv | img_fhd | 1 | 128 | 9 | 2300 | 118.5 | 74.1 | 34.7 | 4.21 | 237.7 | 4.25/3.83 | 162.35 | 1.007 (0.878) | 253.4 | 2437 | 3.26 |
| sdpa_repeat_kv | img_small | 1 | 1 | 1 | 234 | 20.7 | 12.1 | 5.3 | – | – | –/– | 3.30 | – (–) | 147.8 | 2595 | 3.02 |
| sdpa_repeat_kv | img_small | 1 | 128 | 1 | 234 | 20.7 | 12.1 | 5.3 | 4.24 | 236.1 | 4.23/3.81 | 125.94 | 0.966 (0.836) | 219.3 | 2527 | 3.02 |
