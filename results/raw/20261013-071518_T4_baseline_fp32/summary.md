# 20261013-071518_T4_baseline_fp32

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 13.8 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 64 | 1 | 234 | – | 305.2 | 189.8 | – | – | 31.95/27.67 | 170.73 | – (–) | 67.6 | 630 | 6.02 |
| baseline | text | 1 | 64 | 0 | 0 | – | 0.0 | 27.5 | – | – | 22.03/19.36 | 94.66 | – (–) | 67.6 | 1155 | 5.97 |
