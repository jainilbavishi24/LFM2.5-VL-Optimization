# 20261013-071626_T4_baseline_nohooks

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 16.2 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 109.5 | – | – | – | – | –/– | 8.58 | – (–) | 56.7 | 645 | 3.03 |
| baseline | img_small | 1 | 128 | 1 | 234 | 109.5 | – | – | 23.82 | 42.0 | –/– | 216.29 | 1.636 (1.249) | 65.5 | 540 | 3.03 |
