# 20261013-071728_T4_baseline_hooks

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 21.8 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 108.1 | 58.9 | 34.8 | – | – | –/– | 8.08 | – (–) | 69.0 | 735 | 3.03 |
| baseline | img_small | 1 | 128 | 1 | 234 | 108.1 | 50.1 | 40.1 | 23.77 | 42.1 | 23.24/21.32 | 220.06 | 1.669 (1.150) | 66.8 | 660 | 3.03 |
