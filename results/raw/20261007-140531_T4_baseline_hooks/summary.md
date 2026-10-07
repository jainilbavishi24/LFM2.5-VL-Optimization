# 20261007-140531_T4_baseline_hooks

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 13.4 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 111.8 | 61.4 | 35.4 | – | – | –/– | 8.03 | – (–) | 69.8 | 735 | 3.03 |
| baseline | img_small | 1 | 128 | 1 | 234 | 111.8 | 55.6 | 43.4 | 22.14 | 45.2 | 20.31/18.63 | 197.97 | 1.496 (1.198) | 65.2 | 570 | 3.03 |
