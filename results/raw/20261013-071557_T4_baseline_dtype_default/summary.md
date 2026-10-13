# 20261013-071557_T4_baseline_dtype_default

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 13.5 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 64 | 1 | 234 | – | 52.8 | 39.2 | – | – | 23.56/21.55 | 119.91 | – (–) | 65.8 | 705 | 3.03 |
| baseline | text | 1 | 64 | 0 | 0 | – | 0.0 | 22.9 | – | – | 23.05/21.07 | 108.25 | – (–) | 65.9 | 1440 | 3.00 |
