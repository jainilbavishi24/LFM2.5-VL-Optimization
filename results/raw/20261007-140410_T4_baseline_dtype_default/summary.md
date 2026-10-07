# 20261007-140410_T4_baseline_dtype_default

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 22.3 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 64 | 1 | 234 | – | 56.3 | 40.7 | – | – | 26.56/24.41 | 121.00 | – (–) | 64.9 | 660 | 3.03 |
| baseline | text | 1 | 64 | 0 | 0 | – | 0.0 | 20.1 | – | – | 19.74/18.09 | 88.94 | – (–) | 66.0 | 1290 | 3.00 |
