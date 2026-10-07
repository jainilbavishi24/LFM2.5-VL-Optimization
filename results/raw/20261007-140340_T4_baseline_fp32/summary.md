# 20261007-140340_T4_baseline_fp32

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 22.1 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 64 | 1 | 234 | – | 312.8 | 187.7 | – | – | 22.27/19.60 | 127.68 | – (–) | 66.3 | 645 | 6.02 |
| baseline | text | 1 | 64 | 0 | 0 | – | 0.0 | 27.9 | – | – | 21.70/19.06 | 94.66 | – (–) | 66.8 | 1050 | 5.97 |
