# 20261013-084012_T4_baseline_long

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 13.3 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 129.1 | 56.2 | 48.0 | – | – | –/– | 9.22 | – (–) | 59.5 | 690 | 3.03 |
| baseline | img_small | 1 | 1024 | 1 | 234 | 129.1 | 50.8 | 43.3 | 25.45 | 39.3 | 23.14/21.23 | 1721.62 | 1.674 (1.336) | 65.1 | 600 | 3.05 |
| baseline | text | 1 | 1 | 0 | 0 | 63.8 | 0.0 | 43.5 | – | – | –/– | 3.05 | – (–) | 36.7 | 585 | 2.99 |
| baseline | text | 1 | 1024 | 0 | 0 | 63.8 | 0.0 | 22.1 | 25.40 | 39.4 | 22.99/21.10 | 1694.74 | 1.654 (1.316) | 65.0 | 1200 | 3.03 |
