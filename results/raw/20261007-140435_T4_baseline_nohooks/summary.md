# 20261007-140435_T4_baseline_nohooks

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 16.3 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 113.9 | – | – | – | – | –/– | 8.61 | – (–) | 66.8 | 675 | 3.03 |
| baseline | img_small | 1 | 128 | 1 | 234 | 113.9 | – | – | 20.75 | 48.2 | –/– | 191.09 | 1.437 (1.099) | 64.8 | 540 | 3.03 |
