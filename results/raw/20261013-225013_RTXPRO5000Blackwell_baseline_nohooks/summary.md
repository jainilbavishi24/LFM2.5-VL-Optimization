# 20261013-225013_RTXPRO5000Blackwell_baseline_nohooks

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 27.6 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 20.3 | – | – | – | – | –/– | 4.07 | – (–) | 45.8 | 2385 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 20.3 | – | – | 3.79 | 263.7 | –/– | 111.94 | 0.849 (0.745) | 222.9 | 2542 | 3.02 |
