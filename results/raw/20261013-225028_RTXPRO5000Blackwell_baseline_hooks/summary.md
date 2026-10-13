# 20261013-225028_RTXPRO5000Blackwell_baseline_hooks

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 28.4 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 20.3 | 12.0 | 5.2 | – | – | –/– | 2.90 | – (–) | 48.8 | 2392 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 20.3 | 12.0 | 5.2 | 3.86 | 259.2 | 3.85/3.46 | 110.33 | 0.846 (0.736) | 218.9 | 2542 | 3.02 |
