# 20261013-230511_RTXPRO5000Blackwell_baseline_long

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 22.1 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 1 | 1 | 234 | 20.6 | 12.0 | 5.3 | – | – | –/– | 6.24 | – (–) | 219.8 | 2535 | 3.02 |
| baseline | img_small | 1 | 1024 | 1 | 234 | 20.6 | 12.1 | 5.3 | 3.86 | 258.8 | 3.85/3.46 | 933.68 | 0.907 (0.821) | 234.9 | 2535 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 6.9 | 0.0 | 4.0 | – | – | –/– | 0.00 | – (–) | 47.5 | 2392 | 2.99 |
| baseline | text | 1 | 1024 | 0 | 0 | 6.9 | 0.0 | 4.0 | 3.86 | 259.1 | 3.84/3.45 | 924.16 | 0.903 (0.818) | 232.3 | 2535 | 3.01 |
