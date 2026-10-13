# 20261013-225004_RTXPRO5000Blackwell_baseline_dtype_default

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 26.6 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 64 | 1 | 234 | – | 12.1 | 5.2 | – | – | 3.83/3.44 | 66.50 | – (–) | 148.8 | 2542 | 3.02 |
| baseline | text | 1 | 64 | 0 | 0 | – | 0.0 | 4.0 | – | – | 3.82/3.42 | 41.45 | – (–) | 66.9 | 2400 | 3.00 |
