# 20261013-224953_RTXPRO5000Blackwell_baseline_fp32

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 29.7 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_small | 1 | 64 | 1 | 234 | – | 37.7 | 18.9 | – | – | 5.18/4.58 | 99.03 | – (–) | 209.7 | 2580 | 6.02 |
| baseline | text | 1 | 64 | 0 | 0 | – | 0.0 | 6.4 | – | – | 5.14/4.54 | 94.64 | – (–) | 103.9 | 2385 | 5.97 |
