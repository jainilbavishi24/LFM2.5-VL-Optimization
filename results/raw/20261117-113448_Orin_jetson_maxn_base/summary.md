# 20261117-113448_Orin_jetson_maxn_base

GPU: Orin (sm_87), driver None, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 8.3 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 1823.1 | – | – | – | – | –/– | 53.65 | – (–) | 29.4 | 1173 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 1823.1 | – | – | 40.52 | 24.7 | –/– | 156.96 | 0.814 (0.477) | 22.5 | 1173 | 3.26 |
| baseline | img_hd | 1 | 1 | 3 | 764 | 624.8 | – | – | – | – | –/– | 18.08 | – (–) | 28.9 | 1173 | 3.08 |
| baseline | img_hd | 1 | 128 | 3 | 764 | 624.8 | – | – | 40.50 | 24.7 | –/– | 120.40 | 0.806 (0.470) | 20.9 | 1173 | 3.08 |
| baseline | img_small | 1 | 1 | 1 | 234 | 226.2 | – | – | – | – | –/– | 5.91 | – (–) | 26.4 | 1173 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 226.2 | – | – | 40.57 | 24.6 | –/– | 107.99 | 0.804 (0.467) | 20.1 | 1173 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 54.1 | – | – | – | – | –/– | 0.72 | – (–) | 13.3 | 1173 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 54.1 | – | – | 40.53 | 24.7 | –/– | 102.51 | 0.801 (0.465) | 19.7 | 1173 | 3.00 |
