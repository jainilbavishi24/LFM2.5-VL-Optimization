# 20261022-104401_L40S_l40_3b_base

GPU: NVIDIA L40S (sm_89), driver 610.57.04, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 33.5 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 159.9 | – | – | – | – | –/– | 56.95 | – (–) | 252.1 | 2100 | 6.11 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 159.9 | – | – | 15.23 | 65.6 | –/– | 445.36 | 3.058 (2.549) | 215.5 | 2085 | 6.11 |
| baseline | img_small | 1 | 1 | 1 | 234 | 31.8 | – | – | – | – | –/– | 12.96 | – (–) | 181.8 | 2520 | 5.87 |
| baseline | img_small | 1 | 128 | 1 | 234 | 31.8 | – | – | 15.21 | 65.8 | –/– | 396.86 | 3.023 (2.514) | 199.9 | 2520 | 5.87 |
| baseline | text | 1 | 1 | 0 | 0 | 31.0 | – | – | – | – | –/– | 5.03 | – (–) | 90.4 | 2520 | 5.84 |
| baseline | text | 1 | 128 | 0 | 0 | 31.0 | – | – | 15.07 | 66.4 | –/– | 385.78 | 2.998 (2.494) | 197.0 | 2520 | 5.84 |
