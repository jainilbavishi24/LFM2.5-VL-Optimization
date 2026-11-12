# 20261112-190517_GB10_spark_base

GPU: NVIDIA GB10 (sm_121), driver None, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 4.7 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 371.1 | – | – | – | – | –/– | 18.29 | – (–) | 48.7 | 2164 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 371.1 | – | – | 15.60 | 64.1 | –/– | 62.26 | 0.346 (0.274) | 26.4 | 2164 | 3.26 |
| baseline | img_small | 1 | 1 | 1 | 234 | 50.2 | – | – | – | – | –/– | 2.17 | – (–) | 24.1 | 2190 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 50.2 | – | – | 15.22 | 65.7 | –/– | 44.15 | 0.331 (0.260) | 22.3 | 2184 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 18.1 | – | – | – | – | –/– | 0.32 | – (–) | 9.8 | 2190 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 18.1 | – | – | 15.16 | 66.0 | –/– | 41.72 | 0.326 (0.255) | 21.5 | 2184 | 3.00 |
