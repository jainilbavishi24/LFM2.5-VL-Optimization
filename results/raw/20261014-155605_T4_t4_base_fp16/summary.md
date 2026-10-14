# 20261014-155605_T4_t4_base_fp16

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 29.1 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 1148.0 | – | – | – | – | –/– | 77.38 | – (–) | 67.1 | 675 | 4.68 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 1148.0 | – | – | 21.15 | 47.3 | –/– | 257.98 | 1.422 (0.806) | 66.9 | 630 | 4.68 |
| baseline | img_small | 1 | 1 | 1 | 234 | 110.8 | – | – | – | – | –/– | 7.99 | – (–) | 68.7 | 690 | 3.03 |
| baseline | img_small | 1 | 128 | 1 | 234 | 110.8 | – | – | 21.13 | 47.3 | –/– | 187.11 | 1.410 (0.795) | 67.1 | 705 | 3.03 |
| baseline | text | 1 | 1 | 0 | 0 | 33.1 | – | – | – | – | –/– | 1.65 | – (–) | 44.1 | 585 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 33.1 | – | – | 21.63 | 46.2 | –/– | 189.17 | 1.477 (0.846) | 68.0 | 1320 | 3.00 |
