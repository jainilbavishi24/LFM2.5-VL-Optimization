# 20261014-172057_RTXPRO5000Blackwell_final_base_r1

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 13.7 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 115.3 | – | – | – | – | –/– | 34.73 | – (–) | 201.1 | 2257 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 115.3 | – | – | 3.70 | 270.1 | –/– | 144.37 | 0.863 (0.812) | 248.7 | 2460 | 3.26 |
| baseline | img_hd | 1 | 1 | 3 | 764 | 39.5 | – | – | – | – | –/– | 11.97 | – (–) | 207.6 | 2505 | 3.08 |
| baseline | img_hd | 1 | 128 | 3 | 764 | 39.5 | – | – | 3.73 | 268.0 | –/– | 116.76 | 0.825 (0.774) | 227.9 | 2527 | 3.08 |
| baseline | img_small | 1 | 1 | 1 | 234 | 20.0 | – | – | – | – | –/– | 4.57 | – (–) | 214.1 | 2557 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 20.0 | – | – | 3.72 | 268.5 | –/– | 106.38 | 0.802 (0.750) | 219.3 | 2542 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 6.5 | – | – | – | – | –/– | 1.55 | – (–) | 37.4 | 2407 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 6.5 | – | – | 3.71 | 269.6 | –/– | 101.10 | 0.784 (0.733) | 213.1 | 2550 | 3.00 |
