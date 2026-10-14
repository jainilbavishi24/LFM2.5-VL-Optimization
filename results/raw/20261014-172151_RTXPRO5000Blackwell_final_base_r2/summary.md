# 20261014-172151_RTXPRO5000Blackwell_final_base_r2

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 30.2 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 117.6 | – | – | – | – | –/– | 37.19 | – (–) | 182.5 | 2205 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 117.6 | – | – | 3.68 | 272.1 | –/– | 151.39 | 0.899 (0.788) | 261.7 | 2437 | 3.26 |
| baseline | img_hd | 1 | 1 | 3 | 764 | 40.5 | – | – | – | – | –/– | 12.37 | – (–) | 221.4 | 2362 | 3.08 |
| baseline | img_hd | 1 | 128 | 3 | 764 | 40.5 | – | – | 3.70 | 269.9 | –/– | 122.05 | 0.864 (0.752) | 242.7 | 2512 | 3.08 |
| baseline | img_small | 1 | 1 | 1 | 234 | 20.1 | – | – | – | – | –/– | 3.65 | – (–) | 230.3 | 2550 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 20.1 | – | – | 3.69 | 271.0 | –/– | 116.40 | 0.888 (0.776) | 236.0 | 2542 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 6.5 | – | – | – | – | –/– | 0.81 | – (–) | 51.5 | 2407 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 6.5 | – | – | 3.69 | 270.9 | –/– | 109.91 | 0.859 (0.748) | 229.9 | 2550 | 3.00 |
