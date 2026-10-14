# 20261014-172247_RTXPRO5000Blackwell_final_base_r3

GPU: NVIDIA RTX PRO 5000 Blackwell (sm_120), driver 595.84, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 39.6 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 118.5 | – | – | – | – | –/– | 33.90 | – (–) | 223.1 | 2182 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 118.5 | – | – | 3.68 | 271.7 | –/– | 153.75 | 0.944 (0.798) | 263.6 | 2430 | 3.26 |
| baseline | img_hd | 1 | 1 | 3 | 764 | 40.7 | – | – | – | – | –/– | 9.53 | – (–) | 212.6 | 2347 | 3.08 |
| baseline | img_hd | 1 | 128 | 3 | 764 | 40.7 | – | – | 3.70 | 270.5 | –/– | 127.44 | 0.928 (0.782) | 247.7 | 2512 | 3.08 |
| baseline | img_small | 1 | 1 | 1 | 234 | 20.1 | – | – | – | – | –/– | 3.22 | – (–) | 237.5 | 2542 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 20.1 | – | – | 3.69 | 270.9 | –/– | 118.97 | 0.911 (0.765) | 241.4 | 2535 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 6.5 | – | – | – | – | –/– | 0.83 | – (–) | 55.5 | 2400 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 6.5 | – | – | 3.69 | 271.0 | –/– | 112.63 | 0.880 (0.734) | 236.8 | 2542 | 3.00 |
