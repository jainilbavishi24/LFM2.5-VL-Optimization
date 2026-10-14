# 20261014-155736_T4_t4_base_bf16

GPU: Tesla T4 (sm_75), driver 580.82.07, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 12.6 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 6443.3 | – | – | – | – | –/– | 425.66 | – (–) | 66.1 | 795 | 4.68 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 6443.3 | – | – | 21.77 | 45.9 | –/– | 610.06 | 1.452 (1.177) | 66.1 | 735 | 4.68 |
| baseline | img_small | 1 | 1 | 1 | 234 | 717.0 | – | – | – | – | –/– | 48.27 | – (–) | 65.1 | 870 | 3.17 |
| baseline | img_small | 1 | 128 | 1 | 234 | 717.0 | – | – | 20.78 | 48.1 | –/– | 226.55 | 1.404 (1.141) | 66.6 | 780 | 3.17 |
| baseline | text | 1 | 1 | 0 | 0 | 63.6 | – | – | – | – | –/– | 4.85 | – (–) | 57.5 | 1050 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 63.6 | – | – | 21.26 | 47.0 | –/– | 186.66 | 1.432 (1.163) | 67.1 | 1155 | 3.00 |
