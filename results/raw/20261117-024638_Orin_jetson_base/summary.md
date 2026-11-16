# 20261117-024638_Orin_jetson_base

GPU: Orin (sm_87), driver None, torch 2.11.0+cu130, transformers 5.19.0, causal_conv1d: False, idle power: 5.1 W

| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | img_fhd | 1 | 1 | 9 | 2300 | 1854.2 | – | – | – | – | –/– | 53.20 | – (–) | 28.7 | 1173 | 3.26 |
| baseline | img_fhd | 1 | 128 | 9 | 2300 | 1854.2 | – | – | 51.86 | 19.3 | –/– | 166.26 | 0.890 (0.627) | 19.7 | 1173 | 3.26 |
| baseline | img_hd | 1 | 1 | 3 | 764 | 649.9 | – | – | – | – | –/– | 17.99 | – (–) | 27.7 | 1173 | 3.08 |
| baseline | img_hd | 1 | 128 | 3 | 764 | 649.9 | – | – | 51.94 | 19.3 | –/– | 129.34 | 0.877 (0.613) | 17.9 | 1173 | 3.08 |
| baseline | img_small | 1 | 1 | 1 | 234 | 250.9 | – | – | – | – | –/– | 6.02 | – (–) | 24.3 | 1173 | 3.02 |
| baseline | img_small | 1 | 128 | 1 | 234 | 250.9 | – | – | 51.57 | 19.4 | –/– | 116.66 | 0.871 (0.609) | 17.2 | 1173 | 3.02 |
| baseline | text | 1 | 1 | 0 | 0 | 68.8 | – | – | – | – | –/– | 0.72 | – (–) | 10.4 | 918 | 2.99 |
| baseline | text | 1 | 128 | 0 | 0 | 68.8 | – | – | 51.42 | 19.4 | –/– | 111.13 | 0.869 (0.608) | 16.8 | 1173 | 3.00 |
