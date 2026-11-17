"""Vision-encoder CUDA graph (lfm2opt.vision, graphs=True) vs the same fused encoder run eagerly: features must be
bit-identical (the graph replays the same kernels); also wall time per image, eager vs replay (median of N,
runs interleaved).

    python tests/check_vision_graph.py [--int8] [--repeats 20]
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.model import build_inputs, load  # noqa: E402
from lfm2opt.preprocess import GpuImageProcessor  # noqa: E402
from lfm2opt.vision import FusedSiglip2Encoder  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402


def wall(fn, n):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    net, processor = load()
    processor = GpuImageProcessor.wrap(processor)
    vt = net.model.vision_tower
    eager = FusedSiglip2Encoder(vt, int8=a.int8, graphs=False)
    graph = FusedSiglip2Encoder(vt, int8=a.int8, graphs=True)
    fails = 0
    for wl in ("img_small", "img_hd", "img_fhd"):
        inp = build_inputs(processor, WORKLOADS[wl].image(), WORKLOADS[wl].prompt)
        args = dict(pixel_values=inp["pixel_values"].to(torch.bfloat16), pixel_attention_mask=inp["pixel_attention_mask"],
                    spatial_shapes=inp["spatial_shapes"])
        ref = eager(**args).last_hidden_state.clone()
        outs = [graph(**args).last_hidden_state.clone() for _ in range(3)]  # eager, capture, replay
        same = all(torch.equal(o, ref) for o in outs)
        fails += not same
        te, tg = [], []
        for _ in range(a.repeats):  # interleaved, so clock / temperature drift hits both alike
            te.append(wall(lambda: eager(**args), 1))
            tg.append(wall(lambda: graph(**args), 1))
        t_e, t_g = statistics.median(te), statistics.median(tg)
        print(f"{'ok  ' if same else 'FAIL'} {wl:9s} tiles={args['pixel_values'].shape[0]}  bit-identical={same}  "
              f"eager {t_e:7.2f} ms  graph {t_g:7.2f} ms  ({t_e / t_g:.2f}x)", flush=True)
    print(f"{fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
