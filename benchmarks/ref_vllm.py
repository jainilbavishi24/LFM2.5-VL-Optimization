"""O0c reference baseline: the same workloads through vLLM (offline LLM API, CUDA graphs on).

Definitions match benchmarks/bench.py: TTFT = wall time of a request with max_tokens=1,
TPOT = (wall(N) - wall(1)) / (N - 1), greedy decoding, EOS ignored so every run makes N tokens.
Runs in its own venv (vLLM pins its own torch):

    python benchmarks/ref_vllm.py --model models/LFM2.5-VL-1.6B --gen-lens 1,128,512 --repeats 5
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm2opt.energy import EnergyMeter  # noqa: E402  (pynvml only, no torch import side effects)
from lfm2opt.workloads import WORKLOADS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "LFM2.5-VL-1.6B"))
    p.add_argument("--workloads", default="text,img_small,img_hd,img_fhd")
    p.add_argument("--gen-lens", default="1,128,512")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--eager", action="store_true", help="enforce_eager=True (no CUDA graphs)")
    p.add_argument("--tag", default="ref_vllm")
    args = p.parse_args()

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    gpu = torch.cuda.get_device_name(0)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{gpu.replace('NVIDIA ', '').replace(' ', '')}_{args.tag}"
    out_dir = ROOT / "results" / "raw" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vllm] {run_id}: vllm {vllm.__version__}, torch {torch.__version__}, {gpu}")

    t0 = time.perf_counter()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.6,
              limit_mm_per_prompt={"image": 1}, enforce_eager=args.eager, seed=0)
    load_s = time.perf_counter() - t0
    print(f"[vllm] loaded in {load_s:.1f}s")

    meter = EnergyMeter()
    rows = []
    for wname in args.workloads.split(","):
        wl = WORKLOADS[wname]
        image = wl.image()
        content = ([{"type": "image_pil", "image_pil": image}] if image is not None else []) + [
            {"type": "text", "text": wl.prompt}]
        messages = [{"role": "user", "content": content}]
        walls = {}
        for n in [int(x) for x in args.gen_lens.split(",")]:
            sp = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True)
            for _ in range(args.warmup):
                llm.chat(messages, sp, use_tqdm=False)
            times = []
            meter.start()
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = llm.chat(messages, sp, use_tqdm=False)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            energy = meter.stop()
            o = out[0]
            walls[n] = statistics.median(times)
            row = dict(engine="vllm", workload=wname, gen=n, prompt_tokens=len(o.prompt_token_ids),
                       out_tokens=len(o.outputs[0].token_ids), wall_ms_median=walls[n] * 1e3,
                       wall_ms_all=[t * 1e3 for t in times], energy=energy, text=o.outputs[0].text[:300])
            rows.append(row)
            print(f"[vllm]   {wname:9s} gen={n:<4d} prompt={row['prompt_tokens']:<5d} wall={row['wall_ms_median']:9.1f} ms")
        if 1 in walls:
            for n, w in walls.items():
                if n > 1:
                    print(f"[vllm]   {wname:9s} TTFT={walls[1]*1e3:.1f} ms  TPOT(gen {n})={(w - walls[1]) / (n - 1) * 1e3:.2f} ms")
    (out_dir / "runs.json").write_text(json.dumps(
        dict(vllm=vllm.__version__, torch=torch.__version__, gpu=gpu, load_s=load_s, args=vars(args), rows=rows),
        indent=2, default=str))
    print(f"[vllm] done -> {out_dir}")


if __name__ == "__main__":
    main()
