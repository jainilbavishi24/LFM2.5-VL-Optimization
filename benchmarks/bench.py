"""End-to-end latency / throughput / energy benchmark for LFM2.5-VL.

Grid: variant x workload x batch size x generation length, each with warm-up + repeats.
Greedy decoding with a fixed number of new tokens (min_new_tokens = max_new_tokens), so every
repeat does identical work and outputs can be compared across variants/GPUs.

Derived metrics (computed in summarize.py):
  TTFT   = wall time of generate(max_new_tokens=1)              (vision + projector + prefill + 1 token)
  TPOT   = (wall(N) - wall(1)) / (N - 1)                        (per decode step)
  J/tok  = (E(N) - E(1)) / (N - 1) / batch

Output: results/raw/<run_id>/{env.json, config.json, runs.jsonl, outputs.jsonl}

Examples:
  python benchmarks/bench.py --preset quick
  python benchmarks/bench.py --workloads img_small --gen-lens 1,128 --repeats 10 --tag baseline
"""

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lfm2opt.energy import EnergyMeter  # noqa: E402
from lfm2opt.env import REPO_ROOT, collect_env, gpu_short_name  # noqa: E402
from lfm2opt.instrument import PhaseInstrumentor  # noqa: E402
from lfm2opt.model import DEFAULT_MODEL, build_inputs, describe_inputs, dtype_name, load  # noqa: E402
from lfm2opt.variants import apply_variant  # noqa: E402
from lfm2opt.workloads import WORKLOADS  # noqa: E402

PRESETS = {
    # ~5 min on a T4: smoke test of the whole pipeline
    "quick": dict(workloads="text,img_small,img_fhd", gen_lens="1,64", batch_sizes="1", warmup=1, repeats=3),
    # Baseline characterization used for Evaluation 1
    "full": dict(workloads="text,img_small,img_hd,img_fhd", gen_lens="1,32,128,512", batch_sizes="1",
                 warmup=2, repeats=10),
    # Batch-size sweep on one workload
    "batch": dict(workloads="img_small", gen_lens="1,128", batch_sizes="1,2,4,8,16", warmup=2, repeats=5),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, help="fills in grid options (explicit flags override)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--attn", default="sdpa", help="attn_implementation: sdpa | eager | flash_attention_2")
    p.add_argument("--variants", default="baseline", help="comma-separated, see lfm2opt/variants.py")
    p.add_argument("--workloads", help=f"comma-separated from {list(WORKLOADS)}")
    p.add_argument("--gen-lens", help="comma-separated new-token counts; include 1 to get TTFT")
    p.add_argument("--batch-sizes")
    p.add_argument("--warmup", type=int)
    p.add_argument("--repeats", type=int)
    p.add_argument("--idle-seconds", type=float, default=5.0, help="idle-power measurement before the run")
    p.add_argument("--no-phases", action="store_true", help="disable hook instrumentation (to check its overhead)")
    p.add_argument("--tag", default="", help="free-text label added to the run id")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "raw"))
    args = p.parse_args()
    defaults = dict(workloads="img_small", gen_lens="1,128", batch_sizes="1", warmup=2, repeats=5)
    defaults.update(PRESETS.get(args.preset, {}))
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def csv_ints(s):
    return [int(x) for x in str(s).split(",") if x]


def run_generate(net, inputs, n_new, pad_id):
    return net.generate(
        **inputs,
        max_new_tokens=n_new,
        min_new_tokens=n_new,
        do_sample=False,
        pad_token_id=pad_id,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required")

    env = collect_env()
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[bench] run {run_id} on {env.get('gpu_name')} ({env.get('gpu_capability')}), "
          f"torch {env['torch']}, transformers {env['transformers']}")
    print(f"[bench] results -> {out_dir}")

    meter = EnergyMeter()
    if meter.available and args.idle_seconds > 0:
        idle = meter.measure_idle(args.idle_seconds)
        print(f"[bench] idle power {idle.get('idle_power_w', float('nan')):.1f} W")
        with open(out_dir / "idle.json", "w") as f:
            json.dump(idle, f, indent=2)

    runs_f = open(out_dir / "runs.jsonl", "a")
    outs_f = open(out_dir / "outputs.jsonl", "a")
    workloads = [WORKLOADS[w] for w in args.workloads.split(",")]

    for variant in args.variants.split(","):
        t_load = time.perf_counter()
        net, processor = load(args.model, args.dtype, args.attn)
        net = apply_variant(variant, net, processor)
        load_s = time.perf_counter() - t_load
        dtype = dtype_name(next(net.parameters()).dtype)
        pad_id = processor.tokenizer.pad_token_id
        image_token_id = net.config.image_token_id
        print(f"[bench] variant={variant} dtype={dtype} attn={args.attn} loaded in {load_s:.1f}s")

        instr = PhaseInstrumentor(net, modes=() if args.no_phases else ("events",))
        instr.attach()

        for wl in workloads:
            image = wl.image()
            for bs in csv_ints(args.batch_sizes):
                # CPU preprocessing time (chat template + image processor + H2D copy)
                pre_ms = []
                for _ in range(max(3, args.repeats)):
                    t = time.perf_counter()
                    inputs = build_inputs(processor, image, wl.prompt, bs)
                    torch.cuda.synchronize()
                    pre_ms.append((time.perf_counter() - t) * 1e3)
                shape = describe_inputs(inputs, image_token_id)

                for n_new in csv_ints(args.gen_lens):
                    cfg = dict(variant=variant, workload=wl.name, batch_size=bs, gen_len=n_new, dtype=dtype,
                               attn=args.attn, **{k: v for k, v in shape.items() if k != "batch_size"})
                    try:
                        with torch.inference_mode():
                            for _ in range(args.warmup):
                                run_generate(net, inputs, n_new, pad_id)
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()

                            e_loop0 = meter.energy_mj() if meter.has_energy_counter else None
                            t_loop0 = time.perf_counter()
                            walls = []
                            for rep in range(args.repeats):
                                instr.reset()
                                meter.start()
                                torch.cuda.synchronize()
                                t0 = time.perf_counter()
                                out = run_generate(net, inputs, n_new, pad_id)
                                torch.cuda.synchronize()
                                wall_ms = (time.perf_counter() - t0) * 1e3
                                energy = meter.stop()
                                phases = instr.summary() if not args.no_phases else {}
                                new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
                                walls.append(wall_ms)
                                row = dict(kind="run", **cfg, rep=rep, wall_ms=wall_ms, new_tokens=new_tokens,
                                           **{f"energy_{k}": v for k, v in energy.items()}, **phases)
                                runs_f.write(json.dumps(row) + "\n")
                                if rep == 0:
                                    ids = out[0, inputs["input_ids"].shape[1]:].tolist()
                                    outs_f.write(json.dumps(dict(**cfg, token_ids=ids, text=processor.decode(
                                        ids, skip_special_tokens=True))) + "\n")
                            loop_s = time.perf_counter() - t_loop0
                            e_loop = ((meter.energy_mj() - e_loop0) / 1000) if e_loop0 is not None else None
                        summary = dict(kind="config", **cfg, repeats=args.repeats, preprocess_ms=statistics.median(pre_ms),
                                       wall_ms_median=statistics.median(walls), wall_ms_min=min(walls),
                                       wall_ms_max=max(walls), loop_s=loop_s, loop_energy_j=e_loop,
                                       peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                        runs_f.write(json.dumps(summary) + "\n")
                        runs_f.flush()
                        outs_f.flush()
                        print(f"[bench] {variant:>10} {wl.name:>9} bs={bs:<3} gen={n_new:<4} "
                              f"tiles={shape['vision_tiles']:<2} img_tok={shape['image_tokens']:<5} "
                              f"wall={summary['wall_ms_median']:9.1f} ms  "
                              f"E/run={(e_loop or float('nan')) / args.repeats:7.2f} J  "
                              f"mem={summary['peak_mem_gb']:.2f} GB")
                    except torch.cuda.OutOfMemoryError:
                        print(f"[bench] OOM: {cfg}")
                        runs_f.write(json.dumps(dict(kind="oom", **cfg)) + "\n")
                        torch.cuda.empty_cache()

        instr.detach()
        del net
        gc.collect()
        torch.cuda.empty_cache()

    runs_f.close()
    outs_f.close()
    print(f"[bench] done -> {out_dir}")


if __name__ == "__main__":
    main()
