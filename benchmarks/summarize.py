"""Summarize a bench.py run directory into derived metrics (markdown + CSV).

  python benchmarks/summarize.py results/raw/<run_id>

Writes <run_id>/summary.csv and <run_id>/summary.md, and prints the markdown table.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def load_rows(run_dir: Path):
    rows = [json.loads(line) for line in open(run_dir / "runs.jsonl")]
    return [r for r in rows if r["kind"] == "run"], [r for r in rows if r["kind"] == "config"]


def summarize(run_dir: Path):
    runs, configs = load_rows(run_dir)
    idle_w = None
    if (run_dir / "idle.json").exists():
        idle_w = json.loads((run_dir / "idle.json").read_text()).get("idle_power_w")

    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r["variant"], r["workload"], r["batch_size"], r["gen_len"])].append(r)
    cfg_rows = {(c["variant"], c["workload"], c["batch_size"], c["gen_len"]): c for c in configs}

    stats = {}
    for key, rs in by_cfg.items():
        c = cfg_rows.get(key, {})
        decode_gpu = [x for r in rs for x in (r.get("lm_decode_ms") or []) if x is not None]
        decode_host = [x for r in rs for x in (r.get("decode_step_host_ms") or [])[1:]]  # [0] spans the prefill step
        energy_run = (c["loop_energy_j"] / c["repeats"]) if c.get("loop_energy_j") is not None else \
            med([r.get("energy_energy_j") for r in rs])
        stats[key] = dict(
            wall_ms=c.get("wall_ms_median", med([r["wall_ms"] for r in rs])),
            energy_j=energy_run,
            power_avg_w=med([r.get("energy_power_avg_w") for r in rs]),
            sm_clock_min=min([r["energy_sm_clock_min_mhz"] for r in rs if "energy_sm_clock_min_mhz" in r], default=None),
            vision_ms=med([r.get("vision_ms") for r in rs]),
            projector_ms=med([r.get("projector_ms") for r in rs]),
            lm_prefill_ms=med([r.get("lm_prefill_ms") for r in rs]),
            decode_step_gpu_ms=med(decode_gpu),
            decode_step_host_ms=med(decode_host),
            preprocess_ms=c.get("preprocess_ms"),
            peak_mem_gb=c.get("peak_mem_gb"),
            tiles=rs[0].get("vision_tiles"),
            image_tokens=rs[0].get("image_tokens"),
            prompt_tokens=rs[0].get("prompt_tokens"),
            dtype=rs[0].get("dtype"),
        )

    out = []
    for (variant, wl, bs, n), s in sorted(stats.items()):
        base = stats.get((variant, wl, bs, 1))
        row = dict(variant=variant, workload=wl, batch_size=bs, gen_len=n, dtype=s["dtype"], tiles=s["tiles"],
                   image_tokens=s["image_tokens"], prompt_tokens=s["prompt_tokens"],
                   preprocess_ms=s["preprocess_ms"], wall_ms=s["wall_ms"],
                   ttft_ms=base["wall_ms"] if base else None,
                   vision_ms=s["vision_ms"], projector_ms=s["projector_ms"], lm_prefill_ms=s["lm_prefill_ms"],
                   tpot_ms=None, decode_tok_s=None, decode_step_host_ms=s["decode_step_host_ms"],
                   decode_step_gpu_ms=s["decode_step_gpu_ms"], energy_run_j=s["energy_j"],
                   decode_j_per_tok=None, decode_dyn_j_per_tok=None,
                   power_avg_w=s["power_avg_w"], sm_clock_min_mhz=s["sm_clock_min"], peak_mem_gb=s["peak_mem_gb"])
        if base and n > 1:
            tpot = (s["wall_ms"] - base["wall_ms"]) / (n - 1)
            row["tpot_ms"] = tpot
            row["decode_tok_s"] = bs * 1000 / tpot if tpot > 0 else None
            if s["energy_j"] is not None and base["energy_j"] is not None:
                e_tok = (s["energy_j"] - base["energy_j"]) / (n - 1) / bs
                row["decode_j_per_tok"] = e_tok
                if idle_w is not None:
                    row["decode_dyn_j_per_tok"] = e_tok - idle_w * tpot / 1000 / bs
        out.append(row)

    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    header = ("| variant | workload | bs | gen | tiles | img tok | TTFT ms | vision ms | prefill ms | "
              "TPOT ms | tok/s | step host/gpu ms | J/run | J/tok (dyn) | avg W | min SM MHz | mem GB |")
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for r in out:
        lines.append(
            f"| {r['variant']} | {r['workload']} | {r['batch_size']} | {r['gen_len']} | {r['tiles']} | "
            f"{r['image_tokens']} | {fmt(r['ttft_ms'])} | {fmt(r['vision_ms'])} | {fmt(r['lm_prefill_ms'])} | "
            f"{fmt(r['tpot_ms'], 2)} | {fmt(r['decode_tok_s'])} | "
            f"{fmt(r['decode_step_host_ms'], 2)}/{fmt(r['decode_step_gpu_ms'], 2)} | {fmt(r['energy_run_j'], 2)} | "
            f"{fmt(r['decode_j_per_tok'], 3)} ({fmt(r['decode_dyn_j_per_tok'], 3)}) | {fmt(r['power_avg_w'])} | "
            f"{fmt(r['sm_clock_min_mhz'], 0)} | {fmt(r['peak_mem_gb'], 2)} |"
        )
    env = json.loads((run_dir / "env.json").read_text())
    md = (f"# {run_dir.name}\n\nGPU: {env.get('gpu_name')} ({env.get('gpu_capability')}), driver {env.get('driver')}, "
          f"torch {env.get('torch')}, transformers {env.get('transformers')}, causal_conv1d: "
          f"{env.get('causal_conv1d_installed')}, idle power: {fmt(idle_w)} W\n\n" + "\n".join(lines) + "\n")
    (run_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    for d in ap.parse_args().run_dirs:
        summarize(d)
