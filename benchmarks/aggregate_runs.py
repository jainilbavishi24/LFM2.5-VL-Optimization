"""Run-to-run variability: mean ± std of TPOT / J per token / TTFT over every full-grid run of each variant.

    python benchmarks/aggregate_runs.py [--gpu RTXPRO5000Blackwell] [--gen 128] [--after 20261014-0225]
    python benchmarks/aggregate_runs.py --pattern 'final_*'      # the final repeats (run summarize.py on each first)

Uses every results/raw/*/summary.csv of batch-1 runs whose tag matches --pattern (default: the full-grid runs,
tags *_nohooks_full and *_fill_nohooks_full), so repeat runs made in different slots/sessions are included.
"First token e2e" = TTFT + the preprocessing time measured by bench.py (CPU or GPU, see --preprocess).
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTXPRO5000Blackwell")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--pattern", default="*nohooks_full", help="run-tag glob")
    ap.add_argument("--after", default="", help="only runs whose id (YYYYMMDD-HHMMSS) is >= this, e.g. after a code change")
    args = ap.parse_args()
    vals = defaultdict(lambda: defaultdict(list))  # (variant, workload) -> metric -> values
    runs = defaultdict(set)
    for f in sorted(RAW.glob(f"*_{args.gpu}_{args.pattern}/summary.csv")):
        if args.after and f.parent.name[:15] < args.after:
            continue
        for r in csv.DictReader(open(f)):
            if int(r["batch_size"]) != 1:
                continue
            key = (r["variant"], r["workload"])
            if int(r["gen_len"]) == args.gen and r["tpot_ms"]:
                vals[key]["tpot_ms"].append(float(r["tpot_ms"]))
                if r["decode_j_per_tok"]:
                    vals[key]["j_per_tok"].append(float(r["decode_j_per_tok"]))
                runs[r["variant"]].add(f.parent.name)
            if int(r["gen_len"]) == 1:
                vals[key]["ttft_ms"].append(float(r["ttft_ms"]))
                if r.get("preprocess_ms"):
                    vals[key]["first_e2e_ms"].append(float(r["ttft_ms"]) + float(r["preprocess_ms"]))

    def ms(xs):
        if not xs:
            return "–"
        return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"

    print(f"| variant | workload | runs | TPOT ms (gen {args.gen}) | J/token | TTFT ms | first token e2e ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for (v, w), m in sorted(vals.items()):
        print(f"| {v} | {w} | {len(m['tpot_ms'])} | {ms(m['tpot_ms'])} | {ms(m['j_per_tok'])} | {ms(m['ttft_ms'])} "
              f"| {ms(m['first_e2e_ms'])} |")


if __name__ == "__main__":
    main()
