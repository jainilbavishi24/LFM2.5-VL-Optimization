"""Paired comparison of eval_quality.py runs: does a variant answer like the baseline?

Per benchmark and variant: score, identical-answer rate vs the reference variant, the discordant pairs
(reference right / variant wrong and the reverse) and the exact two-sided McNemar p-value on per-question
correctness (score >= 0.5 counts as correct for TextVQA's fractional VQA accuracy). Also latency and energy.

    python benchmarks/eval_compare.py results/eval/<run> [more runs ...] [--ref baseline] [--md]
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test: P(X <= min(b, c)) * 2 for X ~ Binomial(b + c, 1/2), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--eos", default="7", help="EOS token ids (comma-separated) for runs whose config.json has no "
                    "eos_ids: token_ids are cut at the first EOS before comparing (1.6B: 7, 3B: 124900)")
    args = ap.parse_args()

    # rows[variant][bench][id] = row ; later runs override earlier ones
    rows = defaultdict(lambda: defaultdict(dict))
    summ = {}
    for run in args.runs:
        run = Path(run)
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        eos = set(cfg.get("eos_ids") or [int(t) for t in args.eos.split(",")])
        for f in sorted(run.glob("*.jsonl")):
            if f.name == "summary.jsonl":
                for line in f.read_text().splitlines():
                    s = json.loads(line)
                    summ[(s["variant"], s["bench"])] = s
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                ids = r["token_ids"]
                r["token_ids"] = ids[: next((j for j, t in enumerate(ids) if t in eos), len(ids))]
                rows[f.stem][r["bench"]][r["id"]] = r

    ref = rows.get(args.ref)
    if ref is None:
        raise SystemExit(f"reference variant {args.ref!r} not found; have {sorted(rows)}")
    hdr = ["variant", "bench", "n", "score %", "Δ vs ref (pp)", "same answer %", "ref✓ var✗", "ref✗ var✓",
           "McNemar p", "e2e ms/q", "J/q"]
    out = []
    for var in sorted(rows, key=lambda v: (v != args.ref, v)):
        for bench, by_id in rows[var].items():
            ids = [i for i in by_id if i in ref.get(bench, {})]
            if not ids:
                continue
            sv = [by_id[i]["score"] for i in ids]
            sr = [ref[bench][i]["score"] for i in ids]
            same = sum(by_id[i]["token_ids"] == ref[bench][i]["token_ids"] for i in ids) / len(ids)
            b = sum(r >= 0.5 > v for r, v in zip(sr, sv))
            c = sum(v >= 0.5 > r for r, v in zip(sr, sv))
            s = summ.get((var, bench), {})
            e2e = statistics.mean(by_id[i]["pre_ms"] + by_id[i]["gen_ms"] for i in ids)
            out.append([var, bench, len(ids), f"{100 * statistics.mean(sv):.2f}",
                        f"{100 * (statistics.mean(sv) - statistics.mean(sr)):+.2f}", f"{100 * same:.2f}", b, c,
                        f"{mcnemar_exact(b, c):.3f}", f"{e2e:.1f}",
                        f"{s['energy_j_per_q']:.3f}" if s.get("energy_j_per_q") else "-"])
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for r in out:
            print("| " + " | ".join(str(x) for x in r) + " |")
    else:
        w = [max(len(str(x)) for x in col) for col in zip(hdr, *out)]
        for r in [hdr] + out:
            print("  ".join(str(x).rjust(n) for x, n in zip(r, w)))


if __name__ == "__main__":
    main()
