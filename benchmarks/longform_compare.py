"""Long free-form answers (eval_quality.py --benches longform): how close is each variant to a reference run?

Greedy BF16 generation drifts from FP32 at near-ties, so differences alone prove nothing. The test is whether our
variant stays as close to the FP32 reference as HF's own BF16 baseline does. Per variant vs the reference (token ids
cut at EOS): identical answers, mean common-prefix length, token-level ROUGE-L F1, mean length.

    python benchmarks/longform_compare.py results/eval/<run> [--ref baseline@fp32] [--eos 7] [--md]
"""

import argparse
import json
import statistics
from pathlib import Path


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(a, b):
    if not a or not b:
        return float(a == b)
    m = lcs(a, b)
    p, r = m / len(a), m / len(b)
    return 0.0 if m == 0 else 2 * p * r / (p + r)


def prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default="baseline@fp32")
    ap.add_argument("--eos", default="7", help="EOS ids if the run's config.json has none (1.6B: 7, 3B: 124900)")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    eos = set(cfg.get("eos_ids") or [int(t) for t in args.eos.split(",")])
    rows = {}
    for f in sorted(run.glob("*.jsonl")):
        if f.name == "summary.jsonl":
            continue
        d = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["bench"] == "longform":
                ids = r["token_ids"]
                d[r["id"]] = ids[: next((j for j, t in enumerate(ids) if t in eos), len(ids))]
        rows[f.stem] = d
    ref = rows[args.ref]
    hdr = ["variant", "n", "identical %", "common prefix (tokens)", "ROUGE-L F1 vs ref", "mean length"]
    out = []
    for var in sorted(rows, key=lambda v: (v != args.ref, v)):
        ids = [i for i in rows[var] if i in ref]
        a, b = [rows[var][i] for i in ids], [ref[i] for i in ids]
        out.append([var, len(ids), f"{100 * statistics.mean(x == y for x, y in zip(a, b)):.1f}",
                    f"{statistics.mean(prefix(x, y) for x, y in zip(a, b)):.1f}",
                    f"{statistics.mean(rouge_l(x, y) for x, y in zip(a, b)):.3f}",
                    f"{statistics.mean(len(x) for x in a):.1f}"])
    if args.md:
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
        for r in out:
            print("| " + " | ".join(str(x) for x in r) + " |")
    else:
        w = [max(len(str(x)) for x in col) for col in zip(hdr, *out)]
        for r in [hdr] + out:
            print("  ".join(str(x).rjust(n) for x, n in zip(r, w)))


if __name__ == "__main__":
    main()
