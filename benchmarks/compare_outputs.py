"""Compare greedy outputs of two bench.py runs (quality check for optimizations, dtypes, GPUs).

  python benchmarks/compare_outputs.py results/raw/<ref_run> results/raw/<test_run>
  python benchmarks/compare_outputs.py <run> <run> --ref-variant baseline --test-variant fused_shortconv

For each (workload, batch, gen_len) present in both: exact match, and the first token where they diverge.
Greedy decoding is deterministic, so any divergence comes from numerics (dtype, kernel, accumulation order).
"""

import argparse
import json
from pathlib import Path


def load(run_dir: Path, variant: str | None):
    out = {}
    for line in open(run_dir / "outputs.jsonl"):
        r = json.loads(line)
        if variant and r["variant"] != variant:
            continue
        out[(r["workload"], r["batch_size"], r["gen_len"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--ref-variant")
    ap.add_argument("--test-variant")
    a = ap.parse_args()
    ref, test = load(a.ref, a.ref_variant), load(a.test, a.test_variant)
    n_match = 0
    keys = sorted(set(ref) & set(test))
    for key in keys:
        r, t = ref[key]["token_ids"], test[key]["token_ids"]
        div = next((i for i, (x, y) in enumerate(zip(r, t)) if x != y), None)
        same = div is None and len(r) == len(t)
        n_match += same
        status = "MATCH" if same else f"DIVERGES at token {div} / {len(r)}"
        print(f"{key[0]:>9} bs={key[1]:<3} gen={key[2]:<4} [{ref[key]['dtype']} vs {test[key]['dtype']}]  {status}")
        if not same and key[2] > 1:
            print(f"    ref : {ref[key]['text'][:200]!r}")
            print(f"    test: {test[key]['text'][:200]!r}")
    print(f"\n{n_match}/{len(keys)} configurations identical")


if __name__ == "__main__":
    main()
