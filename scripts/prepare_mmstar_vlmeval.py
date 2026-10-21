"""Build VLMEvalKit's MMStar.tsv (separate A-D option columns) from the MMStar authors' TSV on the Hugging Face Hub.

VLMEvalKit's own copy is hosted on opencompass.openxlab.space, which is unreachable from our machines. The authors'
file (Lin-Chen/MMStar, MMStar.tsv) embeds the options in the question text in two styles:
  "...\\nOptions: A: x, B: y, C: z, D: w"       (most sources)
  "...\\nChoices:\\n(A) x\\n(B) y ..."            (MathVista-derived; sometimes without "Choices:", or 3 options)
All 1500 questions parse; every answer letter is among the parsed options.

    python scripts/prepare_mmstar_vlmeval.py <authors MMStar.tsv> <LMUData dir>   # writes <LMUData>/MMStar.tsv
"""

import re
import sys

import pandas as pd

P_OPTIONS = re.compile(r"^(.*?)\n?Options:\s*A:\s*(.*?),\s*B:\s*(.*?),\s*C:\s*(.*?),\s*D:\s*(.*?)\s*$", re.S)


def parse(q):
    m = P_OPTIONS.match(q)
    if m:
        stem, *opts = m.groups()
        return stem, dict(zip("ABCD", opts))
    k = q.find("\n(A)")
    if k < 0:
        return None
    stem = re.sub(r"\n?Choices:\s*$", "", q[:k])
    opts = dict(re.findall(r"\(([A-E])\)\s*(.*?)(?=\n\([A-E]\)|\s*$)", q[k:], re.S))
    return (stem, opts) if "A" in opts and "B" in opts else None


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    d = pd.read_csv(src, sep="\t")
    rows, bad = [], []
    for _, r in d.iterrows():
        p = parse(str(r["question"]))
        if not p:
            bad.append(r["index"])
            continue
        stem, opts = p
        rows.append(dict(index=r["index"], question=stem.strip(), **{k: v.strip() for k, v in opts.items()},
                         answer=r["answer"], category=r["category"], **{"l2-category": r["l2_category"]},
                         image=r["image"]))
    out = pd.DataFrame(rows)
    missing = sum(1 for _, r in out.iterrows() if pd.isna(r.get(r["answer"])))
    print(f"parsed {len(rows)}/{len(d)}, unparsed {bad}, answers outside options {missing}")
    out.to_csv(f"{out_dir}/MMStar.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
