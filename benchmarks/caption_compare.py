"""Reference-scored captions and descriptions (eval_quality.py --benches coco_cap,coco_desc): do our variants caption as
well as the baseline, and do they hallucinate more objects in long descriptions?

  coco_cap   CIDEr-D, BLEU-4, ROUGE-L vs the 5 human captions (pycocoevalcap, PTB tokenizer), per variant
  coco_desc  CHAIR (Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018): the COCO objects a
             description mentions that are neither annotated in the image (val2017 instances) nor named in its human
             captions. CHAIR_s = share of descriptions with >= 1 hallucinated object, CHAIR_i = share of mentioned objects
             that are hallucinated; also recall = share of the image's ground-truth objects that are mentioned.
             Port of the authors' rules (synonym list, double words, baby / adult / passenger qualifiers, toilet seat);
             nouns are singularized with the WordNet lemmatizer instead of pattern.en.
Differences to the reference variant: paired bootstrap over images (95 % CI).

    python benchmarks/caption_compare.py results/eval/<run> [more runs] [--ref baseline] [--md]
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eval"

DOUBLE = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light',
          'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove',
          'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier',
          'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'home plate', 'train track']
ANIMALS = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']


class Chair:
    def __init__(self, synonyms_path, instances_path):
        from nltk.stem import WordNetLemmatizer

        self.lemma = WordNetLemmatizer()
        syn = [line.strip().split(", ") for line in open(synonyms_path) if line.strip()]
        self.objects = {w for group in syn for w in group}
        self.node = {w: group[0] for group in syn for w in group}
        self.double = {d: d for d in DOUBLE}
        for a in ANIMALS:
            self.double[f"baby {a}"] = a
            self.double[f"adult {a}"] = a
        for v in ("jet", "train"):
            self.double[f"passenger {v}"] = v
        self.double.update({"bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass"})
        inst = json.load(open(instances_path))
        names = {c["id"]: c["name"] for c in inst["categories"]}
        self.gt = defaultdict(set)
        for a in inst["annotations"]:
            self.gt[a["image_id"]].add(self.node[names[a["category_id"]]])

    def words(self, text):
        """COCO object node words mentioned in `text` (the authors' caption_to_words)."""
        # WordNet keeps "men" (a lemma itself): -men -> -man as pattern.en singularize did (men, women, policemen)
        ws = [w[:-3] + "man" if w.endswith("men") and len(w) > 3 or w == "men" else self.lemma.lemmatize(w)
              for w in re.findall(r"[a-z]+", text.lower())]
        out, i = [], 0
        while i < len(ws):
            dw = " ".join(ws[i:i + 2])
            if dw in self.double:
                out.append(self.double[dw])
                i += 2
            else:
                out.append(ws[i])
                i += 1
        if "toilet" in out and "seat" in out:
            out = [w for w in out if w != "seat"]
        return [self.node[w] for w in out if w in self.objects]

    def score(self, image_id, refs, text):
        gt = self.gt[image_id] | {w for r in refs for w in self.words(r)}
        mentioned = self.words(text)
        hall = [w for w in mentioned if w not in gt]
        return dict(n_obj=len(mentioned), n_hall=len(hall), any_hall=int(bool(hall)),
                    recall=len(set(mentioned) & gt) / len(gt) if gt else None)


def load(runs):
    rows = defaultdict(dict)  # variant -> id -> row
    for run in runs:
        for f in sorted(Path(run).glob("*.jsonl")):
            if f.name == "summary.jsonl":
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["bench"] in ("coco_cap", "coco_desc"):
                    rows[f.stem][r["id"]] = r
    return rows


def boot(ids, fn, n=2000, seed=0):
    """95 % CI of fn(sample of ids) over bootstrap resamples."""
    rng = random.Random(seed)
    vals = sorted(fn([rng.choice(ids) for _ in ids]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    rows = load(args.runs)
    variants = sorted(rows, key=lambda v: (v != args.ref, v))
    out = []

    # ---------------------------------------------------------------- captions: CIDEr-D / BLEU-4 / ROUGE-L
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tok = PTBTokenizer()
    cap = {}
    for v in variants:
        rs = {k: r for k, r in rows[v].items() if r["bench"] == "coco_cap"}
        if not rs:
            continue
        gts = tok.tokenize({k: [{"caption": c} for c in r["gt"]] for k, r in rs.items()})
        res = tok.tokenize({k: [{"caption": r["pred"]}] for k, r in rs.items()})
        cider, per = Cider().compute_score(gts, res)
        bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
        rouge, _ = Rouge().compute_score(gts, res)
        cap[v] = dict(zip(gts.keys(), per))
        out.append(dict(bench="coco_cap", variant=v, n=len(rs), CIDEr=100 * cider, BLEU4=100 * bleu[3], ROUGE_L=100 * rouge,
                        words=sum(len(r["pred"].split()) for r in rs.values()) / len(rs)))
    if args.ref in cap:
        for o in out:
            if o["bench"] == "coco_cap" and o["variant"] != args.ref:
                ids = sorted(set(cap[o["variant"]]) & set(cap[args.ref]))
                d = {k: cap[o["variant"]][k] - cap[args.ref][k] for k in ids}
                o["dCIDEr"] = 100 * sum(d.values()) / len(ids)
                lo, hi = boot(ids, lambda s: 100 * sum(d[k] for k in s) / len(s))
                o["dCIDEr_CI"] = f"[{lo:+.1f}, {hi:+.1f}]"

    # ---------------------------------------------------------------- descriptions: CHAIR
    if any(r["bench"] == "coco_desc" for v in variants for r in rows[v].values()):
        ch = Chair(DATA / "chair_synonyms.txt", DATA / "instances_val2017.json")
        per = {}
        for v in variants:
            rs = {k: r for k, r in rows[v].items() if r["bench"] == "coco_desc"}
            if not rs:
                continue
            per[v] = {k: ch.score(int(k.split("/")[1]), r["gt"], r["pred"]) for k, r in rs.items()}
            p = list(per[v].values())
            rec = [x["recall"] for x in p if x["recall"] is not None]
            out.append(dict(bench="coco_desc", variant=v, n=len(p),
                            CHAIR_s=100 * sum(x["any_hall"] for x in p) / len(p),
                            CHAIR_i=100 * sum(x["n_hall"] for x in p) / max(1, sum(x["n_obj"] for x in p)),
                            recall=100 * sum(rec) / len(rec),
                            words=sum(len(r["pred"].split()) for r in rs.values()) / len(rs)))
        if args.ref in per:
            ref = per[args.ref]
            for o in out:
                v = o["variant"]
                if o["bench"] != "coco_desc" or v == args.ref:
                    continue
                ids = sorted(set(per[v]) & set(ref))

                def chair_i(vv, s):
                    return 100 * sum(per[vv][k]["n_hall"] for k in s) / max(1, sum(per[vv][k]["n_obj"] for k in s))

                o["dCHAIR_i"] = chair_i(v, ids) - chair_i(args.ref, ids)
                lo, hi = boot(ids, lambda s: chair_i(v, s) - chair_i(args.ref, s))
                o["dCHAIR_i_CI"] = f"[{lo:+.2f}, {hi:+.2f}]"
                ds = {k: per[v][k]["any_hall"] - ref[k]["any_hall"] for k in ids}
                o["dCHAIR_s"] = 100 * sum(ds.values()) / len(ids)
                lo, hi = boot(ids, lambda s: 100 * sum(ds[k] for k in s) / len(s))
                o["dCHAIR_s_CI"] = f"[{lo:+.1f}, {hi:+.1f}]"

    for bench, cols in (("coco_cap", ["variant", "n", "CIDEr", "dCIDEr", "dCIDEr_CI", "BLEU4", "ROUGE_L", "words"]),
                        ("coco_desc", ["variant", "n", "CHAIR_s", "dCHAIR_s", "dCHAIR_s_CI", "CHAIR_i", "dCHAIR_i",
                                       "dCHAIR_i_CI", "recall", "words"])):
        sel = [o for o in out if o["bench"] == bench]
        if not sel:
            continue
        fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)  # noqa: E731
        print(f"\n{bench} (reference variant: {args.ref})")
        if args.md:
            print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        for o in sel:
            vals = [fmt(o.get(c, "")) for c in cols]
            print(("| " + " | ".join(vals) + " |") if args.md else "  ".join(f"{c}={x}" for c, x in zip(cols, vals)))


if __name__ == "__main__":
    main()
