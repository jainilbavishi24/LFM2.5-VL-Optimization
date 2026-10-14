"""Output quality + latency + energy of each variant on standard VLM benchmarks (our optimizations must not change
the answers). Same prompts, greedy decoding, stop at EOS, batch 1; images decoded outside the timed region.

Benchmarks (test/val parquet files from scripts/download_eval_data.sh, under data/eval/):
  mmstar    1500 multiple-choice questions (6 capabilities)          accuracy (option letter)
  pope      3 x 3000 object-hallucination yes/no (adversarial/popular/random)   accuracy, F1
  ocrbench  1000 OCR questions (text recognition, scene/doc VQA, KIE, formulas)  OCRBench score (substring match)
  chartqa   2500 chart questions                                      relaxed accuracy (5 % numeric tolerance)
  textvqa   val shard 0 (~1670 scene-text questions)                  VQA accuracy (min(#humans agreeing / 3, 1))
  textvqa_full  all 5000 TextVQA val questions (3 shards)

    python benchmarks/eval_quality.py --variants baseline,fused_gemv_vcudnn+gpuprep [--limit pope=3000]
    python benchmarks/eval_compare.py results/eval/<run>      # paired comparison + McNemar test

Writes results/eval/<run>/<variant>.jsonl (one row per question) and summary.jsonl (per variant x benchmark).
"""

import argparse
import gc
import io
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.energy import EnergyMeter  # noqa: E402
from lfm2opt.env import REPO_ROOT, collect_env, gpu_short_name  # noqa: E402
from lfm2opt.model import DEFAULT_MODEL, build_inputs, load  # noqa: E402
from lfm2opt.variants import apply_variant  # noqa: E402

DATA = REPO_ROOT / "data" / "eval"
SHORT = "\nAnswer the question using a single word or phrase."
MC = "\nAnswer with the option's letter from the given choices directly."


# ----------------------------------------------------------------------------------------------- datasets
def _rows(name):
    import pyarrow.parquet as pq
    return pq.read_table(DATA / name).to_pylist()


def _img(v):
    if isinstance(v, dict):
        v = v.get("bytes")
    return v  # encoded bytes; decoded per item


def load_bench(name):
    """-> list of dict(id, image (bytes), prompt, max_new, gt, extra)"""
    items = []
    if name == "mmstar":
        for r in _rows("mmstar.parquet"):
            items.append(dict(id=f"mmstar/{r['index']}", image=_img(r["image"]), prompt=r["question"] + MC, max_new=8,
                              gt=r["answer"].strip(), cat=r.get("category")))
    elif name == "pope":
        for split in ("adversarial", "popular", "random"):
            for r in _rows(f"pope_{split}.parquet"):
                items.append(dict(id=f"pope/{split}/{r['question_id']}", image=_img(r["image"]),
                                  prompt=r["question"] + SHORT, max_new=8, gt=r["answer"].strip().lower(), cat=split))
    elif name == "ocrbench":
        for i, r in enumerate(_rows("ocrbench.parquet")):
            ans = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            items.append(dict(id=f"ocrbench/{i}", image=_img(r["image"]), prompt=r["question"], max_new=64,
                              gt=ans, cat=r.get("question_type") or r.get("dataset")))
    elif name == "chartqa":
        for i, r in enumerate(_rows("chartqa_test.parquet")):
            items.append(dict(id=f"chartqa/{i}", image=_img(r["image"]), prompt=r["question"] + SHORT, max_new=16,
                              gt=r["answer"], cat=r.get("type")))
    elif name in ("textvqa", "textvqa_full"):
        shards = ["textvqa_val0.parquet"] + (["textvqa_val1.parquet", "textvqa_val2.parquet"] if name == "textvqa_full" else [])
        for r in (r for sh in shards for r in _rows(sh)):
            items.append(dict(id=f"textvqa/{r['question_id']}", image=_img(r["image"]), prompt=r["question"] + SHORT,
                              max_new=16, gt=list(r["answers"]), cat=None))
    else:
        raise KeyError(name)
    return items


def parse_limits(spec, benches):
    """'500' -> every benchmark 500; 'pope=3000,chartqa=1000' -> those only (others complete)."""
    if not spec or spec == "0":
        return {}
    if "=" not in spec:
        return {b: int(spec) for b in benches}
    return {k: int(v) for k, v in (kv.split("=") for kv in spec.split(","))}


def subsample(items, limit):
    """Evenly spaced subset (keeps the category mix of datasets stored grouped by category)."""
    if not limit or limit >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


# ----------------------------------------------------------------------------------------------- scoring
def score_mmstar(pred, gt):
    m = re.search(r"\b([A-D])\b", pred.strip()) or re.match(r"\s*\(?([A-D])", pred.strip())
    return float(bool(m) and m.group(1) == gt)


def score_pope(pred, gt):
    p = pred.strip().lower()
    yes = p.startswith("yes") or (re.search(r"\byes\b", p) is not None and re.search(r"\bno\b", p) is None)
    return float(("yes" if yes else "no") == gt)


def score_ocrbench(pred, gts, cat):
    if cat == "Handwritten Mathematical Expression Recognition":
        p = re.sub(r"\s+", "", pred)
        return float(any(re.sub(r"\s+", "", g) in p for g in gts))
    p = pred.lower().strip().replace("\n", " ")
    return float(any(g.lower().strip().replace("\n", " ") in p for g in gts))


def _to_float(s):
    s = s.strip().rstrip("%").replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def score_chartqa(pred, gt, tol=0.05):
    p, g = pred.strip().rstrip("."), str(gt).strip()
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        return float(abs(pf - gf) <= tol * abs(gf)) if gf != 0 else float(pf == gf)
    return float(p.lower() == g.lower())


_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$|~:]")
_NUMS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
         "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def _vqa_norm(s):
    s = s.lower().replace("\n", " ").replace("\t", " ").strip()
    s = _PUNCT.sub("", re.sub(r"(?<=\d),(?=\d)", "", s))
    s = re.sub(r"(?<!\d)\.(?!\d)", "", s)
    return " ".join(_NUMS.get(w, w) for w in s.split() if w not in _ARTICLES)


def score_textvqa(pred, gts):
    """Standard VQA accuracy: averaged over the 10 leave-one-out subsets of the human answers."""
    p = _vqa_norm(pred)
    g = [_vqa_norm(a) for a in gts]
    accs = []
    for i in range(len(g)):
        others = g[:i] + g[i + 1:]
        accs.append(min(1.0, sum(a == p for a in others) / 3))
    return float(sum(accs) / len(accs)) if accs else 0.0


def score(bench, item, pred):
    if bench == "mmstar":
        return score_mmstar(pred, item["gt"])
    if bench == "pope":
        return score_pope(pred, item["gt"])
    if bench == "ocrbench":
        return score_ocrbench(pred, item["gt"], item["cat"])
    if bench == "chartqa":
        return score_chartqa(pred, item["gt"])
    return score_textvqa(pred, item["gt"])  # textvqa, textvqa_full


def bench_metrics(bench, rows):
    m = {"n": len(rows), "score": statistics.mean(r["score"] for r in rows)}
    if bench == "pope":
        tp = sum(r["score"] == 1 and r["gt"] == "yes" for r in rows)
        fp = sum(r["score"] == 0 and r["gt"] == "no" for r in rows)
        fn = sum(r["score"] == 0 and r["gt"] == "yes" for r in rows)
        m["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
    return m


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variants", default="baseline,fused_gemv_vcudnn+gpuprep",
                    help="comma-separated; suffix +gpuprep = our GPU image preprocessing for that variant")
    ap.add_argument("--benches", default="mmstar,pope,ocrbench,chartqa,textvqa")
    ap.add_argument("--limit", default="0", help="questions per benchmark (evenly spaced subset): N for all, or "
                    "bench=N,...; 0 = all")
    ap.add_argument("--warmup", type=int, default=3, help="untimed questions before each benchmark")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "eval"))
    args = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(collect_env(), indent=2))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"[eval] -> {out}")

    names = args.benches.split(",")
    limits = parse_limits(args.limit, names)
    benches = {b: subsample(load_bench(b), limits.get(b, 0)) for b in names}
    print("[eval] " + "  ".join(f"{b}={len(v)}" for b, v in benches.items()))
    meter = EnergyMeter()
    summary_f = open(out / "summary.jsonl", "a")

    for variant in args.variants.split(","):
        base, _, opt = variant.partition("+")
        net, processor = load(args.model, "auto")
        net = apply_variant(base, net, processor)
        if opt == "gpuprep":
            from lfm2opt.preprocess import GpuImageProcessor
            processor = GpuImageProcessor.wrap(processor)
        dec = getattr(net, "_static_decoder", None)
        if dec is not None:  # capture the CUDA graph of every length bucket now, not inside timed questions
            max_new = max(it["max_new"] for items in benches.values() for it in items)
            for need in range(dec.bucket, 4097, dec.bucket):
                dec._get(1, need, max_new - 1)
        eos = net.generation_config.eos_token_id
        rows_f = open(out / f"{variant}.jsonl", "a")
        for bench, items in benches.items():
            def run(item, img):
                t0 = time.perf_counter()
                inputs = build_inputs(processor, img, item["prompt"])
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.inference_mode():
                    o = net.generate(**inputs, max_new_tokens=item["max_new"], do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                ids = o[0, inputs["input_ids"].shape[1]:].tolist()
                if eos in ids:
                    ids = ids[: ids.index(eos)]
                return ids, (t1 - t0) * 1e3, (t2 - t1) * 1e3, int(inputs["input_ids"].shape[1])

            for item in items[: args.warmup]:
                run(item, Image.open(io.BytesIO(item["image"])).convert("RGB"))
            rows, e0, tb = [], meter.energy_mj() if meter.has_energy_counter else None, time.perf_counter()
            busy_ms = 0.0
            for item in items:
                img = Image.open(io.BytesIO(item["image"])).convert("RGB")
                ids, pre_ms, gen_ms, n_prompt = run(item, img)
                busy_ms += pre_ms + gen_ms
                pred = processor.decode(ids, skip_special_tokens=True)
                r = dict(id=item["id"], bench=bench, cat=item["cat"], gt=item["gt"], pred=pred, token_ids=ids,
                         score=score(bench, item, pred), pre_ms=pre_ms, gen_ms=gen_ms, prompt_tokens=n_prompt,
                         new_tokens=len(ids), image_size=list(img.size))
                rows_f.write(json.dumps(r) + "\n")
                rows.append(r)
            wall_s = time.perf_counter() - tb
            energy_j = (meter.energy_mj() - e0) / 1000 if e0 is not None else None
            rows_f.flush()
            s = dict(variant=variant, bench=bench, **bench_metrics(bench, rows),
                     pre_ms_mean=statistics.mean(r["pre_ms"] for r in rows),
                     gen_ms_mean=statistics.mean(r["gen_ms"] for r in rows),
                     e2e_ms_mean=busy_ms / len(rows), new_tokens_mean=statistics.mean(r["new_tokens"] for r in rows),
                     wall_s=wall_s, energy_j=energy_j, energy_j_per_q=energy_j / len(rows) if energy_j else None)
            summary_f.write(json.dumps(s) + "\n")
            summary_f.flush()
            print(f"[eval] {variant:>20} {bench:>8} n={s['n']:<5} score={s['score'] * 100:6.2f}"
                  + (f" f1={s['f1'] * 100:6.2f}" if "f1" in s else "")
                  + f"  pre={s['pre_ms_mean']:6.2f} ms gen={s['gen_ms_mean']:7.2f} ms"
                  + (f"  E/q={s['energy_j_per_q']:.3f} J" if energy_j else ""))
        rows_f.close()
        del net
        gc.collect()
        torch.cuda.empty_cache()
    summary_f.close()


if __name__ == "__main__":
    main()
