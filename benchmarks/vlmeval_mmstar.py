"""MMStar through VLMEvalKit itself: LiquidAI's own LFM2-VL integration (vlmeval/vlm/liquid.py, config
"LFM2.5-VL-1.6B": use_custom_prompt=False, greedy, max_new_tokens 8192), VLMEvalKit's MMStar TSV, prompt builder
and scorer. The model card's MMStar number (50.67) was produced with VLMEvalKit.

Our variants keep that class unchanged and only replace the model it wraps (lfm2opt.variants), optionally with our GPU
image preprocessing, so the comparison isolates our optimizations.

    VLMEVAL_DIR=/path/to/VLMEvalKit LMUData=/path/to/data \\
    python benchmarks/vlmeval_mmstar.py --variants baseline,fused_gemv_pf_vfused+gpuprep [--judge exact_matching]

VLMEvalKit's full package imports every dataset and model it supports; only the modules needed here are loaded
(sub-packages registered without running their __init__). Scoring uses VLMEvalKit's MCQ evaluation; without an
OpenAI key its answer extraction is "exact_matching" (rule-based), the official no-API setting.
"""

import argparse
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def import_vlmeval(vlmeval_dir):
    sys.path.insert(0, vlmeval_dir)
    # vlmeval/__init__.py imports its CLI, which imports the registry of every model; register the package and the
    # dataset / vlm sub-packages without running their __init__, so only the modules used below are loaded
    for name, sub in (("vlmeval", ""), ("vlmeval.dataset", "dataset"), ("vlmeval.vlm", "vlm")):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(vlmeval_dir, "vlmeval", sub)]
        sys.modules[name] = m
    sys.modules["vlmeval.dataset"].DATASET_TYPE = lambda name, **kw: "MCQ"  # only MMStar is run here
    # vlmeval.api = clients for judge LLMs (OpenAI, Gemini, ...) with heavy dependencies; the "exact_matching"
    # scorer never instantiates one, so names imported from it resolve to a placeholder that fails if used
    api = types.ModuleType("vlmeval.api")

    def _placeholder(name):
        if name.startswith("__"):
            raise AttributeError(name)

        def missing(*a, **k):
            raise RuntimeError(f"vlmeval.api.{name} is not loaded (only --judge exact_matching is supported)")
        return missing
    api.__getattr__ = _placeholder
    sys.modules["vlmeval.api"] = api
    from vlmeval.dataset.image_mcq import ImageMCQDataset
    from vlmeval.vlm.liquid import LFM2VL
    return ImageMCQDataset, LFM2VL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline,fused_gemv_pf_vfused+gpuprep")
    ap.add_argument("--dataset", default="MMStar")
    ap.add_argument("--judge", default="exact_matching")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--local-tsv", action="store_true",
                    help="use a locally prepared $LMUData/<dataset>.tsv (VLMEvalKit's download host unreachable)")
    ap.add_argument("--out", default=str(ROOT / "results" / "vlmeval"))
    args = ap.parse_args()

    import pandas as pd
    import torch

    ImageMCQDataset, LFM2VL = import_vlmeval(os.environ["VLMEVAL_DIR"])
    from lfm2opt.model import resolve_model_path
    from lfm2opt.variants import apply_variant

    if args.local_tsv:  # use $LMUData/<dataset>.tsv as is (VLMEvalKit's own file is checked against its md5)
        ImageMCQDataset.DATASET_MD5 = {**ImageMCQDataset.DATASET_MD5, args.dataset: None}
    ds = ImageMCQDataset(args.dataset)
    data = ds.data if not args.limit else ds.data.iloc[: args.limit]
    path, _ = resolve_model_path("LiquidAI/LFM2.5-VL-1.6B")
    out = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[vlmeval] {args.dataset}: {len(data)} questions -> {out}", flush=True)

    for variant in args.variants.split(","):
        base, _, opt = variant.partition("+")
        model = LFM2VL(model_path=path, use_custom_prompt=False)  # = vlmeval/config.py "LFM2.5-VL-1.6B"
        if base != "baseline":
            model.model = apply_variant(base, model.model, model.processor)
        if opt == "gpuprep":
            from lfm2opt.preprocess import GpuImageProcessor
            model.processor = GpuImageProcessor.wrap(model.processor)
        preds, t0 = [], time.perf_counter()
        for i in range(len(data)):  # = vlmeval/inference.py infer_data
            line = data.iloc[i]
            struct = model.build_prompt(line, dataset=args.dataset) if model.use_custom_prompt(args.dataset) \
                else ds.build_prompt(line)
            with torch.inference_mode():
                preds.append(model.generate(message=struct, dataset=args.dataset))
        secs = time.perf_counter() - t0
        res = data.copy()
        res["prediction"] = [str(p) for p in preds]
        res = res.drop(columns=[c for c in ("image",) if c in res])
        f = out / f"LFM2.5-VL-1.6B_{variant}_{args.dataset}.xlsx"
        res.to_excel(f, index=False)
        acc = ds.evaluate(str(f), model=args.judge)
        print(f"[vlmeval] {variant}: {secs:.0f} s ({1000 * secs / len(data):.0f} ms/question)\n{acc}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
