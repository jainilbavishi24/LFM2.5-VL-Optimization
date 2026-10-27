"""Record the demo: each case (image + prompt, demo/cases.json) answered by each variant on the GPU, with a GPU
timestamp for every phase and every generated token, so demo/render.py can replay it as a video without a GPU.

Same request path as benchmarks/eval_quality.py: greedy, stop at EOS, batch 1, warm model (graphs captured, each
case run --warmup times first). The clock starts with the decoded image in memory (JPEG decoding excluded, the same
for every variant) and covers preprocessing, vision encoder, prompt, every answer token, and the return of generate().

Timestamps are CUDA events (no host sync added): t=0 is recorded on the idle GPU when the request starts; then
  pre      inputs on the GPU (HF: CPU processor + copy; ours: GPU preprocessing)
  vision   image features ready (get_image_features returned: vision encoder + projector)
  token i  logits of answer token i ready (HF: after each lm_head call; ours: after prefill and after each graph
           replay, StaticDecoder.trace)
  end      generate() returned; wall_ms = host clock after a final synchronize (check: matches end).
Per case: --repeats traced runs (the video replays the median one by wall time), peak memory over them, then an
untraced loop of >= --energy-s seconds for energy per answer (NVML counter) and its mean time (no instrumentation).

    python demo/record.py --tag l40_demo
Writes results/demo/<run>/: env.json, config.json, traces.json, images/<case>.<ext> (inputs, not committed).
"""

import argparse
import functools
import gc
import io
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lfm2opt.energy import EnergyMeter  # noqa: E402
from lfm2opt.env import REPO_ROOT, collect_env, gpu_short_name  # noqa: E402
from lfm2opt.model import DEFAULT_MODEL, build_inputs, load  # noqa: E402
from lfm2opt.variants import apply_variant  # noqa: E402

DATA = REPO_ROOT / "data" / "eval"
VARIANTS = "baseline,fused_gemv_pf_vfused+gpuprep,fused_w8all_pf_vfused+gpuprep"


def load_case_image(src):
    """-> (encoded bytes, PIL format) of a case's input image."""
    if "file" in src:
        raw = (REPO_ROOT / src["file"]).read_bytes()
    else:
        import pyarrow.parquet as pq
        v = pq.read_table(DATA / src["parquet"]).slice(src["row"], 1).column("image")[0].as_py()
        raw = v["bytes"] if isinstance(v, dict) else v
    return raw, Image.open(io.BytesIO(raw)).format


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cases", default=str(REPO_ROOT / "demo" / "cases.json"))
    ap.add_argument("--only", default="", help="comma-separated case keys (default: all)")
    ap.add_argument("--variants", default=VARIANTS)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--energy-s", type=float, default=3.0, help="min. seconds of untraced runs for energy per answer")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "demo"))
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text())
    if args.only:
        cases = [c for c in cases if c["key"] in args.only.split(",")]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out) / run_id
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(collect_env(), indent=2))
    (out / "config.json").write_text(json.dumps(dict(vars(args), cases=cases), indent=2))
    print(f"[demo] -> {out}")
    images = {}
    for c in cases:
        raw, fmt = load_case_image(c["src"])
        (out / "images" / f"{c['key']}.{fmt.lower().replace('jpeg', 'jpg')}").write_bytes(raw)
        images[c["key"]] = raw

    meter = EnergyMeter()
    traces = []
    for variant in args.variants.split(","):
        base, _, opt = variant.partition("+")
        net, processor = load(args.model)
        net = apply_variant(base, net, processor)
        if opt == "gpuprep":
            from lfm2opt.preprocess import GpuImageProcessor
            processor = GpuImageProcessor.wrap(processor)
        dec = getattr(net, "_static_decoder", None)
        model_mem = torch.cuda.memory_allocated() / 2**30
        eos = net.generation_config.eos_token_id
        eos = set([eos] if isinstance(eos, int) else eos)

        rec = {"on": False, "vision": [], "tokens": []}
        inner = net.model
        get_feats = inner.get_image_features

        @functools.wraps(get_feats)  # HF generate() reads its signature to pre-encode the image
        def get_image_features(*a, **k):
            r = get_feats(*a, **k)
            if rec["on"]:
                rec["vision"].append(event())
            return r

        inner.get_image_features = get_image_features
        if dec is None:  # HF decode: one lm_head call per generated token
            net.lm_head.register_forward_hook(lambda m, i, o: rec["tokens"].append(event()) if rec["on"] else None)

        for c in cases:
            img = Image.open(io.BytesIO(images[c["key"]])).convert("RGB")

            def run(traced):
                rec.update(on=traced, vision=[], tokens=[])
                if dec is not None:
                    dec.trace = rec["tokens"] if traced else None
                torch.cuda.synchronize()
                e0 = event()
                t0 = time.perf_counter()
                inputs = build_inputs(processor, img, c["prompt"])
                e_pre = event()
                with torch.inference_mode():
                    o = net.generate(**inputs, max_new_tokens=c["max_new"], do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
                e_end = event()
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) * 1e3
                rec["on"] = False
                if dec is not None:
                    dec.trace = None
                n_prompt = int(inputs["input_ids"].shape[1])
                ids = o[0, n_prompt:].tolist()
                r = dict(wall_ms=wall, pre_ms=e0.elapsed_time(e_pre), end_ms=e0.elapsed_time(e_end),
                         vision_ms=[e0.elapsed_time(e) for e in rec["vision"]],
                         token_ms=[e0.elapsed_time(e) for e in rec["tokens"]])
                return ids, n_prompt, r

            if dec is not None:  # keep only this request's CUDA graph + buffers (captured by the warm-up): peak memory
                dec.graphs.clear()  # is then what one request needs, as in benchmarks/bench.py
            gc.collect()
            torch.cuda.empty_cache()
            for _ in range(args.warmup):
                run(False)
            torch.cuda.reset_peak_memory_stats()
            runs, all_ids = [], []
            for _ in range(args.repeats):
                ids, n_prompt, r = run(True)
                runs.append(r)
                all_ids.append(ids)
            peak = torch.cuda.max_memory_allocated() / 2**30
            ids = all_ids[0]
            cut = next((j for j, t in enumerate(ids) if t in eos), None)
            answer = ids if cut is None else ids[: cut + 1]  # incl. EOS: its logits are the last thing computed
            same = all(a[: len(answer)] == answer for a in all_ids)
            pieces = [processor.decode(answer[:k], skip_special_tokens=True) for k in range(1, len(answer) + 1)]

            n_e = max(5, math.ceil(args.energy_s * 1e3 / statistics.median(r["wall_ms"] for r in runs)))
            meter.start()
            t0 = time.perf_counter()
            for _ in range(n_e):
                run(False)
            untraced_ms = (time.perf_counter() - t0) * 1e3 / n_e
            en = meter.stop()

            walls = [r["wall_ms"] for r in runs]
            med = sorted(range(len(runs)), key=lambda i: walls[i])[len(runs) // 2]
            t = dict(variant=variant, key=c["key"], image_size=list(img.size), prompt=c["prompt"], max_new=c["max_new"],
                     prompt_tokens=n_prompt, answer_ids=answer, ended_with_eos=cut is not None,
                     same_tokens_all_runs=same, text=pieces[-1] if pieces else "", pieces=pieces, runs=runs,
                     median_run=med, wall_ms_median=walls[med], wall_ms_min=min(walls), wall_ms_max=max(walls),
                     peak_mem_gb=peak, model_mem_gb=model_mem, energy_runs=n_e, untraced_ms_mean=untraced_ms,
                     energy_j_per_answer=en["energy_j"] / n_e if "energy_j" in en else None,
                     power_avg_w=en.get("power_avg_w"), sm_clock_avg_mhz=en.get("sm_clock_avg_mhz"))
            traces.append(t)
            (out / "traces.json").write_text(json.dumps(traces))
            r = runs[med]
            print(f"[demo] {variant:>32} {c['key']:>18} {img.size[0]:>4}x{img.size[1]:<4} T={n_prompt:<5} "
                  f"new={len(answer):<3} pre={r['pre_ms']:7.2f} vis={(r['vision_ms'] or [0])[0]:7.2f} "
                  f"ttft={r['token_ms'][0]:7.2f} end={r['end_ms']:7.2f} wall={walls[med]:7.2f} "
                  f"(untraced {untraced_ms:7.2f}) mem={peak:.2f} GB E={t['energy_j_per_answer'] or 0:.2f} J "
                  f"same={same} | {t['text'][:60]!r}")
        del net, dec, inner, get_feats
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
