"""Live check of the demo video: re-run the video's cases on the GPU right now and print every number next to the
number the video shows, plus whether each answer is token-for-token the one in the video.

The measurement is demo/record.py's Runner, i.e. exactly the code the video's numbers came from; the reference values
are read from the committed traces.json the video was rendered from (demo/video.json names the run and the cases).
Same inputs: the cases (image source, prompt, max tokens) are taken from that run's config.json.

    bash demo/live_l40.sh                     # on the L40S via SLURM (from a laptop or the login node), streamed here
    bash demo/live_l40.sh --quick             # 3 runs per case, no energy loop (~2 min)
    bash demo/live_l40.sh --only cats,receipt_35mp
    bash demo/live_l40.sh --image some.jpg --prompt "What is in this picture?"   # any image: no video reference

What reproduces exactly: every answer, token for token, and the peak memory. GPU-bound times (our decode per token,
mid-size images) reproduce within a few %. CPU-bound times do not: node13's CPUs scale their clock with load
(schedutil governor, 0.8-3.5 GHz, no root to pin it), so tiny images (kernel-launch bound: BF16 148x60 measured 16 or
20 ms over 14 processes) and very large ones (CPU preprocessing / 104 MB copies) vary between sessions; the script
flags every case that differs by more than 5 % and says why.

Writes results/demo/<run>_live/ (same format as record.py, so render.py can draw a video from the live run too).
"""

import argparse
import hashlib
import io
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from record import VARIANTS, Runner, load_case_image  # noqa: E402
from lfm2opt.energy import EnergyMeter  # noqa: E402
from lfm2opt.env import REPO_ROOT, collect_env, gpu_short_name  # noqa: E402

NAMES = {"baseline": "HF baseline", "fused_gemv_pf_vfused+gpuprep": "Ours BF16",
         "fused_w8all_pf_vfused+gpuprep": "Ours INT8"}
TITLES = {}
COLOR = sys.stdout.isatty()
CPU_NOTE = ("note: CPU-side work (image prep, kernel launches, HF's Python loop) runs at whatever clock node13's governor "
            "gives it (schedutil, 0.8-3.5 GHz; no root to pin it), so tiny and very large images vary between sessions. "
            "GPU-bound numbers (ours: ms/token) and the answers do not.")


def c(s, code):
    return f"\033[{code}m{s}\033[0m" if COLOR else str(s)


def bold(s):
    return c(s, "1")


def dim(s):
    return c(s, "2")


def delta(live, ref, tol=(5, 10)):
    """'+1.2 %' colored green / yellow / red by |difference|."""
    if ref is None or live is None or ref == 0:
        return dim("   n/a ")
    d = (live - ref) / ref * 100
    code = "32" if abs(d) <= tol[0] else "33" if abs(d) <= tol[1] else "31"
    return c(f"{d:+6.1f} %", code)


def fms(ms):
    return "    —    " if ms is None else f"{ms:8.1f} ms" if ms < 1000 else f"{ms / 1000:8.3f} s "


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=REPO_ROOT).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"({e})"


def header(ref_dir, ref_env):
    print(bold("=" * 100))
    print(bold("  LIVE CHECK OF THE DEMO VIDEO") + "  ·  LFM2.5-VL-1.6B  ·  HF baseline vs ours BF16 vs ours INT8")
    print(bold("=" * 100))
    print(f"  now        {datetime.now():%Y-%m-%d %H:%M:%S}  on {socket.gethostname()}"
          + (f"  (SLURM job {os.environ['SLURM_JOB_ID']})" if "SLURM_JOB_ID" in os.environ else ""))
    print("  GPU        " + sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,"
                                "temperature.gpu", "--format=csv,noheader"]))
    print(f"  software   torch {torch.__version__} (CUDA {torch.version.cuda})  ·  python {sys.version.split()[0]}")
    dirty = [ln for ln in sh(["git", "status", "--porcelain", "--untracked-files=no"]).splitlines() if ln]
    print(f"  code       git {sh(['git', 'rev-parse', '--short', 'HEAD'])}  ·  "
          + (c("working tree clean", "32") if not dirty else c(f"modified ({len(dirty)}): ", "33")
             + " ".join(ln.split(maxsplit=1)[1] for ln in dirty)))
    if ref_dir is not None:
        tr = ref_dir / "traces.json"
        sha = hashlib.sha256(tr.read_bytes()).hexdigest()[:16]
        commit = sh(["git", "log", "-1", "--format=%h %ci", "--", str(tr.relative_to(REPO_ROOT))])
        print(f"  video data {tr.relative_to(REPO_ROOT)}  sha256 {sha}…  committed {commit or c('NOT COMMITTED', '31')}")
        print(f"             recorded {ref_env.get('timestamp_utc', '?')} UTC on {ref_env.get('gpu_name')} "
              f"(driver {ref_env.get('driver')}, torch {ref_env.get('torch')})")
    print(dim("  Timer: decoded image in memory -> answer complete (CUDA events), greedy, batch 1, median of the timed "
              "runs.\n  Δ = live vs video; green within 5 %, yellow within 10 %."))
    print()


def per_token(t):
    """Decode time per answer token of the median run (first to last answer token)."""
    tok = t["runs"][t["median_run"]]["token_ms"][: len(t["answer_ids"])]
    return (tok[-1] - tok[0]) / (len(tok) - 1)


def case_block(t, ref):
    """Print one variant x case result, with the video's numbers beside it."""
    r = t["runs"][t["median_run"]]
    ttft = r["token_ms"][0] if r["token_ms"] else None
    print(f"    {bold(NAMES.get(t['variant'], t['variant']).ljust(11))} total {bold(fms(t['wall_ms_median']))}"
          + (f"   video {fms(ref['wall_ms_median'])}  Δ {delta(t['wall_ms_median'], ref['wall_ms_median'])}" if ref else "")
          + dim(f"   (min {t['wall_ms_min']:.1f}, max {t['wall_ms_max']:.1f} ms over {len(t['runs'])} runs)"))
    rr = ref["runs"][ref["median_run"]] if ref else None
    parts = [f"first token {ttft:.1f} ms" + (f" (video {rr['token_ms'][0]:.1f})" if rr else ""),
             f"{len(t['answer_ids'])} tokens" + (f" (video {len(ref['answer_ids'])})" if ref else "")]
    if len(r["token_ms"]) > 2 and len(t["answer_ids"]) > 2:
        parts.append(f"{per_token(t):.2f} ms/token" + (f" (video {per_token(ref):.2f})" if ref else ""))
    parts.append(f"peak memory {t['peak_mem_gb']:.2f} GB" + (f" (video {ref['peak_mem_gb']:.2f})" if ref else ""))
    if t["energy_j_per_answer"] is not None:
        parts.append(f"energy {t['energy_j_per_answer']:.1f} J"
                     + (f" (video {ref['energy_j_per_answer']:.1f})" if ref and ref["energy_j_per_answer"] else ""))
    print("                " + dim(" · ".join(parts)))
    if ref is not None:
        same = t["answer_ids"] == ref["answer_ids"]
        tag = c("answer identical to the video, token for token", "32") if same else \
            c("answer differs from the video", "31")
        print(f"                {tag}" + ("" if t["same_tokens_all_runs"] else c("  (runs disagree!)", "31")))
    if ref and abs(t["wall_ms_median"] / ref["wall_ms_median"] - 1) > 0.05:
        print("                " + c(CPU_NOTE, "33"))
    print(f"                {c('»', '36')} {t['text']!r}" if len(t["text"]) < 300 else
          f"                {c('»', '36')} {t['text'][:300]!r}…")


def summary(order, live, refs):
    print()
    print(bold("=" * 100))
    print(bold("  SUMMARY: time to the complete answer, live now vs in the video"))
    print(bold("=" * 100))
    vs = [v for v in VARIANTS.split(",") if any((v, k) in live for k in order)]
    print("  " + "case".ljust(26) + "".join(bold(NAMES[v].ljust(36)) for v in vs))
    print("  " + " " * 26 + dim("".join(("live ms / video ms      Δ" if refs else "live ms").ljust(36) for _ in vs)))
    n_same = n_all = 0
    for k in order:
        row = "  " + TITLES.get(k, k)[:25].ljust(26)
        for v in vs:
            t, ref = live.get((v, k)), refs.get((v, k))
            if t is None:
                row += " " * 36
            elif ref is None:
                row += f"{t['wall_ms_median']:8.1f}".ljust(36)
            else:
                row += f"{t['wall_ms_median']:8.1f} / {ref['wall_ms_median']:8.1f}  " \
                    + delta(t["wall_ms_median"], ref["wall_ms_median"]) + " " * 8
                n_all += 1
                n_same += t["answer_ids"] == ref["answer_ids"]
        print(row)
    base = "baseline"
    for v in vs:
        if v == base:
            continue
        sp = [live[(base, k)]["wall_ms_median"] / live[(v, k)]["wall_ms_median"] for k in order
              if (base, k) in live and (v, k) in live]
        spr = [refs[(base, k)]["wall_ms_median"] / refs[(v, k)]["wall_ms_median"] for k in order
               if (base, k) in refs and (v, k) in refs and (base, k) in live]
        if sp:
            g = statistics.geometric_mean(sp)
            print(f"  {NAMES[v]} vs HF: {bold(f'{g:.2f}×')} faster (geometric mean, live)"
                  + (f"   video: {statistics.geometric_mean(spr):.2f}×" if spr else ""))
    if n_all:
        col = "32" if n_same == n_all else "33"
        print(f"  answers identical to the video: {c(f'{n_same} / {n_all}', col)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(REPO_ROOT / "demo" / "video.json"))
    ap.add_argument("--only", default="", help="comma-separated case keys (default: the video's cases)")
    ap.add_argument("--variants", default=VARIANTS)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--energy-s", type=float, default=2.0, help="0 = skip the energy loop")
    ap.add_argument("--quick", action="store_true", help="2 warm-up + 3 timed runs, no energy loop")
    ap.add_argument("--image", default="", help="your own image: run it through all three (no video reference)")
    ap.add_argument("--prompt", default="Describe this image in detail.")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()
    if args.quick:
        args.warmup, args.repeats, args.energy_s = 2, 3, 0.0

    video = json.loads(Path(args.video).read_text())
    if args.image:
        raw = Path(args.image).read_bytes()
        cases = [dict(key="custom", title=Path(args.image).name, prompt=args.prompt, max_new=args.max_new)]
        images, refs, ref_dir, ref_env = {"custom": raw}, {}, None, {}
    else:
        ref_dir = REPO_ROOT / video["run"]
        ref_env = json.loads((ref_dir / "env.json").read_text())
        rcfg = json.loads((ref_dir / "config.json").read_text())
        by_key = {cc["key"]: cc for cc in rcfg["cases"]}
        keys = args.only.split(",") if args.only else video["cases"]
        cases = [by_key[k] for k in keys]
        refs = {(t["variant"], t["key"]): t for t in json.loads((ref_dir / "traces.json").read_text())}
        images = {cc["key"]: load_case_image(cc["src"])[0] for cc in cases}
    TITLES.update({cc["key"]: cc["title"] for cc in cases})

    header(ref_dir, ref_env)
    print(bold(f"  {len(cases)} case{'s' if len(cases) > 1 else ''}") + dim(
        f"  ·  {args.warmup} warm-up + {args.repeats} timed runs per model"
        + (f" + ≥{args.energy_s:g} s for energy" if args.energy_s > 0 else ", energy skipped")))
    for i, cc in enumerate(cases):
        w, h = Image.open(io.BytesIO(images[cc["key"]])).size
        print(f"   {i + 1}. {cc['title']:<30} {w}×{h} px   {cc['prompt']!r}")
    print()

    out = REPO_ROOT / "results" / "demo" / (f"{datetime.now():%Y%m%d-%H%M%S}_{gpu_short_name()}_live")
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(collect_env(), indent=2))
    (out / "config.json").write_text(json.dumps(dict(vars(args), cases=cases, reference=video.get("run")), indent=2))
    meter = EnergyMeter() if args.energy_s > 0 else None
    live, traces = {}, []
    for v in args.variants.split(","):
        t0 = time.perf_counter()
        print(bold(c(f"── {NAMES.get(v, v)} ", "36") + c("─" * (80 - len(NAMES.get(v, v))), "36")) + dim("  loading…"),
              flush=True)
        runner = Runner(v)
        print(dim(f"   loaded in {time.perf_counter() - t0:.1f} s, weights in GPU memory {runner.model_mem:.2f} GB"))
        for i, cc in enumerate(cases):
            print(f"  {i + 1}. {cc['title']}", flush=True)
            t = runner.measure(cc, images[cc["key"]], args.warmup, args.repeats, args.energy_s, meter)
            live[(v, cc["key"])] = t
            traces.append(t)
            (out / "traces.json").write_text(json.dumps(traces))
            case_block(t, refs.get((v, cc["key"])))
        runner.close()
        print()
    summary([cc["key"] for cc in cases], live, refs)
    print(dim(f"  live results saved to {out.relative_to(REPO_ROOT)}"))


if __name__ == "__main__":
    main()
