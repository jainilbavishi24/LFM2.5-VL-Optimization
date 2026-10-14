"""Figures for the report / evaluation slides from committed result files (no GPU needed).

    python benchmarks/plot_results.py            # -> report/figures/*.png

Run directories are found by tag (latest Blackwell run wins), so re-running benchmarks updates the figures.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "report" / "figures"
GPU = "RTXPRO5000Blackwell"

VARIANTS = [  # (label, run tag, variant name in the CSV)
    ("HF eager", "final_base_r3", "baseline"),
    ("torch.compile", "o0b_compile_nohooks_full", "compile_static"),
    ("O1 CUDA graph", "o1_cudagraph_nohooks_full", "cudagraph"),
    ("O1-O4 fused (BF16)", "o2g_fused_gemv_nohooks_full", "fused_gemv"),
    ("ours: full BF16 stack", "final_stack_r3", "fused_gemv_pf_vfused"),
    ("ext.: INT8 weights", "final_w8_stack", "fused_w8_pf_vfused"),
]
EVAL = ROOT / "results" / "eval"
BENCH_LABELS = {"mmstar": "MMStar", "pope": "POPE", "ocrbench": "OCRBench", "chartqa": "ChartQA", "textvqa": "TextVQA"}


def run_dir(tag):
    runs = sorted(RAW.glob(f"*_{GPU}_{tag}"))
    return runs[-1] if runs else None


def rows(tag):
    d = run_dir(tag)
    if d is None or not (d / "summary.csv").exists():
        return []
    return list(csv.DictReader(open(d / "summary.csv")))


def pick(rs, variant, workload, gen=128, bs=1):
    for r in rs:
        if r["variant"] == variant and r["workload"] == workload and int(r["gen_len"]) == gen and int(r["batch_size"]) == bs:
            return r
    return None


def references():
    ref = {}
    d = run_dir("ref_vllm")
    if d and (d / "runs.json").exists():
        rs = json.load(open(d / "runs.json"))["rows"]
        w = {(r["workload"], r["gen"]): r["wall_ms_median"] for r in rs}
        for wl in ("text", "img_fhd"):
            if (wl, 1) in w and (wl, 128) in w:
                ref.setdefault("vLLM 0.31", {})[wl] = (w[(wl, 128)] - w[(wl, 1)]) / 127
    d = run_dir("ref_llamacpp")
    for q, label in (("BF16", "llama.cpp BF16"), ("Q8_0", "llama.cpp Q8_0")):
        f = d / f"bench_{q}.json" if d else None
        if f and f.exists():
            for r in json.load(open(f)):
                if r["n_prompt"] == 0 and r["n_gen"] == 128:
                    ref.setdefault(label, {})["text"] = 1000.0 / r["avg_ts"]
    return ref


def fig_tpot():
    labels, text, fhd = [], [], []
    for label, tag, var in VARIANTS:
        rs = rows(tag)
        a, b = pick(rs, var, "text"), pick(rs, var, "img_fhd")
        if a and b:
            labels.append(label)
            text.append(float(a["tpot_ms"]))
            fhd.append(float(b["tpot_ms"]))
    for label, v in references().items():
        labels.append(label)
        text.append(v.get("text", float("nan")))
        fhd.append(v.get("img_fhd", float("nan")))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], text, 0.4, label="text prompt")
    ax.bar([i + 0.2 for i in x], fhd, 0.4, label="1080p image (2.3k-token context)")
    for i, (t, f) in enumerate(zip(text, fhd)):
        if t == t:
            ax.text(i - 0.2, t + 0.04, f"{t:.2f}", ha="center", fontsize=7)
    ax.axhline(1.9, ls="--", c="gray", lw=0.8)
    ax.text(len(labels) - 0.5, 1.93, "BF16 weight-read floor (1.23 TB/s)", ha="right", fontsize=7, color="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("time per output token (ms)")
    ax.set_title("LFM2.5-VL-1.6B decode, batch 1, RTX PRO 5000 Blackwell")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "decode_tpot.png", dpi=160)


def fig_energy():
    labels, jt = [], []
    for label, tag, var in VARIANTS:
        r = pick(rows(tag), var, "img_small")
        if r and r["decode_j_per_tok"]:
            labels.append(label)
            jt.append(float(r["decode_j_per_tok"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, jt, color="tab:green")
    for i, v in enumerate(jt):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("GPU energy per output token (J)")
    ax.set_title("Decode energy (img_small, batch 1, NVML)")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "decode_energy.png", dpi=160)


def fig_ttft():
    """First token including preprocessing: HF (CPU preprocessing) vs our full BF16 stack (GPU preprocessing)."""
    wls = ["text", "img_small", "img_hd", "img_fhd"]
    runs = [("HF eager", "final_base_r", "baseline"), ("ours (BF16)", "final_stack_r", "fused_gemv_pf_vfused")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (name, tag, var) in enumerate(runs):
        for i, wl in enumerate(wls):
            pre, tt = [], []
            for d in sorted(RAW.glob(f"*_{GPU}_{tag}*")):
                r = pick(list(csv.DictReader(open(d / "summary.csv"))), var, wl, gen=1) if (d / "summary.csv").exists() else None
                if r:
                    pre.append(float(r["preprocess_ms"]))
                    tt.append(float(r["ttft_ms"]))
            if not tt:
                continue
            p, t = sum(pre) / len(pre), sum(tt) / len(tt)
            xpos = i + (j - 0.5) * 0.38
            ax.bar(xpos, p, 0.36, color="tab:gray", label="image preprocessing" if i == 0 and j == 0 else None)
            ax.bar(xpos, t, 0.36, bottom=p, color="tab:blue" if j == 0 else "tab:green",
                   label=f"{name}: vision + prefill" if i == 0 else None)
            ax.text(xpos, p + t + 2, f"{p + t:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels(["text", "640x480 (1 tile)", "1280x720 (3 tiles)", "1920x1080 (9 tiles)"])
    ax.set_ylabel("time to first token incl. preprocessing (ms)")
    ax.set_title("First token: left = HF eager (CPU preprocessing), right = ours (all BF16)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_breakdown.png", dpi=160)


def fig_quality():
    """Benchmark scores: baseline vs our BF16 stack (and the INT8 extension), from results/eval/*/summary.jsonl."""
    scores = {}
    for f in sorted(EVAL.glob(f"*_{GPU}_*/summary.jsonl")):
        for line in open(f):
            r = json.loads(line)
            scores[(r["variant"].split("+")[0], r["bench"])] = 100 * r["score"]
    series = [("HF eager", "baseline", "tab:blue"), ("ours: full BF16 stack", "fused_gemv_pf_vfused", "tab:green"),
              ("ext.: INT8 weights", "fused_w8_pf_vfused", "tab:olive")]
    series = [s for s in series if any((s[1], b) in scores for b in BENCH_LABELS)]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(series)
    for j, (label, var, color) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * w for i in range(len(BENCH_LABELS))]
        ys = [scores.get((var, b), float("nan")) for b in BENCH_LABELS]
        ax.bar(xs, ys, w, label=label, color=color)
        for x, y in zip(xs, ys):
            if y == y:
                ax.text(x, y + 0.8, f"{y:.1f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(BENCH_LABELS)))
    ax.set_xticklabels(list(BENCH_LABELS.values()))
    ax.set_ylabel("score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Output quality is unchanged (all differences within ±0.3 pp, McNemar p > 0.3)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "quality.png", dpi=160)


def fig_batch():
    rs = rows("batch_sweep_final")
    fig, ax = plt.subplots(figsize=(7, 4))
    rs = rs + rows("batch_stack")
    for var, label in (("baseline", "HF eager"), ("fused_gemv", "O1-O4 fused (BF16)"),
                       ("fused_gemv_pf_vfused", "ours: full BF16 stack"), ("fused_w8", "ext.: INT8 weights")):
        pts = sorted((int(r["batch_size"]), float(r["decode_tok_s"])) for r in rs
                     if r["variant"] == var and int(r["gen_len"]) == 128 and r["decode_tok_s"])
        if pts:
            ax.plot(*zip(*pts), marker="o", label=label)
    ax.set_xlabel("batch size")
    ax.set_ylabel("decode throughput (tokens/s)")
    ax.set_title("Batched decode (img_small, 128 tokens)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "batch_throughput.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_tpot, fig_energy, fig_ttft, fig_batch, fig_quality):
        f()
    print("figures ->", OUT)
