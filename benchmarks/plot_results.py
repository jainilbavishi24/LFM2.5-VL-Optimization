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
    ("HF eager", "o0a_ctrl_nohooks_full", "baseline"),
    ("torch.compile", "o0b_compile_nohooks_full", "compile_static"),
    ("O1 CUDA graph", "o1_cudagraph_nohooks_full", "cudagraph"),
    ("O1-O4 fused (BF16)", "o2g_fused_gemv_nohooks_full", "fused_gemv"),
    ("O1-O6 INT8 weights", "o6_w8_nohooks_full", "fused_w8"),
    ("O1-O6 INT4 (lossy)", "o6b_w4_nohooks_full", "fused_w4"),
]


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
    base = rows("baseline_full")
    if not base:
        base = rows("o0a_ctrl_full")
    best = rows("o7_w8_vcudnn_full")
    wls = ["img_small", "img_hd", "img_fhd"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (name, rs, var) in enumerate([("HF eager", base, "baseline"), ("+ cuDNN vision attention", best, "fused_w8_vcudnn")]):
        for i, wl in enumerate(wls):
            r = pick(rs, var, wl, gen=1)
            if not r:
                continue
            vis, pre, tt = float(r["vision_ms"] or 0), float(r["lm_prefill_ms"] or 0), float(r["ttft_ms"])
            xpos = i + (j - 0.5) * 0.38
            ax.bar(xpos, vis, 0.36, color="tab:blue", label="vision tower" if i == 0 and j == 0 else None)
            ax.bar(xpos, pre, 0.36, bottom=vis, color="tab:orange", label="LM prefill" if i == 0 and j == 0 else None)
            ax.bar(xpos, tt - vis - pre, 0.36, bottom=vis + pre, color="tab:gray", label="preprocess/other" if i == 0 and j == 0 else None)
            ax.text(xpos, tt + 1.5, f"{tt:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(wls)))
    ax.set_xticklabels(["640x480 (1 tile)", "1280x720 (3 tiles)", "1920x1080 (9 tiles)"])
    ax.set_ylabel("time to first token (ms)")
    ax.set_title("TTFT: left = HF eager, right = with cuDNN vision attention")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ttft_breakdown.png", dpi=160)


def fig_batch():
    rs = rows("batch_sweep_final")
    fig, ax = plt.subplots(figsize=(7, 4))
    for var, label in (("baseline", "HF eager"), ("fused_gemv", "O1-O4 fused (BF16)"), ("fused_w8", "O1-O6 INT8")):
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
    for f in (fig_tpot, fig_energy, fig_ttft, fig_batch):
        f()
    print("figures ->", OUT)
