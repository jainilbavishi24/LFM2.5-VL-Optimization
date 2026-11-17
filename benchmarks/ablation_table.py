"""Ablation table per GPU: one optimization added at a time (scripts/ablation.sh runs; the Blackwell rows come from the
runs of the first optimization night). Time per output token (text), first token incl. preprocessing (VGA, 1080p),
energy per token. Means over all runs of a tag.

    python benchmarks/ablation_table.py            # markdown to stdout
"""

import csv
import glob
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"

STEPS = ["HF transformers (baseline)", "+ O1 static decode + CUDA graph", "+ O2–O4 fused kernels (cuBLAS GEMV)",
         "+ our GEMV (O2d epilogues)", "+ O3b fused LM prefill", "+ O7d fused vision encoder",
         "+ P1 GPU preprocessing (full BF16 stack)", "+ vision-encoder CUDA graph"]
VARS = ["baseline", "cudagraph", "fused", "fused_gemv", "fused_gemv_pf", "fused_gemv_pf_vfused", "fused_gemv_pf_vfused",
        "fused_gemv_pf_vfused"]


def ablation_runs(gpu, tag):
    suffix = ["_ablation"] * 6 + ["_ablation_prep", "_ablation_vgraph"]
    return [(f"{tag}{s}", v) for s, v in zip(suffix, VARS)]


GPUS = {
    "RTX PRO 5000 Blackwell": ("RTXPRO5000Blackwell", [("final_base_r*", "baseline"), ("o1_cudagraph_nohooks_full", "cudagraph"),
                                                      ("o2_fused_nohooks_full", "fused"), ("o2g_fused_gemv_nohooks_full", "fused_gemv"),
                                                      None, None, ("final_stack_r*", "fused_gemv_pf_vfused"), None]),
    "L40S": ("L40S", ablation_runs("L40S", "l40")),
    "Jetson Orin NX (40 W)": ("Orin", ablation_runs("Orin", "jetson")),
}


def value(gpu, tag, variant, wl, gen, field):
    vals = []
    for d in sorted(glob.glob(str(RAW / f"*_{gpu}_{tag}"))):
        p = Path(d) / "summary.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if r["variant"] == variant and r["workload"] == wl and r["gen_len"] == str(gen) and r["batch_size"] == "1":
                try:
                    v = float(r[field])
                except ValueError:
                    continue
                if field == "ttft_ms":
                    v += float(r["preprocess_ms"] or 0)
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


def fmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def main():
    for name, (gpu, runs) in GPUS.items():
        print(f"\n**{name}**\n")
        print("| step | TPOT text (ms) | first token VGA (ms) | first token 1080p (ms) | J/token |")
        print("|---|---|---|---|---|")
        base = None
        for step, run in zip(STEPS, runs):
            if run is None:
                print(f"| {step} | – | – | – | – |")
                continue
            tag, var = run
            tp = value(gpu, tag, var, "text", 128, "tpot_ms")
            base = tp if base is None else base
            sp = f" ({base / tp:.2f}×)" if tp and base and step != STEPS[0] else ""
            print(f"| {step} | {fmt(tp, 2)}{sp} | {fmt(value(gpu, tag, var, 'img_small', 1, 'ttft_ms'))} | "
                  f"{fmt(value(gpu, tag, var, 'img_fhd', 1, 'ttft_ms'), 0)} | "
                  f"{fmt(value(gpu, tag, var, 'text', 128, 'decode_j_per_tok'), 3)} |")


if __name__ == "__main__":
    main()
