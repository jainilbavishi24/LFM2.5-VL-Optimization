"""Aggregate an ncu metrics CSV (from profile_ncu.sh) per phase and kernel: time, DRAM traffic, bandwidth.

  python profiling/ncu/analyze_ncu.py profiling/reports/<run>/ncu_decode_metrics.csv [--steps N]

Achieved bandwidth = DRAM bytes / kernel time. Compare with the GPU's peak (T4 ~320 GB/s,
RTX PRO 5000 ~1.3 TB/s) to see how far each kernel is from the memory roof.
"""

import argparse
import csv
import re
from collections import defaultdict


def short(name: str) -> str:
    """Readable kernel name, e.g. 'std::enable_if<..>::type internal::gemvx::kernel<..>(..)' -> 'gemvx::kernel'."""
    prev = None
    while prev != name:  # drop template arguments, innermost first
        prev, name = name, re.sub(r"<[^<>]*>", "", name)
    name = name.split("(")[0].strip().split(" ")[-1]  # drop arguments and return type
    parts = [p for p in name.split("::") if p]
    if len(parts) >= 2 and parts[-1] in ("kernel", "Kernel", "run", "operator"):
        return "::".join(parts[-2:])
    return parts[-1] if parts else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--steps", type=float, default=None, help="divide totals by this (e.g. decode steps captured)")
    a = ap.parse_args()
    rows = list(csv.reader(open(a.csv)))
    hdr, units, data = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(hdr)}
    range_col = next(i for i, h in enumerate(hdr) if "Push/Pop_Range" in h)

    def num(r, key, scale=1.0):
        try:
            return float(r[col[key]].replace(",", "")) * scale
        except (KeyError, ValueError):
            return 0.0

    t_scale = {"ms": 1e-3, "us": 1e-6, "usecond": 1e-6, "msecond": 1e-3, "ns": 1e-9, "nsecond": 1e-9, "s": 1.0}
    unit_t = units[col["gpu__time_duration.sum"]]
    b_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}

    agg = defaultdict(lambda: dict(n=0, t=0.0, bytes=0.0, pct=[]))
    for r in data:
        m = re.search(r":([A-Za-z_]+):none", r[range_col])
        phase = m.group(1) if m else "?"
        key = (phase, short(r[col["Kernel Name"]]))
        t = num(r, "gpu__time_duration.sum", t_scale.get(unit_t, 1e-3))
        nbytes = sum(num(r, k, b_scale.get(units[col[k]], 1)) for k in ("dram__bytes_read.sum", "dram__bytes_write.sum")
                     if k in col)
        g = agg[key]
        g["n"] += 1
        g["t"] += t
        g["bytes"] += nbytes
        if "dram__throughput.avg.pct_of_peak_sustained_elapsed" in col:
            g["pct"].append(num(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed"))

    div = a.steps or 1.0
    tot_t = sum(g["t"] for g in agg.values())
    tot_b = sum(g["bytes"] for g in agg.values())
    print(f"{len(data)} kernels, {tot_t * 1e3 / div:.3f} ms, {tot_b / 1e6 / div:.1f} MB DRAM"
          f"{' per step' if a.steps else ''}; overall {tot_b / tot_t / 1e9 if tot_t else 0:.1f} GB/s\n")
    print(f"{'phase':<16}{'kernel':<42}{'n':>6}{'time ms':>10}{'%time':>7}{'MB':>9}{'GB/s':>8}{'%peakBW':>9}")
    for (phase, k), g in sorted(agg.items(), key=lambda kv: -kv[1]["t"]):
        pct = sum(g["pct"]) / len(g["pct"]) if g["pct"] else float("nan")
        print(f"{phase:<16}{k[:41]:<42}{g['n'] / div:>6.0f}{g['t'] * 1e3 / div:>10.3f}{100 * g['t'] / tot_t:>7.1f}"
              f"{g['bytes'] / 1e6 / div:>9.2f}{(g['bytes'] / g['t'] / 1e9) if g['t'] else 0:>8.1f}{pct:>9.1f}")


if __name__ == "__main__":
    main()
