"""Evaluation 2 deck (same look as presentations/eval1/PAA_Eval1_v2_Edge_VLM.pptx: Cambria / Calibri, teal + orange).
Numbers are read from the committed result files where they exist, so re-running after new measurements updates it.

    python3 presentations/eval2/build_deck.py      # -> presentations/eval2/PAA_Eval2_Working_Prototype.pptx
"""

import csv
import glob
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT = Path(__file__).resolve().parent / "PAA_Eval2_Working_Prototype.pptx"

INK, TEAL, ORANGE, BLUE, GOLD, GRAY, GREEN = "1B2428", "0F4C5C", "EB6834", "2A78D6", "C98A1B", "5C6B73", "1B9E70"
TINT, WHITE, PURPLE, MUTED = "EDF3F4", "FFFFFF", "7A5AC8", "6B7A80"
HEAD, BODY = "Cambria", "Calibri"
FOOT = "Optimizing a VLM for Edge Devices · CS3.406 Evaluation 2"


def rgb(h):
    return RGBColor.from_string(h)


# ----------------------------------------------------------------------------------------------- data
def summary(tag, gpu="Orin"):
    """{(variant, workload, gen): row} from the latest run with this tag."""
    runs = sorted(glob.glob(str(RAW / f"*_{gpu}_{tag}")))
    if not runs:
        return {}
    rows = list(csv.DictReader(open(Path(runs[-1]) / "summary.csv")))
    return {(r["variant"], r["workload"], int(r["gen_len"])): r for r in rows if r["batch_size"] == "1"}


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


J_BASE, J_OURS, J_BP = summary("jetson_base"), summary("jetson_ours"), summary("jetson_final") or summary("jetson_ours_bp")
M_BASE, M_OURS = summary("jetson_maxn_base"), summary("jetson_maxn_ours")   # after sudo nvpmodel -m 0 + jetson_clocks


def maxn(variant, field="tpot_ms", wl="text", gen=128):
    r = (M_BASE if variant == "baseline" else M_OURS).get((variant, wl, gen))
    return f(r, field) if r else None


def jet(variant, field, wl="text", gen=128, after=False):
    src = J_BP if (after and (variant, wl, gen) in J_BP) else (J_BASE if variant == "baseline" else J_OURS)
    r = src.get((variant, wl, gen))
    if r is None:
        return None
    v = f(r, field)
    if field == "ttft_ms":
        v += f(r, "preprocess_ms") or 0.0
    return v


def eval_subset():
    """Jetson 5-benchmark subset: {variant: {bench: score}} from the latest *_jetson_subset run."""
    runs = sorted(glob.glob(str(ROOT / "results" / "eval" / "*_jetson_subset")))[-1:]
    runs += sorted(glob.glob(str(ROOT / "results" / "eval" / "*_jetson_subset_q")))
    out = {}
    for run in runs:
        for line in open(Path(run) / "summary.jsonl"):
            s = json.loads(line)
            out.setdefault(s["variant"].split("+")[0], {})[s["bench"]] = 100 * s["score"]
    return out


def gemv_table():
    """GEMV microbenchmark before/after the byte_perm conversion, from the Jetson run log (if present)."""
    log = RAW / "jetson_after_log.txt"
    if not log.exists():
        return []
    txt = log.read_text()
    parts = re.split(r"\[jetson\] [\d:]+ gemv_(old|new) start", txt)
    res = {}
    for i in range(1, len(parts) - 1, 2):
        tag, body = parts[i], parts[i + 1]
        for m in re.finditer(r"^(\S[\w /|]+?)\s+(\d+)x(\d+)\s+\|.*?\|.*?\|\s*([\d.]+)us.*?int4\s+([\d.]+)us", body, re.M):
            res.setdefault(m.group(1).strip(), {})[tag] = (float(m.group(4)), float(m.group(5)))
    rows2 = {}
    rl = RAW / "jetson_gemv_w4_rows.txt"
    if rl.exists():
        for m in re.finditer(r"^(\S[\w /|]+?)\s+\d+x\d+\s+rows 1/2/4:\s+[\d.]+us /\s+([\d.]+)us", rl.read_text(), re.M):
            rows2[m.group(1).strip()] = float(m.group(2))
    return [(k, v["old"][1], v["new"][1], rows2.get(k)) for k, v in res.items() if "old" in v and "new" in v]


# ----------------------------------------------------------------------------------------------- drawing helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


def bullet(p, color=ORANGE):
    """Real bullet on a paragraph (hanging indent, colored dot)."""
    from pptx.oxml.ns import qn
    from lxml import etree
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.22)))
    pPr.set("indent", str(-Inches(0.22)))
    for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", color)
    etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(pPr, qn("a:buChar")).set("char", "•")


def text(sl, x, y, w, h, s, size=13, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         italic=False, bullets=False):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = line if isinstance(line, list) else [(line, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", italic)
            r.font.name = o.get("font", font)
            r.font.color.rgb = rgb(o.get("color", color))
        p.space_after = Pt(6 if bullets else 4)
        if bullets:
            bullet(p)
    return tb


def box(sl, x, y, w, h, fill=TINT, line=None, rounded=True):
    shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                              Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def slide(eyebrow, title, dark=False):
    sl = prs.slides.add_slide(BLANK)
    N[0] += 1
    if dark:
        bg = sl.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(INK)
    text(sl, 0.6, 0.32, 12.1, 0.32, eyebrow.upper(), size=12, bold=True, color=ORANGE)
    text(sl, 0.6, 0.62, 12.1, 0.75, title, size=32, bold=True, font=HEAD, color=WHITE if dark else TEAL)
    if not dark:
        text(sl, 0.6, 7.05, 8, 0.3, FOOT, size=10, color=MUTED)
        text(sl, 12.2, 7.05, 0.55, 0.3, str(N[0]), size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    return sl


def stat(sl, x, y, w, big, label, color=ORANGE, sub=None, dark=False):
    text(sl, x, y, w, 0.65, big, size=30, bold=True, font=HEAD, color=color)
    text(sl, x, y + 0.68, w, 0.6, [label] + ([sub] if sub else []), size=12, color=WHITE if dark else INK)


def table(sl, x, y, w, rows, widths, size=12, height=0.36, bold_first_col=False, highlight_col=None):
    t = sl.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(height * len(rows))).table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(height)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(TEAL if i == 0 else (TINT if (highlight_col == j) else WHITE))
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = BODY
            r.font.bold = i == 0 or (bold_first_col and j == 0)
            r.font.color.rgb = rgb(WHITE if i == 0 else INK)
    return t


def bars(sl, x, y, w, h, cats, series, colors, title=None, fmt='0.0', point_colors=None, legend=False, size=11):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    ch = sl.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_legend = legend
    if legend:
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)
        ch.legend.font.name = BODY
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title
        tp = ch.chart_title.text_frame.paragraphs[0]
        tp.runs[0].font.size = Pt(12)
        tp.runs[0].font.bold = True
        tp.runs[0].font.name = BODY
        tp.runs[0].font.color.rgb = rgb(INK)
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E3E8EA")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.name = BODY
    ca.tick_labels.font.color.rgb = rgb(INK)
    ca.format.line.color.rgb = rgb("C9D1D4")
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
        s.data_labels.show_value = True
        s.data_labels.number_format = fmt
        s.data_labels.number_format_is_linked = False
        s.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        s.data_labels.font.size = Pt(size)
        s.data_labels.font.name = BODY
        s.data_labels.font.color.rgb = rgb(INK)
    if point_colors:
        for i, c in enumerate(point_colors):
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(c)
    return ch


def notes(sl, s):
    sl.notes_slide.notes_text_frame.text = s


def fmt_ms(v):
    return "–" if v is None else (f"{v:.1f}" if v < 100 else f"{v:.0f}")


# ----------------------------------------------------------------------------------------------- numbers
hf_tpot = jet("baseline", "tpot_ms")
bf_tpot = jet("fused_gemv_pf_vfused", "tpot_ms")
w8_tpot = jet("fused_w8_pf_vfused", "tpot_ms", after=True)
w8a_tpot = jet("fused_w8all_pf_vfused", "tpot_ms", after=True)
w4_tpot = jet("fused_w4gptq_pf_vfused", "tpot_ms", after=True)
w4_before = jet("fused_w4gptq_pf_vfused", "tpot_ms")
w8_before = jet("fused_w8_pf_vfused", "tpot_ms")
hf_j, bf_j = jet("baseline", "decode_j_per_tok"), jet("fused_gemv_pf_vfused", "decode_j_per_tok")
w8_j = jet("fused_w8_pf_vfused", "decode_j_per_tok", after=True)
w4_j = jet("fused_w4gptq_pf_vfused", "decode_j_per_tok", after=True)
mem = {v: jet(v, "peak_mem_gb", "img_fhd") for v in ("baseline", "fused_gemv_pf_vfused", "fused_w8all_pf_vfused")}
ttft = {v: [jet(v, "ttft_ms", wl, 1) for wl in ("text", "img_small", "img_fhd")]
        for v in ("baseline", "fused_gemv_pf_vfused", "fused_w8all_pf_vfused")}
EV = eval_subset()


def demo_summary():
    """Geometric-mean speed-up and energy share vs HF over the Jetson demo cases (demo/video_jetson.json)."""
    import math
    cfg = json.loads((ROOT / "demo" / "video_jetson.json").read_text())
    tr = json.loads((ROOT / cfg["run"] / "traces.json").read_text())
    by = {(x["variant"], x["key"]): x for x in tr}
    out = {}
    for v in ("fused_gemv_pf_vfused+gpuprep", "fused_w8all_pf_vfused+gpuprep"):
        sp = [by[("baseline", k)]["wall_ms_median"] / by[(v, k)]["wall_ms_median"] for k in cfg["cases"]]
        en = [by[(v, k)]["energy_j_per_answer"] / by[("baseline", k)]["energy_j_per_answer"] for k in cfg["cases"]]
        out[v] = (math.exp(sum(map(math.log, sp)) / len(sp)), 100 * math.exp(sum(map(math.log, en)) / len(en)))  # as the video
    return out


DEMO = demo_summary()
GEMV = gemv_table()

# ----------------------------------------------------------------------------------------------- 1 title
sl = prs.slides.add_slide(BLANK)
N[0] += 1
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = rgb(INK)
text(sl, 0.6, 0.55, 12, 0.3, "CS3.406 PROGRAMMING AI ACCELERATORS · EVALUATION 2", size=12, bold=True, color=ORANGE)
text(sl, 0.6, 1.1, 8.9, 1.4, ["Optimizing a Vision-Language Model", "for Edge Devices"], size=32, bold=True, font=HEAD,
     color=WHITE)
text(sl, 0.6, 2.75, 8.9, 0.5, "Working prototype: LiquidAI LFM2.5-VL-1.6B, now running on a Jetson Orin NX 16GB",
     size=18, color="C9D6DA")
text(sl, 0.6, 3.75, 8.4, 1.0, [[("Key question: ", {"bold": True, "color": ORANGE}),
                                 ("does our approach actually work? Same CUDA optimizations, measured against the "
                                  "standard Hugging Face baseline on five GPUs, from a datacenter Blackwell to a 40 W "
                                  "Jetson, with the answers checked on standard benchmarks.", {"color": "E6EEF0"})]],
     size=15)
for i, (n, r) in enumerate([("Jainil Bavishi", "2023102045"), ("Gautam Gandhi", "2023102059"),
                            ("Bibek Singh Dhody", "2023101054")]):
    text(sl, 0.6 + 3.0 * i, 5.55, 2.8, 0.6, [[(n, {"bold": True, "color": WHITE})], [(r, {"size": 11, "color": "9FB1B7"})]],
         size=14)
text(sl, 0.6, 6.6, 9, 0.35, [[("Repository ", {"bold": True, "color": ORANGE}),
                              ("github.com/jainilbavishi24/LFM2.5-VL-Optimization", {"color": "C9D6DA"})]], size=12)
for i, (big, lab) in enumerate([("1.6–2.0×", "faster per word on the Jetson (BF16, MAXN / 40 W)"),
                                ("3–4×", "with INT8 weights (extension)"),
                                ("±0.3 pts", "benchmark accuracy")]):
    box(sl, 9.75, 1.15 + 1.45 * i, 3.0, 1.25, fill="24323A")
    text(sl, 9.95, 1.25 + 1.45 * i, 2.7, 0.6, big, size=30, bold=True, font=HEAD, color=ORANGE)
    text(sl, 9.95, 1.85 + 1.45 * i, 2.7, 0.45, lab, size=12, color="E6EEF0")
notes(sl, "Eval 1 showed the baseline and the bottleneck. Today: the prototype works, on the real edge board too.")

# ----------------------------------------------------------------------------------------------- 2 since eval 1
sl = slide("Since Evaluation 1", "From Plan to Working Prototype")
items = [
    ("Edge board running", "Jetson Orin NX 16GB (JetPack 7.2): all our kernels pass their tests on sm_87; energy from the "
                           "board's power sensor", TEAL),
    ("Same code, five GPUs", "RTX PRO 5000 Blackwell, L40S, T4, DGX Spark (GB10), Jetson Orin NX: the speed-up holds on "
                             "every one", BLUE),
    ("Quality proven", "5 standard benchmarks (19,000 questions), COCO captions vs human references, object hallucination:"
                       " no significant difference", GREEN),
    ("Memory for the edge", "INT8 everywhere: 2.1 GB peak instead of 3.3 GB (HF); fits small Jetsons", PURPLE),
    ("4-bit without training", "GPTQ INT4: plain rounding loses accuracy, GPTQ keeps it; no retraining needed", GOLD),
    ("Profiling on the board", "found that the Orin's 8 SMs make weight conversion, not memory, the INT4 limit; fixed in "
                               "the GEMV kernel", ORANGE),
]
for i, (h, d, c) in enumerate(items):
    x, y = 0.6 + (i % 3) * 4.1, 1.65 + (i // 3) * 2.55
    box(sl, x, y, 3.85, 2.25)
    circ = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.25), Inches(y + 0.25), Inches(0.42), Inches(0.42))
    circ.fill.solid()
    circ.fill.fore_color.rgb = rgb(c)
    circ.line.fill.background()
    text(sl, x + 0.25, y + 0.29, 0.42, 0.34, str(i + 1), size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    text(sl, x + 0.85, y + 0.28, 2.85, 0.4, h, size=15, bold=True, color=INK)
    text(sl, x + 0.25, y + 0.85, 3.4, 1.3, d, size=13, color=INK)
notes(sl, "Six things that are new since Eval 1. Everything is in the repo with its raw results and the experiment log.")

# ----------------------------------------------------------------------------------------------- 3 the prototype
sl = slide("Implementation", "The Prototype: One Fix per Measured Bottleneck")
rows = [["Measured bottleneck (Eval 1)", "Our fix (code)", "Result (RTX PRO 5000)"],
        ["703 kernel launches per word, GPU idle 55 %", "static decode loop replayed as one CUDA graph (runtime.py)",
         "3.83 → 2.91 ms/word"],
        ["~600 tiny norm / conv / RoPE kernels", "fused CUDA kernels, merged projections (lfm2_decode.cu)",
         "703 → 123 kernels; 2.07 ms"],
        ["weight-streaming GEMVs at 70 % of DRAM", "our GEMV: 4 loads in flight, epilogue fusion, PDL",
         "1.09× the memory floor"],
        ["CPU image preprocessing 25–63 ms", "2 CUDA kernels, bit-exact with HF (image_prep.cu)", "→ 3.9 ms"],
        ["padded vision attention, 24 % occupancy", "padding-free flash attention, fused LN / GELU (vision.py)",
         "77 → 54 ms (1080p)"],
        ["prompt through eager PyTorch", "fused prefill into the decode buffers", "37 → 31 ms (1080p)"]]
table(sl, 0.6, 1.6, 12.1, rows, [4.1, 5.0, 3.0], size=13, height=0.6)
text(sl, 0.6, 6.05, 12.1, 0.8, [[("All lossless BF16. ", {"bold": True, "color": TEAL}),
                                 ("Every kernel has a unit test against PyTorch (109/109, mostly bit-exact); the whole "
                                  "runtime is checked token by token against Hugging Face.", {})]], size=13)
notes(sl, "Recap of what we built; the numbers here are from the datacenter Blackwell. Next: the Jetson.")

# ----------------------------------------------------------------------------------------------- 4 jetson setup
sl = slide("Edge deployment · Jetson Orin NX 16GB", "On the Edge: the Same Code on a 40 W Board")
specs = [("8 SMs", "Ampere GPU, sm_87"), ("102 GB/s", "LPDDR5, shared by CPU + GPU"), ("15 GB", "memory for everything"),
         ("40 W", "power mode (nvpmodel)")]
for i, (big, lab) in enumerate(specs):
    box(sl, 0.6 + i * 3.05, 1.6, 2.8, 1.35)
    text(sl, 0.8 + i * 3.05, 1.7, 2.5, 0.6, big, size=28, bold=True, font=HEAD, color=TEAL)
    text(sl, 0.8 + i * 3.05, 2.35, 2.5, 0.5, lab, size=13)
text(sl, 0.6, 3.25, 6.0, 0.4, "What it took", size=16, bold=True, color=INK)
text(sl, 0.6, 3.7, 6.0, 3.0, [
    "JetPack 7.2 / CUDA 13.2; PyTorch 2.11 aarch64 (its sm_80 code runs on sm_87)",
    "no pip and no pypi.org on the board: uv + the Jetson AI Lab mirror",
    "our CUDA kernels JIT-compiled for sm_87, no source change",
    "no NVML energy counter: we sample the INA3221 VDD_IN rail (whole module) every 5 ms and integrate it",
], size=13, bullets=True)
text(sl, 6.9, 3.25, 5.8, 0.4, "Correct on the board", size=16, bold=True, color=INK)
rows = [["Check", "Result"], ["kernel unit tests", "109 / 109"], ["INT8 GEMM tests", "36 / 36"],
        ["next word vs HF, BF16 (381 steps)", "377 / 381 (flips at near-ties)"],
        ["next word vs HF, INT8", "376 / 381"], ["image preprocessing", "≤ 1 pixel level (Arm CPU resize)"]]
table(sl, 6.9, 3.7, 5.8, rows, [3.3, 2.5], size=12, height=0.42)
notes(sl, "Preprocessing differs by one pixel level because PyTorch's CPU resize on Arm takes a different code path "
          "than on x86; our GPU kernels match the x86 HF output exactly.")

# ----------------------------------------------------------------------------------------------- 5 jetson results
sl = slide("Edge results · Jetson Orin NX 16GB", "Jetson: 1.6–2× Faster per Word, Same Weights")
labels = ["HF\ntransformers", "ours BF16", "ours INT8\n(decode)", "ours INT8\neverywhere", "ours INT4\nGPTQ"]
vals = [hf_tpot, bf_tpot, w8_tpot, w8a_tpot, w4_tpot]
VARS = ["baseline", "fused_gemv_pf_vfused", "fused_w8_pf_vfused", "fused_w8all_pf_vfused", "fused_w4gptq_pf_vfused"]
mvals = [maxn(v) for v in VARS]
series = [("40 W mode", [v or 0 for v in vals])] + ([("MAXN (max clocks)", [v or 0 for v in mvals])] if all(mvals) else [])
bars(sl, 0.5, 1.45, 7.4, 4.6, labels, series, [ORANGE, TEAL],
     title="Time per output word (ms, batch 1)", legend=len(series) > 1, size=12)
y = 1.6
for big, lab, sub, c in [
    (f"{mvals[0] / mvals[1]:.1f}–{hf_tpot / bf_tpot:.1f}×" if all(mvals) else f"{hf_tpot / bf_tpot:.2f}×",
     "faster per word, BF16 (lossless)", "MAXN / 40 W: ours 26 ms in both modes, HF 40.5 / 51.4 ms (CPU-bound)", ORANGE),
    (f"{mvals[0] / mvals[2]:.0f}–{hf_tpot / w8_tpot:.1f}×" if all(mvals) else f"{hf_tpot / w8_tpot:.1f}×",
     "with INT8 weights (extension)", f"{fmt_ms(w8_tpot)} ms; INT4 {fmt_ms(w4_tpot)} ms ({mvals[0] / w4_tpot:.1f}–{hf_tpot / w4_tpot:.1f}×)" if all(mvals) else "", GREEN),
    (f"−{100 * (1 - bf_j / hf_j):.0f} %" if hf_j and bf_j else "–", "energy per word, whole module",
     f"40 W: {hf_j:.2f} → {bf_j:.2f} J (INT8 {w8_j:.2f} J)" if hf_j and bf_j and w8_j else "", TEAL),
    (f"{mem['fused_w8all_pf_vfused']:.1f} GB" if mem["fused_w8all_pf_vfused"] else "–", "peak memory, INT8 everywhere",
     f"HF {mem['baseline']:.1f} GB · ours BF16 {mem['fused_gemv_pf_vfused']:.1f} GB" if mem["baseline"] else "", PURPLE),
]:
    text(sl, 8.3, y, 1.9, 0.6, big, size=26, bold=True, font=HEAD, color=c)
    text(sl, 10.25, y + 0.02, 2.5, 0.9, [[(lab, {"bold": True})], [(sub, {"size": 11, "color": GRAY})]], size=13)
    y += 1.18
text(sl, 0.6, 6.15, 12.1, 0.75, [
    [("Nsight Systems on the board: ", {"bold": True, "color": TEAL, "italic": False}),
     ("HF issues 22,644 kernel launches for a 32-word answer at 27 µs each (≈ 19 ms of CPU per word, GPU idle half "
      "the time); ours replays one CUDA graph per word (31 launches).", {"italic": False, "color": INK})],
    "Text prompt, 128 new words, median of 5 runs after 2 warm-ups. INT8 / INT4 change the weights and are reported "
    "separately; quality on the next slides."], size=11, color=GRAY, italic=True)
notes(sl, "Hugging Face spends about half of every word on Python and kernel-launch overhead on the Jetson's Arm CPU. "
          "Our BF16 runtime is at 1.15x the memory floor: 2.34 GB of weights per word at 102 GB/s = 22.9 ms. "
          "Power modes: our runtime does not depend on the CPU clock (26.0 vs 26.2 ms); HF gets 21 % faster in MAXN, "
          "so the speed-up is 1.6x in MAXN and 2.0x in the 40 W mode, where battery-limited devices run.")

# ----------------------------------------------------------------------------------------------- 6 jetson profiling
sl = slide("Profiling on the board", "The Edge Has a Different Bottleneck")
text(sl, 0.6, 1.55, 6.0, 0.4, "What we saw", size=16, bold=True)
text(sl, 0.6, 2.0, 6.0, 2.3, [
    [("INT4 weights are half the bytes of INT8, ", {}), ("yet INT4 was barely faster: ", {"bold": True}),
     (f"{fmt_ms(w4_before)} vs {fmt_ms(w8_before)} ms per word (floor ~6 ms).", {})],
    [("On a datacenter GPU INT4 is 1.65× faster than INT8; on the Orin's 8 SMs the GEMV is ", {}),
     ("compute-bound, not memory-bound.", {"bold": True})],
    [("The cause: every weight went through I2F (int → float), a quarter-rate instruction on Ampere: 1.17 G "
      "conversions per word ≈ 10 ms on 8 SMs.", {})],
], size=13, bullets=True)
text(sl, 0.6, 4.3, 6.0, 0.4, "The fix (lfm2_decode.cu)", size=16, bold=True)
text(sl, 0.6, 4.75, 6.0, 1.9, [
    "int → float without I2F: one byte permute puts the 4- or 8-bit value into the mantissa of 2^23, one FADD removes it",
    "on ≤ 16 SMs: two rows per warp, so each x load and conversion serves two weight rows",
    "both exact: output bit-identical, all 109 kernel tests unchanged",
], size=13, bullets=True)
if GEMV:
    rows = [["INT4 GEMV, µs", "before", "+ byte_perm", "+ 2 rows / warp"]]
    for name, old, new, r2 in GEMV:
        rows.append([name, f"{old:.0f}", f"{new:.0f}", "already 2" if name == "lm_head" else (f"{r2:.0f}" if r2 else "–")])
    table(sl, 7.0, 1.6, 5.7, rows, [1.7, 1.2, 1.3, 1.5], size=12, height=0.45, highlight_col=3)
else:
    box(sl, 7.0, 1.6, 5.7, 3.0)
    text(sl, 7.2, 2.6, 5.3, 1.0, "GEMV before / after: measurement pending", size=14, color=GRAY, align=PP_ALIGN.CENTER)
y0 = 1.6 + 0.45 * (len(GEMV) + 1) + 0.35 if GEMV else 4.9
for i, (big, lab) in enumerate([
    (f"{fmt_ms(w4_before)} → {fmt_ms(w4_tpot)} ms", "INT4 per word on the Jetson"),
    (f"{fmt_ms(w8_before)} → {fmt_ms(w8_tpot)} ms", "INT8 per word on the Jetson")]):
    text(sl, 7.0 + 2.9 * i, y0, 2.8, 0.5, big, size=20, bold=True, font=HEAD, color=ORANGE)
    text(sl, 7.0 + 2.9 * i, y0 + 0.5, 2.8, 0.4, lab, size=12)
notes(sl, "This is the profiling-driven step for the edge: the same kernel that is bandwidth-bound on a datacenter "
          "GPU is ALU-bound on 8 SMs. GEMV timings from profiling/bench_gemv.py at the model's real shapes.")

# ----------------------------------------------------------------------------------------------- 7 first token
sl = slide("First token on the Jetson", "First Word: 8 SMs Make It Compute-Bound")
rows = [["First word incl. image preprocessing", "Text only", "640×480 (1 tile)", "1920×1080 (9 tiles)"]]
for v, name in (("baseline", "HF transformers"), ("fused_gemv_pf_vfused", "ours BF16"), ("fused_w8all_pf_vfused", "ours INT8 everywhere")):
    rows.append([name] + [f"{fmt_ms(t)} ms" for t in ttft[v]])
if ttft["baseline"][0]:
    rows.append(["speed-up, ours BF16"] + [f"{a / b:.1f}×" for a, b in zip(ttft["baseline"], ttft["fused_gemv_pf_vfused"])])
table(sl, 0.6, 1.6, 12.1, rows, [4.6, 2.5, 2.5, 2.5], size=13, height=0.5)
text(sl, 0.6, 4.4, 5.8, 0.4, "Why the gain is smaller at 1080p", size=16, bold=True)
text(sl, 0.6, 4.85, 5.8, 1.9, [
    "a 1080p photo is 2,300 image tokens through a 27-layer vision encoder: large matrix multiplies",
    "on 8 SMs those are limited by arithmetic, not by overhead, so removing overhead helps less",
    "INT8 tensor cores gain only 5–29 % per matrix here, and the extra quantize kernels cost more",
], size=13, bullets=True)
text(sl, 6.9, 4.4, 5.8, 0.4, "Next for the first word", size=16, bold=True)
text(sl, 6.9, 4.85, 5.8, 1.9, [
    "CUDA graph for the vision encoder (launch-bound at one tile)",
    "MAXN power mode + locked clocks (needs sudo on the board)",
    "fewer image tiles for small screens: an accuracy trade-off we would measure, not assume",
], size=13, bullets=True)
notes(sl, "Short prompts get 2x; huge images are compute-bound on this small GPU.")

# ----------------------------------------------------------------------------------------------- 8 five GPUs
sl = slide("Generalization", "Same Code, Five GPUs: Faster Everywhere")
gpus = ["RTX PRO 5000\nBlackwell", "L40S\n(Ada)", "T4\n(Turing)", "DGX Spark\n(GB10)", "Jetson Orin NX\n(40 W mode)"]
hf = [3.70, 8.98, 21.6, 15.2, hf_tpot or 51.4]
ours = [2.07, 3.29, 9.03, 9.88, bf_tpot or 26.2]
bars(sl, 0.5, 1.45, 8.0, 5.1, gpus, [("HF transformers", hf), ("ours (BF16)", ours)], [BLUE, ORANGE],
     title="Time per output word (ms, batch 1, text prompt)", legend=True, size=12)
rows = [["GPU", "Speed-up", "vs memory floor"]]
for g, a, b, fl in zip(["Blackwell", "L40S", "T4", "DGX Spark", "Jetson 40 W"], hf, ours, [1.9, 2.7, 7.3, 8.6, 22.9]):
    rows.append([g, f"{a / b:.1f}×", f"{b / fl:.2f}×"])
table(sl, 8.8, 1.6, 3.9, rows, [1.5, 1.1, 1.3], size=13, height=0.48)
text(sl, 8.8, 4.75, 3.9, 1.6, "HF's time varies 14× for the same work, limited by each machine's CPU. Ours follows each "
                              "GPU's memory bandwidth: 1.1–1.2× the floor everywhere.", size=13)
notes(sl, "Floor = weight bytes per word divided by the practical DRAM bandwidth of each GPU.")

# ----------------------------------------------------------------------------------------------- 9 quality
sl = slide("Quality", "Same Answers: Benchmarks, Captions, Hallucination")
rows = [["Full sets, L40S (19,000 questions)", "HF", "ours BF16", "INT8 everywhere", "INT4 GPTQ"],
        ["MMStar (1500)", "49.20", "49.27", "49.13", "49.27"],
        ["POPE (9000)", "89.12", "89.10", "89.20", "89.10"],
        ["OCRBench (1000)", "79.80", "80.00", "79.60", "79.30"],
        ["ChartQA (2500)", "80.08", "80.00", "79.72", "80.04"],
        ["TextVQA (5000)", "81.16", "81.03", "80.98", "80.83"],
        ["COCO CIDEr (1000 captions)", "122.8", "123.3", "123.3", "122.2"],
        ["CHAIR_i, hallucinated objects (500)", "5.78 %", "5.69 %", "5.86 %", "5.89 %"]]
table(sl, 0.6, 1.55, 8.2, rows, [3.4, 1.1, 1.2, 1.3, 1.2], size=12, height=0.43)
text(sl, 0.6, 5.1, 8.2, 1.0, "Every difference vs HF: not significant (exact McNemar p ≥ 0.09; 95 % bootstrap intervals "
                             "for CIDEr / CHAIR contain 0). Plain-rounding INT4 does lose (TextVQA −0.6 pts, p = 0.002; "
                             "CIDEr −4.5): GPTQ is what makes 4-bit usable.", size=12, color=INK)
if EV.get("baseline"):
    b = EV["baseline"]
    vs = [v for v in ("fused_gemv_pf_vfused", "fused_w8all_pf_vfused", "fused_w4gptq_pf_vfused") if v in EV]
    heads = {"fused_gemv_pf_vfused": "BF16", "fused_w8all_pf_vfused": "INT8", "fused_w4gptq_pf_vfused": "INT4"}
    rows = [["Jetson, 200 each", "HF"] + [heads[v] for v in vs]]
    for k, lab in (("mmstar", "MMStar"), ("pope", "POPE"), ("ocrbench", "OCRBench"), ("chartqa", "ChartQA"), ("textvqa", "TextVQA")):
        rows.append([lab, f"{b.get(k, 0):.1f}"] + [f"{EV[v][k]:.1f}" if k in EV[v] else "–" for v in vs])
    cw = 2.5 / (1 + len(vs))
    table(sl, 9.0, 1.55, 3.7, rows, [1.2] + [cw] * (1 + len(vs)), size=11, height=0.43)
text(sl, 9.0, 4.3 if EV.get("baseline") else 1.6, 3.7, 1.5,
     "Long answers: our BF16 stays as close to FP32 as HF's own BF16. Quantized versions reword more, but score the same "
     "against human captions.", size=12, color=GRAY, italic=True)
notes(sl, "Answers are checked at four levels: kernel tests, logits vs HF, 5 benchmarks, reference-scored long answers.")

# ----------------------------------------------------------------------------------------------- 10 quantization
sl = slide("Extension · fewer bytes per weight", "Quantization: Measured Like Everything Else")
cols = [("INT8 decode", GREEN, [f"Jetson: {fmt_ms(w8_tpot)} ms per word ({hf_tpot / w8_tpot:.1f}× HF)" if w8_tpot else "",
                                "weights: one scale per row, our GEMV",
                                "5 benchmarks, captions, hallucination: n.s."]),
        ("INT8 everywhere", PURPLE, ["vision + prompt on our INT8 tensor-core GEMM too",
                                     f"peak memory {mem['fused_w8all_pf_vfused']:.1f} GB (HF {mem['baseline']:.1f})" if mem["baseline"] else "",
                                     "long answers reworded, but CIDEr / CHAIR unchanged"]),
        ("INT4 with GPTQ", GOLD, ["no training: rounding error pushed onto the remaining weights",
                                  f"Jetson: {fmt_ms(w4_tpot)} ms per word ({hf_tpot / w4_tpot:.1f}× HF); L40S: 1.06 ms (8×)" if w4_tpot else "",
                                  "plain rounding fails; GPTQ: all checks n.s."])]
for i, (h, c, lines) in enumerate(cols):
    x = 0.6 + 4.1 * i
    box(sl, x, 1.6, 3.85, 3.3)
    text(sl, x + 0.3, 1.8, 3.3, 0.5, h, size=20, bold=True, font=HEAD, color=c)
    text(sl, x + 0.3, 2.5, 3.3, 3.2, [ln for ln in lines if ln], size=13, bullets=True)
text(sl, 0.6, 5.25, 12.1, 0.7, "Quantization changes the model, so it is never counted as one of our optimizations; the "
                              "headline results are lossless BF16. NVIDIA's NVFP4 4-bit format was simulated: no more "
                              "accurate than INT4 with small groups, so no kernel was written for it.", size=12, color=GRAY,
     italic=True)
notes(sl, "GPTQ: Frantar et al. 2023; calibration on COCO test2017 images, which none of our benchmarks use.")

# ----------------------------------------------------------------------------------------------- 11 demo
sl = slide("Demonstration", "Demo: Measured on the Jetson, Re-runnable Live")
still = Path(__file__).resolve().parent / "demo_frame.png"   # demo/render.py --still graph:2200 of the Jetson run
if still.exists():
    sl.shapes.add_picture(str(still), Inches(0.6), Inches(1.55), width=Inches(7.6))
else:
    box(sl, 0.6, 1.55, 7.6, 4.3)
    text(sl, 0.8, 3.4, 7.2, 0.6, "demo video frame", size=16, color=GRAY, align=PP_ALIGN.CENTER)
text(sl, 8.6, 1.55, 4.1, 0.4, "The video", size=16, bold=True)
text(sl, 8.6, 2.0, 4.1, 2.9, [
    "8 real images (148×60 px to 35 MP), answered by HF, ours BF16 and ours INT8 side by side",
    "every word at its GPU timestamp measured on the Jetson",
    "here: function plot at 2.2 s; INT8 done, same answer as HF",
    "over the 8 cases: BF16 {:.2f}× faster at {:.0f} % of HF's energy, INT8 {:.2f}× at {:.0f} %".format(
        *DEMO["fused_gemv_pf_vfused+gpuprep"], *DEMO["fused_w8all_pf_vfused+gpuprep"]),
], size=13, bullets=True)
text(sl, 8.6, 5.0, 4.1, 0.4, "The live proof", size=16, bold=True)
text(sl, 8.6, 5.45, 4.1, 1.4, [
    [("bash demo/live_jetson.sh", {"font": "Courier New", "size": 12, "bold": True})],
    "re-runs the video's cases on the board with the same measuring code and prints live vs video",
], size=13, bullets=True)
notes(sl, "If the board is available during the evaluation, run demo/live_jetson.sh --quick or --image with your own photo.")

# ----------------------------------------------------------------------------------------------- 12 feasibility
sl = slide("Feasibility & plan to the final", "Does It Work? Yes. What Is Left")
box(sl, 0.6, 1.45, 5.95, 3.9)
box(sl, 6.85, 1.45, 5.85, 3.9)
text(sl, 0.85, 1.65, 5.5, 0.4, "Works today", size=16, bold=True, color=GREEN)
text(sl, 0.85, 2.15, 5.5, 4.0, [
    "full prototype on 5 GPUs incl. the Jetson: 1.5–2.7× faster per word, lossless",
    "decode at 1.1–1.2× each GPU's memory floor",
    "answers unchanged on 19,000 benchmark questions and reference-scored captions",
    "INT8 / INT4 extensions with their own quality evidence; 2.1 GB peak memory; INT4 10.4 ms per word on the Jetson",
    "demo video + live re-run script",
], size=13, bullets=True)
text(sl, 7.1, 1.65, 5.4, 0.4, "Left for the final evaluation", size=16, bold=True, color=ORANGE)
text(sl, 7.1, 2.15, 5.4, 4.0, [
    "first word on the Jetson: vision-encoder CUDA graph, MAXN power mode",
    "INT4 for prompt + vision (memory), only if quality holds",
    "full-size quality run on the Jetson; ablation table per GPU",
    "Nsight profiles on the Jetson (needs counter access)",
    "technical report",
], size=13, bullets=True)
notes(sl, "Feasibility: the approach is validated end to end; remaining work is polish on the edge first-token path.")

# ----------------------------------------------------------------------------------------------- 13 summary
sl = slide("Summary", "Faster, Leaner VLM Inference on the Edge, Same Answers", dark=True)
stats = [("1.6–2.0×", "faster per word on the Jetson", "lossless BF16, MAXN / 40 W"),
         ("1.1–1.2×", "of the memory floor", "on every GPU"),
         (f"−{100 * (1 - bf_j / hf_j):.0f} %" if hf_j and bf_j else "−32 %", "energy per word", "Jetson, whole module"),
         ("±0.3 pts", "benchmark accuracy", "19,000 questions, n.s.")]
for i, (big, lab, sub) in enumerate(stats):
    x = 0.6 + 3.1 * i
    text(sl, x, 2.0, 2.9, 0.8, big, size=40, bold=True, font=HEAD, color=ORANGE)
    text(sl, x, 2.95, 2.9, 0.8, [[(lab, {"bold": True, "color": WHITE})], [(sub, {"size": 12, "color": "9FB1B7"})]], size=15)
text(sl, 0.6, 4.4, 12.1, 1.6, [
    [("Approach: ", {"bold": True, "color": ORANGE}),
     ("CUDA graph + fused kernels + our weight-streaming GEMV for each word; padding-free fused vision and GPU "
      "preprocessing for the first word.", {"color": "E6EEF0"})],
    [("On the edge: ", {"bold": True, "color": ORANGE}),
     ("profiling on the board found the INT4 GEMV conversion-bound on 8 SMs; two exact kernel changes (byte-permute "
      "conversion, two rows per warp) took it from 13.5 to 10.4 ms per word.",
      {"color": "E6EEF0"})],
], size=15)
notes(sl, "Answer to the key question: yes, the approach works, and the evidence is measured, not estimated.")

# ----------------------------------------------------------------------------------------------- backup
sl = slide("Backup · for questions", "What Did Not Work (All Measured)")
rows = [["Attempt", "Result", "Why"],
        ["INT4, plain rounding", "TextVQA −0.6 pts (p = 0.002), CIDEr −4.5", "too coarse without error compensation → GPTQ"],
        ["NVFP4 (Blackwell 4-bit)", "no more accurate than INT4 group 32", "no kernel written"],
        ["INT8 for the first word on the Jetson", "1080p first word slower (1.64 vs 1.49 s)", "8 SMs: quantize kernels cost more than they save"],
        ["FP8 vision matrices", "slower and inaccurate", "separate conversion kernels"],
        ["torch.compile on the vision encoder", "97.8 → 118 ms", "its kernel choices lose to cuBLAS + cuDNN"],
        ["first fused-epilogue GEMV", "3 % slower", "fewer loads in flight per row; redesigned"]]
table(sl, 0.6, 1.6, 12.1, rows, [3.6, 4.0, 4.5], size=12, height=0.55)
notes(sl, "Negative results are logged with the same care as positive ones (docs/experiment_log.md).")

prs.save(OUT)
print("deck ->", OUT, f"({N[0]} slides)")
