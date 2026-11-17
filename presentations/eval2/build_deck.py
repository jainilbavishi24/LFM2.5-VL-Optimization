"""Evaluation 2 deck (same look as presentations/eval1/PAA_Eval1_v2_Edge_VLM.pptx: Cambria / Calibri, teal + orange).
Numbers are read from the committed result files where they exist, so re-running after new measurements updates it.

    python3 presentations/eval2/build_deck.py      # -> presentations/eval2/PAA_Eval2_Working_Prototype.pptx
"""

import csv
import glob
import json
import re
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
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
        ch.legend.position = XL_LEGEND_POSITION.TOP
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



def demo_summary(run=None):
    """Geometric-mean speed-up and energy share vs HF over the Jetson demo cases (demo/video_jetson.json)."""
    import math
    cfg = json.loads((ROOT / "demo" / "video_jetson.json").read_text())
    tr = json.loads((ROOT / (run or cfg["run"]) / "traces.json").read_text())
    by = {(x["variant"], x["key"]): x for x in tr}
    out = {}
    for v in ("fused_gemv_pf_vfused+gpuprep", "fused_w8all_pf_vfused+gpuprep"):
        sp = [by[("baseline", k)]["wall_ms_median"] / by[(v, k)]["wall_ms_median"] for k in cfg["cases"]]
        en = [by[(v, k)]["energy_j_per_answer"] / by[("baseline", k)]["energy_j_per_answer"] for k in cfg["cases"]]
        out[v] = (math.exp(sum(map(math.log, sp)) / len(sp)), 100 * math.exp(sum(map(math.log, en)) / len(en)))  # as the video
    return out



def ms(v):
    return "–" if v is None else (f"{v:.2f} ms" if v < 10 else f"{v:.1f} ms" if v < 100 else f"{v:,.0f} ms")


def arrow(sl, x, y, w=0.35, h=0.4, color="C9D1D4"):
    a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = rgb(color)
    a.line.fill.background()
    return a


def card(sl, x, y, w, h, big, label, color=TEAL, sub=None, big_size=28, fill=TINT):
    box(sl, x, y, w, h, fill=fill)
    text(sl, x + 0.2, y + 0.15, w - 0.4, 0.6, big, size=big_size, bold=True, font=HEAD, color=color)
    text(sl, x + 0.2, y + 0.15 + big_size / 72 + 0.12, w - 0.4, h - 0.9,
         [[(label, {"bold": True})]] + ([[(sub, {"size": 11, "color": GRAY})]] if sub else []), size=13)


def footnote(sl, s, y=6.55):
    text(sl, 0.6, y, 12.1, 0.45, s, size=10.5, color=GRAY, italic=True)


# ----------------------------------------------------------------------------------------------- numbers
# Jetson: one session in MAXN mode (sudo nvpmodel -m 0 + jetson_clocks) for every variant, so the bars compare like
# with like. The 40 W session of the same day gives the same ours-numbers (26.0 ms) and HF 42.4 ms.
VARS = ["baseline", "fused_gemv_pf_vfused", "fused_w8_pf_vfused", "fused_w8all_pf_vfused", "fused_w4gptq_pf_vfused"]
TP = {v: maxn(v) for v in VARS}
EJ = {v: maxn(v, "decode_j_per_tok") for v in VARS}
MEM = {v: maxn(v, "peak_mem_gb", "img_fhd", 1) for v in VARS}
FW = {v: [(maxn(v, "ttft_ms", wl, 1) or 0) + (maxn(v, "preprocess_ms", wl, 1) or 0) for wl in ("text", "img_small", "img_fhd")]
      for v in VARS}
HF, BF, W8, W8A, W4 = (TP[v] for v in VARS)
FLOOR_JET = 22.9          # 2.34 GB of BF16 weights per word / ~102 GB/s
EV = eval_subset()
DEMO = demo_summary("results/demo/20261117-114427_Orin_jetson_maxn_demo")   # the recording the frame is from
GEMV = gemv_table()
W4_BEFORE = jet("fused_w4gptq_pf_vfused", "tpot_ms")
L40 = {"fused_gemv_pf_vfused": 3.30, "fused_w8_pf_vfused": 1.79, "fused_w8all_pf_vfused": 1.79, "fused_w4gptq_pf_vfused": 1.06}

sys.path.insert(0, str(ROOT / "benchmarks"))
import ablation_table as AB  # noqa: E402


def ab(gname, i, wl="text", gen=128, field="tpot_ms"):
    gpu, runs = AB.GPUS[gname]
    r = runs[i]
    return AB.value(gpu, r[0], r[1], wl, gen, field) if r else None


def ncu_summary(label, bytes_per_w):
    """Achieved memory bandwidth (GB/s) of the two largest GEMVs (lm_head 65536x2048, w1|w3 16384x2048) in the real
    model, from the latest Jetson ncu run: weight bytes / kernel time. The Orin's throughput counters read 0 under ncu,
    so the kernel times are what we use. None while the run is missing."""
    runs = sorted(glob.glob(str(ROOT / "profiling" / "reports" / f"*_Orin_ncu_ours_{label}" / "ncu_ours_metrics.csv")))
    if not runs:
        return None
    rows = list(csv.DictReader(open(runs[-1])))
    to_ms = {"ns": 1e-6, "us": 1e-3, "usecond": 1e-3, "ms": 1.0, "msecond": 1.0}[rows[0]["gpu__time_duration.sum"]]
    grp = {}
    for r in rows[1:]:
        if "gemv" in r["Kernel Name"]:
            grp.setdefault((r["Kernel Name"], r["launch__grid_size"]), []).append(
                float(r["gpu__time_duration.sum"].replace(",", "")) * to_ms)
    t = sorted((sum(v) / len(v) for v in grp.values()), reverse=True)
    if len(t) < 2:
        return None
    return {"lm_head": 65536 * 2048 * bytes_per_w / t[0] / 1e6, "w1|w3": 16384 * 2048 * bytes_per_w / t[1] / 1e6,
            "t": t[:2]}


NCU = {"BF16": ncu_summary("bf16_vga", 2), "INT8": ncu_summary("int8_vga", 1), "INT4": ncu_summary("int4_vga", 0.5)}

# Vision-encoder CUDA graph, Jetson, eager vs graph replay, interleaved (tests/check_vision_graph.py). None = pending.
VGRAPH = None   # e.g. {"VGA": 1.0x, "720p": ..., "1080p": ...}

# ----------------------------------------------------------------------------------------------- 1 title
sl = prs.slides.add_slide(BLANK)
N[0] += 1
sl.background.fill.solid()
sl.background.fill.fore_color.rgb = rgb(INK)
text(sl, 0.6, 0.55, 12, 0.3, "CS3.406 PROGRAMMING AI ACCELERATORS · EVALUATION 2", size=12, bold=True, color=ORANGE)
text(sl, 0.6, 1.1, 8.9, 1.4, ["Optimizing a Vision-Language Model", "for Edge Devices"], size=32, bold=True, font=HEAD,
     color=WHITE)
text(sl, 0.6, 2.75, 8.9, 0.5, "Working prototype: LiquidAI LFM2.5-VL-1.6B, now on a Jetson Orin NX 16GB", size=18,
     color="C9D6DA")
text(sl, 0.6, 3.75, 8.4, 1.0, [[("Key question: ", {"bold": True, "color": ORANGE}),
                                 ("does our approach work on a real edge board, with the same answers?",
                                  {"color": "E6EEF0"})]], size=16)
for i, (n, r) in enumerate([("Jainil Bavishi", "2023102045"), ("Gautam Gandhi", "2023102059"),
                            ("Bibek Singh Dhody", "2023101054")]):
    text(sl, 0.6 + 3.0 * i, 5.55, 2.8, 0.6, [[(n, {"bold": True, "color": WHITE})], [(r, {"size": 11, "color": "9FB1B7"})]],
         size=14)
text(sl, 0.6, 6.6, 9, 0.35, [[("Repository ", {"bold": True, "color": ORANGE}),
                              ("github.com/jainilbavishi24/LFM2.5-VL-Optimization", {"color": "C9D6DA"})]], size=12)
for i, (big, lab) in enumerate([("1.5–2.7×", "faster per word, same weights, 5 GPUs"),
                                (f"{HF / W4:.1f}×", "on the Jetson with 4-bit weights"),
                                ("±0.3 pts", "accuracy on 19,000 questions")]):
    box(sl, 9.75, 1.15 + 1.45 * i, 3.0, 1.25, fill="24323A")
    text(sl, 9.95, 1.25 + 1.45 * i, 2.7, 0.6, big, size=30, bold=True, font=HEAD, color=ORANGE)
    text(sl, 9.95, 1.85 + 1.45 * i, 2.7, 0.45, lab, size=12, color="E6EEF0")
notes(sl, "Eval 1: baseline and bottlenecks. Today: the working prototype, on the edge board, with quality evidence.")

# ----------------------------------------------------------------------------------------------- 2 since eval 1
sl = slide("Since Evaluation 1", "From Plan to Working Prototype")
items = [("Runs on the edge", "Jetson Orin NX 16GB", f"{HF / BF:.1f}× faster per word", TEAL),
         ("Same code, 5 GPUs", "Blackwell → Jetson", "faster on every one", BLUE),
         ("Quality proven", "5 benchmarks · 19,000 questions", "no significant change", GREEN),
         ("Memory for the edge", "INT8 everywhere", f"{MEM['fused_w8all_pf_vfused']:.1f} GB peak (HF {MEM['baseline']:.1f})", PURPLE),
         ("4-bit, no training", "GPTQ INT4", f"{HF / W4:.1f}× on the Jetson", GOLD),
         ("Profiled on the board", "found a new bottleneck", "INT4 13.5 → 10.4 ms", ORANGE)]
for i, (h, d, r, c) in enumerate(items):
    x, y = 0.6 + (i % 3) * 4.1, 1.6 + (i // 3) * 2.6
    box(sl, x, y, 3.85, 2.3)
    circ = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.25), Inches(y + 0.28), Inches(0.42), Inches(0.42))
    circ.fill.solid()
    circ.fill.fore_color.rgb = rgb(c)
    circ.line.fill.background()
    text(sl, x + 0.25, y + 0.32, 0.42, 0.34, str(i + 1), size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    text(sl, x + 0.85, y + 0.3, 2.85, 0.4, h, size=17, bold=True, color=INK)
    text(sl, x + 0.25, y + 0.95, 3.4, 0.4, d, size=13, color=GRAY)
    text(sl, x + 0.25, y + 1.45, 3.4, 0.6, r, size=18, bold=True, font=HEAD, color=c)
notes(sl, "Six new things since Eval 1; each has a slide. Everything is in the repo with raw results and the log.")

# ----------------------------------------------------------------------------------------------- 3 prototype
sl = slide("Implementation", "The Prototype: One Fix per Measured Bottleneck")
stages = [("1  Image prep", "CPU resize + normalize", "2 GPU kernels, bit-exact", "25–63 → 3.9 ms"),
          ("2  Vision encoder", "padded attention, 24 % occupancy", "padding-free flash attention + fused LN / GELU",
           "77 → 54 ms"),
          ("3  Read the prompt", "eager PyTorch, many small kernels", "fused prefill into the decode buffers", "37 → 31 ms"),
          ("4  Write each word", "703 launches per word, GPU idle 55 %", "CUDA graph + fused kernels + our GEMV",
           "3.83 → 2.07 ms")]
w, gap = 2.75, 0.37
for i, (name, prob, fix, res) in enumerate(stages):
    x = 0.6 + i * (w + gap)
    box(sl, x, 1.6, w, 0.6, fill=TEAL)
    text(sl, x + 0.15, 1.6, w - 0.3, 0.6, name, size=15, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
    box(sl, x, 2.3, w, 3.55)
    text(sl, x + 0.18, 2.45, w - 0.36, 0.3, "PROBLEM", size=10, bold=True, color=MUTED)
    text(sl, x + 0.18, 2.75, w - 0.36, 0.75, prob, size=13)
    text(sl, x + 0.18, 3.55, w - 0.36, 0.3, "OUR FIX", size=10, bold=True, color=MUTED)
    text(sl, x + 0.18, 3.85, w - 0.36, 0.95, fix, size=13, bold=True, color=TEAL)
    text(sl, x + 0.18, 4.85, w - 0.36, 0.3, "RESULT", size=10, bold=True, color=MUTED)
    text(sl, x + 0.18, 5.12, w - 0.36, 0.6, res, size=21, bold=True, font=HEAD, color=ORANGE)
    if i < 3:
        arrow(sl, x + w + 0.03, 1.7, 0.31, 0.4)
text(sl, 0.6, 6.1, 12.1, 0.4, [[("All lossless BF16 · ", {"bold": True, "color": TEAL}),
                                ("109 kernel unit tests vs PyTorch · whole runtime checked token by token vs Hugging Face",
                                 {})]], size=13)
footnote(sl, "Numbers: RTX PRO 5000 Blackwell, 1920×1080 image (stages 1–3), text prompt (stage 4).", y=6.6)
notes(sl, "The pipeline from Eval 1, now with each stage's fix and measured result.")

# ----------------------------------------------------------------------------------------------- 4 ablation
sl = slide("Ablation · one optimization at a time", "Which Step Gives What?")
steps = ["+ CUDA graph", "+ fused kernels", "+ our GEMV", "full stack"]
idx = [1, 2, 3, 6]
series = []
for gname, short in (("RTX PRO 5000 Blackwell", "Blackwell"), ("L40S", "L40S"), ("Jetson Orin NX (40 W)", "Jetson (40 W)")):
    base = ab(gname, 0)
    series.append((short, [base / ab(gname, i) if ab(gname, i) else 0 for i in idx]))
bars(sl, 0.5, 1.45, 7.6, 4.9, steps, series, [BLUE, TEAL, ORANGE], title="Speed-up per word vs HF (×, cumulative)",
     fmt='0.00"×"', legend=True, size=12)
text(sl, 8.5, 1.55, 4.2, 0.4, "First word, incl. image prep", size=16, bold=True)
fw = [("L40S · 1080p", ab("L40S", 0, "img_fhd", 1, "ttft_ms"), ab("L40S", 7, "img_fhd", 1, "ttft_ms")),
      ("Jetson 40 W · 1080p", ab("Jetson Orin NX (40 W)", 0, "img_fhd", 1, "ttft_ms"), ab("Jetson Orin NX (40 W)", 7, "img_fhd", 1, "ttft_ms")),
      ("Jetson 40 W · 640×480", ab("Jetson Orin NX (40 W)", 0, "img_small", 1, "ttft_ms"), ab("Jetson Orin NX (40 W)", 7, "img_small", 1, "ttft_ms"))]
for i, (lab, a, b) in enumerate(fw):
    y = 2.05 + i * 1.3
    box(sl, 8.5, y, 4.2, 1.1)
    text(sl, 8.7, y + 0.1, 2.6, 0.4, lab, size=13, bold=True)
    text(sl, 8.7, y + 0.5, 2.6, 0.5, f"{a:,.0f} → {b:,.0f} ms", size=19, bold=True, font=HEAD, color=TEAL)
    text(sl, 11.3, y + 0.3, 1.25, 0.6, f"{a / b:.2f}×", size=22, bold=True, font=HEAD, color=ORANGE, align=PP_ALIGN.RIGHT)
text(sl, 8.5, 5.95, 4.2, 0.5, "from fused vision + GPU preprocessing", size=12, color=GRAY)
footnote(sl, "Each bar adds one optimization to the one before; all BF16, same weights. Text prompt, 128 words, median of "
             "5 runs (scripts/ablation.sh). Full table in the backup.")
notes(sl, "The CUDA graph is the largest single step everywhere. On the Jetson our GEMV is slightly slower than cuBLAS "
          "in BF16 (26.4 vs 25.4 ms) but uses less energy (0.60 vs 0.64 J per word).")

# ----------------------------------------------------------------------------------------------- 5 five GPUs
sl = slide("Generalization", "Same Code, Five GPUs: Faster on Every One")
gpus = ["RTX PRO 5000\nBlackwell", "L40S\n(Ada)", "T4\n(Turing)", "DGX Spark\n(GB10)", "Jetson\nOrin NX"]
hf = [3.70, 8.98, 21.6, 15.2, HF]
ours = [2.07, 3.29, 9.03, 9.88, BF]
bars(sl, 0.5, 1.45, 8.0, 5.0, gpus, [("HF transformers", hf), ("ours (BF16)", ours)], [BLUE, ORANGE],
     title="Time per output word (ms, lower is better)", legend=True, size=12)
rows = [["GPU", "Speed-up", "vs floor"]]
for g, a, b, fl in zip(["Blackwell", "L40S", "T4", "DGX Spark", "Jetson"], hf, ours, [1.9, 2.7, 7.3, 8.6, FLOOR_JET]):
    rows.append([g, f"{a / b:.1f}×", f"{b / fl:.2f}×"])
table(sl, 8.8, 1.6, 3.9, rows, [1.5, 1.2, 1.2], size=14, height=0.5, highlight_col=1)
text(sl, 8.8, 4.85, 3.9, 1.3, [[("Floor", {"bold": True, "color": TEAL}),
                                (" = weight bytes per word ÷ memory bandwidth: the best any software can do.", {})]],
     size=13)
footnote(sl, "Batch 1, text prompt, 128 words. Jetson in MAXN mode.")
notes(sl, "HF's time follows each machine's CPU; ours follows the GPU's memory bandwidth, near the floor everywhere.")

# ----------------------------------------------------------------------------------------------- 6 quality
sl = slide("Quality", "Same Answers as the Original Model")
rows = [["Benchmark (full set, L40S)", "HF", "ours BF16", "Δ"]]
for name, a, b in [("MMStar (1,500)", 49.20, 49.27), ("POPE (9,000)", 89.12, 89.10), ("OCRBench (1,000)", 79.80, 80.00),
                   ("ChartQA (2,500)", 80.08, 80.00), ("TextVQA (5,000)", 81.16, 81.03)]:
    rows.append([name, f"{a:.2f}", f"{b:.2f}", f"{b - a:+.2f}"])
rows.append(["COCO captions, CIDEr ↑", "122.8", "123.3", "+0.5"])
rows.append(["Hallucinated objects, CHAIR ↓", "5.78 %", "5.69 %", "−0.09"])
table(sl, 0.6, 1.6, 7.6, rows, [3.7, 1.3, 1.4, 1.2], size=14, height=0.5, highlight_col=2)
card(sl, 8.6, 1.6, 4.1, 1.7, "n.s.", "no difference is significant", GREEN,
     "McNemar p ≥ 0.09 on every benchmark; bootstrap CIs contain 0")
ev_b, ev_o = EV.get("baseline", {}), EV.get("fused_gemv_pf_vfused", {})
dmax = max((abs(ev_o[k] - ev_b[k]) for k in ev_b if k in ev_o), default=None)
card(sl, 8.6, 3.5, 4.1, 1.7, f"≤ {dmax:.1f} pts" if dmax is not None else "–", "on the Jetson too", TEAL,
     "same 5 benchmarks, 1,000-question subset")
footnote(sl, "Answers checked at four levels: kernel tests → logits vs HF → 19,000 benchmark questions → captions scored "
             "against human references.", y=6.3)
notes(sl, "Quantized versions are on the quantization slide.")

# ----------------------------------------------------------------------------------------------- 7 jetson setup
sl = slide("Edge deployment", "The Board: Jetson Orin NX 16GB")
specs = [("8 SMs", "Ampere GPU (sm_87)"), ("102 GB/s", "memory, shared CPU + GPU"), ("15 GB", "for everything"),
         ("no fan", "passive heatsink only")]
for i, (big, lab) in enumerate(specs):
    card(sl, 0.6 + i * 3.05, 1.6, 2.8, 1.4, big, lab, ORANGE if big == "no fan" else TEAL)
text(sl, 0.6, 3.35, 5.8, 0.4, "Same code, compiled for the board", size=16, bold=True)
text(sl, 0.6, 3.85, 5.8, 2.4, ["JetPack 7.2 · CUDA 13.2 · PyTorch 2.11",
                               "our CUDA kernels: no source change",
                               "energy: board power sensor, sampled every 5 ms",
                               "heat: jobs paused above 85 °C (thermal guard)"], size=14, bullets=True)
rows = [["Correctness check on the board", "Result"], ["kernel unit tests", "109 / 109"], ["INT8 GEMM tests", "36 / 36"],
        ["next word = HF (BF16)", "377 / 381 *"]]
table(sl, 6.9, 3.4, 5.8, rows, [3.6, 2.2], size=14, height=0.52)
text(sl, 6.9, 5.6, 5.8, 0.6, "* the 4 flips are near-ties between two words", size=12, color=GRAY, italic=True)
notes(sl, "Preprocessing differs by at most one pixel level: PyTorch's CPU resize takes another code path on Arm.")

# ----------------------------------------------------------------------------------------------- 8 jetson per word
sl = slide("Edge results · Jetson Orin NX", f"Jetson: {HF / BF:.1f}× Faster per Word, Same Weights")
labels = ["HF", "ours\nBF16", "ours\nINT8", "ours INT8\neverywhere", "ours\nINT4 GPTQ"]
bars(sl, 0.5, 1.45, 7.6, 5.0, labels, [("ms per word", [TP[v] for v in VARS])], [ORANGE],
     title="Time per output word (ms, lower is better)", point_colors=[BLUE, ORANGE, GREEN, PURPLE, GOLD], size=12)
card(sl, 8.5, 1.55, 4.2, 1.45, f"{HF / BF:.1f}×", "lossless BF16", ORANGE, f"{FLOOR_JET} ms floor: we are at {BF / FLOOR_JET:.2f}×")
card(sl, 8.5, 3.15, 4.2, 1.45, f"{HF / W8:.1f}× · {HF / W4:.1f}×", "INT8 · INT4 weights", GOLD, "quantized: own quality checks")
card(sl, 8.5, 4.75, 4.2, 1.45, f"−{100 * (1 - EJ[VARS[1]] / EJ['baseline']):.0f} %", "energy per word, BF16", TEAL,
     f"{EJ['baseline']:.2f} → {EJ[VARS[1]]:.2f} J (INT4 {EJ[VARS[4]]:.2f} J)")
footnote(sl, "Text prompt, 128 words, median of 5. MAXN mode. In the 40 W mode ours stays at 26.0 ms, HF slows to 42.4 ms "
             "(it is CPU-bound).")
notes(sl, "HF spends about half of every word on Python and kernel launches on the Arm CPU. "
          "Ours is at 1.13x the memory floor.")

# ----------------------------------------------------------------------------------------------- 9 jetson first word
sl = slide("Edge results · Jetson Orin NX", "First Word: Smaller Gain for Big Images")
for i, (lab, k) in enumerate((("Text only", 0), ("640×480 photo", 1), ("1920×1080 photo", 2))):
    a, b = FW["baseline"][k], FW[VARS[1]][k]
    x = 0.6 + i * 4.1
    box(sl, x, 1.6, 3.85, 2.4)
    text(sl, x + 0.25, 1.75, 3.4, 0.4, lab, size=16, bold=True)
    text(sl, x + 0.25, 2.25, 3.4, 0.8, f"{a / b:.2f}×", size=40, bold=True, font=HEAD, color=ORANGE)
    text(sl, x + 0.25, 3.2, 3.4, 0.5, f"HF {ms(a)} → ours {ms(b)}", size=14, color=TEAL, bold=True)
text(sl, 0.6, 4.35, 12.1, 0.4, "Why", size=16, bold=True)
text(sl, 0.6, 4.8, 12.1, 1.5, ["1080p photo = 2,300 image tokens through a 27-layer vision encoder: large matrix multiplies",
                               "on only 8 SMs these are limited by arithmetic, so removing overhead helps less",
                               "our gain here comes from GPU preprocessing and the padding-free vision encoder"],
     size=14, bullets=True)
footnote(sl, "First word includes image preprocessing. MAXN mode, median of 5.")
notes(sl, "Short prompts: overhead-bound, big gain. Big images: compute-bound on this small GPU.")

# ----------------------------------------------------------------------------------------------- 10 profiling on board
sl = slide("Profiling on the board", "The Edge Has a Different Bottleneck")
box(sl, 0.6, 1.55, 3.3, 4.75)
text(sl, 0.8, 1.7, 3.0, 0.4, "Nsight Systems", size=16, bold=True, color=TEAL)
text(sl, 0.8, 2.15, 3.0, 0.4, "kernel launches, 32-word answer", size=12, color=GRAY)
text(sl, 0.8, 2.6, 3.0, 0.6, [[("HF  ", {"size": 14, "bold": True}), ("22,644", {})]], size=28, bold=True, font=HEAD,
     color=BLUE)
text(sl, 0.8, 3.3, 3.0, 0.6, [[("ours  ", {"size": 14, "bold": True}), ("31", {})]], size=28, bold=True, font=HEAD,
     color=ORANGE)
text(sl, 0.8, 4.25, 2.9, 1.9, ["HF: 27 µs of CPU per launch ≈ 19 ms of every word", "ours: one CUDA graph per word"],
     size=13, bullets=True)

text(sl, 4.2, 1.6, 4.7, 0.4, "INT4 GEMV: compute-bound on 8 SMs", size=16, bold=True, color=TEAL)
if GEMV:
    rows = [["INT4 GEMV, µs", "before", "fix 1", "fix 1+2"]]
    for name, old, new, r2 in GEMV:
        rows.append([name, f"{old:.0f}", f"{new:.0f}", f"{new:.0f}" if name == "lm_head" else (f"{r2:.0f}" if r2 else "–")])
    table(sl, 4.2, 2.05, 4.7, rows, [1.55, 1.0, 1.0, 1.15], size=12, height=0.36, highlight_col=3)
yy = 2.05 + 0.36 * (len(GEMV) + 1) + 0.15
text(sl, 4.2, yy, 4.7, 0.8, [[("fix 1: ", {"bold": True}), ("int → float by byte permute, not the slow I2F", {})],
                             [("fix 2: ", {"bold": True}), ("two weight rows per warp", {})]], size=12, bullets=True)
text(sl, 4.2, yy + 0.8, 4.7, 0.5, [[(f"{W4_BEFORE:.1f} → {W4:.1f} ms", {}),
                                    (" per word, bit-identical", {"size": 13, "font": BODY, "color": GRAY, "bold": False})]],
     size=20, bold=True, font=HEAD, color=ORANGE)

box(sl, 9.2, 1.55, 3.5, 4.75)
text(sl, 9.4, 1.7, 3.1, 0.4, "Nsight Compute", size=16, bold=True, color=TEAL)
text(sl, 9.4, 2.15, 3.1, 0.5, "memory bandwidth reached inside the model (peak 102 GB/s)", size=12, color=GRAY)
if any(NCU.values()):
    rows = [["GB/s", "lm_head", "w1|w3"]]
    for lab, v in NCU.items():
        if v:
            rows.append([lab, f"{v['lm_head']:.0f}", f"{v['w1|w3']:.0f}"])
    table(sl, 9.4, 2.75, 3.1, rows, [1.0, 1.05, 1.05], size=13, height=0.45)
    text(sl, 9.4, 2.9 + 0.45 * len(rows), 3.1, 1.6, "BF16 / INT8 near the memory limit; INT4 reads fewer bytes but at a "
                                                    "lower rate: the arithmetic is the limit on 8 SMs", size=12)
else:
    text(sl, 9.4, 3.2, 3.1, 1.5, "running on the board now; result added before the evaluation", size=13, color=GRAY,
         italic=True)
footnote(sl, "GEMV = the matrix-vector products that read every weight once per word. Nsight Compute: weight bytes ÷ "
             "kernel time (the Orin's throughput counters read 0).")
notes(sl, "On a datacenter GPU INT4 is 1.65x faster than INT8. On the Orin it was barely faster because every weight "
          "went through a quarter-rate I2F instruction on 8 SMs. Both fixes are exact.")

# ----------------------------------------------------------------------------------------------- 11 quantization
sl = slide("Extension · fewer bytes per weight", "Quantization: Same Measurements, Every Column")
Q = VARS[1:]
rows = [["", "ours BF16", "INT8 (decode)", "INT8 everywhere", "INT4 GPTQ"],
        [f"Jetson, time per word (HF {HF:.1f} ms)"] + [f"{ms(TP[v])}  ({HF / TP[v]:.1f}×)" for v in Q],
        [f"L40S, time per word (HF 8.98 ms)"] + [f"{ms(L40[v])}  ({8.98 / L40[v]:.1f}×)" for v in Q],
        [f"Jetson, energy per word (HF {EJ['baseline']:.2f} J)"] + [f"{EJ[v]:.2f} J" for v in Q],
        [f"Peak memory, 1080p photo (HF {MEM['baseline']:.1f} GB)"] + [f"{MEM[v]:.1f} GB" for v in Q],
        ["5 benchmarks vs HF (19,000 questions)", "n.s.", "n.s.", "n.s.", "n.s."],
        ["COCO captions, CIDEr (HF 122.8)", "123.3", "123.3", "123.3", "122.2"],
        ["Training needed?", "no", "no", "no", "no (GPTQ)"]]
table(sl, 0.6, 1.55, 12.1, rows, [4.1, 2.0, 2.0, 2.0, 2.0], size=13, height=0.5, highlight_col=1)
text(sl, 0.6, 5.7, 12.1, 0.6, [[("BF16 is our headline: ", {"bold": True, "color": TEAL}),
                                 ("lossless. Quantized columns change the weights, so they are an extension with their "
                                  "own quality checks.", {})]], size=13)
footnote(sl, "INT8 decode: language-model weights only · INT8 everywhere: also vision + prompt · INT4 without "
             "GPTQ loses accuracy (TextVQA −0.6, p = 0.002) · n.s. = not significant (McNemar)", y=6.25)
notes(sl, "Peak memory of ours BF16 is above HF's because of the static decode buffers and the CUDA graph pool.")

# ----------------------------------------------------------------------------------------------- 12 demo
sl = slide("Demonstration", "Demo: Real Images, Measured on the Jetson")
still = Path(__file__).resolve().parent / "demo_frame.png"   # demo/render.py --still graph:2100 of the Jetson MAXN run
if still.exists():
    sl.shapes.add_picture(str(still), Inches(0.6), Inches(1.55), width=Inches(7.8))
else:
    box(sl, 0.6, 1.55, 7.8, 4.4)
text(sl, 8.8, 1.55, 3.9, 2.4, ["8 real images, from a 148×60 icon to a 35 MP photo",
                               "HF, ours BF16 and ours INT8 answer side by side",
                               "every word appears at its measured time"], size=14, bullets=True)
s_bf, e_bf = DEMO["fused_gemv_pf_vfused+gpuprep"]
s_w8, e_w8 = DEMO["fused_w8all_pf_vfused+gpuprep"]
card(sl, 8.8, 3.95, 1.85, 1.6, f"{s_bf:.2f}×", "BF16", ORANGE, f"{e_bf:.0f} % energy", big_size=24)
card(sl, 10.85, 3.95, 1.85, 1.6, f"{s_w8:.2f}×", "INT8", GREEN, f"{e_w8:.0f} % energy", big_size=24)
footnote(sl, "Whole answers incl. image preprocessing; geometric mean over the 8 cases vs HF, MAXN mode.", y=6.3)
notes(sl, "The frame: function-plot question at 2.1 s; INT8 already done, same answer as HF.")

# ----------------------------------------------------------------------------------------------- 13 feasibility
sl = slide("Feasibility & plan to the final", "Does It Work? Yes. What Is Left")
box(sl, 0.6, 1.5, 5.95, 4.6)
box(sl, 6.85, 1.5, 5.85, 4.6)
text(sl, 0.85, 1.7, 5.5, 0.4, "Works today", size=18, bold=True, color=GREEN)
text(sl, 0.85, 2.25, 5.5, 3.8, ["full prototype on 5 GPUs, incl. the Jetson",
                                "1.5–2.7× faster per word, lossless",
                                "near each GPU's memory floor",
                                "same answers on 19,000 questions",
                                "INT8 / INT4: up to 3.9× on the Jetson",
                                "demo video from the board"], size=17, bullets=True)
text(sl, 7.1, 1.7, 5.4, 0.4, "Left for the final", size=18, bold=True, color=ORANGE)
text(sl, 7.1, 2.25, 5.4, 3.8, ["first word on the Jetson (compute-bound vision encoder)",
                               "INT4 for vision + prompt, only if quality holds",
                               "remaining full-size quantized quality runs",
                               "technical report"], size=17, bullets=True)
notes(sl, "The approach is validated end to end; the remaining work is the edge first-word path and the report.")

# ----------------------------------------------------------------------------------------------- 14 summary
sl = slide("Summary", "Faster VLM Inference on the Edge, Same Answers", dark=True)
stats = [(f"{HF / BF:.1f}×", "faster per word", "Jetson, lossless BF16"),
         (f"{BF / FLOOR_JET:.2f}×", "of the memory floor", "Jetson; 1.1–1.2× on every GPU"),
         (f"−{100 * (1 - EJ[VARS[1]] / EJ['baseline']):.0f} %", "energy per word", "Jetson, whole module"),
         ("±0.3 pts", "benchmark accuracy", "19,000 questions, n.s.")]
for i, (big, lab, sub) in enumerate(stats):
    x = 0.6 + 3.1 * i
    text(sl, x, 2.0, 2.9, 0.8, big, size=40, bold=True, font=HEAD, color=ORANGE)
    text(sl, x, 2.95, 2.9, 0.8, [[(lab, {"bold": True, "color": WHITE})], [(sub, {"size": 12, "color": "9FB1B7"})]], size=15)
text(sl, 0.6, 4.4, 12.1, 1.6, [
    [("How: ", {"bold": True, "color": ORANGE}),
     ("CUDA graph + fused kernels + our GEMV for each word; GPU preprocessing + padding-free vision for the first word.",
      {"color": "E6EEF0"})],
    [("Edge: ", {"bold": True, "color": ORANGE}),
     (f"profiling on the board found INT4 compute-bound on 8 SMs; two exact fixes: {W4_BEFORE:.1f} → {W4:.1f} ms per word.",
      {"color": "E6EEF0"})]], size=15)
notes(sl, "Answer to the key question: yes, and every number is measured.")

# ----------------------------------------------------------------------------------------------- backup
sl = slide("Backup · ablation", "Full Ablation Table")
cols = [("L40S", "L40S", "text", 128, "tpot_ms"), ("L40S", "L40S", "img_fhd", 1, "ttft_ms"),
        ("Jetson Orin NX (40 W)", "Orin", "text", 128, "tpot_ms"), ("Jetson Orin NX (40 W)", "Orin", "img_small", 1, "ttft_ms"),
        ("Jetson Orin NX (40 W)", "Orin", "img_fhd", 1, "ttft_ms"), ("RTX PRO 5000 Blackwell", "RTXPRO5000Blackwell", "text", 128, "tpot_ms")]
rows = [["Step (all BF16)", "L40S per word", "L40S 1st word 1080p", "Jetson per word", "Jetson 1st word VGA",
         "Jetson 1st word 1080p", "Blackwell per word"]]
for i, step in enumerate(AB.STEPS):
    rows.append([step] + [ms(ab(g, i, wl, gen, fl)) for g, _, wl, gen, fl in cols])
table(sl, 0.6, 1.5, 12.1, rows, [3.7, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4], size=11, height=0.5)
footnote(sl, "Jetson in the 40 W mode; first word includes image preprocessing. Blackwell rows from the first optimization "
             "night. benchmarks/ablation_table.py", y=6.4)

sl = slide("Backup · for questions", "What Did Not Work (All Measured)")
rows = [["Attempt", "Result", "Why"],
        ["INT4, plain rounding", "TextVQA −0.6 pts (p = 0.002)", "too coarse → GPTQ"],
        ["NVFP4 (Blackwell 4-bit)", "no better than INT4 group 32", "no kernel written"],
        ["INT8 first word on the Jetson", "1080p slower (1.65 vs 1.48 s)", "quantize kernels cost more on 8 SMs"],
        ["our GEMV on the Jetson, BF16", "26.4 vs 25.4 ms (cuBLAS)", "kept: 7 % less energy"],
        ["vision-encoder CUDA graph", "≈ 1 % (1,411 → 1,410 ms)" if not VGRAPH else VGRAPH, "vision is compute-bound"],
        ["torch.compile, vision encoder", "97.8 → 118 ms", "loses to cuBLAS + cuDNN"]]
table(sl, 0.6, 1.6, 12.1, rows, [3.9, 4.2, 4.0], size=13, height=0.6)
notes(sl, "Negative results are logged like positive ones (docs/experiment_log.md).")

print("NCU", NCU)
prs.save(OUT)
print("deck ->", OUT, f"({N[0]} slides)")
