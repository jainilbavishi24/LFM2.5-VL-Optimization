"""Render the demo video from a demo/record.py run: the three variants answer the same image + prompt side by side,
replayed from the measured GPU timestamps (no GPU needed). Nothing is re-timed or re-worded: every token appears at
its measured time, the text is the variant's own output, and every number on screen comes from traces.json.

Each case plays once at real speed, then slowed down by a factor chosen per case (shown on screen) so the slowest
lane takes ~5-9 s. Runs were measured one after another on the same GPU and are shown from a common start.

    python demo/render.py results/demo/<run> --cases receipt,cats,... --out presentations/demo
    python demo/render.py results/demo/<run> --still receipt:120     # one PNG at 120 ms real time (layout check)

Writes <out>/clips/NN_<case>.mp4 (one per case) and <out>/<name>.mp4 (title, how-to, all cases, summary).
Fonts: demo/fonts/Inter-*.woff and JetBrainsMono-*.woff if present (Google Fonts, SIL OFL; not committed), else DejaVu.
"""

import argparse
import json
import math
import subprocess
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
ROOT = Path(__file__).resolve().parent

BG = (9, 12, 19)
PANEL = (17, 22, 33)
PANEL_HI = (22, 29, 43)
BORDER = (38, 47, 66)
TEXT = (232, 236, 243)
MUTED = (150, 161, 181)
DIM = (92, 103, 124)
GOOD = (74, 222, 128)
WARN = (251, 191, 36)

LANES = [  # variant, name, subtitle, accent
    ("baseline", "Hugging Face baseline", "transformers as shipped · CPU image prep", (148, 163, 184)),
    ("fused_gemv_pf_vfused+gpuprep", "Ours · BF16", "CUDA graphs + fused CUDA kernels · same weights", (96, 165, 250)),
    ("fused_w8all_pf_vfused+gpuprep", "Ours · INT8", "+ INT8 weights in vision, prompt and decode", (74, 222, 128)),
]
PHASES = [("Image prep", (192, 132, 252)), ("Vision encoder", (251, 191, 36)), ("Reading prompt", (251, 113, 133)),
          ("Writing answer", (34, 211, 238))]
FACTORS = [2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
GPU, N_RUNS = "GPU", 0  # set from the run's env.json / config.json


# ----------------------------------------------------------------------------------------------- drawing helpers
@lru_cache(None)
def font(kind, size):
    names = {"regular": ["Inter-Regular.woff"], "medium": ["Inter-Medium.woff"], "semibold": ["Inter-SemiBold.woff"],
             "bold": ["Inter-Bold.woff"], "mono": ["JetBrainsMono-Regular.woff"],
             "monobold": ["JetBrainsMono-Bold.woff"]}[kind]
    for n in names:
        if (ROOT / "fonts" / n).exists():
            return ImageFont.truetype(str(ROOT / "fonts" / n), size)
    fb = {"mono": "DejaVuSansMono.ttf", "monobold": "DejaVuSansMono-Bold.ttf", "regular": "DejaVuSans.ttf",
          "medium": "DejaVuSans.ttf"}.get(kind, "DejaVuSans-Bold.ttf")
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{fb}", size)


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def tw(d, s, f):
    return d.textlength(s, font=f)


def wrap(d, text, f, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else cur + " " + word
            if tw(d, cand, f) <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = word
            while tw(d, cur, f) > width:  # a single over-long word (e.g. LaTeX): hard break
                k = len(cur)
                while k > 1 and tw(d, cur[:k], f) > width:
                    k -= 1
                lines.append(cur[:k])
                cur = cur[k:]
        lines.append(cur)
    return lines


def check(d, x, y, s, color, w=3):
    d.line([(x, y + s * 0.55), (x + s * 0.38, y + s * 0.9), (x + s, y + s * 0.1)], fill=color, width=w, joint="curve")


def fmt_ms(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"


def background():
    im = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(glow)
    g.ellipse((-400, -500, 900, 500), fill=(18, 28, 52))
    g.ellipse((1200, 600, 2400, 1500), fill=(14, 30, 38))
    return Image.blend(im, glow.filter(ImageFilter.GaussianBlur(220)), 1.0)


BG_IMG = None


def canvas():
    global BG_IMG
    if BG_IMG is None:
        BG_IMG = background()
    im = BG_IMG.copy()
    return im, ImageDraw.Draw(im)


@lru_cache(None)
def thumb(path, bw, bh):
    im = Image.open(path).convert("RGB")
    s = min(bw / im.width, bh / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), 14, fill=255)
    return im, mask


# ----------------------------------------------------------------------------------------------- data
class Lane:
    def __init__(self, tr, meta):
        self.variant, self.name, self.sub, self.color = meta
        r = tr["runs"][tr["median_run"]]
        n = len(tr["answer_ids"])
        self.tok = r["token_ms"][:n]
        self.pieces = tr["pieces"]
        self.total = tr["wall_ms_median"]
        self.text = tr["text"]
        vis = r["vision_ms"][0] if r["vision_ms"] else r["pre_ms"]
        self.bounds = [0.0, r["pre_ms"], vis, self.tok[0], self.total]  # phase k = [bounds[k], bounds[k+1]]
        self.ttft = self.tok[0]
        self.n = n
        self.per_tok = (self.tok[-1] - self.tok[0]) / (n - 1) if n > 1 else None
        self.mem = tr["peak_mem_gb"]
        self.energy = tr["energy_j_per_answer"]
        self.tr = tr

    def shown_tokens(self, t):
        return sum(1 for x in self.tok if x <= t)

    def phase(self, t):
        for k in range(4):
            if t < self.bounds[k + 1]:
                return k
        return 4  # done


def load_run(run):
    run = Path(run)
    traces = json.loads((run / "traces.json").read_text())
    env = json.loads((run / "env.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    by = {(t["variant"], t["key"]): t for t in traces}
    imgs = {p.stem: p for p in (run / "images").iterdir()}
    return by, env, cfg, imgs


def pick_factor(ms, target=(4500, 9000)):
    ok = [f for f in FACTORS if ms * f <= target[1]]
    return max(ok) if ok else 1


# ----------------------------------------------------------------------------------------------- case frames
def draw_top(d, title, subtitle, badge, badge_color, clock_ms, step):
    d.text((60, 38), title, font=font("bold", 40), fill=TEXT)
    d.text((60, 92), subtitle, font=font("regular", 22), fill=MUTED)
    if step:
        f = font("medium", 18)
        d.text((60 + tw(d, title, font("bold", 40)) + 24, 54), step, font=f, fill=DIM)
    # slow-motion badge + real-time clock (top right)
    f = font("bold", 26)
    bw = tw(d, badge, f) + 44
    x1 = W - 60
    d.rounded_rectangle((x1 - bw, 34, x1, 82), 24, fill=mix(BG, badge_color, 0.18), outline=badge_color, width=2)
    d.text((x1 - bw + 22, 41), badge, font=f, fill=badge_color)
    if clock_ms is not None:
        s = f"real time  {clock_ms:7.1f} ms"
        fm = font("mono", 24)
        d.text((x1 - tw(d, s, fm), 96), s, font=fm, fill=MUTED)


def draw_case(case, lanes, imgs, t, badge, badge_color, show_clock=True, step="", overlay=None, fade_metrics=1.0):
    im, d = canvas()
    w_, h_ = lanes[0].tr["image_size"]
    mp = w_ * h_ / 1e6
    size = f"{w_}×{h_} px" + (f" ({mp:.0f} MP)" if mp >= 1.5 else "")
    draw_top(d, case["title"], f"{size} · prompt {lanes[0].tr['prompt_tokens']} tokens incl. image", badge, badge_color,
             t if show_clock else None, step)

    # left: image + prompt
    lx, ly, lw = 60, 150, 560
    d.rounded_rectangle((lx, ly, lx + lw, 1000), 22, fill=PANEL, outline=BORDER, width=1)
    th, mask = thumb(str(imgs[case["key"]]), lw - 40, 520)
    ix = lx + (lw - th.width) // 2
    iy = ly + 20 + (520 - th.height) // 2
    im.paste(th, (ix, iy), mask)
    py = ly + 560
    d.text((lx + 24, py), "PROMPT", font=font("semibold", 16), fill=DIM)
    lines = wrap(d, case["prompt"], font("medium", 26), lw - 48)
    for i, ln in enumerate(lines[:5]):
        d.text((lx + 24, py + 30 + i * 36), ln, font=font("medium", 26), fill=TEXT)
    gy = py + 30 + min(len(lines), 5) * 36 + 22
    d.text((lx + 24, gy), f"max {case['max_new']} new tokens · greedy · batch 1", font=font("regular", 18), fill=DIM)

    # right: three lanes
    rx, rw = 650, W - 60 - 650
    lane_h, gap, top = 270, 15, 150
    tmax = max(L.total for L in lanes)
    base = lanes[0]
    for i, L in enumerate(lanes):
        y = top + i * (lane_h + gap)
        done = t >= L.total
        fill = mix(PANEL, L.color, 0.07) if done else PANEL
        d.rounded_rectangle((rx, y, rx + rw, y + lane_h), 22, fill=fill, outline=mix(BORDER, L.color, 0.6 if done else 0.25),
                            width=2 if done else 1)
        d.rounded_rectangle((rx, y + 18, rx + 6, y + lane_h - 18), 3, fill=L.color)
        d.text((rx + 30, y + 16), L.name, font=font("bold", 28), fill=L.color)
        sub_x = rx + 30 + tw(d, L.name, font("bold", 28)) + 16
        if not done or sub_x + tw(d, L.sub, font("regular", 17)) < rx + rw - 420:  # keep clear of the pill + timer
            d.text((sub_x, y + 26), L.sub, font=font("regular", 17), fill=DIM)
        # live timer
        shown = min(t, L.total)
        tf = font("monobold", 40)
        s = fmt_ms(shown)
        x_t = rx + rw - 30 - tw(d, s, tf)
        d.text((x_t, y + 12), s, font=tf, fill=TEXT if done else mix(TEXT, L.color, 0.3))
        if done:
            check(d, x_t - 40, y + 22, 24, L.color, 4)
        # phase bar on the shared time axis
        bx, by_, bw, bh = rx + 30, y + 70, rw - 60, 16
        d.rounded_rectangle((bx, by_, bx + bw, by_ + bh), 8, fill=(28, 35, 50))
        sc = bw / tmax
        for k in range(4):
            a, b = L.bounds[k], min(L.bounds[k + 1], t)
            if b > a:
                x0, x1 = bx + a * sc, bx + max(b * sc, a * sc + 1.5)
                d.rectangle((x0, by_, x1, by_ + bh), fill=PHASES[k][1])
        # status line
        k = L.phase(t)
        nt = L.shown_tokens(t)
        if done:  # speed-up pill left of the timer; the status line becomes the lane's metrics
            pill = "baseline" if L is base else f"{base.total / L.total:.1f}× faster"
            pf = font("bold", 24)
            pw = tw(d, pill, pf) + 36
            px1 = x_t - 56
            pc = MUTED if L is base else L.color
            d.rounded_rectangle((px1 - pw, y + 16, px1, y + 56), 20, fill=mix(PANEL, pc, 0.16), outline=pc, width=2)
            d.text((px1 - pw + 18, y + 20), pill, font=pf, fill=pc)
            m = [f"first token {fmt_ms(L.ttft)}", f"{L.n} tokens"]
            if L.per_tok:
                m.append(f"{L.per_tok:.1f} ms/token")
            m.append(f"peak memory {L.mem:.2f} GB")
            if L.energy:
                m.append(f"energy {L.energy:.0f} J" if L.energy >= 100 else f"energy {L.energy:.1f} J")
            d.text((bx, y + 95), "  ·  ".join(m), font=font("medium", 19), fill=mix(PANEL, TEXT, 0.85 * fade_metrics))
            if L is not base:
                same = L.tr["answer_ids"] == base.tr["answer_ids"]
                tag = "same answer as baseline" if same else "worded differently from baseline"
                tf3 = font("semibold", 18)
                d.text((bx + bw - tw(d, tag, tf3), y + 96), tag, font=tf3, fill=GOOD if same else WARN)
        else:
            if k == 3:
                status, scol = f"Writing answer · {nt} token{'s' if nt != 1 else ''}", PHASES[3][1]
            else:
                status, scol = (PHASES[k][0] + " …", PHASES[k][1]) if t > 0 else ("ready", DIM)
            d.text((bx, y + 94), status, font=font("semibold", 20), fill=scol)
        # answer text: the last 4 lines while it grows (first line dimmed once it has scrolled)
        txt = L.pieces[nt - 1] if nt else ""
        tf2 = font("regular", 22)
        lines = wrap(d, txt, tf2, bw - 10) if txt else [""]
        vis = lines[-4:]
        for j, ln in enumerate(vis):
            d.text((bx, y + 128 + j * 30), ln, font=tf2, fill=DIM if (j == 0 and len(lines) > 4) else TEXT)
        if not done and k >= 3:  # cursor while writing
            cx = bx + tw(d, vis[-1], tf2) + 3
            cy = y + 128 + (len(vis) - 1) * 30
            d.rectangle((cx, cy + 3, cx + 11, cy + 26), fill=mix(L.color, TEXT, 0.3))

    # left panel foot: how it was measured; bottom: phase legend
    note = (f"Measured on {GPU}: median of {N_RUNS} runs per model after warm-up. "
            "Models ran one after another; shown here from a common start.")
    for j, ln in enumerate(wrap(d, note, font("regular", 17), lw - 48)):
        d.text((lx + 24, 1000 - 24 - 22 * (len(wrap(d, note, font("regular", 17), lw - 48)) - j)), ln,
               font=font("regular", 17), fill=DIM)
    d.text((60, 1018), "LiquidAI LFM2.5-VL-1.6B", font=font("medium", 19), fill=DIM)
    lx0 = 650
    for name, col in PHASES:
        d.rounded_rectangle((lx0, 1021, lx0 + 18, 1039), 4, fill=col)
        d.text((lx0 + 26, 1018), name, font=font("medium", 19), fill=MUTED)
        lx0 += 26 + tw(d, name, font("medium", 19)) + 34

    if overlay:
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        o = ImageDraw.Draw(ov)
        a = overlay[1]
        o.rectangle((0, 0, W, H), fill=(5, 7, 12, int(170 * a)))
        f1 = font("bold", 76)
        f2 = font("regular", 30)
        o.text(((W - o.textlength(overlay[0], font=f1)) / 2, H / 2 - 80), overlay[0], font=f1, fill=(*TEXT, int(255 * a)))
        o.text(((W - o.textlength(overlay[2], font=f2)) / 2, H / 2 + 20), overlay[2], font=f2, fill=(*MUTED, int(255 * a)))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    return im


def case_frames(case, lanes, imgs, factor, step):
    """Yields frames: intro, real-speed pass, 'slowed down' card, slow-motion pass, hold."""
    tmax = max(L.total for L in lanes)
    rt_badge, rt_col = "REAL SPEED  1×", (226, 232, 240)
    sm_badge, sm_col = f"SLOW MOTION  {factor}× slower", (251, 191, 36)
    for _ in range(int(1.2 * FPS)):
        yield draw_case(case, lanes, imgs, 0.0, rt_badge, rt_col, step=step)
    n = max(1, math.ceil(tmax / 1000 * FPS))
    for i in range(n + 1):
        yield draw_case(case, lanes, imgs, min(tmax, i * 1000 / FPS), rt_badge, rt_col, step=step)
    for _ in range(int(1.0 * FPS)):
        yield draw_case(case, lanes, imgs, tmax, rt_badge, rt_col, step=step)
    msg = f"Again, {factor}× slower"
    sub = f"1 second of video = {1000 / factor:g} ms of real time"
    k = int(1.4 * FPS)
    for i in range(k):
        a = min(1.0, i / 8, (k - i) / 8)
        yield draw_case(case, lanes, imgs, 0.0, sm_badge, sm_col, step=step, overlay=(msg, a, sub))
    n = math.ceil(tmax * factor / 1000 * FPS)
    for i in range(n + 1):
        yield draw_case(case, lanes, imgs, min(tmax, i * 1000 / FPS / factor), sm_badge, sm_col, step=step)
    for _ in range(int(4.0 * FPS)):
        yield draw_case(case, lanes, imgs, tmax, sm_badge, sm_col, step=step)


# ----------------------------------------------------------------------------------------------- cards
def title_frames(env):
    for i in range(int(5 * FPS)):
        a = min(1.0, i / 15)
        im, d = canvas()
        c = lambda col: mix(BG, col, a)  # noqa: E731
        d.text((140, 300), "CS3.406 · Optimizing a Vision-Language Model for Edge Devices", font=font("semibold", 28),
               fill=c(MUTED))
        d.text((140, 360), "Same image, same question,", font=font("bold", 84), fill=c(TEXT))
        d.text((140, 460), "three ways to run the model", font=font("bold", 84), fill=c((96, 165, 250)))
        d.text((140, 600), "LiquidAI LFM2.5-VL-1.6B · measured on " + GPU, font=font("medium", 32),
               fill=c(MUTED))
        y = 700
        for _, name, sub, col in LANES:
            d.rounded_rectangle((140, y + 8, 158, y + 26), 5, fill=c(col))
            d.text((176, y), name, font=font("semibold", 28), fill=c(TEXT))
            d.text((176 + tw(d, name, font("semibold", 28)) + 16, y + 6), sub, font=font("regular", 22), fill=c(DIM))
            y += 46
        d.text((140, 960), "Jainil Bavishi · Gautam Gandhi · Bibek Singh Dhody", font=font("medium", 24), fill=c(DIM))
        yield im


def howto_frames(n_runs, warmup):
    items = [
        ("What you see", "Each model gets the same image and prompt. Words appear at the exact time the GPU produced "
                         "them, and the bar fills by phase on one shared time axis."),
        ("Measured, not simulated", f"Every case ran {warmup} warm-up + {n_runs} timed times per model on the GPU; "
                                    "the video replays the median run from its CUDA-event timestamps. The clock starts "
                                    "with the decoded image in memory and stops when the answer is complete."),
        ("Slow motion", "Answers take tens to hundreds of milliseconds. Each case plays once at real speed, then "
                        "slowed down by the factor shown at the top right; the real-time clock keeps counting real "
                        "milliseconds."),
        ("Fair comparison", "Greedy decoding, batch 1, stop at end of answer, measured one after another on the same "
                            "GPU. Answers are shown exactly as each model produced them. Ours · BF16 keeps LiquidAI's "
                            "weights; its fused kernels round slightly differently, so a long answer can be worded "
                            "differently (benchmark accuracy unchanged)."),
        ("Memory and energy", "Peak GPU memory during the request. Energy: whole GPU board (NVML counter), averaged "
                              "over at least 3 s of repeated runs of the same request."),
    ]
    for i in range(int(16 * FPS)):
        a = min(1.0, i / 12)
        im, d = canvas()
        d.text((140, 90), "How to read this video", font=font("bold", 56), fill=mix(BG, TEXT, a))
        x, y = 140, 200
        for col, (name, _) in zip([p[1] for p in PHASES], PHASES):
            d.rounded_rectangle((x, y + 4, x + 24, y + 28), 6, fill=mix(BG, col, a))
            d.text((x + 36, y), name, font=font("semibold", 26), fill=mix(BG, TEXT, a))
            x += 36 + tw(d, name, font("semibold", 26)) + 50
        y = 280
        for j, (h, body) in enumerate(items):
            aj = max(0.0, min(1.0, (i - 10 - j * 10) / 12))
            d.text((140, y), h, font=font("bold", 30), fill=mix(BG, (96, 165, 250), aj))
            for k, ln in enumerate(wrap(d, body, font("regular", 26), 1600)):
                d.text((140, y + 44 + k * 36), ln, font=font("regular", 26), fill=mix(BG, MUTED, aj))
            y += 44 + len(wrap(d, body, font("regular", 26), 1600)) * 36 + 26
        yield im


def summary_frames(rows, env):
    """rows: [(case, [Lane x3])]"""
    for i in range(int(12 * FPS)):
        a = min(1.0, i / 12)
        grow = min(1.0, max(0.0, (i - 6) / 30))
        grow = 1 - (1 - grow) ** 3
        im, d = canvas()
        d.text((80, 50), "All cases: time to the complete answer", font=font("bold", 48), fill=mix(BG, TEXT, a))
        d.text((80, 112), f"median of {N_RUNS} runs on {GPU} · bars relative to the Hugging Face baseline of each case",
               font=font("regular", 22), fill=mix(BG, MUTED, a))
        x0, x1 = 520, 1380
        y = 170
        rh = (880 - 170) / len(rows)
        bh = min(18, (rh - 14) / 3)
        for case, lanes in rows:
            base = lanes[0].total
            w_, h_ = lanes[0].tr["image_size"]
            d.text((80, y + 2), case["title"], font=font("semibold", 22), fill=mix(BG, TEXT, a))
            d.text((80, y + 30), f"{w_}×{h_} · {lanes[0].n} tokens", font=font("regular", 18), fill=mix(BG, DIM, a))
            for j, L in enumerate(lanes):
                yy = y + j * (bh + 4)
                ww = (x1 - x0) * min(1.0, L.total / base) * grow
                d.rounded_rectangle((x0, yy, x0 + max(ww, 4), yy + bh), 4, fill=mix(BG, L.color, a))
                s = fmt_ms(L.total) + ("" if j == 0 else f"   {base / L.total:.1f}×")
                d.text((x0 + max(ww, 4) + 10, yy - 3), s, font=font("medium", 17), fill=mix(BG, MUTED, a * grow))
            y += rh
        # right column: averages
        px = 1500
        d.rounded_rectangle((px - 30, 170, W - 60, 880), 22, fill=mix(BG, PANEL, a), outline=mix(BG, BORDER, a))
        yy = 200
        d.text((px, yy), "Across these cases", font=font("bold", 28), fill=mix(BG, TEXT, a))
        yy += 56

        def gmean(xs):
            return math.exp(sum(math.log(x) for x in xs) / len(xs))

        for j, (_, name, _, col) in enumerate(LANES):
            sp = gmean([ls[0].total / ls[j].total for _, ls in rows])
            mem = max(ls[j].mem for _, ls in rows)
            en = [ls[j].energy / ls[0].energy for _, ls in rows if ls[j].energy and ls[0].energy]
            d.text((px, yy), name, font=font("semibold", 24), fill=mix(BG, col, a))
            d.text((px, yy + 36), "speed-up (geo. mean)", font=font("regular", 18), fill=mix(BG, DIM, a))
            d.text((px + 230, yy + 32), f"{sp:.2f}×", font=font("monobold", 24), fill=mix(BG, TEXT, a))
            d.text((px, yy + 68), "peak GPU memory (max)", font=font("regular", 18), fill=mix(BG, DIM, a))
            d.text((px + 230, yy + 64), f"{mem:.2f} GB", font=font("monobold", 24), fill=mix(BG, TEXT, a))
            if en:
                d.text((px, yy + 100), "energy per answer", font=font("regular", 18), fill=mix(BG, DIM, a))
                d.text((px + 230, yy + 96), f"{gmean(en) * 100:.0f} % of HF", font=font("monobold", 24), fill=mix(BG, TEXT, a))
            yy += 200
        d.text((80, 930), "INT8 is a separately reported extension (changes the weights); on 5 standard benchmarks and "
                          "COCO captions its accuracy matched the baseline.", font=font("regular", 20), fill=mix(BG, DIM, a))
        d.text((80, 962), "Ours · BF16 keeps LiquidAI's weights unchanged.", font=font("regular", 20), fill=mix(BG, DIM, a))
        yield im


# ----------------------------------------------------------------------------------------------- encode
def encode(frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        p.stdin.write(f.tobytes())
        n += 1
    p.stdin.close()
    p.wait()
    return n


def job(spec):
    kind, path = spec[0], Path(spec[-1])
    S = STATE
    if kind == "title":
        n = encode(title_frames(S["env"]), path)
    elif kind == "howto":
        n = encode(howto_frames(S["cfg"]["repeats"], S["cfg"]["warmup"]), path)
    elif kind == "summary":
        n = encode(summary_frames([(c, S["lanes"][c["key"]]) for c in S["cases"]], S["env"]), path)
    else:
        c, step = spec[1], spec[2]
        lanes = S["lanes"][c["key"]]
        n = encode(case_frames(c, lanes, S["imgs"], pick_factor(max(L.total for L in lanes)), step), path)
    return path.name, n / FPS


STATE = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cases", default="", help="comma-separated case keys in play order (default: all recorded)")
    ap.add_argument("--out", default="presentations/demo")
    ap.add_argument("--name", default="LFM2.5-VL_demo_L40S")
    ap.add_argument("--still", default="", help="case:ms -> write one PNG instead of the video")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    by, env, cfg, imgs = load_run(args.run)
    global GPU, N_RUNS
    GPU, N_RUNS = env["gpu_name"], cfg["repeats"]
    cases_cfg = {c["key"]: c for c in cfg["cases"]}
    keys = args.cases.split(",") if args.cases else [c["key"] for c in cfg["cases"]]
    cases = [cases_cfg[k] for k in keys]
    lanes = {c["key"]: [Lane(by[(v, c["key"])], (v, n, s, col)) for v, n, s, col in LANES] for c in cases}
    STATE.update(env=env, cfg=cfg, imgs=imgs, cases=cases, lanes=lanes)
    out = Path(args.out)

    if args.still:
        key, _, ms = args.still.partition(":")
        c = cases_cfg[key]
        ls = lanes[key] if key in lanes else [Lane(by[(v, key)], (v, n, s, col)) for v, n, s, col in LANES]
        f = pick_factor(max(L.total for L in ls))
        if ms == "summary":
            im = list(summary_frames([(c2, lanes[c2["key"]]) for c2 in cases], env))[-1]
        elif ms == "title":
            im = list(title_frames(env))[-1]
        elif ms == "howto":
            im = list(howto_frames(cfg["repeats"], cfg["warmup"]))[-1]
        else:
            im = draw_case(c, ls, imgs, float(ms), f"SLOW MOTION  {f}× slower", (251, 191, 36), step="")
        p = out / f"still_{key}_{ms}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p)
        print(p)
        return

    specs = [("title", out / "parts" / "00_title.mp4"), ("howto", out / "parts" / "01_howto.mp4")]
    for i, c in enumerate(cases):
        specs.append(("case", c, f"case {i + 1} of {len(cases)}", out / "clips" / f"{i + 1:02d}_{c['key']}.mp4"))
    specs.append(("summary", out / "parts" / "99_summary.mp4"))
    for c in cases:
        ls = lanes[c["key"]]
        print(f"[render] {c['key']:>20}: " + "  ".join(f"{L.name} {fmt_ms(L.total)}" for L in ls)
              + f"  -> {pick_factor(max(L.total for L in ls))}× slower")
    with Pool(args.jobs) as pool:
        for name, sec in pool.imap(job, specs):
            print(f"[render] {name}: {sec:.1f} s")
    lst = out / "parts" / "concat.txt"
    lst.write_text("".join(f"file '{Path(s[-1]).resolve()}'\n" for s in specs))
    final = out / f"{args.name}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(final)], check=True)
    print(f"[render] -> {final}")


if __name__ == "__main__":
    main()
