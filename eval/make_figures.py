"""Figures for the article and README, drawn from the saved evaluation runs (no model calls).

    uv run python eval/make_figures.py

    docs/img/two_conversations.png   one memory, two conversations, two decisions
    docs/img/architecture.png        the read path and the write path
    docs/img/cover.png               the article's cover image
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))

from memory_gate import MemoryGate  # noqa: E402

IMG = ROOT / "docs" / "img"
LIGHT = {"surface": "#fcfcfb", "card": "#f4f3f0", "ink": "#0b0b0b", "secondary": "#52514e",
         "muted": "#898781", "hairline": "#e1e0d9"}
DARK = {"surface": "#1a1a19", "card": "#242422", "ink": "#f0efec", "secondary": "#c3c2b7",
        "muted": "#898781", "hairline": "#383835"}
# The three lanes take slots 1-3 of the validated categorical palette, in this fixed order
# (all pairs pass the colour-blindness checks in both modes).
LANE = {"mention": "#2a78d6", "quiet": "#eb6834", "skip": "#1baf7a"}
LANE_DARK = {"mention": "#3987e5", "quiet": "#d95926", "skip": "#199e70"}
SAY = {"mention": "may mention it", "quiet": "uses it quietly", "skip": "leaves it out"}
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def canvas(width: int, height: int, colors: dict):
    """A figure whose data units are output pixels (at dpi 200), so shapes are exact."""
    fig = plt.figure(figsize=(width / 200, height / 200), dpi=200)
    fig.patch.set_facecolor(colors["surface"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def luminance(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def text_on(fill: str, colors: dict) -> str:
    """White or ink, whichever contrasts more with the fill."""
    lum = luminance(fill)
    white = 1.05 / (lum + 0.05)
    ink = (lum + 0.05) / (luminance(LIGHT["ink"]) + 0.05)
    return "#ffffff" if white >= ink else LIGHT["ink"]


def stacked_bar(ax, fig, x: float, y: float, w: float, h: float, p: dict, lanes: dict, colors: dict,
                labels: bool = True):
    """A 100% bar of P(mention), P(quiet), P(skip): 2px surface gaps, rounded ends, and a
    label inside a segment only when it fits."""
    outline = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                             facecolor="none", edgecolor="none")
    ax.add_patch(outline)
    renderer = fig.canvas.get_renderer()
    left, gap = x, 4  # 2 css px at 2x
    for label in ("mention", "quiet", "skip"):
        seg = w * p[label]
        rect = Rectangle((left, y), max(seg - gap, 0), h, facecolor=lanes[label], edgecolor="none")
        ax.add_patch(rect)
        rect.set_clip_path(outline)
        if labels:
            t = ax.text(left + 16, y + h / 2, f"{label} {p[label]:.0%}", va="center", ha="left",
                        fontsize=8.5, color=text_on(lanes[label], colors))
            if t.get_window_extent(renderer).width + 32 > seg - gap:
                t.remove()  # never clip a label: the legend and the caption carry it
        left += seg


def legend(ax, x: float, y: float, colors: dict, lanes: dict, items: list[tuple[str, str]]):
    for label, words in items:
        ax.add_patch(FancyBboxPatch((x, y - 9), 18, 18, boxstyle="round,pad=0,rounding_size=4",
                                    facecolor=lanes[label], edgecolor="none"))
        t = ax.text(x + 28, y, words, va="center", fontsize=8.5, color=colors["secondary"])
        x = t.get_window_extent(ax.figure.canvas.get_renderer()).x1 + 40


def inline(ax, x: float, y: float, parts: list[tuple[str, dict]]) -> None:
    """Several differently styled pieces of text on one line."""
    renderer = ax.figure.canvas.get_renderer()
    for words, style in parts:
        t = ax.text(x, y, words, va="center", **style)
        x = t.get_window_extent(renderer).x1 + 10


def two_conversations() -> None:
    """The same memory (Biscuit died) gets two decisions in two conversations."""
    runs = {"s12": ("scenarios.json", "results/run-20261021-202009.json"),
            "f08": ("fresh_scenarios.json", "results/run-20261022-001433.json")}
    gate = MemoryGate(ask=None)
    rows = []
    for sid, (scen, run) in runs.items():
        message = next(s["message"] for s in json.loads((ROOT / "eval" / scen).read_text(encoding="utf-8"))
                       if s["id"] == sid)
        saved = next(r for r in json.loads((ROOT / "eval" / run).read_text(encoding="utf-8")) if r["id"] == sid)
        p = saved["systems"]["tact"]["gate"]["m05"]["p"]
        rows.append((sid, message, p, gate.decide(p)[0]))
    memory = next(m["text"] for m in json.loads((ROOT / "data" / "memories.json").read_text(encoding="utf-8"))
                  if m["id"] == "m05")

    c = LIGHT
    fig, ax = canvas(1600, 800, c)
    ax.text(64, 725, "Same memory, two conversations", fontsize=15, weight="semibold", color=c["ink"])
    ax.text(64, 675, f"Remembered: “{memory}”", fontsize=9.5, color=c["secondary"])
    legend(ax, 64, 622, c, LANE, [("mention", "mention: say it"), ("quiet", "quiet: use it, never say it"),
                                  ("skip", "skip: leave it out")])
    why = {"s12": "he didn't bring it up: it may shape the reply, but is never said",
           "f08": "he brought it up himself, so ignoring it would be cold"}
    for i, (sid, message, p, action) in enumerate(rows):
        top = 535 - i * 225
        ax.text(64, top, f"User: “{message}”", fontsize=10, color=c["ink"], va="center")
        stacked_bar(ax, fig, 64, top - 95, 1472, 48, p, LANE, c)
        ax.add_patch(FancyBboxPatch((64, top - 140), 18, 18, boxstyle="round,pad=0,rounding_size=9",
                                    facecolor=LANE[action], edgecolor="none"))
        inline(ax, 96, top - 131, [(f"Tact {SAY[action]}", {"fontsize": 10, "weight": "semibold",
                                                            "color": c["ink"]}),
                                   (f"— {why[sid]}", {"fontsize": 9, "color": c["secondary"]})])
    ax.text(64, 52, "Bars: Strands Decider 2B's probabilities. Decision: Tact's expected-cost policy (saying "
            "something painful costs 3, losing a useful fact 1.5).\nEvaluation scenarios s12 and f08; f08 is "
            "from the fresh set, written after the configuration was frozen.",
            fontsize=7, color=c["muted"], linespacing=1.6)
    fig.savefig(IMG / "two_conversations.png", facecolor=c["surface"])
    plt.close(fig)
    print("wrote docs/img/two_conversations.png:", [(sid, action) for sid, _, _, action in rows])


def box(ax, x, top, w, title, body, colors, accent=None, step=None) -> float:
    """A card that grows with its lines of text. Returns its bottom edge."""
    h = 66 + 34 * len(body) + (8 if body else 0)
    y = top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                facecolor=colors["card"], edgecolor=colors["hairline"], linewidth=1))
    if accent:
        ax.add_patch(FancyBboxPatch((x, y), 10, h, boxstyle="round,pad=0,rounding_size=5",
                                    facecolor=accent, edgecolor="none"))
    tx = x + (34 if accent else 26)
    head = f"{step}  {title}" if step else title
    ax.text(tx, top - 34, head, fontsize=10, weight="semibold", color=colors["ink"], va="center")
    for i, line in enumerate(body):
        ax.text(tx, top - 76 - i * 34, line, fontsize=8, color=colors["secondary"], va="center")
    return y


def arrow(ax, start, end, colors):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>,head_length=8,head_width=5",
                                 color=colors["muted"], linewidth=1.6, shrinkA=0, shrinkB=4))


def architecture() -> None:
    c = LIGHT
    fig, ax = canvas(1760, 1180, c)
    ax.text(64, 1110, "How Tact decides what to remember, and what to say", fontsize=14,
            weight="semibold", color=c["ink"])
    ax.text(64, 1062, "Every user message takes the read path. When the message is news about the user, "
            "the write path runs too.", fontsize=9, color=c["secondary"])
    L, LW, R, RW, GAP = 64, 820, 1084, 612, 46
    read = [("User message", [], None, None),
            ("Embeddings shortlist the 8 most similar memories",
             ["similar is not the same as appropriate: that is the next step's job"], None, "1"),
            ("Strands Decider 2B, one batched request",
             ["8 × choice: mention / quiet / skip, one per memory",
              "+ remember?   + drop a subject?   + tone?",
              "the conversation is read once; each question adds a few tokens"], LANE["mention"], "2"),
            ("Expected-cost policy, in plain Python",
             ["saying something painful costs 3× a missed detail,",
              "so when in doubt: use it quietly"], LANE["quiet"], "3"),
            ("The chat model writes the reply",
             ["Amazon Bedrock (Nova Lite), or Ollama where Bedrock is unavailable",
              "it sees only what the gate allowed: facts it may mention,",
              "and sensitive background it must never mention"], None, "4")]
    top, tops = 990, []
    for title, body, accent, step in read:
        tops.append(top)
        bottom = box(ax, L, top, LW, title, body, c, accent, step)
        if len(tops) < len(read):
            arrow(ax, (L + LW / 2, bottom), (L + LW / 2, bottom - GAP), c)
        top = bottom - GAP

    write = [("Strands Decider 2B, write side", ["for the 5 closest stored memories:",
              "outdated / repeat / unaffected", "+ is it a past event that stays true?"], LANE["mention"]),
             ("Three bands, not one threshold", ["retire at ≥ 0.6 · flag 0.4–0.6 · keep",
              "a past event is never retired, only flagged"], LANE["quiet"]),
             ("Memory store", ["“I moved to Bangalore” retires “lives in Pune”",
              "retired facts leave retrieval, kept in history;",
              "flagged ones are marked “may be out of date”"], None)]
    for i, (title, body, accent) in enumerate(write):
        bottom = box(ax, R, tops[i + 2], RW, title, body, c, accent)
        if i < len(write) - 1:
            arrow(ax, (R + RW / 2, bottom), (R + RW / 2, tops[i + 3]), c)
    y = tops[2] - 90
    arrow(ax, (L + LW, y), (R, y), c)
    ax.text(L + LW + (R - L - LW) / 2, y + 22, "news about himself", fontsize=7.5, ha="center",
            color=c["secondary"])
    ax.text(64, 62, "Blue bar: a Strands Decider 2B decision. Orange bar: plain Python that turns its "
            "probabilities into an action.\nIf the decider is unreachable: no memories this turn, and no "
            "change to the store.", fontsize=7.5, color=c["muted"], linespacing=1.6)
    fig.savefig(IMG / "architecture.png", facecolor=c["surface"])
    plt.close(fig)
    print("wrote docs/img/architecture.png")


def cover() -> None:
    c = DARK
    fig, ax = canvas(1600, 900, c)
    ax.text(108, 610, "Tact", fontsize=60, weight="bold", color=c["ink"])
    for i, line in enumerate(("An assistant that", "remembers everything,", "and knows when", "to keep quiet.")):
        ax.text(114, 520 - i * 56, line, fontsize=17, color=c["secondary"])
    ax.text(114, 250, "Strands Decider 2B  ·  Strands Agents  ·  #context-memory", fontsize=9.5,
            color=c["muted"])

    saved = next(r for r in json.loads((ROOT / "eval" / "results" / "run-20261021-202009.json")
                                       .read_text(encoding="utf-8")) if r["id"] == "s12")
    p = saved["systems"]["tact"]["gate"]["m05"]["p"]
    x, y, w, h = 820, 150, 690, 600
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=22",
                                facecolor=c["card"], edgecolor=c["hairline"], linewidth=1.2))
    ax.text(x + 44, y + h - 64, "USER", fontsize=8, weight="semibold", color=c["muted"])
    ax.text(x + 44, y + h - 112, "“I'm thinking of getting a puppy. Thoughts?”", fontsize=11,
            color=c["ink"])
    ax.text(x + 44, y + h - 200, "TACT REMEMBERS", fontsize=8, weight="semibold", color=c["muted"])
    ax.text(x + 44, y + h - 248, "Arjun's dog Biscuit passed away last month.", fontsize=11,
            color=c["secondary"])
    stacked_bar(ax, fig, x + 44, y + 210, w - 88, 40, p, LANE_DARK, c, labels=False)
    ax.add_patch(FancyBboxPatch((x + 44, y + 122), 18, 18, boxstyle="round,pad=0,rounding_size=9",
                                facecolor=LANE_DARK["quiet"], edgecolor="none"))
    ax.text(x + 78, y + 131, "Tact uses it quietly", va="center", fontsize=12.5, weight="semibold",
            color=c["ink"])
    ax.text(x + 78, y + 78, "it shapes the reply, but is never said", va="center", fontsize=10.5,
            color=c["secondary"])
    fig.savefig(IMG / "cover.png", facecolor=c["surface"])
    plt.close(fig)
    print("wrote docs/img/cover.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    two_conversations()
    architecture()
    cover()
