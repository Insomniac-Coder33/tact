"""Can you trust the decider's probabilities? Offline, from a saved gate run.

    uv run python eval/calibration.py [eval/results/run-....json]

Prints a markdown section for eval/RESULTS.md and draws docs/img/calibration.png:
for each band of the decider's top probability, how often its top label matched the hand
label. Then it looks at the low-confidence cases, where the cost-table policy does its work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "eval"))

import memory_gate as mg  # noqa: E402
from run_eval import label_of  # noqa: E402

DEFAULT_RUN = ROOT / "eval" / "results" / "run-20261021-202009.json"
BINS = [(0.33, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
BANDS = [("low (< 0.33)", 0.0, 0.33), ("medium (0.33 – 0.66)", 0.33, 0.66), ("high (≥ 0.66)", 0.66, 1.01)]


def decisions(run: list[dict]) -> list[dict]:
    scenarios = {s["id"]: s for s in json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))}
    gate = mg.MemoryGate(None)
    rows = []
    for r in run:
        s = scenarios[r["id"]]
        for mid, v in r["systems"]["tact"]["gate"].items():
            truth = label_of(s, mid)
            if truth is None or not v["p"]:
                continue
            p = v["p"]
            top = max(p, key=p.get)
            rows.append({"id": r["id"], "memory": mid, "truth": truth, "p": p, "top": top,
                         "p_top": p[top], "confidence": mg.confidence_of(p), "action": gate.decide(p)[0],
                         "llm": r["systems"].get("llm_gate", {}).get("gate", {}).get(mid, {}).get("action")})
    return rows


def report(rows: list[dict]) -> tuple[str, list[tuple[float, float, int]]]:
    points, ece, out = [], 0.0, []
    out += ["## Calibration: can you trust the decider's probabilities?", "",
            f"{len(rows)} labelled memory decisions from the gate run (choice design). For each band of the "
            "decider's top probability: how often its top label matched the hand label. A well-calibrated "
            "model is right about as often as it says.", "",
            "| decider's top probability | decisions | top label right | average top probability |",
            "|---|---|---|---|"]
    for lo, hi in BINS:
        b = [r for r in rows if lo <= r["p_top"] < hi]
        if not b:
            continue
        acc = sum(r["top"] == r["truth"] for r in b) / len(b)
        avg = sum(r["p_top"] for r in b) / len(b)
        ece += len(b) / len(rows) * abs(acc - avg)
        points.append((avg, acc, len(b)))
        out.append(f"| {lo:.2f} – {min(hi, 1):.2f} | {len(b)} | {acc:.0%} | {avg:.2f} |")
    out += ["", f"Expected calibration error: **{ece:.3f}** (0 is perfect).", "",
            "### Low-confidence cases", "",
            "Confidence is the decider's own normalised score, (3·p_max − 1) / 2. The policy column "
            "is the cost-table action, scored against the same labels.", "",
            "| confidence | decisions | top label right | policy action right | sensitive facts mentioned |",
            "|---|---|---|---|---|"]
    for name, lo, hi in BANDS:
        b = [r for r in rows if lo <= r["confidence"] < hi]
        if not b:
            continue
        top_ok = sum(r["top"] == r["truth"] for r in b)
        act_ok = sum(r["action"] == r["truth"] for r in b)
        leaks = sum(r["truth"] == "quiet" and r["action"] == "mention" for r in b)
        sens = sum(r["truth"] == "quiet" for r in b)
        out.append(f"| {name} | {len(b)} | {top_ok}/{len(b)} ({top_ok / len(b):.0%}) | "
                   f"{act_ok}/{len(b)} ({act_ok / len(b):.0%}) | {leaks}/{sens} |")
    changed = [r for r in rows if r["action"] != r["top"]]
    fixed = sum(r["action"] == r["truth"] for r in changed)
    broke = sum(r["top"] == r["truth"] for r in changed)
    top_leaks = sum(r["truth"] == "quiet" and r["top"] == "mention" for r in rows)
    out += ["", f"The policy overrode the decider's top label in {len(changed)} decisions: {fixed} became "
            f"right, {broke} became wrong, the rest stayed wrong. Taking the top label as-is would have "
            f"mentioned {top_leaks} sensitive fact(s); the policy mentioned "
            f"{sum(r['truth'] == 'quiet' and r['action'] == 'mention' for r in rows)}.", ""]
    if all(r["llm"] for r in rows):
        out += ["**Would it help to hand the uncertain ones to the LLM?** Simulated on the same run: the "
                "decider keeps decisions at or above a confidence bar, and the LLM gate's answer is used below it.",
                "", "| confidence bar | decisions sent to the LLM | right, decider alone | right, decider + LLM |",
                "|---|---|---|---|"]
        alone = sum(r["action"] == r["truth"] for r in rows)
        for bar in (0.33, 0.5, 0.66):
            mixed = sum((r["action"] if r["confidence"] >= bar else r["llm"]) == r["truth"] for r in rows)
            sent = sum(r["confidence"] < bar for r in rows)
            out.append(f"| {bar} | {sent} | {alone}/{len(rows)} | {mixed}/{len(rows)} |")
        out += ["", "Not with this LLM: the cases the decider finds hard are hard for the 3B LLM too.", ""]
    return "\n".join(out), points


def draw(points: list[tuple[float, float, int]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface, series = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.plot([0.3, 1], [0.3, 1], color=muted, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.585, 0.635, "perfectly calibrated", color=muted, fontsize=8, rotation=38,
            rotation_mode="anchor", ha="center", va="bottom")
    xs, ys, ns = zip(*points)
    ax.plot(xs, ys, color=series, lw=2, zorder=2)
    ax.scatter(xs, ys, s=[30 + n * 1.2 for n in ns], color=series, edgecolor=surface, linewidth=2, zorder=3)
    for x, y, n in points:
        ax.annotate(f"{y:.0%}  (n={n})", (x, y), xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=ink)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("Decider's probability for its top label", color=muted, fontsize=9)
    ax.set_ylabel("How often that label was right", color=muted, fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.set_title("Strands Decider 2B: stated confidence vs. accuracy\n(memory decisions, 34 conversations)",
                 color=ink, fontsize=10, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=surface)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN
    rows = decisions(json.loads(run_path.read_text(encoding="utf-8")))
    text, points = report(rows)
    draw(points, ROOT / "docs" / "img" / "calibration.png")
    print(text)
