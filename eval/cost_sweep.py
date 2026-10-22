"""Re-score the saved decider answers at every cost of saying something painful, with no
model calls, and compare the cost table with the decider's own top label.

    uv run python eval/cost_sweep.py      # prints the markdown section used in RESULTS.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval as ev  # noqa: E402
from memory_gate import COST, MemoryGate  # noqa: E402

SETS = {"original": ("eval/scenarios.json", "eval/results/run-20261021-202009.json"),
        "fresh": ("eval/fresh_scenarios.json", "eval/results/run-20261022-001433.json")}
SWEEP = (0, 0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.5, 4, 5, 6)


def load() -> dict[str, list[tuple[dict, dict]]]:
    out = {}
    for name, (scenario_file, run_file) in SETS.items():
        scenarios = {s["id"]: s for s in json.loads((ROOT / scenario_file).read_text(encoding="utf-8"))}
        out[name] = [(scenarios[r["id"]], r) for r in json.loads((ROOT / run_file).read_text(encoding="utf-8"))]
    return out


def score(pairs: list[tuple[dict, dict]], act) -> dict:
    """Scored like run_eval.gate_table: `act(p)` gives the action for one memory."""
    n = dict.fromkeys(("right", "total", "bad", "bad_n", "flip", "flip_n", "keep", "keep_n"), 0)
    for s, row in pairs:
        for mid, v in row["systems"]["tact"]["gate"].items():
            gold = ev.label_of(s, mid)
            if gold is None:
                continue
            a = act(v["p"])
            n["total"] += 1
            n["right"] += a == gold
            if mid in ev.SENSITIVE and gold != "mention":
                n["bad_n"] += 1
                n["bad"] += a == "mention"
            if mid in ev.SENSITIVE and gold == "mention":
                n["flip_n"] += 1
                n["flip"] += a == "mention"
            if gold == "mention":
                n["keep_n"] += 1
                n["keep"] += a in ("mention", "quiet")
    return n


def wilson(k: int, n: int, z: float = 1.96) -> str:
    """A 95% confidence interval for k out of n."""
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{max(0.0, centre - half):.0%}–{min(1.0, centre + half):.0%}"


def policy(c: float):
    cost = {a: dict(row) for a, row in COST.items()}
    cost["mention"]["quiet"] = c
    gate = MemoryGate(ask=None, cost=cost)
    return lambda p: gate.decide(p)[0]


def top_label(p: dict) -> str:
    return max(p, key=p.get)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = load()
    both = data["original"] + data["fresh"]
    lines = ["## The cost of saying something painful: which mistakes, not how many", "",
             "The saved decider answers, re-scored with `COST[\"mention\"][\"quiet\"]` from 0 to 6 "
             "(3 is shipped), on all 50 conversations. No model calls.", "",
             "| cost | right decisions | sensitive facts let through ↓ | user-raised subjects mentioned ↑ | "
             "useful facts kept ↑ |", "|---|---|---|---|---|"]
    for c in SWEEP:
        n = score(both, policy(c))
        mark = "**" if c == COST["mention"]["quiet"] else ""
        lines.append(f"| {mark}{c}{mark} | {n['right']}/{n['total']} ({n['right'] / n['total']:.0%}) | "
                     f"{n['bad']}/{n['bad_n']} | {n['flip']}/{n['flip_n']} | {n['keep']}/{n['keep_n']} |")
    lines += ["", "Accuracy stays within 80–85%; the cost moves the kind of mistake. 3 was chosen on the "
              "10-scenario dev split (`eval/tune.py`); on all 50 it is the lowest value in this sweep that "
              "lets no sensitive fact through.", "",
              "### The cost table vs. the decider's top label", "",
              "| scenarios | gate | right decisions (95% CI) | sensitive facts let through ↓ | "
              "user-raised subjects mentioned ↑ | useful facts kept ↑ |", "|---|---|---|---|---|---|"]
    for name, pairs in (("34 original", data["original"]), ("16 fresh", data["fresh"]), ("all 50", both)):
        for gate, act in (("top label", top_label), ("cost table (3)", policy(COST["mention"]["quiet"]))):
            n = score(pairs, act)
            lines.append(f"| {name} | {gate} | {n['right']}/{n['total']} ({n['right'] / n['total']:.0%}, "
                         f"{wilson(n['right'], n['total'])}) | {n['bad']}/{n['bad_n']} "
                         f"({wilson(n['bad'], n['bad_n'])}) | {n['flip']}/{n['flip_n']} | "
                         f"{n['keep']}/{n['keep_n']} |")
    lines += ["", "Confidence intervals are Wilson 95% intervals. Made by `eval/cost_sweep.py`.", ""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
