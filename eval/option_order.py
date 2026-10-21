"""Does the decider's answer depend on the order the options are listed in?

    uv run python eval/option_order.py            # asks the decider (3 orders x 34 scenarios)
    uv run python eval/option_order.py --rescore  # re-scores the saved answers

Every memory question is asked with the three options in three rotations. A robust decision
gives the same policy action whatever the order; a model with a position bias picks the
first-listed option more often than its share.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import memory_gate as mg  # noqa: E402

SAVED = ROOT / "eval" / "results" / "option_order.json"
ORDERS = [("mention", "quiet", "skip"), ("quiet", "skip", "mention"), ("skip", "mention", "quiet")]


def collect() -> list[dict]:
    from run_eval import conversation_of
    from tact.agent import decider_url
    from tact.memory import MemoryStore

    scenarios = json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))
    store = MemoryStore()
    ask = mg.http_ask(decider_url())
    state_of = mg.MemoryGate.render_state
    rows = []
    for s in scenarios:
        candidates = store.candidates(s["message"], k=8)
        row = {"id": s["id"], "orders": []}
        for order in ORDERS:
            options = {k: mg.OPTIONS[k] for k in order}
            questions = {m["id"]: mg.choice(mg.INSTRUCTIONS.format(memory=m["text"]), options)
                         for m in candidates}
            answers = ask(state_of(conversation_of(s)), questions)
            row["orders"].append({m: {k: float(a["probabilities"][k]) for k in mg.LABELS}
                                  for m, a in answers.items()})
        print(f"[{s['id']}] {len(candidates)} memories x {len(ORDERS)} orders", flush=True)
        rows.append(row)
    SAVED.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def report(rows: list[dict]) -> str:
    gate = mg.MemoryGate(None)
    n = top_changed = action_changed = first_picked = asked = 0
    max_shift = []
    for row in rows:
        for mid in row["orders"][0]:
            ps = [o[mid] for o in row["orders"]]
            tops = [max(p, key=p.get) for p in ps]
            actions = [gate.decide(p)[0] for p in ps]
            n += 1
            top_changed += len(set(tops)) > 1
            action_changed += len(set(actions)) > 1
            for order, top in zip(ORDERS, tops):
                asked += 1
                first_picked += top == order[0]
            max_shift.append(max(max(p[k] for p in ps) - min(p[k] for p in ps) for k in mg.LABELS))
    max_shift.sort()
    return "\n".join([
        "## Robustness: does the order of the options matter?", "",
        f"Each of the {n} memory decisions was asked three times, with the options in three rotations "
        "(mention/quiet/skip, quiet/skip/mention, skip/mention/quiet).", "",
        "| check | result |", "|---|---|",
        f"| policy action changed with the order | {action_changed}/{n} ({action_changed / n:.1%}) |",
        f"| decider's top label changed with the order | {top_changed}/{n} ({top_changed / n:.1%}) |",
        f"| first-listed option picked as top label | {first_picked}/{asked} ({first_picked / asked:.1%}); "
        "a bias-free model picks it as often as that option happens to be right |",
        f"| median / largest probability shift between orders | {max_shift[len(max_shift) // 2]:.2f} / "
        f"{max_shift[-1]:.2f} |", "",
        "Made by `eval/option_order.py`; answers saved in `eval/results/option_order.json`.", ""])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = json.loads(SAVED.read_text(encoding="utf-8")) if "--rescore" in sys.argv else collect()
    print("\n" + report(rows))
