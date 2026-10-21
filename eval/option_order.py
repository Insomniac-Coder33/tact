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
    from run_eval import SENSITIVE, label_of

    scenarios = {s["id"]: s for s in json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))}
    gate = mg.MemoryGate(None)
    n = top_changed = action_changed = moved = moved_to_first = 0
    max_shift = []
    # per answer set: [right, scored, sensitive facts said]
    scores = {name: [0, 0, 0] for name in ("mention / quiet / skip (shipped)", "quiet / skip / mention",
                                           "skip / mention / quiet", "average of the three")}
    for row in rows:
        s = scenarios[row["id"]]
        for mid in row["orders"][0]:
            ps = [o[mid] for o in row["orders"]]
            tops = [max(p, key=p.get) for p in ps]
            actions = [gate.decide(p)[0] for p in ps]
            n += 1
            top_changed += len(set(tops)) > 1
            action_changed += len(set(actions)) > 1
            for order, top in zip(ORDERS[1:], tops[1:]):
                if top != tops[0]:
                    moved += 1
                    moved_to_first += top == order[0]
            max_shift.append(max(max(p[k] for p in ps) - min(p[k] for p in ps) for k in mg.LABELS))
            truth = label_of(s, mid)
            if truth is None:
                continue
            avg = {k: sum(p[k] for p in ps) / len(ps) for k in mg.LABELS}
            for name, p in zip(scores, ps + [avg]):
                action = gate.decide(p)[0]
                scores[name][0] += action == truth
                scores[name][1] += 1
                scores[name][2] += mid in SENSITIVE and truth != "mention" and action == "mention"
    max_shift.sort()
    out = [
        "## Robustness: does the order of the options matter?", "",
        f"Each of the {n} memory decisions in the 34 scenarios was asked three times, with the options "
        "in three rotations.", "",
        "| check | result |", "|---|---|",
        f"| decider's top label changed with the order | {top_changed}/{n} ({top_changed / n:.1%}) |",
        f"| … and moved to whichever option was listed first | {moved_to_first} of {moved} moves |",
        f"| policy action changed with the order | {action_changed}/{n} ({action_changed / n:.1%}) |",
        f"| median / largest probability shift between orders | {max_shift[len(max_shift) // 2]:.2f} / "
        f"{max_shift[-1]:.2f} |", "",
        "| options listed as | policy right | sensitive facts mentioned |", "|---|---|---|"]
    for name, (right, total, said) in scores.items():
        out.append(f"| {name} | {right}/{total} ({right / total:.1%}) | {said} |")
    out += ["", "No sign of a bias towards the first-listed option. But the cost table draws its lines "
            "where small probability shifts matter, so the final action moves about three times as often "
            "as the top label, and in the other two orders 1 and 2 sensitive facts got through. Averaging "
            "the three orders gains little for three times the questions, so the shipped gate asks once.",
            "", "Made by `eval/option_order.py`; answers saved in `eval/results/option_order.json`.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = json.loads(SAVED.read_text(encoding="utf-8")) if "--rescore" in sys.argv else collect()
    print("\n" + report(rows))
