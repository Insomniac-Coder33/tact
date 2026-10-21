"""Tune the question wording and the cost table on a small dev split of the scenarios.

    uv run python eval/tune.py collect V1 V2 V3   # ask the decider, save probabilities
    uv run python eval/tune.py sweep              # replay saved probabilities under several cost tables

The dev split is kept small on purpose; eval/run_eval.py reports the held-out scenarios
separately, so the numbers in RESULTS.md are not just the ones the wording was tuned on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "eval"))

import memory_gate as mg  # noqa: E402
from run_eval import DEV, SENSITIVE, conversation_of, label_of  # noqa: E402

OUT = ROOT / "eval" / "results" / "tune"

VARIANTS = {
    "V1": (
        "An AI assistant is about to reply to the user's latest message. Should it use this "
        "fact it remembers about the user, and may it say it out loud?\nRemembered fact: {memory}",
        {"mention": "the fact directly helps answer the latest message and is comfortable to bring up",
         "quiet": "the fact matters for a careful answer, but saying it would be intrusive, "
                  "painful or awkward, so it should only shape the answer silently",
         "skip": "the fact has nothing to do with the latest message"},
    ),
    "V2": (  # the wording memory_gate.py ships with
        "An AI assistant is about to reply to the user's latest message. It remembers this fact "
        "about the user:\n{memory}\nHow should the assistant treat this fact in its reply?",
        {"mention": "use it openly: it makes the reply more useful or personal (a preference, "
                    "constraint, plan or interest that fits the request), and bringing it up feels natural",
         "quiet": "use it silently: it is related to the request, but saying it out loud could hurt, "
                  "embarrass or upset the user, so it should only make the reply more careful",
         "skip": "ignore it: it has no bearing on what a good reply would say"},
    ),
    "V3": (
        "A thoughtful friend who knows this about the user is answering the user's latest "
        "message:\n{memory}\nWhat would the friend do with this fact?",
        {"mention": "bring it up, because it helps tailor the answer and the user would be glad they remembered",
         "quiet": "keep it in mind but not say it, because mentioning it now would be tactless or painful",
         "skip": "not think of it at all, because it does not matter for this message"},
    ),
}


def collect(names: list[str]) -> None:
    from tact.agent import decider_url
    from tact.memory import MemoryStore

    scenarios = {s["id"]: s for s in json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))}
    store = MemoryStore()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        design = "split" if name.startswith("S") else "choice"
        if design == "choice":
            mg.INSTRUCTIONS, mg.OPTIONS = VARIANTS[name]
        gate = mg.MemoryGate(mg.http_ask(decider_url()), design=design)
        rows = []
        for sid in DEV:
            s = scenarios[sid]
            cands = store.candidates(s["message"], k=8)
            r = gate.run(conversation_of(s), cands)
            rows.append({"id": sid, "latency_ms": r.latency_ms,
                         "p": {v.memory_id: v.probabilities for v in r.verdicts}})
            print(f"{name} {sid} {r.latency_ms:.0f} ms", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def score(rows: list[dict], gate: mg.MemoryGate, scenarios: dict) -> dict:
    m = dict(correct=0, total=0, bad=0, bad_n=0, flip=0, flip_n=0, keep=0, keep_n=0, said=0, said_n=0)
    for row in rows:
        s = scenarios[row["id"]]
        for mid, p in row["p"].items():
            gold = label_of(s, mid)
            if gold is None:
                continue
            action = gate.decide(p)[0]
            m["total"] += 1
            m["correct"] += action == gold
            if mid in SENSITIVE and gold != "mention":
                m["bad_n"] += 1
                m["bad"] += action == "mention"
            if mid in SENSITIVE and gold == "mention":
                m["flip_n"] += 1
                m["flip"] += action == "mention"
            if gold == "mention":
                m["keep_n"] += 1
                m["keep"] += action in ("mention", "quiet")
                m["said_n"] += 1
                m["said"] += action == "mention"
    return m


def sweep() -> None:
    scenarios = {s["id"]: s for s in json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))}
    print("variant  cost(mention|quiet)  acc    sensitive-said  flips-said  helpful-kept  helpful-said")
    for path in sorted(OUT.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        for c in (1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
            cost = {a: dict(r) for a, r in mg.COST.items()}
            cost["mention"]["quiet"] = c
            m = score(rows, mg.MemoryGate(None, cost=cost), scenarios)
            print(f"{path.stem:8} {c:>5}                {m['correct']}/{m['total']} "
                  f"({m['correct'] / m['total']:.0%})   {m['bad']}/{m['bad_n']}             "
                  f"{m['flip']}/{m['flip_n']}         {m['keep']}/{m['keep_n']}         {m['said']}/{m['said_n']}")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    collect(sys.argv[2:]) if sys.argv[1] == "collect" else sweep()
