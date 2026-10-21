"""Evaluate the write-side decision: does a new message make a stored memory outdated?

    uv run python eval/run_curator.py            # asks the decider, saves the answers
    uv run python eval/run_curator.py --rescore  # re-scores the saved answers offline

Each scenario is one user message and three stored memories, hand-labelled outdated /
repeat / unaffected ("either" = not scored). The "dev" cases were used while writing the
question; the "test" cases were written before the first run and never tuned on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "src"))

import memory_gate as mg  # noqa: E402

SAVED = ROOT / "eval" / "results" / "curator.json"


def collect() -> list[dict]:
    from tact.agent import decider_url

    scenarios = json.loads((ROOT / "eval" / "curator_scenarios.json").read_text(encoding="utf-8"))
    memories = {m["id"]: m for m in json.loads((ROOT / "data" / "memories.json").read_text(encoding="utf-8"))}
    curator = mg.MemoryCurator(mg.http_ask(decider_url()))
    rows = []
    for s in scenarios:
        result = curator.run([("user", s["message"])], [memories[k] for k in s["labels"]])
        print(f"[{s['id']}] {result.latency_ms:6.0f} ms  {s['message']}")
        rows.append({**s, "latency_ms": result.latency_ms,
                     "probabilities": {u.memory_id: u.probabilities for u in result.updates},
                     "p_past": {u.memory_id: u.p_past for u in result.updates}})
    SAVED.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return rows


def score(rows: list[dict], curator: mg.MemoryCurator) -> dict:
    m = {"n": 0, "argmax": 0, "outdated": 0, "retired": 0, "doubted": 0,
         "true": 0, "wrongly_retired": 0, "wrongly_doubted": 0, "repeats": 0, "repeat_caught": 0}
    for r in rows:
        for mid, truth in r["labels"].items():
            if truth == "either":
                continue
            p = r["probabilities"][mid]
            action = curator.decide(p, r.get("p_past", {}).get(mid, 0.0))
            m["n"] += 1
            m["argmax"] += max(p, key=p.get) == truth
            if truth == "outdated":
                m["outdated"] += 1
                m["retired"] += action == "retire"
                m["doubted"] += action == "doubt"
            else:
                m["true"] += 1
                m["wrongly_retired"] += action == "retire"
                m["wrongly_doubted"] += action == "doubt"
            if truth == "repeat":
                m["repeats"] += 1
                m["repeat_caught"] += p["repeat"] >= curator.repeat_at
    return m


def report(rows: list[dict]) -> str:
    curator = mg.MemoryCurator(None)
    unguarded = mg.MemoryCurator(None, past_at=None)
    out = ["## Write side: which stored memories does a new message make outdated?", "",
           f"{len(rows)} messages × 3 stored memories, one decider call per message. Policy: retire at "
           f"P(outdated) ≥ {curator.retire_at}, flag as \"may be out of date\" from {curator.doubt_at}; "
           f"a message that repeats a memory (P(repeat) ≥ {curator.repeat_at}) isn't stored again. "
           f"With the past-event guard, a memory with P(past event) ≥ {curator.past_at} is never retired, "
           "only flagged. Both policies are scored on the same decider answers.", "",
           "| cases | policy | 3-way accuracy | outdated memories retired ↑ | … or flagged | still-true memories "
           "wrongly retired ↓ | … or flagged | repeats caught |", "|---|---|---|---|---|---|---|---|"]
    for name, part in (("dev (used to write the question)", [r for r in rows if r["split"] == "dev"]),
                       ("test (never tuned on)", [r for r in rows if r["split"] == "test"]),
                       ("all", rows)):
        for policy, c in (("no guard", unguarded), ("**past-event guard**", curator)):
            s = score(part, c)
            out.append(f"| {name} | {policy} | {s['argmax']}/{s['n']} ({s['argmax'] / s['n']:.0%}) | "
                       f"{s['retired']}/{s['outdated']} | {s['doubted']}/{s['outdated']} | "
                       f"{s['wrongly_retired']}/{s['true']} | {s['wrongly_doubted']}/{s['true']} | "
                       f"{s['repeat_caught']}/{s['repeats']} |")
    lat = sorted(r["latency_ms"] for r in rows)
    out += ["", f"Median decider latency per message (3 memories): {lat[len(lat) // 2]:.0f} ms.", "",
            "Misses (outdated memories kept, still-true memories retired or flagged):", ""]
    for r in rows:
        for mid, truth in r["labels"].items():
            p = r["probabilities"][mid]
            action = curator.decide(p, r.get("p_past", {}).get(mid, 0.0))
            if (truth == "outdated" and action != "retire") or (truth in ("unaffected", "repeat") and action != "keep"):
                out.append(f"- {r['id']} \"{r['message']}\" → {mid} labelled {truth}, "
                           f"{action} at P(outdated)={p['outdated']:.2f}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = json.loads(SAVED.read_text(encoding="utf-8")) if "--rescore" in sys.argv else collect()
    text = report(rows)
    # Paste into eval/RESULTS.md when the numbers change.
    print(text)
