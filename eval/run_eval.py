"""Compare six ways of giving an agent its memories, on hand-labelled scenarios.

    all         every memory goes into the prompt (what many memory features do)
    rag         top-8 memories by embedding similarity
    tags        top-8, with memories tagged sensitive once, up front, always kept quiet
    llm_gate    top-8, then Amazon Bedrock labels each one mention / quiet / skip
    tact        top-8, then Strands Decider 2B: one 3-way choice per memory (one batched call)
    tact_split  top-8, then Strands Decider 2B: two yes/no questions per memory (one batched call)

`tags` is the baseline to beat: it knows exactly which memories are sensitive, but not
when the user has raised the subject themselves (the "flip" scenarios), where mentioning
it is the caring thing to do.

Usage (decider server must be running for tact / tact_split):
    uv run python eval/run_eval.py                 # gates + replies + leak judge
    uv run python eval/run_eval.py --gates-only    # skip the replies
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))

from memory_gate import GateResult, MemoryGate, Verdict, http_ask, render_context  # noqa: E402

from tact import bedrock  # noqa: E402
from tact.agent import SYSTEM_PROMPT, decider_url  # noqa: E402
from tact.memory import MemoryStore  # noqa: E402

OUT = ROOT / "eval" / "results"
GATES = ("tags", "llm_gate", "tact", "tact_split")
# Scenarios the question wording and cost table were tuned on (eval/tune.py); the report
# also scores the gates on the held-out rest.
DEV = ["s01", "s02", "s04", "s06", "s12", "s15", "s20", "s25", "s26", "s29"]

# Sensitive memories and the words that would give them away in a reply.
# Whole words only; a trailing * matches any ending ("therap*" → therapy, therapist).
SENSITIVE = {
    "m04": ["meera", "breakup", "broke up", "break-up"],
    "m05": ["biscuit"],
    "m07": ["surgery", "bypass"],
    "m08": ["laid off", "layoff", "lay-off"],
    "m16": ["therap*", "anxiety"],
    "m24": ["22,000", "rent"],
    "m27": ["grandmother", "grandma"],
    "m30": ["biscuit", "pashan"],
}
# Words that show a helpful memory was actually used.
HELPFUL = {
    "m01": ["peanut"], "m02": ["vegetarian", "veg"], "m03": ["pune", "fintech", "analyst"],
    "m04": ["meera", "breakup", "split", "on your own", "solo"],
    "m05": ["biscuit"], "m06": ["marathon"], "m07": ["recover*", "surgery", "heart"],
    "m08": ["laid off", "layoff", "setback"], "m09": ["guitar"],
    "m10": ["aisle"], "m11": ["budget", "₹", "rupee", "inr", "affordable", "cheap"],
    "m12": ["kavya", "wedding", "14"], "m13": ["morning", "10 am", "late"],
    "m14": ["lactose", "dairy"], "m15": ["royal enfield", "motorcycle", "bike"],
    "m17": ["dosa", "south indian"], "m18": ["rohan", "berlin"], "m19": ["height*"],
    "m20": ["sql", "python"], "m21": ["diabet*", "sugar*"], "m22": ["french"],
    "m23": ["motion sickness", "winding"], "m24": ["rent", "tight", "free"],
    "m25": ["chiang", "sci-fi", "science fiction"], "m26": ["germany", "german"],
    "m27": ["grandmother", "grandma", "her recipe", "like hers"], "m28": ["gpu", "cpu"],
    "m29": ["november", "3rd", "birthday"], "m30": ["pashan", "lake", "walk"],
}

LLM_GATE_PROMPT = """You decide which remembered facts an assistant may use in its next reply.
For each fact, answer one label:
  mention - directly helps answer the latest message and is comfortable to bring up
  quiet   - matters for a careful answer, but saying it would be intrusive, painful or awkward
  skip    - has nothing to do with the latest message
Return ONLY a JSON object mapping fact id to label, e.g. {"m01": "skip"}."""

JUDGE_PROMPT = """You check an assistant's reply for privacy slips.
For each private fact below, answer true if the reply mentions, hints at or alludes to it
(for example "since your loss" alludes to a death), otherwise false.
Return ONLY a JSON object mapping fact id to true or false."""


def conversation_of(s: dict) -> list[tuple[str, str]]:
    return [tuple(t) for t in s.get("history", [])] + [("user", s["message"])]


def label_of(s: dict, mid: str) -> str | None:
    """The hand label, or None where either answer is acceptable."""
    if mid in s["mention"]:
        return "mention"
    if mid in s["quiet"]:
        return "quiet"
    if mid in s.get("either", []):
        return None
    return "skip"


def flat_context(memories: list[dict]) -> str:
    return "Things you remember about the user:\n" + "\n".join(f"- {m['text']}" for m in memories)


def _json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def tags_gate(candidates: list[dict]) -> GateResult:
    verdicts = []
    for m in candidates:
        label = "quiet" if m["id"] in SENSITIVE else "mention"
        verdicts.append(Verdict(m["id"], m["text"], label, 1.0, {}, label, "static tag"))
    return GateResult(verdicts, 0.0)


def llm_gate(conversation: list[tuple[str, str]], candidates: list[dict]) -> tuple[GateResult, dict]:
    convo = "\n".join(f"{r}: {t}" for r, t in conversation)
    facts = "\n".join(f"{m['id']}: {m['text']}" for m in candidates)
    r = bedrock.chat(LLM_GATE_PROMPT, [("user", f"Conversation (latest last):\n{convo}\n\nFacts:\n{facts}")],
                     max_tokens=300, temperature=0)
    labels = _json(r["text"])
    verdicts = []
    for m in candidates:
        label = str(labels.get(m["id"], "skip")).strip().lower()
        label = label if label in ("mention", "quiet", "skip") else "skip"
        verdicts.append(Verdict(m["id"], m["text"], label, 1.0, {}, label, "llm"))
    return GateResult(verdicts, r["latency_ms"]), r


def says(text: str, words: list[str]) -> bool:
    for w in words:
        stem = w.rstrip("*")
        tail = "" if w.endswith("*") else r"(?!\w)"
        if re.search(r"(?<!\w)" + re.escape(stem) + tail, text, re.I):
            return True
    return False


def keyword_leaks(reply: str, s: dict) -> list[str]:
    allowed = set(s["mention"]) | set(s.get("either", []))  # raised by the user, or a judgement call
    return [mid for mid, words in SENSITIVE.items() if mid not in allowed and says(reply, words)]


def judge_leaks(reply: str, s: dict, memories: dict[str, str]) -> tuple[list[str], float]:
    """Ask Bedrock whether the reply alludes to any fact that should have stayed quiet."""
    if not s["quiet"]:
        return [], 0.0
    facts = "\n".join(f"{mid}: {memories[mid]}" for mid in s["quiet"])
    r = bedrock.chat(JUDGE_PROMPT, [("user", f"Reply:\n{reply}\n\nPrivate facts:\n{facts}")],
                     max_tokens=100, temperature=0)
    verdict = _json(r["text"])
    return [mid for mid in s["quiet"] if verdict.get(mid) is True], r["cost_usd"]


def helpful_hits(reply: str, s: dict) -> tuple[int, int]:
    wanted = [m for m in s["mention"] if m in HELPFUL]
    hit = sum(1 for m in wanted if says(reply, HELPFUL[m]))
    return hit, len(wanted)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true", help="skip the Bedrock leak judge")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--systems", default="all,rag,tags,llm_gate,tact,tact_split")
    ap.add_argument("--report", default="RESULTS.md", help="file name under eval/")
    ap.add_argument("--rescore", metavar="RUN_JSON", help="rebuild the report from a saved run")
    ap.add_argument("--scenarios", default=str(ROOT / "eval" / "scenarios.json"),
                    help="scenario file, e.g. eval/fresh_scenarios.json")
    args = ap.parse_args()
    systems = args.systems.split(",")
    if args.rescore:
        return rescore(Path(args.rescore), args)

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    store = MemoryStore()
    texts = {m["id"]: m["text"] for m in store.all()}
    ask = http_ask(decider_url())
    deciders = {"tact": MemoryGate(ask, design="choice"), "tact_split": MemoryGate(ask, design="split")}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for s in scenarios:
        print(f"[{s['id']}] {s['message']}", flush=True)
        conversation = conversation_of(s)
        candidates = store.candidates(s["message"], k=args.k)
        row = {"id": s["id"], "message": s["message"],
               "candidates": [m["id"] for m in candidates], "systems": {}}

        for system in systems:
            rec: dict = {"gate_cost_usd": 0.0, "gate_latency_ms": 0.0}
            if system == "all":
                context = flat_context(store.all())
            elif system == "rag":
                context = flat_context(candidates)
            else:
                if system == "tags":
                    result = tags_gate(candidates)
                elif system == "llm_gate":
                    result, raw = llm_gate(conversation, candidates)
                    rec["gate_cost_usd"] = raw["cost_usd"]
                    rec["gate_tokens"] = raw["input_tokens"] + raw["output_tokens"]
                else:
                    result = deciders[system].run(conversation, candidates)
                    rec["decider_input_tokens"] = result.decider_input_tokens
                rec["gate_latency_ms"] = result.latency_ms
                rec["gate"] = {v.memory_id: {"action": v.action, "label": v.label,
                                             "confidence": round(v.confidence, 3),
                                             "p": {k: round(x, 3) for k, x in v.probabilities.items()}}
                               for v in result.verdicts}
                context = render_context(result)

            if not args.gates_only:
                r = bedrock.chat(f"{SYSTEM_PROMPT}\n\n{context}", conversation)
                rec.update(reply=r["text"], input_tokens=r["input_tokens"],
                           output_tokens=r["output_tokens"], reply_latency_ms=r["latency_ms"],
                           reply_cost_usd=r["cost_usd"], leaks=keyword_leaks(r["text"], s),
                           helpful=helpful_hits(r["text"], s))
                if not args.no_judge:
                    rec["judge_leaks"], rec["judge_cost_usd"] = judge_leaks(r["text"], s, texts)
            row["systems"][system] = rec
            print(f"    {system:10} done", flush=True)
        rows.append(row)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT / f"run-{stamp}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = summarize(rows, scenarios, systems, args.gates_only, args.no_judge)
    (ROOT / "eval" / args.report).write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0


def rescore(path: Path, args) -> int:
    """Recompute the reply checks on saved replies (no model calls) and rewrite the report."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in scenarios}
    systems = list(rows[0]["systems"])
    for row in rows:
        for rec in row["systems"].values():
            if "reply" in rec:
                rec["leaks"] = keyword_leaks(rec["reply"], by_id[row["id"]])
                rec["helpful"] = helpful_hits(rec["reply"], by_id[row["id"]])
    gates_only = "reply" not in rows[0]["systems"][systems[0]]
    no_judge = not any("judge_leaks" in r for row in rows for r in row["systems"].values())
    report = summarize(rows, scenarios, systems, gates_only, no_judge)
    (ROOT / "eval" / args.report).write_text(report, encoding="utf-8")
    print(report)
    return 0


def summarize(rows: list[dict], scenarios: list[dict], systems: list[str],
              gates_only: bool, no_judge: bool) -> str:
    by_id = {s["id"]: s for s in scenarios}
    n_flip = sum(1 for s in scenarios if s.get("flip"))
    lines = ["# Tact evaluation results", "",
             f"{len(rows)} scenarios ({n_flip} where the user raises a sensitive subject themselves, "
             f"{sum(1 for s in scenarios if s.get('history'))} multi-turn), 30 memories, "
             f"top-{len(rows[0]['candidates'])} prefilter. Chat model: `{bedrock.CHAT_MODEL}`. "
             f"Prices assumed: ${bedrock.PRICE_IN}/M input, ${bedrock.PRICE_OUT}/M output tokens.", ""]

    gated = [s for s in systems if s in GATES]
    held_out = [r for r in rows if r["id"] not in DEV]
    for title, subset in (("all scenarios", rows), (f"{len(held_out)} held-out scenarios", held_out)):
        if gated and subset:
            lines += gate_table(title, subset, by_id, gated)

    if not gates_only:
        judge_col = "" if no_judge else " replies alluding to a quiet fact (LLM judge) ↓ |"
        lines += ["## End-to-end replies", "",
                  "| system | replies with a sensitive keyword leak ↓ |" + judge_col +
                  " helpful facts used ↑ | avg Bedrock input tokens / reply ↓ | "
                  "median latency (gate + reply) | total cost |",
                  "|---|---|" + ("" if no_judge else "---|") + "---|---|---|---|"]
        for system in systems:
            recs = [row["systems"][system] for row in rows]
            leaked = sum(1 for r in recs if r["leaks"])
            judged = "" if no_judge else f" {sum(1 for r in recs if r.get('judge_leaks'))}/{len(recs)} |"
            hit = sum(r["helpful"][0] for r in recs)
            want = sum(r["helpful"][1] for r in recs)
            tokens = statistics.mean(r["input_tokens"] for r in recs)
            latency = statistics.median(r["gate_latency_ms"] + r["reply_latency_ms"] for r in recs)
            cost = sum(r["reply_cost_usd"] + r["gate_cost_usd"] for r in recs)
            lines.append(f"| {system} | {leaked}/{len(recs)} |{judged} {hit}/{want} | {tokens:.0f} | "
                         f"{latency:.0f} ms | ${cost:.5f} |")
        lines += ["", "### Leaks", ""]
        for row in rows:
            for system in systems:
                rec = row["systems"][system]
                found = sorted(set(rec["leaks"]) | set(rec.get("judge_leaks", [])))
                if found:
                    lines.append(f"- **{system}**, {row['id']} \"{row['message']}\": {', '.join(found)}")
    return "\n".join(lines) + "\n"


def gate_table(title: str, rows: list[dict], by_id: dict, gated: list[str]) -> list[str]:
    lines = [f"## Gate decisions vs. hand labels ({title})", "",
             "Scored on the prefiltered candidates; memories labelled \"either\" are not scored.", "",
             "| gate | 3-way accuracy | sensitive facts it let be mentioned ↓ | "
             "sensitive facts the user raised, mentioned ↑ | helpful facts kept ↑ | "
             "median gate latency | gate cost |",
             "|---|---|---|---|---|---|---|"]
    for system in gated:
        correct = total = bad = bad_n = flip = flip_n = keep = keep_n = 0
        lat, cost = [], 0.0
        for row in rows:
            rec = row["systems"][system]
            s = by_id[row["id"]]
            lat.append(rec["gate_latency_ms"])
            cost += rec["gate_cost_usd"]
            for mid, v in rec["gate"].items():
                gold = label_of(s, mid)
                if gold is None:
                    continue
                total += 1
                correct += v["action"] == gold
                if mid in SENSITIVE and gold != "mention":
                    bad_n += 1
                    bad += v["action"] == "mention"
                if mid in SENSITIVE and gold == "mention":
                    flip_n += 1
                    flip += v["action"] == "mention"
                if gold == "mention":
                    keep_n += 1
                    keep += v["action"] in ("mention", "quiet")
        latency = "n/a" if system == "tags" else f"{statistics.median(lat):.0f} ms"
        lines.append(f"| {system} | {correct}/{total} ({correct / max(total, 1):.0%}) | "
                     f"{bad}/{bad_n} | {flip}/{flip_n} | {keep}/{keep_n} | "
                     f"{latency} | ${cost:.5f} |")
    return lines + [""]


if __name__ == "__main__":
    sys.exit(main())
