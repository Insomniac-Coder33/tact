"""Re-run the evaluation's LLM parts on other chat models, reusing the decider's saved answers.

Strands Decider 2B is deterministic: two separate runs gave identical probabilities on all
272 memory decisions. So the saved runs (each scenario's 8 shortlisted memories and the
decider's answers for them) stand in for a live server, and only the chat models are called.
Nothing in Tact changes: the question wording, cost table and thresholds are the ones frozen
at git tag `eval-freeze-2026-10-22`, and the script stops if any saved decision no longer
matches the shipped policy.

    gates     each model in --models labels the same 8 shortlisted memories (the LLM-gate
              baseline), on the 34 original and the 16 fresh scenarios
    escalate  the decider keeps its confident decisions and hands the unsure ones to a model,
              scored from the answers above, with no extra calls
    replies   the chat model writes a reply for each way of giving it memories (all, rag,
              tags, llm_gate, tact), checked for leaks by keyword and by an LLM judge

The plan, committed before the first run, is in eval/BEDROCK_PLAN.md.

    uv run python -m tact.bedrock                              # can this account use Bedrock?
    TACT_PROVIDER=bedrock uv run python eval/replay_eval.py    # ~600 calls, a few cents
    uv run python eval/replay_eval.py --dry-run                # no model calls: checks the plumbing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "eval"))

import run_eval as ev  # noqa: E402
from memory_gate import GateResult, MemoryGate, Verdict, render_context  # noqa: E402

from tact import bedrock  # noqa: E402
from tact.agent import SYSTEM_PROMPT  # noqa: E402
from tact.memory import MemoryStore  # noqa: E402

SETS = {  # name: (scenarios, saved gate run with each shortlist and the decider's answers)
    "original": ("eval/scenarios.json", "eval/results/run-20261021-202009.json"),
    "fresh": ("eval/fresh_scenarios.json", "eval/results/run-20261022-001433.json"),
}
SAVED_LLM = "Qwen2.5 3B (saved run)"  # the LLM gate in the saved runs, served by Ollama
NAMES = {"us.amazon.nova-micro-v1:0": "Nova Micro", "us.amazon.nova-lite-v1:0": "Nova Lite",
         "us.amazon.nova-pro-v1:0": "Nova Pro",
         "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude Haiku 4.5"}
REPLY_SYSTEMS = ["all", "rag", "tags", "llm_gate", "tact"]
BARS = (0.33, 0.5, 0.66)
# The decider ran on a c7i.2xlarge (on-demand, us-east-1) because the account had no GPU
# quota. Its cost per turn is that hourly price times the measured gate time, as if the
# instance were always busy.
INSTANCE = ("c7i.2xlarge CPU", 0.357)


def name_of(model: str) -> str:
    return NAMES.get(model, model)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_tact(rec: dict, texts: dict[str, str]) -> GateResult:
    """The decider's saved verdicts, re-checked against the shipped policy."""
    gate = MemoryGate(ask=None)
    verdicts = []
    for mid, v in rec["gate"].items():
        action, reason = gate.decide(v["p"])
        if action != v["action"]:
            raise SystemExit(f"{mid}: the saved action is {v['action']}, the shipped policy says "
                             f"{action}. The cost table changed after this run; re-run the decider.")
        verdicts.append(Verdict(mid, texts[mid], v["label"], v["confidence"], v["p"], action, reason))
    return GateResult(verdicts, rec["gate_latency_ms"])


def labelled(rec: dict, texts: dict[str, str]) -> GateResult:
    """An LLM gate's labels as a GateResult, for the reply prompt."""
    return GateResult([Verdict(mid, texts[mid], v["action"], 1.0, {}, v["action"], "llm")
                       for mid, v in rec["gate"].items()], rec["gate_latency_ms"])


def escalate(rows: list[dict], model: str, bar: float) -> str:
    """Add a gate that keeps the decider's decisions at or above `bar` confidence and takes
    `model`'s label for the rest. Returns the new gate's name."""
    name = f"tact, below {bar} to {name_of(model)}"
    for row in rows:
        tact, llm = row["systems"]["tact"], row["systems"][name_of(model)]
        gate = {mid: {"action": v["action"] if v["confidence"] >= bar else llm["gate"][mid]["action"]}
                for mid, v in tact["gate"].items()}
        sent = sum(v["confidence"] < bar for v in tact["gate"].values())
        row["systems"][name] = {"gate": gate, "sent": sent,
                                "gate_latency_ms": tact["gate_latency_ms"] + (llm["gate_latency_ms"] if sent else 0),
                                "gate_cost_usd": llm["gate_cost_usd"] if sent else 0.0}
    return name


def run_gates(data: dict, models: list[str], memories: dict[str, dict]) -> dict[str, list[dict]]:
    """Each model labels every scenario's shortlist. Rows follow run_eval's format, with the
    saved tags, Qwen and decider gates alongside."""
    out = {}
    for set_name, (scenarios, saved) in data.items():
        rows = []
        for s in scenarios:
            row = saved[s["id"]]
            candidates = [memories[mid] for mid in row["candidates"]]
            conversation = ev.conversation_of(s)
            systems = {"tags": row["systems"]["tags"], SAVED_LLM: row["systems"]["llm_gate"],
                       "tact": row["systems"]["tact"]}
            for model in models:
                result, raw = ev.llm_gate(conversation, candidates, model)
                systems[name_of(model)] = {
                    "gate": {v.memory_id: {"action": v.action} for v in result.verdicts},
                    "gate_latency_ms": raw["latency_ms"], "gate_cost_usd": raw["cost_usd"],
                    "gate_tokens": raw["input_tokens"] + raw["output_tokens"],
                    "parsed": bool(ev._json(raw["text"])), "raw": raw["text"]}
            rows.append({"id": s["id"], "message": s["message"], "candidates": row["candidates"],
                         "systems": systems})
            print(f"[gates {set_name} {s['id']}] {s['message'][:60]}", flush=True)
        out[set_name] = rows
    return out


def run_replies(data: dict, gate_rows: dict, chat_model: str, judge_model: str | None,
                store: MemoryStore, texts: dict[str, str]) -> dict[str, list[dict]]:
    """The chat model replies once per way of giving it memories, as in run_eval."""
    memories = {m["id"]: m for m in store.all()}
    out = {}
    for set_name, (scenarios, saved) in data.items():
        gated = {r["id"]: r for r in gate_rows[set_name]}
        rows = []
        for s in scenarios:
            row = saved[s["id"]]
            candidates = [memories[mid] for mid in row["candidates"]]
            conversation = ev.conversation_of(s)
            llm = gated[s["id"]]["systems"][name_of(chat_model)]  # the chat model as its own gate
            tact = row["systems"]["tact"]
            contexts = {
                "all": (ev.flat_context(store.all()), {}),
                "rag": (ev.flat_context(candidates), {}),
                "tags": (render_context(ev.tags_gate(candidates)), {}),
                "llm_gate": (render_context(labelled(llm, texts)),
                             {"gate_latency_ms": llm["gate_latency_ms"], "gate_cost_usd": llm["gate_cost_usd"]}),
                "tact": (render_context(saved_tact(tact, texts)),
                         {"gate_latency_ms": tact["gate_latency_ms"], "gate_cost_usd": 0.0}),
            }
            systems = {}
            for system, (context, gate) in contexts.items():
                r = bedrock.chat(f"{SYSTEM_PROMPT}\n\n{context}", conversation, model=chat_model)
                rec = {"gate_latency_ms": 0.0, "gate_cost_usd": 0.0, **gate,
                       "reply": r["text"], "input_tokens": r["input_tokens"],
                       "output_tokens": r["output_tokens"], "reply_latency_ms": r["latency_ms"],
                       "reply_cost_usd": r["cost_usd"], "leaks": ev.keyword_leaks(r["text"], s),
                       "helpful": ev.helpful_hits(r["text"], s)}
                if judge_model:
                    rec["judge_leaks"], rec["judge_cost_usd"] = ev.judge_leaks(r["text"], s, texts, judge_model)
                systems[system] = rec
            rows.append({"id": s["id"], "message": s["message"], "candidates": row["candidates"],
                         "systems": systems})
            print(f"[replies {set_name} {s['id']}] {s['message'][:60]}", flush=True)
        out[set_name] = rows
    return out


def per_thousand(usd: float) -> str:
    return "n/a" if math.isnan(usd) else f"${usd * 1000:.2f}"


def report(args, data: dict, gate_rows: dict, reply_rows: dict, inputs: list[Path]) -> str:
    models = [name_of(m) for m in args.models]
    title = "Tact on Amazon Bedrock" if bedrock.PROVIDER == "bedrock" else f"Tact against {', '.join(models)}"
    lines = [f"# {title}", "",
             f"Run `{time.strftime('%Y-%m-%d %H:%M')}`, provider `{bedrock.PROVIDER}`. The decider's answers "
             "and each scenario's 8-memory shortlist are replayed from saved runs (the decider is "
             "deterministic), so only these models were called. Plan: `eval/BEDROCK_PLAN.md`.", "",
             "Inputs (sha256):", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}` `{sha256(p)}`" for p in inputs] + [""]

    for set_name, rows in gate_rows.items():
        by_id = {s["id"]: s for s in data[set_name][0]}
        gates = ["tags", SAVED_LLM, *models, "tact"]
        lines += ev.gate_table(f"{set_name} scenarios", rows, by_id, gates)
        if set_name == "original":
            held = [r for r in rows if r["id"] not in ev.DEV]
            lines += ev.gate_table(f"{len(held)} held-out original scenarios", held, by_id, gates)
        bad = {m: sum(not r["systems"][m]["parsed"] for r in rows) for m in models}
        if any(bad.values()):
            lines += [f"Replies that weren't valid JSON (every fact then counts as skip): {bad}", ""]

        lines += [f"### Escalation ({set_name}): the decider keeps its confident decisions, "
                  "a model takes the rest", "",
                  "| gate | decisions sent to the model | 3-way accuracy | sensitive facts let through ↓ | "
                  "user-raised subjects mentioned ↑ | helpful facts kept ↑ | model cost per 1,000 turns |",
                  "|---|---|---|---|---|---|---|"]
        for model in args.models:
            for bar in BARS:
                name = escalate(rows, model, bar)
                n = ev.score_gate(rows, by_id, name)
                sent = sum(r["systems"][name]["sent"] for r in rows)
                total = sum(len(r["systems"][name]["gate"]) for r in rows)
                lines.append(f"| {name} | {sent}/{total} | {n['correct']}/{n['total']} "
                             f"({n['correct'] / n['total']:.0%}) | {n['bad']}/{n['bad_n']} | "
                             f"{n['flip']}/{n['flip_n']} | {n['keep']}/{n['keep_n']} | "
                             f"{per_thousand(n['cost_usd'] / len(rows))} |")
        lines.append("")

    lines += ["## What a gate decision costs, per 1,000 turns", "",
              "| gate | median time per turn | cost per 1,000 turns | how |", "|---|---|---|---|"]
    rows = [r for set_rows in gate_rows.values() for r in set_rows]
    tact_ms = statistics.median(r["systems"]["tact"]["gate_latency_ms"] for r in rows)
    lines.append(f"| tact (decider on a {INSTANCE[0]}) | {tact_ms / 1000:.1f} s | "
                 f"{per_thousand(INSTANCE[1] / 3600 * tact_ms / 1000)} | the instance's hourly price "
                 "× the measured time, as if always busy |")
    for model in models:
        ms = statistics.median(r["systems"][model]["gate_latency_ms"] for r in rows)
        usd = statistics.mean(r["systems"][model]["gate_cost_usd"] for r in rows)
        lines.append(f"| {model} | {ms / 1000:.1f} s | {per_thousand(usd)} | tokens at list price |")
    lines += ["", "The decider's published speed on a GPU is about 115 ms per question; it ran on a CPU "
              "here (no GPU quota), and that is not measured.", ""]

    for set_name, rows in reply_rows.items():
        lines += ev.reply_table(f"Replies written by {name_of(args.chat_model)} ({set_name} scenarios)",
                                rows, REPLY_SYSTEMS, not args.judge_model)
    if reply_rows:
        lines += [f"`llm_gate` here is {name_of(args.chat_model)} labelling the memories itself. "
                  f"Leak judge: {name_of(args.judge_model) if args.judge_model else 'none'}.", ""]
    return "\n".join(lines)


def fake_chat(system: str, messages: list[tuple[str, str]], max_tokens: int = 400,
              temperature: float = 0.3, model: str | None = None) -> dict:
    """Stands in for every model call in --dry-run: every fact is "skip", replies are canned."""
    text = messages[-1][1]
    if system == ev.LLM_GATE_PROMPT:
        out = json.dumps({mid: "skip" for mid in re.findall(r"^(m\d+):", text, re.M)})
    elif system == ev.JUDGE_PROMPT:
        out = "{}"
    else:
        out = "Here is one practical suggestion for this weekend."
    return {"text": out, "input_tokens": len(system + text) // 4, "output_tokens": len(out) // 4,
            "latency_ms": 1.0, "cost_usd": 0.0}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    on_bedrock = bedrock.PROVIDER == "bedrock"
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="us.amazon.nova-lite-v1:0,us.amazon.nova-pro-v1:0" if on_bedrock else "",
                    help="comma-separated model IDs for the LLM-gate baseline")
    ap.add_argument("--chat-model", default=bedrock.CHAT_MODEL, help="the model that writes the replies")
    ap.add_argument("--judge-model", default="us.amazon.nova-pro-v1:0" if on_bedrock else "",
                    help="LLM leak judge for the replies (empty: keyword check only)")
    ap.add_argument("--sets", default="original,fresh")
    ap.add_argument("--parts", default="gates,replies")
    ap.add_argument("--report", default=f"RESULTS-{bedrock.PROVIDER}.md", help="file name under eval/")
    ap.add_argument("--dry-run", action="store_true", help="fake every model call; write nothing")
    args = ap.parse_args()
    args.models = [m for m in args.models.split(",") if m]
    parts = args.parts.split(",")
    if "replies" in parts and args.chat_model not in args.models:
        args.models.append(args.chat_model)  # the chat model is also the replies' llm_gate
    if not args.models:
        ap.error("--models is required with this provider, e.g. --models qwen2.5:7b-instruct")
    if args.dry_run:
        bedrock.chat = fake_chat

    data, inputs = {}, []
    for set_name in args.sets.split(","):
        scenario_file, run_file = (ROOT / p for p in SETS[set_name])
        rows = json.loads(run_file.read_text(encoding="utf-8"))
        data[set_name] = (json.loads(scenario_file.read_text(encoding="utf-8")), {r["id"]: r for r in rows})
        inputs += [scenario_file, run_file]
    store = MemoryStore()
    texts = {m["id"]: m["text"] for m in store.all()}
    for _, saved in data.values():  # stop now, not halfway, if the policy has drifted
        for row in saved.values():
            saved_tact(row["systems"]["tact"], texts)

    gate_rows = run_gates(data, args.models, {m["id"]: m for m in store.all()})
    reply_rows = (run_replies(data, gate_rows, args.chat_model, args.judge_model or None, store, texts)
                  if "replies" in parts else {})
    text = report(args, data, gate_rows, reply_rows, inputs)
    if args.dry_run:
        print("\n" + text)
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ev.OUT / f"{bedrock.PROVIDER}-{stamp}.json"
    out.write_text(json.dumps({"models": args.models, "chat_model": args.chat_model,
                               "judge_model": args.judge_model, "gates": gate_rows,
                               "replies": reply_rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "eval" / args.report).write_text(text + f"\nRaw answers: `eval/results/{out.name}`.\n",
                                             encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
