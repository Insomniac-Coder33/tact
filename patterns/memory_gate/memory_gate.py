"""MemoryGate: let a decision model decide which memories an agent may bring up.

A reusable Strands Decider 2B decision pattern (#context-memory). Standard library only,
so you can copy this one file into your own agent.

For every candidate memory, the decider judges whether it is:

    mention  - relevant to the user's message, and natural to bring up
    quiet    - relevant, but personal or painful: use it as background, never say it
    skip     - not relevant to this message

Two interchangeable ways to ask (pick with `design=`):

    "choice"  one 3-way `choice` question per memory
    "split"   two narrow yes/no `noul` questions per memory ("does it help?" and
              "would saying it be unwelcome right now?"), combined into the same 3 labels

All the questions go in ONE request. The decider reads the conversation once and every
question only adds its own tokens, so judging 8 memories costs far less than 8 calls.

The decider only returns probabilities. What happens next is plain Python: an
expected-cost policy (`MemoryGate.decide`) that knows a painful mention is much worse
than a missed detail. Tune the cost table without retraining anything.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

LABELS = ("mention", "quiet", "skip")

# --- design "choice": one 3-way question per memory ----------------------------------
# The option descriptions carry most of the signal, so they are written as carefully as
# the question itself.
INSTRUCTIONS = (
    "An AI assistant is about to reply to the user's latest message. It remembers this fact "
    "about the user:\n{memory}\nHow should the assistant treat this fact in its reply?"
)
OPTIONS = {
    "mention": "use it openly: it makes the reply more useful or personal (a preference, "
               "constraint, plan or interest that fits the request), and bringing it up feels natural",
    "quiet": "use it silently: it is related to the request, but saying it out loud could hurt, "
             "embarrass or upset the user, so it should only make the reply more careful",
    "skip": "ignore it: it has no bearing on what a good reply would say",
}

# --- design "split": two narrow yes/no questions per memory --------------------------
HELPS = (
    "Would knowing this fact about the user change what a thoughtful assistant suggests "
    "in reply to the user's latest message?\nRemembered fact: {memory}",
    {"true": "a careful reply would be different, safer or more personal because of this fact",
     "false": "the fact is unrelated, and the best reply is the same without it"},
)
UNWELCOME = (
    "If the assistant said this fact out loud in its next reply, would it feel intrusive, "
    "painful or awkward to the user, given how the conversation has gone so far?\n"
    "Remembered fact: {memory}",
    {"true": "saying it would hurt, embarrass or feel like being watched, because the user "
             "has not brought the subject up themselves",
     "false": "saying it is natural and welcome, or the user has raised the subject themselves"},
)

# --- the policy: expected cost of each action, given the true label ------------------
# COST[action][truth]. Saying a fact that should stay quiet is the worst outcome (3);
# dropping a useful fact entirely is a real loss (1.5); keeping it but not saying it is a
# smaller one (1); carrying an irrelevant fact as background costs only tokens (0.4).
# The gate picks the action with the lowest expected cost under the decider's
# probabilities, so a 60% "mention" with 25% on "quiet" still ends up quiet.
# 3 was picked by sweeping 1-5 on a dev split (eval/tune.py): at 5 the gate almost never
# says anything; below 2.5 it starts saying sensitive facts.
COST = {
    "mention": {"mention": 0.0, "quiet": 3.0, "skip": 0.5},
    "quiet":   {"mention": 1.0, "quiet": 0.0, "skip": 0.4},
    "skip":    {"mention": 1.5, "quiet": 1.0, "skip": 0.0},
}

AskFn = Callable[[str, Mapping[str, dict]], Mapping[str, dict]]


@dataclass
class Verdict:
    memory_id: str
    text: str
    label: str                       # the decider's most likely label
    confidence: float
    probabilities: dict[str, float]  # P(mention), P(quiet), P(skip)
    action: str                      # what the policy decided: mention / quiet / skip
    reason: str
    locked: bool = False             # the user asked never to bring this up


@dataclass
class GateResult:
    verdicts: list[Verdict]
    latency_ms: float
    decider_input_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None         # set when the decider was unreachable (fail-safe)

    def by_action(self, action: str) -> list[Verdict]:
        return [v for v in self.verdicts if v.action == action]


def http_ask(url: str = "http://127.0.0.1:8099", timeout: float = 180.0) -> AskFn:
    """An `ask` function for a running `strands-decider serve` server."""

    def ask(state: str, questions: Mapping[str, dict]) -> Mapping[str, dict]:
        body = json.dumps({"state": state, "questions": dict(questions)}).encode()
        req = urllib.request.Request(
            f"{url.rstrip('/')}/v1/systemone", data=body,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        ask.last_usage = payload.get("usage", {})  # type: ignore[attr-defined]
        return payload["answers"]

    ask.last_usage = {}  # type: ignore[attr-defined]
    return ask


def noul(instructions: str, criteria: Mapping[str, str]) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": dict(criteria)}


def choice(instructions: str, options: Mapping[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def confidence_of(probabilities: Mapping[str, float]) -> float:
    """The decider's own confidence formula: (N * p_max - 1) / (N - 1)."""
    n = len(probabilities)
    return max(0.0, (n * max(probabilities.values()) - 1) / (n - 1)) if n > 1 else 1.0


class MemoryGate:
    """Decide, per memory, whether the agent may mention it, use it quietly, or skip it."""

    def __init__(self, ask: AskFn, design: str = "choice",
                 cost: Mapping[str, Mapping[str, float]] = COST):
        if design not in ("choice", "split"):
            raise ValueError(f"design must be 'choice' or 'split', not {design!r}")
        self.ask = ask
        self.design = design
        self.cost = cost

    @staticmethod
    def render_state(conversation: Sequence[tuple[str, str]]) -> str:
        """The conversation the decider reads, newest message last."""
        lines = [f"{role}: {text}" for role, text in conversation]
        return "Conversation so far:\n" + "\n".join(lines)

    def questions(self, memories: Sequence[Mapping[str, str]]) -> dict[str, dict]:
        qs: dict[str, dict] = {}
        for m in memories:
            if self.design == "choice":
                qs[m["id"]] = choice(INSTRUCTIONS.format(memory=m["text"]), OPTIONS)
            else:
                qs[f"{m['id']}:helps"] = noul(HELPS[0].format(memory=m["text"]), HELPS[1])
                qs[f"{m['id']}:unwelcome"] = noul(UNWELCOME[0].format(memory=m["text"]), UNWELCOME[1])
        return qs

    def probabilities(self, memory_id: str, answers: Mapping[str, dict]) -> dict[str, float]:
        """P(mention), P(quiet), P(skip) for one memory, whichever design asked."""
        if self.design == "choice":
            p = answers[memory_id]["probabilities"]
            return {k: float(p.get(k, 0.0)) for k in LABELS}
        helps = float(answers[f"{memory_id}:helps"]["noul"])
        unwelcome = float(answers[f"{memory_id}:unwelcome"]["noul"])
        return {"mention": helps * (1 - unwelcome), "quiet": helps * unwelcome, "skip": 1 - helps}

    def decide(self, probabilities: Mapping[str, float], locked: bool = False) -> tuple[str, str]:
        """The policy: pick the action with the lowest expected cost."""
        expected = {a: sum(probabilities[t] * c for t, c in row.items())
                    for a, row in self.cost.items()}
        if locked:  # the user asked not to hear about it: never mention
            expected.pop("mention")
        action = min(expected, key=expected.get)
        top = max(probabilities, key=probabilities.get)
        if locked and top == "mention":
            return action, "locked by the user, so never mentioned"
        if action == top:
            return action, f"{top} at p={probabilities[top]:.2f}"
        return action, (f"leaned {top} (p={probabilities[top]:.2f}), but {action} has the "
                        f"lower expected cost ({expected[action]:.2f} vs {expected[top]:.2f})")

    def run(self, conversation: Sequence[tuple[str, str]],
            memories: Sequence[Mapping[str, Any]],
            extra_questions: Mapping[str, dict] | None = None) -> GateResult:
        """Judge every memory in one request. `extra_questions` ride along in the same
        request for free (the state is already encoded); their answers land in `raw`.
        A memory with `"locked": True` can be used quietly but is never mentioned."""
        questions = {**self.questions(memories), **(extra_questions or {})}
        if not questions:
            return GateResult([], 0.0)
        start = time.perf_counter()
        answers = self.ask(self.render_state(conversation), questions)
        latency_ms = (time.perf_counter() - start) * 1000
        verdicts = []
        for m in memories:
            p = self.probabilities(m["id"], answers)
            locked = bool(m.get("locked"))
            action, reason = self.decide(p, locked)
            top = max(p, key=p.get)
            verdicts.append(Verdict(m["id"], m["text"], top, confidence_of(p), p, action,
                                    reason, locked))
        tokens = getattr(self.ask, "last_usage", {}).get("input_tokens", 0)
        return GateResult(verdicts, latency_ms, tokens, dict(answers))

    def safe_run(self, *args, **kwargs) -> GateResult:
        """`run`, but if the decider is unreachable, remember nothing rather than guess."""
        try:
            return self.run(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - any failure means: no memories this turn
            return GateResult([], 0.0, error=f"{type(e).__name__}: {e}")


def render_context(result: GateResult) -> str:
    """Turn a gate result into the memory block for the LLM's system prompt."""
    mention = result.by_action("mention")
    quiet = result.by_action("quiet")
    parts = []
    if mention:
        parts.append("Things you remember about the user that you MAY bring up if useful:\n"
                     + "\n".join(f"- {v.text}" for v in mention))
    if quiet:
        parts.append("Sensitive background. Let it shape your answer (avoid suggestions that "
                     "would hurt), but NEVER mention, hint at or allude to these facts:\n"
                     + "\n".join(f"- {v.text}" for v in quiet))
    return "\n\n".join(parts) if parts else "You don't remember anything relevant to this message."


if __name__ == "__main__":
    # Minimal demo: needs `strands-decider serve <model> --port 8099` running.
    gate = MemoryGate(http_ask())
    memories = [
        {"id": "a", "text": "The user is severely allergic to peanuts."},
        {"id": "b", "text": "The user's dog passed away last month."},
        {"id": "c", "text": "The user prefers aisle seats on flights."},
    ]
    result = gate.run([("user", "Can you suggest a Thai dish I could cook tonight?")], memories)
    for v in result.verdicts:
        print(f"{v.action:8} {v.reason:60}  {v.text}")
    print(f"{result.latency_ms:.0f} ms for {len(memories)} memories in one call")
