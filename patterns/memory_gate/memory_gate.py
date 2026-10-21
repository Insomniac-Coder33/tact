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

The write side has its own decision. When the user tells the agent something new,
`MemoryCurator` asks, for each similar memory, whether the message makes it outdated,
repeats it, or leaves it alone, so "I moved to Bangalore" retires "lives in Pune"
instead of leaving two contradicting facts in the store.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
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
    stale: bool = False              # the curator suspects it may be out of date


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


def _request_key(state: str, questions: Mapping[str, dict]) -> str:
    blob = json.dumps([state, dict(questions)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def recording_ask(ask: AskFn, path: str | Path) -> AskFn:
    """Wrap an `ask` function so every request and its answers are saved to a JSON file."""
    path = Path(path)

    def record(state: str, questions: Mapping[str, dict]) -> Mapping[str, dict]:
        answers = ask(state, questions)
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        saved[_request_key(state, questions)] = dict(answers)
        path.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
        return answers

    record.last_usage = {}  # type: ignore[attr-defined]
    return record


def replay_ask(path: str | Path) -> AskFn:
    """An `ask` function that answers from a recording: for demos, tests and CI without a
    decider server. A request that was never recorded raises KeyError."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))

    def replay(state: str, questions: Mapping[str, dict]) -> Mapping[str, dict]:
        key = _request_key(state, questions)
        if key not in saved:
            raise KeyError(f"request {key} is not in {path}; record it with recording_ask()")
        return saved[key]

    replay.last_usage = {}  # type: ignore[attr-defined]
    return replay


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
                                    reason, locked, bool(m.get("stale"))))
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

    def line(v: Verdict) -> str:
        return f"- {v.text}" + (" (may be out of date: don't rely on it)" if v.stale else "")

    parts = []
    if mention:
        parts.append("Things you remember about the user that you MAY bring up if useful:\n"
                     + "\n".join(line(v) for v in mention))
    if quiet:
        parts.append("Sensitive background. Let it shape your answer (avoid suggestions that "
                     "would hurt), but NEVER mention, hint at or allude to these facts:\n"
                     + "\n".join(line(v) for v in quiet))
    return "\n\n".join(parts) if parts else "You don't remember anything relevant to this message."


# --- the write side: does a new message make an old memory outdated? ------------------
UPDATE_INSTRUCTIONS = (
    "The user just sent the message above. The assistant remembers this fact about the user:\n"
    "{memory}\nWhat does the user's message mean for this remembered fact?"
)
UPDATE_OPTIONS = {
    "outdated": "the message shows the fact is no longer true: something changed (a move, a new "
                "job, a cancelled or changed plan, a new preference), so the old fact should be replaced",
    "repeat": "the message says the same thing as the fact again, so there is nothing new to store",
    "unaffected": "the fact is still true after this message; the message is about something else "
                  "or only adds a new detail alongside it",
}
# A past event stays true when life moves on, but "I got a new job" still pushed
# P(outdated) for "was laid off in March 2025" to 0.72. Writing that into the options above
# made things worse (every P(outdated) went up), so it is a separate yes/no question.
PAST_EVENT = (
    "Is this remembered fact about the user a past event that is over and done, rather than "
    "their current situation, plan, habit or preference?\nRemembered fact: {memory}",
    {"true": "it describes something that happened and finished, which stays true forever",
     "false": "it describes how things are now or what the user plans or prefers, which can change"},
)


@dataclass
class Update:
    memory_id: str
    text: str
    probabilities: dict[str, float]  # P(outdated), P(repeat), P(unaffected)
    action: str                      # retire / doubt / keep
    p_past: float = 0.0              # P(the memory is a finished past event)


@dataclass
class CurateResult:
    updates: list[Update]
    store_new: bool                  # False when the message only repeats a known fact
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def by_action(self, action: str) -> list[Update]:
        return [u for u in self.updates if u.action == action]


class MemoryCurator:
    """Decide what a new message does to the memories that are already stored.

    Three bands instead of one threshold, because deleting a true memory and keeping a
    stale one are both mistakes, and the decider is not always sure which one it is:

        P(outdated) >= retire_at        retire: the memory leaves the store (kept in history)
        doubt_at <= P(outdated) < ...   doubt: keep it, but tell the LLM it may be out of date
        below doubt_at                  keep

    If any memory is a likely repeat (P(repeat) >= repeat_at), the new fact isn't stored.
    A memory that is likely a finished past event (P(past) >= past_at) is never retired:
    at most it is flagged. Set `past_at=None` to skip that question.
    """

    def __init__(self, ask: AskFn, retire_at: float = 0.6, doubt_at: float = 0.4,
                 repeat_at: float = 0.5, past_at: float | None = 0.5):
        self.ask = ask
        self.retire_at, self.doubt_at, self.repeat_at = retire_at, doubt_at, repeat_at
        self.past_at = past_at

    def decide(self, probabilities: Mapping[str, float], p_past: float = 0.0) -> str:
        p = probabilities.get("outdated", 0.0)
        action = "retire" if p >= self.retire_at else "doubt" if p >= self.doubt_at else "keep"
        if action == "retire" and self.past_at is not None and p_past >= self.past_at:
            return "doubt"
        return action

    def run(self, conversation: Sequence[tuple[str, str]],
            memories: Sequence[Mapping[str, Any]]) -> CurateResult:
        if not memories:
            return CurateResult([], True, 0.0)
        questions = {m["id"]: choice(UPDATE_INSTRUCTIONS.format(memory=m["text"]), UPDATE_OPTIONS)
                     for m in memories}
        if self.past_at is not None:
            questions.update({f"{m['id']}:past": noul(PAST_EVENT[0].format(memory=m["text"]), PAST_EVENT[1])
                              for m in memories})
        start = time.perf_counter()
        answers = self.ask(MemoryGate.render_state(conversation), questions)
        latency_ms = (time.perf_counter() - start) * 1000
        updates = []
        for m in memories:
            p = {k: float(answers[m["id"]]["probabilities"].get(k, 0.0)) for k in UPDATE_OPTIONS}
            p_past = float(answers.get(f"{m['id']}:past", {}).get("noul", 0.0))
            updates.append(Update(m["id"], m["text"], p, self.decide(p, p_past), p_past))
        repeat = any(u.probabilities["repeat"] >= self.repeat_at for u in updates)
        return CurateResult(updates, not repeat, latency_ms, dict(answers))

    def safe_run(self, *args, **kwargs) -> CurateResult:
        """`run`, but if the decider is unreachable, change nothing and store the new fact."""
        try:
            return self.run(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - failing safe means: touch no stored memory
            return CurateResult([], True, 0.0, error=f"{type(e).__name__}: {e}")

