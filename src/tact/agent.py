"""Tact: a Strands agent on Amazon Bedrock whose memory is gated by Strands Decider 2B."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from strands import Agent
from strands.interventions import InterventionHandler, Transform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))

from memory_gate import GateResult, MemoryGate, choice, http_ask, noul, render_context  # noqa: E402

from . import bedrock  # noqa: E402
from .memory import MemoryStore  # noqa: E402

SYSTEM_PROMPT = (
    "You are Tact, a warm, practical AI assistant chatting with Arjun. Talk to him directly as "
    "\"you\"; you can't meet him in person, so suggest things he can do. Reply in under 120 words. "
    "Use what you remember only when it genuinely helps, and never make the user feel watched."
)

# These ride along in the same decider request as the memory questions, so they cost no
# extra call: the conversation is read once, and each question only adds its own tokens.
REMEMBER = noul(
    "Does the user's latest message reveal a new, lasting fact about the user "
    "(a preference, plan, constraint or life event) worth remembering?",
    {"true": "the latest message states something about the user that will still matter next week",
     "false": "the latest message is only a request or small talk, with nothing lasting to store"},
)
FORGET = noul(
    "Is the user asking the assistant to stop bringing up, or to forget, something about them?",
    {"true": "the user says not to mention, remind them of, or talk about some subject again",
     "false": "the user is not asking the assistant to avoid any subject"},
)
TONE = choice(
    "How should the assistant's next reply feel, given the user's latest message?",
    {"gentle": "the user sounds stressed, sad, anxious or vulnerable, so the reply should be soft and kind",
     "upbeat": "the user sounds excited or is celebrating, so the reply can share the energy",
     "practical": "a plain request with no strong feeling, so the reply should be clear and to the point"},
)
# Asked only when FORGET fires: which of the candidate memories is the user talking about?
FORGET_TARGET = ("Is the user asking the assistant not to bring up this remembered fact?\n"
                 "Remembered fact: {memory}")

REMEMBER_AT = 0.7
FORGET_AT = 0.7
LOCK_AT = 0.6
TONE_AT = 0.5
TONE_STYLE = {
    "gentle": "The user seems to be having a hard time. Be gentle and kind, and keep it simple.",
    "upbeat": "The user is in a good mood. It's fine to be upbeat.",
}

EXTRACT_PROMPT = ("Rewrite the lasting personal fact in the user's message as one short third-person "
                  "sentence about Arjun. Output only the sentence.")


def decider_url() -> str:
    if os.environ.get("TACT_DECIDER_URL"):
        return os.environ["TACT_DECIDER_URL"]
    saved = ROOT / ".cache" / "decider_url"
    return saved.read_text().strip() if saved.exists() else "http://127.0.0.1:8099"


def _text_turns(messages: list[dict]) -> list[tuple[str, str]]:
    turns = []
    for m in messages:
        text = " ".join(b["text"] for b in m.get("content", []) if isinstance(b, dict) and "text" in b)
        if text:
            turns.append((m["role"], text))
    return turns


class MemoryGateIntervention(InterventionHandler):
    """Before each invocation, let the decider pick which memories reach the system prompt."""

    name = "tact-memory-gate"

    def __init__(self, gate: MemoryGate, store: MemoryStore, k: int = 8) -> None:
        self.gate, self.store, self.k = gate, store, k
        self.last: GateResult | None = None
        self.last_candidates: list[dict] = []
        self.last_tone: tuple[str, float] | None = None
        self.last_locked: list[dict] = []

    def before_invocation(self, event, **kwargs):
        new_turns = _text_turns(event.messages or [])
        if not new_turns:
            return Transform(apply=lambda e: None)
        # agent.messages does not hold the new turn yet; Strands appends it after this hook.
        conversation = _text_turns(event.agent.messages)[-6:] + new_turns
        latest = new_turns[-1][1]
        self.last_candidates = self.store.candidates(latest, k=self.k)
        self.last = self.gate.safe_run(conversation, self.last_candidates,
                                       {"_remember": REMEMBER, "_forget": FORGET, "_tone": TONE})
        self.last_locked = self._apply_forget(conversation)
        self.last_tone = None
        tone = self.last.raw.get("_tone")
        if tone and tone["confidence"] >= TONE_AT:
            self.last_tone = (tone["choice"], tone["confidence"])

        # On a decider failure the result is empty, so the prompt carries no memories at all:
        # never a stale block from the previous turn.
        prompt = f"{SYSTEM_PROMPT}\n\n{render_context(self.last)}"
        if self.last_tone and self.last_tone[0] in TONE_STYLE:
            prompt += "\n\n" + TONE_STYLE[self.last_tone[0]]

        def apply(e):
            e.agent.system_prompt = prompt
        return Transform(apply=apply, reason="memory gated by strands-decider")

    def _apply_forget(self, conversation) -> list[dict]:
        """If the user asked us to drop a subject, find which memories they meant and lock them."""
        if self.last.raw.get("_forget", {}).get("noul", 0.0) < FORGET_AT or not self.last.verdicts:
            return []
        questions = {v.memory_id: noul(FORGET_TARGET.format(memory=v.text),
                                       {"true": "the user's request is about this fact",
                                        "false": "the user's request is about something else"})
                     for v in self.last.verdicts}
        try:
            answers = self.gate.ask(self.gate.render_state(conversation), questions)
        except Exception:  # noqa: BLE001 - locking is best effort
            return []
        locked = []
        for v in self.last.verdicts:
            if answers[v.memory_id]["noul"] >= LOCK_AT:
                memory = self.store.lock(v.memory_id)
                v.locked = True
                v.action, v.reason = self.gate.decide(v.probabilities, locked=True)
                locked.append(memory)
        return locked


class TactAgent:
    def __init__(self, store: MemoryStore | None = None, k: int = 8,
                 design: str | None = None) -> None:
        self.store = store or MemoryStore()
        self.gate = MemoryGate(http_ask(decider_url()),
                               design=design or os.environ.get("TACT_GATE_DESIGN", "choice"))
        self.handler = MemoryGateIntervention(self.gate, self.store, k)
        self.agent = Agent(
            model=bedrock.strands_model(temperature=0.3, max_tokens=400),
            system_prompt=SYSTEM_PROMPT,
            interventions=[self.handler],
            callback_handler=None,
        )

    def chat(self, text: str) -> dict:
        before = dict(self.agent.event_loop_metrics.accumulated_usage)
        result = self.agent(text)
        after = self.agent.event_loop_metrics.accumulated_usage
        gate = self.handler.last
        remembered = None
        p_remember = gate.raw.get("_remember", {}).get("noul", 0.0) if gate else 0.0
        # A request to forget something is not a new fact to store.
        if p_remember >= REMEMBER_AT and not self.handler.last_locked:
            fact = bedrock.chat(EXTRACT_PROMPT, [("user", text)], max_tokens=60, temperature=0)["text"].strip()
            remembered = self.store.add(fact)
        return {
            "reply": str(result).strip(),
            "gate": gate,
            "candidates": self.handler.last_candidates,
            "p_remember": p_remember,
            "remembered": remembered,
            "tone": self.handler.last_tone,
            "locked": self.handler.last_locked,
            "input_tokens": after["inputTokens"] - before.get("inputTokens", 0),
            "output_tokens": after["outputTokens"] - before.get("outputTokens", 0),
        }
