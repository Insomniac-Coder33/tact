"""Offline tests: a stub stands in for the decider server.   uv run python -m unittest"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "src"))

from memory_gate import MemoryGate, render_context  # noqa: E402

MEMORIES = [
    {"id": "veg", "text": "The user is vegetarian."},
    {"id": "ex", "text": "The user broke up with Meera in Paris."},
    {"id": "seat", "text": "The user prefers aisle seats."},
]


def choice_answer(mention: float, quiet: float, skip: float) -> dict:
    p = {"mention": mention, "quiet": quiet, "skip": skip}
    top = max(p, key=p.get)
    return {"type": "choice", "choice": top, "probabilities": p, "confidence": (3 * p[top] - 1) / 2}


class StubDecider:
    """Records each request and answers from a fixed table."""

    def __init__(self, answers: dict):
        self.answers, self.calls = answers, []
        self.last_usage = {"input_tokens": 123}

    def __call__(self, state, questions):
        self.calls.append((state, dict(questions)))
        return {qid: self.answers[qid] for qid in questions}


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.gate = MemoryGate(StubDecider({}))

    def test_confident_labels_pass_through(self):
        self.assertEqual(self.gate.decide({"mention": .9, "quiet": .02, "skip": .08})[0], "mention")
        self.assertEqual(self.gate.decide({"mention": .1, "quiet": .85, "skip": .05})[0], "quiet")
        self.assertEqual(self.gate.decide({"mention": .03, "quiet": .02, "skip": .95})[0], "skip")

    def test_a_little_doubt_about_saying_it_means_quiet(self):
        # Most likely "mention", but 25% on "quiet": saying it is not worth the risk.
        action, reason = self.gate.decide({"mention": .6, "quiet": .25, "skip": .15})
        self.assertEqual(action, "quiet")
        self.assertIn("expected cost", reason)

    def test_locked_memory_is_never_mentioned(self):
        action, _ = self.gate.decide({"mention": .95, "quiet": .03, "skip": .02}, locked=True)
        self.assertNotEqual(action, "mention")


class GateTest(unittest.TestCase):
    conversation = [("user", "I'm planning a trip to Paris. What should I see?")]

    def test_choice_design_one_request_with_extras(self):
        stub = StubDecider({"veg": choice_answer(.85, .05, .10), "ex": choice_answer(.15, .75, .10),
                            "seat": choice_answer(.05, .05, .90), "_x": {"type": "noul", "noul": .9}})
        result = MemoryGate(stub).run(self.conversation, MEMORIES, {"_x": {"type": "noul"}})
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual({v.memory_id: v.action for v in result.verdicts},
                         {"veg": "mention", "ex": "quiet", "seat": "skip"})
        self.assertEqual(result.raw["_x"]["noul"], .9)
        self.assertEqual(result.decider_input_tokens, 123)
        self.assertIn("user: I'm planning a trip to Paris", stub.calls[0][0])

    def test_split_design_combines_two_yes_no_answers(self):
        n = lambda p: {"type": "noul", "noul": p}  # noqa: E731
        stub = StubDecider({"veg:helps": n(.9), "veg:unwelcome": n(.05),
                            "ex:helps": n(.8), "ex:unwelcome": n(.9),
                            "seat:helps": n(.05), "seat:unwelcome": n(.1)})
        result = MemoryGate(stub, design="split").run(self.conversation, MEMORIES)
        self.assertEqual(len(stub.calls[0][1]), 6)
        actions = {v.memory_id: v.action for v in result.verdicts}
        self.assertEqual(actions, {"veg": "mention", "ex": "quiet", "seat": "skip"})
        ex = next(v for v in result.verdicts if v.memory_id == "ex")
        self.assertAlmostEqual(sum(ex.probabilities.values()), 1.0)

    def test_render_context_separates_mention_and_quiet(self):
        stub = StubDecider({"veg": choice_answer(.85, .05, .10), "ex": choice_answer(.15, .75, .10),
                            "seat": choice_answer(.05, .05, .90)})
        text = render_context(MemoryGate(stub).run(self.conversation, MEMORIES))
        mention, quiet = text.split("Sensitive background")
        self.assertIn("vegetarian", mention)
        self.assertIn("Meera", quiet)
        self.assertNotIn("aisle", text)

    def test_unreachable_decider_means_no_memories(self):
        def down(state, questions):
            raise ConnectionRefusedError("decider offline")
        result = MemoryGate(down).safe_run(self.conversation, MEMORIES)
        self.assertEqual(result.verdicts, [])
        self.assertIn("ConnectionRefusedError", result.error)
        self.assertIn("don't remember anything", render_context(result))


class ForgetTest(unittest.TestCase):
    """The Strands hook: 'don't bring up Meera again' locks that memory."""

    def test_forget_request_locks_the_memory(self):
        from tact.agent import MemoryGateIntervention

        class Store:
            def __init__(self):
                self.memories = [dict(m) for m in MEMORIES]

            def candidates(self, query, k=8):
                return self.memories

            def lock(self, mid):
                m = next(m for m in self.memories if m["id"] == mid)
                m["locked"] = True
                return m

        answers = {"veg": choice_answer(.05, .05, .90), "ex": choice_answer(.80, .10, .10),
                   "seat": choice_answer(.05, .05, .90),
                   "_remember": {"noul": .2}, "_forget": {"noul": .95},
                   "_tone": {"choice": "gentle", "confidence": .8}}

        def ask(state, questions):
            if "ex" in questions and questions["ex"]["type"] == "noul":  # the follow-up
                return {"veg": {"noul": .05}, "ex": {"noul": .9}, "seat": {"noul": .02}}
            return {q: answers[q] for q in questions}

        store = Store()
        hook = MemoryGateIntervention(MemoryGate(ask), store)
        event = SimpleNamespace(
            messages=[{"role": "user", "content": [{"text": "Please don't bring up Meera again."}]}],
            agent=SimpleNamespace(messages=[], system_prompt=""))
        action = hook.before_invocation(event)
        action.apply(event)

        self.assertEqual([m["id"] for m in hook.last_locked], ["ex"])
        self.assertTrue(store.memories[1]["locked"])
        ex = next(v for v in hook.last.verdicts if v.memory_id == "ex")
        self.assertNotEqual(ex.action, "mention")
        self.assertEqual(hook.last_tone, ("gentle", .8))
        self.assertIn("gentle", event.agent.system_prompt)


if __name__ == "__main__":
    unittest.main()
