"""Offline tests for the replayed evaluation (eval/replay_eval.py): no decider, no Bedrock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "patterns" / "memory_gate"))
sys.path.insert(0, str(ROOT / "eval"))

import replay_eval as rp  # noqa: E402

from tact.memory import MemoryStore  # noqa: E402


class ReplayTest(unittest.TestCase):
    def test_saved_decisions_match_the_shipped_policy(self):
        # The replay stands in for the decider only while the cost table is unchanged.
        texts = {m["id"]: m["text"] for m in MemoryStore().all()}
        for _, run_file in rp.SETS.values():
            for row in json.loads((ROOT / run_file).read_text(encoding="utf-8")):
                result = rp.saved_tact(row["systems"]["tact"], texts)  # SystemExit on drift
                self.assertEqual(len(result.verdicts), len(row["candidates"]))

    def test_escalation_keeps_confident_decisions(self):
        rows = [{"systems": {
            "tact": {"gate_latency_ms": 100.0, "gate": {"a": {"action": "skip", "confidence": 0.9},
                                                        "b": {"action": "skip", "confidence": 0.2}}},
            "Nova Pro": {"gate_latency_ms": 50.0, "gate_cost_usd": 0.001,
                         "gate": {"a": {"action": "mention"}, "b": {"action": "quiet"}}}}}]
        name = rp.escalate(rows, "us.amazon.nova-pro-v1:0", 0.5)
        gate = rows[0]["systems"][name]
        self.assertEqual(gate["gate"], {"a": {"action": "skip"}, "b": {"action": "quiet"}})
        self.assertEqual((gate["sent"], gate["gate_latency_ms"], gate["gate_cost_usd"]), (1, 150.0, 0.001))

    def test_the_bedrock_run_works_end_to_end_with_fake_models(self):
        env = {**os.environ, "TACT_PROVIDER": "bedrock", "PYTHONIOENCODING": "utf-8"}
        out = subprocess.run([sys.executable, str(ROOT / "eval" / "replay_eval.py"), "--dry-run"],
                             capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("# Tact on Amazon Bedrock", out.stdout)
        self.assertIn("Replies written by Nova Lite (fresh scenarios)", out.stdout)


if __name__ == "__main__":
    unittest.main()
