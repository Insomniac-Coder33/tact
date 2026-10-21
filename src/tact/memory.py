"""A tiny long-term memory store with an embedding prefilter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import bedrock

DATA = Path(__file__).resolve().parents[2] / "data"


class MemoryStore:
    def __init__(self, path: Path = DATA / "memories.json"):
        self.path = path
        self.memories: list[dict] = json.loads(path.read_text(encoding="utf-8"))

    def all(self) -> list[dict]:
        return list(self.memories)

    def add(self, text: str) -> dict:
        memory = {"id": f"m{len(self.memories) + 1:02d}", "text": text}
        self.memories.append(memory)
        return memory

    def lock(self, memory_id: str) -> dict:
        """The user asked not to hear about this again: it may still shape answers quietly,
        but the gate will never let it be mentioned."""
        memory = next(m for m in self.memories if m["id"] == memory_id)
        memory["locked"] = True
        return memory

    def candidates(self, query: str, k: int = 8) -> list[dict]:
        """Top-k memories by cosine similarity. Similar is not the same as useful or
        appropriate; that judgement is left to the MemoryGate."""
        q = bedrock.embed(query)
        scored = [(float(np.dot(q, bedrock.embed(m["text"]))), m) for m in self.memories]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [m for _, m in scored[:k]]
