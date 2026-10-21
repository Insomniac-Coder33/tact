"""See the pattern work in ten seconds, with no server, no GPU and no AWS account.

    python patterns/memory_gate/quickstart.py
    python patterns/memory_gate/quickstart.py --live http://127.0.0.1:8099            # ask a real decider
    python patterns/memory_gate/quickstart.py --live http://127.0.0.1:8099 --record   # and refresh the recording

By default the decider's answers are replayed from `quickstart_answers.json`, recorded from a
real Strands Decider 2B server. The policy, the curator bands and the prompt block are computed
live by `memory_gate.py`, so you can change COST or the thresholds and rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from memory_gate import (  # noqa: E402
    MemoryCurator, MemoryGate, http_ask, recording_ask, render_context, replay_ask,
)

MEMORIES = [
    {"id": "m1", "text": "The user has been vegetarian since 2019."},
    {"id": "m2", "text": "The user's dog Biscuit passed away last month."},
    {"id": "m3", "text": "The user used to walk Biscuit every morning by the lake."},
    {"id": "m4", "text": "The user broke up with Meera during a trip to Paris last year."},
    {"id": "m5", "text": "The user prefers aisle seats on flights."},
    {"id": "m6", "text": "The user lives in Pune and works as a data analyst."},
]

SCENES = [
    ("Same memories, a harmless question", [("user", "What should I do this Saturday morning?")]),
    ("A topic close to a painful memory", [("user", "I'm thinking of getting a puppy. Thoughts?")]),
    ("The user opens the door themselves",
     [("user", "It's been a month since Biscuit died and I still can't walk past the lake. How do I cope?")]),
    ("A request where a memory really helps", [("user", "Can you suggest a Thai dish I could cook tonight?")]),
]
NEWS = [("user", "Big news: I just moved to Bangalore for a new job at a bank!")]
# In an agent, the curator only sees the few stored memories closest to the news (by embedding).
NEAR_NEWS = ["m6", "m5", "m2"]

BAR = {"mention": "█", "quiet": "▒", "skip": "·"}


def bar(p: dict[str, float], width: int = 20) -> str:
    cells = "".join(BAR[k] * round(p[k] * width) for k in ("mention", "quiet", "skip"))
    return (cells + "·" * width)[:width]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", metavar="URL", help="ask a running decider server instead of the recording")
    parser.add_argument("--record", action="store_true", help="with --live: save the answers for replay")
    args = parser.parse_args()
    recording = HERE / "quickstart_answers.json"
    if args.live:
        ask = http_ask(args.live)
        if args.record:
            recording.unlink(missing_ok=True)
            ask = recording_ask(ask, recording)
    else:
        ask = replay_ask(recording)

    print(f"Decider answers: {'live from ' + args.live if args.live else 'replayed from ' + recording.name}\n")
    print(f"Memories: {len(MEMORIES)}.  Bars: {BAR['mention']} P(mention)  {BAR['quiet']} P(quiet)  "
          f"{BAR['skip']} P(skip)\n")
    gate = MemoryGate(ask)
    for title, conversation in SCENES:
        result = gate.run(conversation, MEMORIES)
        print(f"── {title}\n   user: {conversation[-1][1]}")
        for v in sorted(result.verdicts, key=lambda v: ("mention", "quiet", "skip").index(v.action)):
            note = "" if v.action == v.label else f"  ← {v.reason}"
            print(f"   {bar(v.probabilities)}  {v.action:7}  {v.text}{note}")
        print()

    print("── What the LLM's system prompt gets for the last scene:\n")
    print("   " + render_context(result).replace("\n", "\n   ") + "\n")

    print(f"── Writing: the user shares news\n   user: {NEWS[0][1]}")
    curation = MemoryCurator(ask).run(NEWS, [m for m in MEMORIES if m["id"] in NEAR_NEWS])
    for u in sorted(curation.updates, key=lambda u: -u.probabilities["outdated"]):
        print(f"   P(outdated)={u.probabilities['outdated']:.2f}  P(past event)={u.p_past:.2f}  "
              f"{u.action:6}  {u.text}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
