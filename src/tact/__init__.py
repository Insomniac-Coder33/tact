"""Tact: Decider-gated long-term memory for a Strands agent on Amazon Bedrock."""


def main() -> None:
    from .agent import TactAgent

    tact = TactAgent()
    print("Tact. Ctrl+C to quit.")
    while True:
        out = tact.chat(input("you> "))
        for v in out["gate"].verdicts:
            print(f"   [{v.action:7} {v.confidence:.2f}] {v.text}")
        print(f"tact> {out['reply']}\n")
