# Plan: Tact on Amazon Bedrock

Written and committed before any Bedrock model is called.

## Why

The evaluation so far used Qwen2.5 3B, served by Ollama on the decider's EC2 instance, both for
the replies and as the "LLM gate" baseline, because this AWS account couldn't call Bedrock. A
3B model is a weak baseline. This plan fixes, in advance, what runs once Bedrock is available
and how it is reported.

## What does not change

- **Tact.** The question wording, cost table, thresholds and memories are the ones frozen at git
  tag `eval-freeze-2026-10-22`.
- **The decider's answers.** They are replayed from the saved runs, not re-run. The decider is
  deterministic (two runs gave identical probabilities on all 272 decisions), and
  `eval/replay_eval.py` stops if any saved decision differs from the shipped policy.
- **Scenarios and labels.** 34 original and 16 fresh conversations, with the same 8-memory
  shortlist per scenario as the saved runs.

| input | sha256 |
|---|---|
| `eval/scenarios.json` | `ff29598588cf96dc815f7b72bf5000e5971fe3d09f524387541c4d4341880f39` |
| `eval/fresh_scenarios.json` | `30661b8d83026327441616dd44d7d078e0137a76e2883090180966fb42010fcc` |
| `eval/results/run-20261021-202009.json` | `20e431019961bd790133fb152be1c718c0dedfd5705fedc25c7862dad07d825c` |
| `eval/results/run-20261022-001433.json` | `8c8d67628ebc2462c7546defa4c1ee0b55589ab7539f1b3d119d36a74dc87669` |

## What runs

One command, in `us-east-1`:

```bash
uv run python -m tact.bedrock                              # access check
TACT_PROVIDER=bedrock uv run python eval/replay_eval.py
```

1. **LLM-gate baselines.** Amazon Nova Lite and Amazon Nova Pro each label every shortlisted
   memory mention / quiet / skip, with the same prompt as before (`LLM_GATE_PROMPT` in
   `eval/run_eval.py`) at temperature 0. Scored against the hand labels exactly like Tact.
2. **Escalation.** The decider keeps its decisions at or above a confidence bar (0.33, 0.5,
   0.66) and each model's label replaces the rest. Scored from the answers in step 1.
3. **Replies.** Nova Lite writes the reply for every way of giving it memories: all memories,
   top-8 retrieval, static tags, Nova Lite as its own gate, and Tact. Same system prompt as the
   agent, temperature 0.3. Leaks are checked by keyword (as before) and by Nova Pro as a judge
   (`JUDGE_PROMPT`).

## Headline comparisons, chosen now

- **Fresh set:** Tact vs. the Nova Lite gate vs. the Nova Pro gate, on 3-way accuracy and
  sensitive facts let through.
- **Replies:** replies that leak a sensitive fact (keyword or judge), and helpful facts used,
  for Tact vs. all memories and top-8 retrieval.
- **Escalation:** whether handing only the decider's unsure decisions to Nova Pro beats both
  on their own.

## Reporting

- Every table the script writes goes into `eval/RESULTS-bedrock.md`, with the raw answers in
  `eval/results/`, whatever they show.
- If a Bedrock model beats Tact on accuracy or on sensitive facts let through, the article
  says so.
- No re-running to get a better result. If a run fails partway (for example throttling on a
  new account's small quota), it is re-run whole and the failure is noted here.

## If Bedrock stays blocked

Then only steps 1 and 2 run, against a larger open model than the 3B baseline: Qwen2.5 7B
Instruct, served by Ollama on the decider's EC2 instance, with the same prompt and temperature 0.

```bash
uv run python infra/decider_ec2.py pull qwen2.5:7b-instruct
TACT_PROVIDER=ollama uv run python eval/replay_eval.py --models qwen2.5:7b-instruct --parts gates
```

The results go into `eval/RESULTS-ollama.md`, reported the same way, whatever they show.

**Status, 2026-10-23:** Bedrock was still blocked at submission time. Every call, including one
from the Bedrock console's playground, returned `ValidationException: Operation not allowed`, also
after a verified card was added. The fallback above ran on 2026-10-22; the
Bedrock steps have not run yet.

## Not in scope

The decider on a GPU (the account has no GPU quota), new scenarios, and any change to Tact.
