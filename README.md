# 🤫 Tact: an assistant that remembers everything, and knows when to keep quiet

**Agent Experience Hackathon: Strands Decider** · `#strands-decider-2b` `#context-memory`

AI assistants with long-term memory have two failure modes: they forget what matters, or they
remember *too well* and bring up your breakup while you plan a trip to Paris. Tact is a
[Strands](https://strandsagents.com) agent whose memory passes through a
gate run by **[Strands Decider 2B](https://github.com/strands-labs/strands-decider)**. Before
every reply, the decider looks at each candidate memory and decides: **mention** it, use it
**quietly**, or **skip** it, and an expected-cost policy turns its calibrated probabilities into
action: when in doubt about saying something, Tact uses it quietly.

Tact isn't a static "sensitive" tag. *"I'm thinking of getting a puppy"* keeps the dog that died
unsaid; *"I still miss Biscuit"* lets Tact acknowledge him. Same memory, different conversation,
different decision. And when Arjun says *"I just moved to Bangalore"*, a second decision retires
"lives in Pune", so old facts stop coming back.

**Try the pattern in 10 seconds**, with no server, GPU or AWS account (it replays answers recorded
from a real Decider 2B):

```bash
git clone https://github.com/Insomniac-Coder33/tact && cd tact
python patterns/memory_gate/quickstart.py
```

```
user message
  │
  ├─ 1. Embeddings shortlist the 8 most similar memories
  │
  ├─ 2. Strands Decider 2B, ONE batched request:
  │       8 × choice "mention / quiet / skip?"
  │       + noul "worth remembering?" + noul "asking me to drop a subject?" + choice "tone?"
  │
  ├─ 3. Expected-cost policy (plain Python): a painful mention costs 3× a missed detail,
  │       so P(quiet) of 25% keeps a fact unsaid. Decider down → no memories (fail-safe)
  │
  └─ 4. The chat model replies, seeing only what the gate allowed
```

Models: **Amazon Bedrock** (Nova Lite + Titan Text Embeddings V2) by default, or
`TACT_PROVIDER=ollama` (Qwen2.5 + nomic-embed-text on the decider's EC2 instance) for accounts
without Bedrock access. The results below used the Ollama setup.

The gate sits in the agent's path as a Strands `InterventionHandler.before_invocation`
([`src/tact/agent.py`](src/tact/agent.py)).

## Reusable decision pattern

**[`patterns/memory_gate/`](patterns/memory_gate/)**: one standard-library Python file plus a
README. Drop it into any agent with long-term memory. It contains the typed question and option
set, the policy, the prompt renderer and the Strands wiring.

## Results

34 hand-labelled scenarios over 30 memories. Ten are built to break static tags: the user raises
the painful subject themselves, or it only becomes sensitive two turns into the conversation.

| | static tags | LLM as the gate | **Tact (Decider 2B)** |
|---|---|---|---|
| gate decisions right | 23% | 60% | **83%** (85% on the 24 untuned scenarios) |
| … on 16 fresh conversations, labels committed before the run | 21% | 70% | **88%** |
| … sensitive facts let through there | 0 / 24 | 3 / 24 | **0 / 24** |
| sensitive facts let through | 0 / 60 | 5 / 60 | **0 / 60** |
| user-raised subjects acknowledged | 0 / 7 | 2 / 7 | **4 / 7** |

In the replies, Tact never said anything sensitive (top-8 retrieval did 4 times out of 34, all
memories in the prompt 3 times), used as many helpful facts as retrieval, and sent the model
**79% fewer input tokens** than the all-memories agent (114 vs 540).

Full tables, including the two-question design and leak examples: [`eval/RESULTS.md`](eval/RESULTS.md),
produced by [`eval/run_eval.py`](eval/run_eval.py). The cost value was tuned on a 10-scenario dev
split with [`eval/tune.py`](eval/tune.py).

## Run it

Prerequisites: Python 3.12 with [uv](https://docs.astral.sh/uv/), AWS credentials with Amazon
Bedrock access in `us-east-1` (Nova Lite and Titan Text Embeddings V2). Without Bedrock, the EC2
launcher also starts Ollama on the instance; then set `TACT_PROVIDER=ollama`.

```bash
uv sync
uv run python -m tact.bedrock        # can this account call Bedrock? says what to fix if not
echo bedrock > .cache/provider       # or "ollama": switches replies + embeddings, nothing else

# 1. A decider server. Either locally (needs ~5 GB download, slow on CPU) ...
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
# ... or on an EC2 GPU instance, reachable only from your IP:
uv run python infra/decider_ec2.py up      # stop with: infra/decider_ec2.py stop
uv run python infra/decider_ec2.py app     # optional: run the demo app on the same instance

# 2. The demo UI
uv run streamlit run app.py

# 3. The evaluation, and offline tests (no server needed)
python patterns/memory_gate/quickstart.py   # the pattern in 10 seconds, replaying recorded answers
uv run python eval/run_eval.py
uv run python eval/replay_eval.py --dry-run  # the LLM parts on other models, replaying the decider
uv run python -m unittest discover -s tests
```

## Repository

| path | what |
|---|---|
| `patterns/memory_gate/` | the reusable Decider 2B pattern |
| `src/tact/agent.py` | Strands agent, `MemoryGateIntervention`, remember / forget / tone |
| `src/tact/memory.py` | memory store with embedding prefilter |
| `src/tact/bedrock.py` | chat + embeddings helpers (Bedrock, or Ollama) |
| `app.py` | Streamlit demo, with a naive-memory agent side by side |
| `eval/` | scenarios, evaluation script, results |
| `tests/` | offline tests with a stub decider |
| `docs/` | Builder Center article |
| `infra/decider_ec2.py` | launch the decider server on EC2 (GPU, CPU fallback) |
| `data/memories.json` | the fictional user Arjun's 30 memories |

All people and memories in this repo are fictional.

## License

Apache-2.0
