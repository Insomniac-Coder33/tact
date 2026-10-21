# MemoryGate: a reusable Strands Decider 2B pattern (#context-memory)

**What it does:** before your agent calls its LLM, MemoryGate decides, for each candidate memory,
whether the agent may **mention** it, should use it **quietly** (as background it never says out
loud), or should **skip** it. Strands Decider 2B gives the probabilities; a small expected-cost
policy in plain Python turns them into the memory block for the system prompt.

Its write-side partner, **MemoryCurator**, decides what a new message does to the memories already
stored: makes them **outdated**, **repeats** them, or leaves them **unaffected**
([below](#the-write-side-memorycurator)).

It solves two problems memory-enabled agents have:

1. **Context bloat.** Dumping every memory into the prompt costs tokens on every turn.
2. **No tact.** Retrieval finds *similar* memories, not *appropriate* ones. Ask for Paris
   sightseeing tips and similarity search happily serves up "broke up with Meera in Paris".

Why not tag sensitive memories once, when they're stored? Because tact depends on the
conversation. "Your dog Biscuit died" must stay unsaid when the user asks about getting a puppy,
and is exactly the right thing to acknowledge when the user says *"I still miss Biscuit"*. A
static tag can't tell those apart; a decision made per message can.

## The decision

All the questions for all the memories go in **one** request. The decider reads the
conversation once, and each extra question only adds its own tokens. There are two
interchangeable ways to ask (`MemoryGate(ask, design=...)`):

**`design="choice"`**: one 3-way `choice` per memory.

```python
{
  "type": "choice",
  "instructions": "An AI assistant is about to reply to the user's latest message. It remembers "
                  "this fact about the user:\n{memory}\nHow should the assistant treat this fact in its reply?",
  "criteria": {
    "mention": "use it openly: it makes the reply more useful or personal (a preference, constraint, "
               "plan or interest that fits the request), and bringing it up feels natural",
    "quiet":   "use it silently: it is related to the request, but saying it out loud could hurt, "
               "embarrass or upset the user, so it should only make the reply more careful",
    "skip":    "ignore it: it has no bearing on what a good reply would say"
  }
}
```

**`design="split"`**: two narrow yes/no `noul` questions per memory, "would this fact change
a thoughtful reply?" and "would saying it out loud be unwelcome, given the conversation so far?".
They combine into the same three probabilities:

```
P(mention) = P(helps) × (1 − P(unwelcome))
P(quiet)   = P(helps) × P(unwelcome)
P(skip)    = 1 − P(helps)
```

**State:** the last few turns of the conversation, newest message last.

## The policy: expected cost, not argmax

A painful mention is much worse than a missed detail, so the gate doesn't just take the most
likely label. It picks the action with the lowest expected cost under the decider's
probabilities (`COST` in `memory_gate.py`, `cost[action][truth]`):

| action ↓ / truth → | mention | quiet | skip |
|---|---|---|---|
| **mention** | 0 | **3** (said something painful) | 0.5 (irrelevant chatter) |
| **quiet** | 1 (held back a useful fact) | 0 | 0.4 (wasted tokens) |
| **skip** | 1.5 (lost a useful fact) | 1 (answer may be careless) | 0 |

The effect: a fact that is 60% "mention" but 25% "quiet" is used **quietly**. The model
leaned towards saying it, but not confidently enough to risk it. Change the table to change
the behaviour, without retraining anything. The 3 came from a sweep on a dev split
([`eval/tune.py`](../../eval/tune.py)): at 5 the gate almost never said anything, and below
2.5 it started saying sensitive facts.

A memory with `"locked": True` (the user asked not to hear about it) can still shape answers
quietly, but the policy never lets it be mentioned.

## Use it in 10 lines

`memory_gate.py` is a single standard-library file. Copy it into your project.

```python
from memory_gate import MemoryGate, http_ask, render_context

gate = MemoryGate(http_ask("http://127.0.0.1:8099"))   # strands-decider serve <model> --port 8099
result = gate.safe_run(                                  # decider down → no memories, never stale ones
    conversation=[("user", "Any tips for my trip to Paris?")],
    memories=[{"id": "a", "text": "User broke up with Meera in Paris last May."},
              {"id": "b", "text": "User is afraid of heights."}],
)
system_prompt = BASE_PROMPT + "\n\n" + render_context(result)
for v in result.verdicts:
    print(v.action, v.probabilities, v.reason)
```

## Wire it into a Strands agent

The seam is a Strands `InterventionHandler.before_invocation`. It runs once per user message,
before any model call, and returns a `Transform` that rewrites the system prompt. See
[`src/tact/agent.py`](../../src/tact/agent.py), `MemoryGateIntervention`:

```python
class MemoryGateIntervention(InterventionHandler):
    def before_invocation(self, event, **kwargs):
        latest = latest_user_text(event)
        result = self.gate.safe_run(conversation, self.store.candidates(latest, k=8))
        prompt = f"{SYSTEM_PROMPT}\n\n{render_context(result)}"
        return Transform(apply=lambda e: setattr(e.agent, "system_prompt", prompt))

agent = Agent(model=BedrockModel(model_id="us.amazon.nova-lite-v1:0"),
              interventions=[MemoryGateIntervention(gate, store)])
```

## Free extra questions

`gate.run(..., extra_questions={...})` sends more questions in the same request; their answers
come back in `result.raw`. Tact uses three, at no extra round trip:

- `_remember` (noul): does the message reveal a lasting fact? Only then does Bedrock write it down.
- `_forget` (noul): is the user asking not to bring something up? A short follow-up call finds
  which memories they mean and locks them.
- `_tone` (choice: gentle / upbeat / practical): adjusts how the reply should feel.

## The write side: MemoryCurator

Saying the right thing also means not saying *old* things. When the user says "I just moved to
Bangalore", a memory store that only appends ends up holding both "lives in Pune" and "lives in
Bangalore", and the agent picks one at random. `MemoryCurator` asks, in one request, what the new
message means for each of the closest stored memories:

```python
{"type": "choice",
 "instructions": "The user just sent the message above. The assistant remembers this fact about "
                 "the user:\n{memory}\nWhat does the user's message mean for this remembered fact?",
 "criteria": {
   "outdated":   "the message shows the fact is no longer true: something changed (a move, a new job, "
                 "a cancelled or changed plan, a new preference), so the old fact should be replaced",
   "repeat":     "the message says the same thing as the fact again, so there is nothing new to store",
   "unaffected": "the fact is still true after this message; the message is about something else "
                 "or only adds a new detail alongside it"}}
```

Deleting a true memory and keeping a stale one are both mistakes, and the decider isn't always
sure which it's looking at, so the policy has three bands instead of one threshold:

| P(outdated) | action |
|---|---|
| ≥ 0.6 | **retire**: out of retrieval, but kept in the list (`retired_by`) so it can be shown and undone |
| 0.4 – 0.6 | **doubt**: kept, and `render_context` tells the LLM it *may be out of date* |
| < 0.4 | keep |

One more yes/no question per memory rides in the same request: *is this a past event that is over
and done?* A past event stays true when life moves on ("was laid off in March 2025" after "I got a
new job"), so a memory with P(past) ≥ 0.5 is never retired, only flagged.

```python
curator = MemoryCurator(http_ask("http://127.0.0.1:8099"))
result = curator.safe_run([("user", "I just moved to Bangalore!")], closest_memories)
for u in result.by_action("retire"):
    store.retire(u.memory_id, replaced_by=new_memory_id)
if not result.store_new:      # the message only repeated something already stored
    ...
```

If the decider is unreachable, `safe_run` changes nothing. On 27 hand-labelled messages it
retired 9 of 17 outdated memories, flagged 5 more, and retired **none of the 60 that were still
true** (without the past-event question: 12 retired, but 2 true memories lost). Details:
[`eval/RESULTS.md`](../../eval/RESULTS.md). Its blind spot is *finished* plans: "I finally bought
the Royal Enfield" doesn't retire "saving up to buy a Royal Enfield".

Lesson from tuning it: when one case went wrong, adding a clause to an option description made
*every* probability drift. A separate narrow yes/no question fixed the case without moving the rest.

## Tuning

- Change the option descriptions first; they carry most of the signal.
- Change `COST` to make the gate more or less cautious.
- Measure on your own traffic: [`eval/run_eval.py`](../../eval/run_eval.py) in this repo compares
  both designs against no gate, retrieval only, static sensitivity tags and an LLM gate.
- Offline tests with a stub decider: [`tests/test_memory_gate.py`](../../tests/test_memory_gate.py).
