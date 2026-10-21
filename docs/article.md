# Tact: teaching a memory agent when to keep quiet, with Strands Decider 2B

*Tags: #strands-decider-2b #context-memory*

## The problem: assistants that remember too well

Ask an AI assistant with long-term memory for sightseeing tips in Paris. It looks through what it
knows about you, finds the memory that matches "Paris" most closely, and helpfully writes:

> "Since your last trip to Paris ended with the breakup with Meera, maybe try a different
> neighbourhood this time..."

The retrieval worked as designed. It found the most *similar* memory, but that memory was not an
*appropriate* one. People with good memories don't say everything they remember. A friend knows
not to suggest a dog park the month your dog died, and doesn't mention why.

Memory-enabled agents usually get one of two jobs wrong:

1. **Too much context.** Everything remembered goes into the prompt on every turn. That costs
   tokens, and the model starts mentioning irrelevant things.
2. **No tact.** Similarity search can't tell "relevant and fine to say" from "relevant and
   painful to say".

## "Just tag the sensitive memories" doesn't work

The obvious fix is to mark painful memories as sensitive when they're saved, and never say them.
But tact depends on the conversation:

- *"I'm thinking of getting a puppy. Thoughts?"* The dog that died last month should shape the
  answer (maybe suggest waiting a little), but saying it out loud would hurt.
- *"It's been a month since Biscuit died and I still can't walk past the lake. How do I cope?"*
  Now **not** acknowledging Biscuit would be cold. The user opened the door.

Same memory, two conversations, two correct answers. A static tag gets one of them wrong every
time. The decision has to be made per message, and it has to be cheap enough to make on every turn.

**Tact** is a Strands agent that makes that decision. Before every reply, **Strands Decider 2B**
judges each candidate memory: **mention** it, use it **quietly**, or **skip** it.

## How it works

```
user message
  │
  ├─ 1. Embeddings shortlist the 8 most similar memories
  │
  ├─ 2. Strands Decider 2B, ONE batched request:
  │       8 × choice "mention / quiet / skip?"
  │       + noul "worth remembering?" + noul "asking me to drop a subject?" + choice "tone?"
  │
  ├─ 3. Expected-cost policy in plain Python: picks mention / quiet / skip per memory
  │
  └─ 4. The chat model replies, seeing only what the gate allowed
```

Retrieval still finds the candidates, because it's fast and good at "what could be related". The
decider then makes the judgment retrieval can't: given this conversation, should this fact be used,
and can it be said out loud?

Everything runs on AWS. Decider 2B is served on an EC2 instance. Tact was built for Amazon Bedrock
(Nova Lite for replies, Titan Text Embeddings V2 for the shortlist), but my brand-new AWS account
couldn't call Bedrock models during the hackathon. So the replies and embeddings in this article
come from Qwen2.5 3B and nomic-embed-text, served by Ollama on the same EC2 instance as the
decider. One setting (`TACT_PROVIDER=bedrock|ollama`) switches between the two; the gate doesn't
change.

## The decision

Each memory becomes one typed `choice` question. All of them go to the decider in a single request,
so the conversation is read once and each memory only adds its own few tokens.

```python
{
  "type": "choice",
  "instructions": "An AI assistant is about to reply to the user's latest message. "
                  "It remembers this fact about the user:\n{memory}\n"
                  "How should the assistant treat this fact in its reply?",
  "criteria": {
    "mention": "use it openly: it makes the reply more useful or personal (a preference, "
               "constraint, plan or interest that fits the request), and bringing it up feels natural",
    "quiet":   "use it silently: it is related to the request, but saying it out loud could hurt, "
               "embarrass or upset the user, so it should only make the reply more careful",
    "skip":    "ignore it: it has no bearing on what a good reply would say"
  }
}
```

I tried three wordings on a 10-scenario development split and kept the best one. The
evaluation below only scores the other 24 scenarios separately, so the wording wasn't tuned on them.

The middle option is the point of the project. A *quiet* memory still reaches the LLM, under a
heading that says "sensitive background, NEVER mention". The Paris reply can then avoid
romantic-couple suggestions without saying why.

I also tried a second design: two narrow yes/no questions per memory, *"would this fact change a
thoughtful reply?"* and *"would saying it out loud be unwelcome, given the conversation so far?"*.
They combine as P(mention) = P(helps) × (1 − P(unwelcome)), P(quiet) = P(helps) × P(unwelcome) and
P(skip) = 1 − P(helps). It looked cleaner on paper, but it lost: 74% accuracy against 83% for the
single 3-way question, one sensitive fact let through, and twice the decider time, because every
memory needs two questions. The repo keeps both designs, and the demo app can switch between them.

## The policy: expected cost, not argmax

The decider returns probabilities, not just a label, and Tact uses all of them. Mistakes don't
cost the same. Saying something painful is far worse than leaving out a useful detail. So the
gate picks the action with the lowest **expected cost**:

| action ↓ / truth → | mention | quiet | skip |
|---|---|---|---|
| **mention** | 0 | **3** (said something painful) | 0.5 |
| **quiet** | 1 (held back a useful fact) | 0 | 0.4 |
| **skip** | 1.5 (lost a useful fact) | 1 | 0 |

```python
def decide(self, probabilities, locked=False):
    expected = {a: sum(probabilities[t] * c for t, c in row.items())
                for a, row in self.cost.items()}
    if locked:                       # the user asked never to hear about it
        expected.pop("mention")
    return min(expected, key=expected.get)
```

A fact that's 60% "mention" but 25% "quiet" ends up **quiet**: the model leaned towards
saying it, but not confidently enough to risk it. This is where a decision model that returns
probabilities pays off over an LLM that just outputs a label. Without probabilities there's
nothing to weigh. And the trade-off lives in a table your team can argue about and change,
not in a prompt.

The 3 wasn't a guess. I started at 5, and on the development split Tact stopped acknowledging
subjects the user had raised (0 of 3) and mentioned only 5 of 18 helpful facts. `eval/tune.py`
replays the saved decider answers with costs from 1 to 5. At 1, 2 of the 20 sensitive facts got
through; 3 was the lowest value that let none through, and it kept 2 of the 3 user-raised subjects.

## Wiring it into a Strands agent

The gate is a Strands `InterventionHandler`. Its `before_invocation` runs once per user message,
before any model call, and returns a `Transform` that rewrites the system prompt:

```python
class MemoryGateIntervention(InterventionHandler):
    def before_invocation(self, event, **kwargs):
        latest = latest_user_text(event)
        result = self.gate.safe_run(conversation, self.store.candidates(latest, k=8))
        prompt = f"{SYSTEM_PROMPT}\n\n{render_context(result)}"
        return Transform(apply=lambda e: setattr(e.agent, "system_prompt", prompt))

agent = Agent(model=model, interventions=[MemoryGateIntervention(gate, store)])
```

If the decider is unreachable, `safe_run` returns no memories. The agent still answers, it just
doesn't remember anything this turn. It never falls back to a stale or ungated memory block.

### Three more decisions, in the same request

Once the conversation is encoded, extra questions are nearly free, so Tact asks three more in the
same call:

- **Remember?** (`noul`) Is the user sharing news about themselves, rather than asking for
  something? Only when the decider is at least 70% sure does Tact spend an LLM call writing it
  down.
- **Forget?** (`noul`) *"Please don't bring up Meera again."* A short follow-up call asks, per
  memory, "is this the one they mean?", and locks the match. A locked memory can still keep
  answers considerate, but the policy will never let it be said again.
- **Tone?** (`choice`: gentle / upbeat / practical) Adjusts how the reply should feel.

### The other half of memory: what to forget on its own

Tact knows "Arjun lives in Pune". Then Arjun says *"Big news: I just moved to Bangalore for a new
job!"* An agent that only appends now holds both facts, and next Saturday it may well suggest a
walk in Pune. Saying the right thing also means not saying *old* things.

So writing gets its own decision. When the remember question fires, a second request asks, for
each of the 5 closest stored memories: does this message make it **outdated**, **repeat** it, or
leave it **unaffected**? Deleting a true memory and keeping a stale one are both mistakes, so the
policy has three bands. At P(outdated) ≥ 0.6 the memory is **retired**: it leaves retrieval but
stays in history. Between 0.4 and 0.6 it's kept but **flagged**, and the LLM is told it *may be
out of date*. Below that it's kept.

The first live test taught me something. "I got a new job" pushed P(outdated) for *"was laid off in
March 2025"* to 0.72, but a past event stays true when life moves on. Writing that into the option
descriptions made every probability drift upwards and caused four wrong retirements. What worked
was a separate yes/no question in the same request: *is this a past event that's over and done?*
If so, the memory can be flagged but never retired.

On 27 hand-labelled messages, the curator retired 9 of 17 outdated memories, flagged 5 more, and
retired **none of the 60 that were still true**. Its blind spot is finished plans: *"I finally
bought the Royal Enfield"* doesn't retire *"saving up to buy a Royal Enfield"*.

## Does it work?

I wrote 34 everyday requests over a fictional user with 30 memories. Some memories are useful,
like allergies, budget and his sister's wedding. Some are painful, like a breakup, a dog that
died, a layoff and a father's surgery. 10 scenarios are built specifically to break static tags:
the user raises the painful subject themselves, or it only becomes sensitive two turns into a
conversation. Each request has hand labels for which memories should be mentioned and which
should stay quiet.

First, the gate decisions themselves, scored against the hand labels. Every gate judges the same
8 shortlisted memories per request:

- **tags**: the static approach. Sensitive memories are tagged when stored and never mentioned.
- **llm_gate**: the same 3-way question, asked of the chat LLM (Qwen2.5 3B) in one prompt.
- **tact**: Decider 2B with the 3-way question and the cost table.
- **tact_split**: Decider 2B with the two yes/no questions.

| gate | 3-way accuracy | sensitive facts let through ↓ | user-raised sensitive facts mentioned ↑ | helpful facts kept ↑ |
|---|---|---|---|---|
| tags | 23% | 0 / 60 | 0 / 7 | 52 / 52 |
| llm_gate | 60% | 5 / 60 | 2 / 7 | 26 / 52 |
| **tact** | **83%** | **0 / 60** | **4 / 7** | 32 / 52 |
| tact_split | 74% | 1 / 60 | 0 / 7 | 29 / 52 |

On the 24 scenarios that weren't used for tuning, Tact scores 85%, the LLM gate 61% and tags 19%.

Tags look safe, and they are, but they get three out of four decisions wrong: every helpful fact
is "mentioned", including the ones that have nothing to do with the question, and a sensitive
fact can never be acknowledged, even when the user brings it up. The LLM gate did let painful
facts through. Asked to suggest activities for Arjun's dad, who is recovering from surgery, it
allowed mentioning Arjun's layoff. Asked about cooking for his dad, it allowed the dog that died.

Then the replies. Five agents answered all 34 requests with the same chat model:

| how the agent gets its memory | replies that said something sensitive ↓ | helpful facts used ↑ | avg input tokens per reply ↓ |
|---|---|---|---|
| all memories in the prompt | 3 / 34 | 30 / 70 | 540 |
| top-8 retrieval (RAG) | 4 / 34 | 32 / 70 | 208 |
| retrieval + static tags | 1 / 34 | 33 / 70 | 243 |
| retrieval + LLM gate | 0 / 34 | 26 / 70 | 143 |
| **Tact** | **0 / 34** | 32 / 70 | **114** |

The leaks are exactly the ones the introduction warned about. Asked *"What should I do this
Saturday morning?"*, the retrieval agent suggested taking Biscuit for a relaxing walk at the lake.
Biscuit is the dog that died. Asked for a sweet to bring to his sister's wedding, the
all-memories agent suggested his late grandmother's rasgulla, "a reminder of her".

**The takeaway:** Tact never said anything painful, used as many helpful facts as plain
retrieval, and sent the chat model 79% fewer input tokens than the all-memories agent, because
it passes on only what the decider allowed. The LLM gate was also leak-free in the replies, but
it threw away more helpful facts and made more wrong calls before the reply was even written.

### Can you trust its confidence?

A probability is only useful if it means something. Across the 260 labelled memory decisions,
when the decider gave its top label 90% or more, it was right **97 times out of 97**. Between 80%
and 90% it was right 87% of the time. Below 80%, though, it was close to a coin flip (50–59%),
whatever it claimed. Its expected calibration error is 0.06.

![Strands Decider 2B: stated confidence vs. accuracy](img/calibration.png)

That's a usable signal. 73% of decisions came with high confidence and were 93% right; the other
27% are where mistakes live. So I tried the obvious next step: send just the uncertain decisions to
the LLM gate. It didn't help (215 right on its own, 217 at best with the LLM), because the cases
the decider finds hard are hard for the 3B LLM too. So for now the signal tells you which
decisions deserve a second look, not who should make them. A bigger Bedrock model might change that.

### What I learned about using Decider 2B

- **Batch everything that reads the same conversation.** Eight memory questions plus remember,
  forget and tone go in one request. The conversation is encoded once, so the extra questions
  cost a few tokens each, not another call.
- **Ask about what's observable, not about value.** "Does this message reveal a fact worth
  remembering?" never reached 0.7 on ten pieces of real news. "Is the user mainly sharing news
  about themselves, rather than asking for something?" caught all ten, with no false alarms on
  eight plain requests.
- **Fix one bad case with a new narrow question, not a longer option.** Adding a clause to an
  option description moved every probability, not just the broken one. The separate past-event
  question fixed the case and left the rest alone.
- **Keep the probabilities and tune on saved answers.** Decider answers are cheap to store, so the
  cost sweep and the calibration chart replay them without calling the model again. With a lenient
  cost table, sensitive facts slipped through on the development split; at 3, none did.
- **It's good at relevance and at noticing when the user opens a subject.** When Arjun says he
  can't walk past the lake since Biscuit died, P(mention) for his lake walks goes from about 0.2
  to about 0.6.
- **It rarely picks "quiet" on its own.** Sensitive facts that relate to the request mostly come
  back as "skip". That's safe, but it means the quiet path relies on the cost table more than on
  the label.

### Limits

- The decider ran on CPU (a c7i.2xlarge), because my account had no GPU quota yet. That made
  each gate call take about 10 seconds, slower than the 3B LLM gate. The published latency on a
  GPU is around 115 ms per question, so this is a deployment issue rather than a model one, but I
  couldn't measure it myself.
- The LLM gate uses the same small 3B model that writes the replies. A larger LLM would likely
  close some of the gap, at a higher cost per turn.
- Tact still missed 3 of the 7 cases where the user raised a sensitive subject, and it only
  mentioned 32 of the 52 helpful facts.
- "Said something sensitive" is a keyword check on the reply, which can miss paraphrases.
- The past-event answers sit close to 0.5 for most memories, so that guard is a blunt tool: it
  protects past events, but it also turns some correct retirements into flags.

The dataset is small and hand-labelled by one person, so read these numbers as a signal, not a
benchmark. The evaluation script is in the repo so you can rerun it on your own memories.

## Reuse it

The pattern is one standard-library Python file, `patterns/memory_gate/memory_gate.py`, plus a
README and offline tests. It contains the questions, both designs, the cost-table policy, the
write-side `MemoryCurator`, the prompt renderer and the Strands wiring. Drop it into any agent with
long-term memory:

```python
from memory_gate import MemoryGate, http_ask, render_context

gate = MemoryGate(http_ask("http://127.0.0.1:8099"))
result = gate.safe_run(conversation=[("user", "Any tips for my trip to Paris?")], memories=memories)
system_prompt = BASE_PROMPT + "\n\n" + render_context(result)
```

- **Code (Apache-2.0):** https://github.com/Insomniac-Coder33/tact
- **Demo video:** `[[VIDEO_URL]]`

A good memory isn't one that remembers everything. It's one that knows what to leave unsaid.
