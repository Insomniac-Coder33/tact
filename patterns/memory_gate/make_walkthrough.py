"""Build and execute walkthrough.ipynb from the recorded decider answers (no server needed).

    uv run python patterns/memory_gate/make_walkthrough.py
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

cells = [
    md("# MemoryGate walkthrough\n\n"
       "This notebook walks through the reusable Strands Decider 2B pattern in this folder, one "
       "decision at a time. The decider's answers are **replayed** from `quickstart_answers.json`, "
       "which was recorded from a real Strands Decider 2B server. Everything else (the policy, the "
       "curator's bands, the prompt block) runs live, so you can edit a cell and rerun it with no "
       "server, GPU or AWS account.\n\n"
       "To ask a live decider instead, replace `replay_ask(...)` with `http_ask(\"http://127.0.0.1:8099\")`."),
    code("from memory_gate import (COST, INSTRUCTIONS, OPTIONS, MemoryCurator, MemoryGate,\n"
         "                         render_context, replay_ask)\n"
         "from quickstart import MEMORIES, NEAR_NEWS, NEWS, SCENES\n\n"
         "ask = replay_ask(\"quickstart_answers.json\")\n"
         "gate = MemoryGate(ask)\n"
         "for m in MEMORIES:\n"
         "    print(m[\"id\"], m[\"text\"])"),
    md("## 1. The question\n\n"
       "Each memory becomes one typed `choice` question. The option descriptions carry most of the "
       "signal, so they're written as carefully as the question. All memories go in **one** request: "
       "the decider reads the conversation once, and each memory only adds its own tokens."),
    code("print(INSTRUCTIONS.format(memory=MEMORIES[1][\"text\"]))\nprint()\n"
         "for label, description in OPTIONS.items():\n    print(f\"{label:8} {description}\")"),
    md("## 2. A harmless question near a painful memory\n\n"
       "The user asks about getting a puppy. Their dog died last month."),
    code("def show(result):\n"
         "    for v in result.verdicts:\n"
         "        p = \"  \".join(f\"{k}={x:.2f}\" for k, x in v.probabilities.items())\n"
         "        print(f\"{v.action:8} {p}   {v.text}\")\n"
         "        if v.action != v.label:\n"
         "            print(f\"         ↳ {v.reason}\")\n\n"
         "title, conversation = SCENES[1]\n"
         "print(\"user:\", conversation[-1][1], \"\\n\")\n"
         "puppy = gate.run(conversation, MEMORIES)\n"
         "show(puppy)"),
    md("The decider's top label for Biscuit is *skip*, but it isn't sure. The policy doesn't just take "
       "the top label: it picks the action with the lowest **expected cost** under the probabilities. "
       "Here is that arithmetic for the Biscuit memory:"),
    code("biscuit = next(v for v in puppy.verdicts if \"Biscuit passed\" in v.text)\n"
         "p = biscuit.probabilities\n"
         "for action, row in COST.items():\n"
         "    terms = \" + \".join(f\"{p[t]:.2f}×{c}\" for t, c in row.items())\n"
         "    print(f\"{action:8} {terms} = {sum(p[t] * c for t, c in row.items()):.2f}\")\n"
         "print(\"\\nchosen:\", biscuit.action)"),
    md("So the memory reaches the LLM as *sensitive background, never mention*: the reply can suggest "
       "waiting a little before a new puppy without bringing up Biscuit.\n\n"
       "## 3. The same memory, when the user opens the door"),
    code("title, conversation = SCENES[2]\n"
         "print(\"user:\", conversation[-1][1], \"\\n\")\n"
         "show(gate.run(conversation, MEMORIES))"),
    md("Now ignoring Biscuit would be cold, and the decider moves both Biscuit memories to *mention*. "
       "A static \"sensitive\" tag can't do this: it would keep them unsaid in both conversations.\n\n"
       "## 4. The cost table is yours to change\n\n"
       "Replay the puppy scene on the **same** decider answers, changing only how much a painful "
       "mention costs (`COST[\"mention\"][\"quiet\"]`, 3 by default):"),
    code("for c in (0.5, 1, 2, 3, 5):\n"
         "    cost = {a: dict(row) for a, row in COST.items()}\n"
         "    cost[\"mention\"][\"quiet\"] = c\n"
         "    v = next(v for v in MemoryGate(ask, cost=cost).run(SCENES[1][1], MEMORIES).verdicts\n"
         "             if \"Biscuit passed\" in v.text)\n"
         "    print(f\"painful mention costs {c:>3}:  {v.action:8} {v.text}\")"),
    md("Make a painful mention cheap and the agent brings up the dead dog when you ask about a puppy. "
       "Taking the decider's top label as-is would *skip* the memory entirely, so the reply couldn't "
       "even be careful. The expected-cost policy lands in between: use it, don't say it."),
    md("No retraining, no new prompt: the trade-off lives in a table your team can read and argue about.\n\n"
       "## 5. What the LLM actually receives"),
    code("title, conversation = SCENES[3]\n"
         "print(\"user:\", conversation[-1][1], \"\\n\")\n"
         "print(render_context(gate.run(conversation, MEMORIES)))"),
    md("## 6. The write side: news makes old memories outdated\n\n"
       "`MemoryCurator` checks the stored memories closest to the news, in one request: outdated, "
       "repeat, or unaffected, plus a yes/no *is this a past event?* that keeps finished events from "
       "being retired."),
    code("print(\"user:\", NEWS[0][1], \"\\n\")\n"
         "nearby = [m for m in MEMORIES if m[\"id\"] in NEAR_NEWS]\n"
         "curation = MemoryCurator(ask).run(NEWS, nearby)\n"
         "for u in curation.updates:\n"
         "    print(f\"{u.action:7} P(outdated)={u.probabilities['outdated']:.2f}  \"\n"
         "          f\"P(past event)={u.p_past:.2f}   {u.text}\")\n"
         "print(\"\\nstore the new fact:\", curation.store_new)"),
    md("## 7. Wiring it into a Strands agent\n\n"
       "The gate runs in a Strands `InterventionHandler.before_invocation`, once per user message and "
       "before any model call. It returns a `Transform` that rewrites the system prompt, so the LLM "
       "only ever sees what the gate allowed. The full version, with remember / forget / tone riding "
       "in the same request, is `src/tact/agent.py`.\n\n"
       "```python\n"
       "class MemoryGateIntervention(InterventionHandler):\n"
       "    def before_invocation(self, event, **kwargs):\n"
       "        result = self.gate.safe_run(conversation, self.store.candidates(latest, k=8))\n"
       "        prompt = f\"{SYSTEM_PROMPT}\\n\\n{render_context(result)}\"\n"
       "        return Transform(apply=lambda e: setattr(e.agent, \"system_prompt\", prompt))\n\n"
       "agent = Agent(model=BedrockModel(model_id=\"us.amazon.nova-lite-v1:0\"),\n"
       "              interventions=[MemoryGateIntervention(gate, store)])\n"
       "```\n\n"
       "`safe_run` means a decider outage gives the agent **no** memories for that turn, never a stale "
       "or ungated block."),
]

nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {
    "name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=60, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
nbformat.write(nb, HERE / "walkthrough.ipynb")
print("wrote", HERE / "walkthrough.ipynb")
