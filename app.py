"""Tact demo UI.   uv run streamlit run app.py"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tact import bedrock  # noqa: E402
from tact.agent import SYSTEM_PROMPT, TactAgent, decider_url  # noqa: E402

st.set_page_config(page_title="Tact", page_icon="🤫", layout="wide")

BADGE = {"mention": ("🟢", "may mention"), "quiet": ("🟡", "use quietly"), "skip": ("⚪", "skip")}
BAR = {"mention": "#2e9e5b", "quiet": "#e0a526", "skip": "#9aa0a6"}
TONE = {"gentle": "🫶 gentle", "upbeat": "🎉 upbeat", "practical": "🧭 practical"}

# One click each, in the order the demo video uses them.
DEMOS = [
    ("☀️ Saturday plans", "What should I do this Saturday morning?"),
    ("🍬 Wedding sweet", "Suggest a sweet I could bring to my sister's wedding."),
    ("🐶 A puppy?", "I'm thinking of getting a puppy. Thoughts?"),
    ("💔 Missing Biscuit", "It's been a month since Biscuit died and I still can't walk past Pashan lake. How do I cope?"),
    ("🔒 Drop a subject", "Please don't bring up Meera again."),
]
HOST = "Amazon Bedrock" if bedrock.PROVIDER == "bedrock" else "Ollama on EC2"
EMBEDDER = "Titan embeddings" if bedrock.PROVIDER == "bedrock" else "Embeddings"
NAIVE = {"Top-8 retrieval": "the 8 most similar memories, like most memory features",
         "All memories": "every memory in the prompt"}


@st.cache_data(ttl=30)
def decider_health(url: str) -> dict | None:
    try:
        return json.loads(urllib.request.urlopen(f"{url}/health", timeout=5).read())
    except Exception:
        return None


def prob_bar(p: dict[str, float]) -> str:
    """A stacked bar of P(mention) / P(quiet) / P(skip)."""
    segs = "".join(
        f'<div title="{k} {p[k]:.0%}" style="width:{p[k] * 100:.1f}%;background:{BAR[k]}"></div>'
        for k in ("mention", "quiet", "skip") if p.get(k, 0) > 0)
    return (f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;'
            f'margin:2px 0 10px 0">{segs}</div>')


if "tact" not in st.session_state:
    st.session_state.tact = TactAgent(design=st.session_state.get("design"))
    st.session_state.turns = []          # list of dicts: user, reply, naive, gate info
    st.session_state.naive_history = []  # the naive agent's own conversation

tact: TactAgent = st.session_state.tact

with st.sidebar:
    st.title("🤫 Tact")
    st.caption("An assistant that remembers everything, and knows when to keep quiet.")
    url = decider_url()
    h = decider_health(url)
    if h:
        st.success(f"Strands Decider 2B online · {h.get('device', '?')}")
    else:
        st.error(f"Decider offline at {url}")
    st.caption(f"Answers: `{bedrock.CHAT_MODEL}` on {HOST}")
    compare = st.toggle("Compare with a typical memory agent", value=True,
                        help="Also ask the same model without the decider, side by side.")
    naive_mode = st.radio("Typical agent gets", list(NAIVE), disabled=not compare,
                          captions=list(NAIVE.values()))
    design = st.radio("Decider question design", ["choice", "split"], horizontal=True,
                      index=["choice", "split"].index(tact.gate.design),
                      help="choice: one 3-way question per memory. "
                           "split: two yes/no questions per memory (helps? unwelcome?).")
    if design != tact.gate.design:
        st.session_state.design = design
        del st.session_state["tact"]
        st.rerun()
    with st.expander(f"What Tact remembers about Arjun ({len(tact.store.memories)})"):
        for m in tact.store.memories:
            st.markdown(f"`{m['id']}` {'🔒 ' if m.get('locked') else ''}{m['text']}")
    if st.button("New conversation"):
        del st.session_state["tact"]
        st.rerun()

st.markdown("### Talk to Tact")
chat_col, panel_col = st.columns([3, 2], gap="large")


def render_panel(turn: dict) -> None:
    gate = turn["gate"]
    if gate is None:
        return
    st.markdown("#### What Strands Decider 2B decided")
    if gate.error:
        st.error(f"Decider unreachable ({gate.error}). Fail-safe: no memories used this turn.")
        return
    tone = f" · tone {TONE[turn['tone'][0]]}" if turn.get("tone") else ""
    st.caption(f"{len(gate.verdicts)} memories judged in **one** call · {gate.latency_ms:.0f} ms{tone}")
    for m in turn.get("locked") or []:
        st.info(f"🔒 Locked at your request, never mentioned again: {m['text']}")
    order = {"mention": 0, "quiet": 1, "skip": 2}
    for v in sorted(gate.verdicts, key=lambda v: (order[v.action], -v.probabilities.get(v.action, 0))):
        icon, label = BADGE[v.action]
        lock = " 🔒" if v.locked else ""
        st.markdown(f"{icon} **{label}**{lock} · <small>{v.text}</small>", unsafe_allow_html=True)
        if v.probabilities:
            st.markdown(prob_bar(v.probabilities), unsafe_allow_html=True)
        if v.label != v.action:
            st.caption(f"↳ {v.reason}")
    st.caption("Bar: 🟩 P(mention) 🟨 P(quiet) ⬜ P(skip)")
    if turn.get("remembered"):
        st.info(f"📝 Remembered (P={turn['p_remember']:.2f}): {turn['remembered']['text']}")


with chat_col:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["user"])
        if turn.get("naive"):
            a, b = st.columns(2)
            with a:
                st.markdown(f"**Typical memory agent** ({turn.get('naive_mode', '').lower()})")
                st.warning(turn["naive"]["text"])
                st.caption(f"{turn['naive']['input_tokens']} input tokens")
            with b:
                st.markdown("**Tact** (Decider-gated memory)")
                st.success(turn["reply"])
                st.caption(f"{turn['input_tokens']} input tokens")
        else:
            with st.chat_message("assistant", avatar="🤫"):
                st.write(turn["reply"])

    clicked = None
    for col, (label, text) in zip(st.columns(len(DEMOS)), DEMOS):
        if col.button(label, width="stretch", help=text):
            clicked = text
    prompt = st.chat_input("Ask Tact anything, e.g. “I'm planning a trip to Paris, what should I see?”") or clicked

if prompt:
    with chat_col:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Decider is judging memories, then the model replies..."):
            out = tact.chat(prompt)
            naive = None
            if compare:
                remembered = (out["candidates"] if naive_mode == "Top-8 retrieval"
                              else tact.store.memories)
                ctx = "Things you remember about the user:\n" + "\n".join(
                    f"- {m['text']}" for m in remembered)
                history = st.session_state.naive_history + [("user", prompt)]
                naive = bedrock.chat(f"{SYSTEM_PROMPT}\n\n{ctx}", history)
                st.session_state.naive_history = history + [("assistant", naive["text"])]
    st.session_state.turns.append({"user": prompt, **out, "naive": naive, "naive_mode": naive_mode})
    st.rerun()

with panel_col:
    if st.session_state.turns:
        render_panel(st.session_state.turns[-1])
    else:
        st.markdown("#### How it works")
        st.markdown(
            f"1. **{EMBEDDER}** shortlist the 8 most similar memories.\n"
            "2. **Strands Decider 2B** reads the conversation once and gives each memory "
            "P(*mention*), P(*quiet*), P(*skip*). One batched call, which also reads the tone "
            "and spots requests to remember or forget.\n"
            "3. An **expected-cost policy** in plain Python picks the action: saying something "
            "painful costs 3× more than missing a detail, so doubt means *quiet*.\n"
            f"4. **{bedrock.CHAT_MODEL}** ({HOST}) writes the reply, seeing only what the gate allowed."
        )
