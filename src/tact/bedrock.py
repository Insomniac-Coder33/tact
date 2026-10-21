"""Thin LLM helpers: chat replies and text embeddings.

Amazon Bedrock by default (Converse API + Titan embeddings). With TACT_PROVIDER=ollama, the
same two calls go to an Ollama server instead, for example the one `infra/decider_ec2.py`
runs on the decider's EC2 instance, for accounts where Bedrock is not available.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("TACT_CACHE", ROOT / ".cache"))


def _saved(name: str, default: str) -> str:
    path = CACHE / name
    return path.read_text().strip() if path.exists() else default


PROVIDER = os.environ.get("TACT_PROVIDER") or _saved("provider", "bedrock")
REGION = os.environ.get("AWS_REGION", "us-east-1")
OLLAMA_URL = os.environ.get("TACT_OLLAMA_URL") or _saved("ollama_url", "http://127.0.0.1:11434")

if PROVIDER == "ollama":
    CHAT_MODEL = os.environ.get("TACT_CHAT_MODEL", "tact-chat")       # an alias set up on the server
    EMBED_MODEL = os.environ.get("TACT_EMBED_MODEL", "nomic-embed-text")
else:
    CHAT_MODEL = os.environ.get("TACT_CHAT_MODEL", "us.amazon.nova-lite-v1:0")
    EMBED_MODEL = os.environ.get("TACT_EMBED_MODEL", "amazon.titan-embed-text-v2:0")

# USD per 1M tokens, used only for the cost estimates in the eval report.
# Check https://aws.amazon.com/bedrock/pricing/ and override if they change.
# A self-hosted model has no per-token price; its cost is the instance's hourly rate.
PRICE_IN = float(os.environ.get("TACT_PRICE_IN", "0.06" if PROVIDER == "bedrock" else "0"))
PRICE_OUT = float(os.environ.get("TACT_PRICE_OUT", "0.24" if PROVIDER == "bedrock" else "0"))


@lru_cache(maxsize=1)
def client():
    import boto3
    return boto3.client("bedrock-runtime", region_name=REGION)


def _post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(system: str, messages: list[tuple[str, str]], max_tokens: int = 400,
         temperature: float = 0.3) -> dict:
    """One chat call. Returns text, token usage, latency and estimated cost."""
    start = time.perf_counter()
    if PROVIDER == "ollama":
        resp = _post("/v1/chat/completions", {
            "model": CHAT_MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system}]
                        + [{"role": r, "content": t} for r, t in messages],
        })
        text = resp["choices"][0]["message"]["content"]
        tokens_in, tokens_out = resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"]
    else:
        resp = client().converse(
            modelId=CHAT_MODEL,
            system=[{"text": system}],
            messages=[{"role": r, "content": [{"text": t}]} for r, t in messages],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        tokens_in, tokens_out = resp["usage"]["inputTokens"], resp["usage"]["outputTokens"]
    return {
        "text": text,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "cost_usd": cost(tokens_in, tokens_out),
    }


def cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens / 1e6 * PRICE_IN + output_tokens / 1e6 * PRICE_OUT


def embed(text: str) -> np.ndarray:
    """A normalized text embedding, cached on disk so repeated runs cost nothing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{EMBED_MODEL}|{text}".encode()).hexdigest()[:24]
    path = CACHE / f"emb_{key}.json"
    if path.exists():
        return np.array(json.loads(path.read_text()))
    if PROVIDER == "ollama":
        vec = np.array(_post("/v1/embeddings", {"model": EMBED_MODEL, "input": text})["data"][0]["embedding"])
        vec = (vec / np.linalg.norm(vec)).tolist()
    else:
        resp = client().invoke_model(
            modelId=EMBED_MODEL,
            body=json.dumps({"inputText": text, "dimensions": 512, "normalize": True}),
        )
        vec = json.loads(resp["body"].read())["embedding"]
    path.write_text(json.dumps(vec))
    return np.array(vec)


def strands_model(temperature: float = 0.3, max_tokens: int = 400):
    """The same backend as a Strands model provider, for the agent."""
    if PROVIDER == "ollama":
        from strands.models.ollama import OllamaModel
        return OllamaModel(OLLAMA_URL, model_id=CHAT_MODEL, temperature=temperature, max_tokens=max_tokens)
    from strands.models import BedrockModel
    return BedrockModel(model_id=CHAT_MODEL, region_name=REGION, temperature=temperature,
                        max_tokens=max_tokens)
