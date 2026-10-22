"""Thin LLM helpers: chat replies and text embeddings.

Amazon Bedrock by default (Converse API + Titan embeddings). With TACT_PROVIDER=ollama, the
same two calls go to an Ollama server instead, for example the one `infra/decider_ec2.py`
runs on the decider's EC2 instance, for accounts where Bedrock is not available.

Switching is one line: `echo bedrock > .cache/provider` (or `echo ollama > ...`), or set
TACT_PROVIDER. Check first that the account can call Bedrock:

    uv run python -m tact.bedrock
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
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

BEDROCK_CHAT = "us.amazon.nova-lite-v1:0"
BEDROCK_EMBED = "amazon.titan-embed-text-v2:0"
if PROVIDER == "ollama":
    CHAT_MODEL = os.environ.get("TACT_CHAT_MODEL", "tact-chat")       # an alias set up on the server
    EMBED_MODEL = os.environ.get("TACT_EMBED_MODEL", "nomic-embed-text")
else:
    CHAT_MODEL = os.environ.get("TACT_CHAT_MODEL", BEDROCK_CHAT)
    EMBED_MODEL = os.environ.get("TACT_EMBED_MODEL", BEDROCK_EMBED)

# USD per 1M tokens (input, output), on-demand in us-east-1, used only for the cost estimates
# in the eval reports. Check https://aws.amazon.com/bedrock/pricing/ and override the chat
# model's with TACT_PRICE_IN / TACT_PRICE_OUT if they change. A model missing here has no
# estimate (NaN), and a self-hosted model has no per-token price: its cost is the
# instance's hourly rate.
PRICES = {
    "us.amazon.nova-micro-v1:0": (0.035, 0.14),
    "us.amazon.nova-lite-v1:0": (0.06, 0.24),
    "us.amazon.nova-pro-v1:0": (0.80, 3.20),
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.00, 5.00),
}
_chat_price = PRICES.get(CHAT_MODEL, (float("nan"),) * 2) if PROVIDER == "bedrock" else (0.0, 0.0)
PRICE_IN = float(os.environ.get("TACT_PRICE_IN", _chat_price[0]))
PRICE_OUT = float(os.environ.get("TACT_PRICE_OUT", _chat_price[1]))


@lru_cache(maxsize=1)
def client():
    import boto3
    from botocore.config import Config
    # New accounts often start with tiny Bedrock quotas: back off and retry on throttling.
    return boto3.client("bedrock-runtime", region_name=REGION,
                        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))


def _post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(system: str, messages: list[tuple[str, str]], max_tokens: int = 400,
         temperature: float = 0.3, model: str | None = None) -> dict:
    """One chat call (with `model`, or the chat model by default). Returns text, token usage,
    latency and estimated cost."""
    model = model or CHAT_MODEL
    start = time.perf_counter()
    if PROVIDER == "ollama":
        resp = _post("/v1/chat/completions", {
            "model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "system", "content": system}]
                        + [{"role": r, "content": t} for r, t in messages],
        })
        text = resp["choices"][0]["message"]["content"]
        tokens_in, tokens_out = resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"]
    else:
        resp = client().converse(
            modelId=model,
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
        "cost_usd": cost(tokens_in, tokens_out, model),
    }


def cost(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    if model is None or model == CHAT_MODEL:
        price_in, price_out = PRICE_IN, PRICE_OUT
    elif PROVIDER == "bedrock":
        price_in, price_out = PRICES.get(model, (float("nan"),) * 2)
    else:
        price_in = price_out = 0.0
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


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


# What each error usually means on a new account, and what fixes it.
HINTS = {
    "ValidationException": "the whole account is blocked from Bedrock (common on brand-new accounts). "
                           "Billing > Payment preferences: add a verified card, or reply on your support case.",
    "ThrottlingException": "the account's Bedrock quota is 0 (common on new accounts). Service Quotas > "
                           "Amazon Bedrock: request more on-demand tokens per minute for this model.",
    "AccessDeniedException": "IAM: the AWS user needs bedrock:InvokeModel for this model.",
    "ResourceNotFoundException": f"this model ID isn't offered in {REGION}.",
}


def check() -> int:
    """Can this account call the Bedrock models Tact needs? Says what to fix if not."""
    from botocore.exceptions import ClientError

    print(f"Bedrock in {REGION} (Tact's provider is currently: {PROVIDER})")
    ok = True
    tests = [(BEDROCK_CHAT, "replies"), ("us.amazon.nova-pro-v1:0", "eval baseline"),
             (BEDROCK_EMBED, "memory shortlist")]
    for model, use in tests:
        try:
            if model == BEDROCK_EMBED:
                client().invoke_model(modelId=model, body=json.dumps({"inputText": "hello"}))
            else:
                client().converse(modelId=model, messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                                  inferenceConfig={"maxTokens": 5})
            print(f"  ok    {model}  ({use})")
        except ClientError as e:
            code, message = e.response["Error"]["Code"], e.response["Error"]["Message"]
            print(f"  FAIL  {model}  ({use}): {code}: {message}\n        -> {HINTS.get(code, 'see the message above')}")
            ok = False
    if ok:
        print("\nBedrock works. Switch Tact to it with:  echo bedrock > .cache/provider")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check())
