"""Prometheus counters. Grafana turns the token rate into USD per hour.

Price: 0.05 USD per 1,000 tokens. That is the documented stand-in for the
Groq Qwen chat price, not a live invoice. The dashboard multiplies
sum(rate(egylaw_tokens_total[1h])) * 3600 / 1000 * 0.05.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, generate_latest

TOKENS = Counter(
    "egylaw_tokens_total",
    "Prompt and completion tokens. A token is one whitespace-separated word.",
    ["direction"],
)
DRIFT = Gauge(
    "egylaw_query_drift",
    "Cosine similarity of the latest query embedding to the index baseline.",
)
# Documented price per 1,000 tokens, in USD. Keep this equal to Settings.token_price_per_1k.
TOKEN_PRICE_PER_1K_USD = 0.05


def count_tokens(text: str) -> int:
    words = [word for word in text.split() if word]
    return len(words)


def record_tokens(question: str, answer: str) -> int:
    prompt = count_tokens(question)
    completion = count_tokens(answer)
    TOKENS.labels(direction="prompt").inc(prompt)
    TOKENS.labels(direction="completion").inc(completion)
    return prompt + completion


def metrics_payload() -> bytes:
    return generate_latest()
