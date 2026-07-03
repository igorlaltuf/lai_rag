from __future__ import annotations

from dataclasses import dataclass

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None


MODEL_PRICES_USD_PER_1M = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
}


@dataclass(frozen=True)
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    usd: float


def count_tokens(text: str, model: str = "gpt-5.4-nano") -> int:
    if not text:
        return 0
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> CostEstimate:
    price = MODEL_PRICES_USD_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    usd = (input_tokens / 1_000_000) * price["input"] + (output_tokens / 1_000_000) * price["output"]
    return CostEstimate(model=model, input_tokens=input_tokens, output_tokens=output_tokens, usd=usd)


def format_usd(value: float) -> str:
    return f"US$ {value:.4f}"
