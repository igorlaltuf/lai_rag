from __future__ import annotations

from functools import lru_cache
import time

from openai import APIError, OpenAI, RateLimitError

from src.config import load_settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI | None:
    settings = load_settings()
    if not settings.openai_api_key:
        return None
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], model: str | None = None, max_retries: int = 8) -> list[list[float]]:
    settings = load_settings()
    client = get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY nao configurada.")
    for attempt in range(max_retries + 1):
        try:
            response = client.embeddings.create(model=model or settings.embedding_model, input=texts)
            return [item.embedding for item in response.data]
        except (RateLimitError, APIError):
            if attempt >= max_retries:
                raise
            time.sleep(min(120, 10 * (2**attempt)))
    raise RuntimeError("Falha inesperada ao gerar embeddings.")
