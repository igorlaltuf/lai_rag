from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = DATA_DIR / "vector" / "chroma"
DB_PATH = PROCESSED_DIR / "lai_2026.sqlite"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    generation_model: str
    embedding_model: str
    falabr_data_page: str
    data_year: int
    rag_top_k: int
    rag_vector_weight: float


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    env_file = dotenv_values(ROOT / ".env")

    def get_env(name: str, default: str | None = None) -> str | None:
        value = os.getenv(name) or env_file.get(name)
        return value if value not in {"", None} else default

    return Settings(
        openai_api_key=get_env("OPENAI_API_KEY"),
        generation_model=get_env("OPENAI_GENERATION_MODEL", "gpt-5.4-nano") or "gpt-5.4-nano",
        embedding_model=get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small",
        falabr_data_page=get_env("FALABR_DATA_PAGE", "https://falabr.cgu.gov.br/web/dadosabertoslai")
        or "https://falabr.cgu.gov.br/web/dadosabertoslai",
        data_year=int(get_env("DATA_YEAR", "2026") or "2026"),
        rag_top_k=int(get_env("RAG_TOP_K", "8") or "8"),
        rag_vector_weight=float(get_env("RAG_VECTOR_WEIGHT", "0.6") or "0.6"),
    )


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, VECTOR_DIR]:
        path.mkdir(parents=True, exist_ok=True)
