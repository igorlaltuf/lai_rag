from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from itertools import islice
from typing import Iterable

import chromadb
import pandas as pd

from src.config import DB_PATH, VECTOR_DIR, ensure_dirs, load_settings
from src.costs import count_tokens, estimate_cost, format_usd
from src.openai_client import embed_texts


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str]


CONTENT_COLUMNS = ["pedido", "resposta", "recurso", "decisao_recurso"]


def _trim_text(value: object, max_chars: int | None = None) -> str:
    text = str(value or "").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def build_content_text(row, field_limits: dict[str, int] | None = None) -> str:
    parts = [
        ("pedido", "Pedido", row.get("pedido", "")),
        ("resposta", "Resposta", row.get("resposta", "")),
        ("recurso", "Recurso", row.get("recurso", "")),
        ("decisao_recurso", "Decisao do recurso", row.get("decisao_recurso", "")),
    ]
    blocks = []
    for field, label, value in parts:
        text = _trim_text(value, field_limits.get(field) if field_limits else None)
        if text:
            blocks.append(f"{label}: {text}")
    return "\n".join(blocks).strip()


def build_context_text(row) -> str:
    return build_content_text(
        row,
        field_limits={
            "pedido": 1200,
            "resposta": 1200,
            "recurso": 900,
            "decisao_recurso": 900,
        },
    )


def load_documents() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError("Banco processado nao encontrado. Rode: uv run python -m src.prepare")
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT rowid AS doc_id, * FROM documents", conn)


def filter_indexable_documents(df: pd.DataFrame) -> pd.DataFrame:
    if not all(column in df.columns for column in CONTENT_COLUMNS):
        return df.iloc[0:0].copy()
    content = df[CONTENT_COLUMNS].fillna("").astype(str)
    has_pedido_resposta = content["pedido"].str.strip().ne("") & content["resposta"].str.strip().ne("")
    has_recurso_decisao = content["recurso"].str.strip().ne("") & content["decisao_recurso"].str.strip().ne("")
    has_content = has_pedido_resposta | has_recurso_decisao
    return df[has_content].copy()


def split_by_tokens(text: str, max_tokens: int = 700, overlap: int = 100) -> list[str]:
    words = text.split()
    if not words:
        return []
    approx_tokens = max(1, len(text) // 4)
    if approx_tokens <= max_tokens:
        return [text]
    words_per_chunk = max_tokens * 3
    overlap_words = overlap * 3
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def make_chunks(df: pd.DataFrame) -> list[Chunk]:
    chunks: list[Chunk] = []
    for _, row in df.iterrows():
        text = build_content_text(row)
        if not text:
            continue
        for idx, chunk_text in enumerate(split_by_tokens(text)):
            chunks.append(
                Chunk(
                    id=f"{row['doc_id']}-{idx}",
                    text=chunk_text,
                    metadata={
                        "doc_id": str(row["doc_id"]),
                        "protocolo": str(row.get("protocolo", "")),
                        "orgao": str(row.get("orgao", "")),
                        "data_pedido": str(row.get("data_pedido", "")),
                        "tema": str(row.get("tema", "")),
                        "status": str(row.get("status", "")),
                    },
                )
            )
    return chunks


def get_collection(reset: bool = False):
    ensure_dirs()
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    if reset:
        try:
            client.delete_collection("lai_2026")
        except Exception:
            pass
    return client.get_or_create_collection("lai_2026", metadata={"hnsw:space": "cosine"})


def batched(items: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def missing_chunks(collection, chunks: list[Chunk]) -> list[Chunk]:
    ids = [chunk.id for chunk in chunks]
    try:
        existing = set(collection.get(ids=ids).get("ids", []))
    except Exception:
        existing = set()
    return [chunk for chunk in chunks if chunk.id not in existing]


def run(reset: bool = True, batch_size: int = 128, limit: int | None = None) -> None:
    settings = load_settings()
    df = load_documents()
    if limit is not None:
        df = df.head(limit)
    total_docs = len(df)
    df = filter_indexable_documents(df)
    skipped_docs = total_docs - len(df)
    if skipped_docs:
        print(
            f"Ignorando {skipped_docs} documentos sem par pedido+resposta ou recurso+decisao de recurso.",
            flush=True,
        )
    chunks = make_chunks(df)
    collection = get_collection(reset=reset)
    texts = [chunk.text for chunk in chunks]
    tokens = sum(count_tokens(text, settings.embedding_model) for text in texts)
    cost = estimate_cost(settings.embedding_model, tokens)
    print(f"Indexando {len(chunks)} chunks. Custo estimado de embeddings: {format_usd(cost.usd)}", flush=True)
    if not texts:
        print("Nenhum chunk gerado.")
        return
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_index, batch in enumerate(batched(chunks, batch_size), start=1):
        batch = missing_chunks(collection, batch)
        if not batch:
            print(f"Lote {batch_index}/{total_batches} ja estava indexado.", flush=True)
            continue
        batch_texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(batch_texts, settings.embedding_model)
        collection.add(
            ids=[chunk.id for chunk in batch],
            documents=batch_texts,
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings,
        )
        print(f"Lote {batch_index}/{total_batches} indexado ({len(batch)} chunks).", flush=True)
    print(f"Indice vetorial salvo em {VECTOR_DIR}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="Nao recria a colecao Chroma.")
    parser.add_argument("--batch-size", type=int, default=128, help="Quantidade de chunks por chamada de embeddings.")
    parser.add_argument("--limit", type=int, default=None, help="Limita documentos para teste.")
    args = parser.parse_args()
    run(reset=not args.no_reset, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
