from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import chromadb
import pandas as pd

from src.config import DB_PATH, VECTOR_DIR, load_settings
from src.openai_client import embed_texts


@dataclass
class SearchResult:
    doc_id: str
    protocolo: str
    orgao: str
    data_pedido: str
    status: str
    text: str
    score: float
    source: str


def _escape_fts_query(query: str) -> str:
    terms = [term.replace('"', "") for term in query.split() if len(term) > 2]
    return " OR ".join(f'"{term}"' for term in terms[:12]) or '""'


def lexical_search(query: str, limit: int = 10) -> list[SearchResult]:
    if not DB_PATH.exists():
        return []
    fts_query = _escape_fts_query(query)
    sql = """
        SELECT d.rowid AS doc_id, d.protocolo, d.orgao, d.data_pedido, d.status, d.document_text,
               bm25(documents_fts) AS rank
        FROM documents_fts
        JOIN documents d ON d.rowid = documents_fts.rowid
        WHERE documents_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        try:
            rows = conn.execute(sql, (fts_query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
    results: list[SearchResult] = []
    for idx, row in enumerate(rows):
        results.append(
            SearchResult(
                doc_id=str(row[0]),
                protocolo=row[1] or "",
                orgao=row[2] or "",
                data_pedido=row[3] or "",
                status=row[4] or "",
                text=row[5] or "",
                score=1.0 / (idx + 1),
                source="keyword",
            )
        )
    return results


def vector_search(query: str, limit: int = 10) -> list[SearchResult]:
    settings = load_settings()
    if not VECTOR_DIR.exists():
        return []
    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        collection = client.get_collection("lai_2026")
        embedding = embed_texts([query], settings.embedding_model)[0]
        response = collection.query(query_embeddings=[embedding], n_results=limit)
    except Exception:
        return []
    results: list[SearchResult] = []
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0] if response.get("distances") else [0] * len(ids)
    for idx, doc_id in enumerate(ids):
        metadata = metadatas[idx] or {}
        results.append(
            SearchResult(
                doc_id=str(metadata.get("doc_id", doc_id)),
                protocolo=str(metadata.get("protocolo", "")),
                orgao=str(metadata.get("orgao", "")),
                data_pedido=str(metadata.get("data_pedido", "")),
                status=str(metadata.get("status", "")),
                text=documents[idx] or "",
                score=1.0 / (1.0 + float(distances[idx] or 0)),
                source="semantic",
            )
        )
    return results


def hybrid_search(query: str, limit: int = 8, vector_weight: float = 0.6) -> list[SearchResult]:
    sem = vector_search(query, limit=limit * 2)
    lex = lexical_search(query, limit=limit * 2)
    scores: dict[str, float] = {}
    picked: dict[str, SearchResult] = {}
    for rank, result in enumerate(sem, start=1):
        key = result.doc_id
        scores[key] = scores.get(key, 0.0) + vector_weight / (60 + rank)
        picked[key] = result
    for rank, result in enumerate(lex, start=1):
        key = result.doc_id
        scores[key] = scores.get(key, 0.0) + (1 - vector_weight) / (60 + rank)
        if key not in picked:
            picked[key] = result
        else:
            picked[key].source = "hybrid"
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    results = []
    for key in ordered:
        result = picked[key]
        result.score = scores[key]
        results.append(result)
    return results


def results_to_frame(results: list[SearchResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])
