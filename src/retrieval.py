from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import chromadb
import pandas as pd

from src.config import DB_PATH, VECTOR_DIR, load_settings
from src.index import build_context_text
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
    term_query = " OR ".join(f'"{term}"' for term in terms[:12]) or '""'
    return f"{{pedido resposta recurso decisao_recurso}} : ({term_query})"


def lexical_search(query: str, limit: int = 10) -> list[SearchResult]:
    if not DB_PATH.exists():
        return []
    fts_query = _escape_fts_query(query)
    sql = """
        SELECT d.rowid AS doc_id, d.protocolo, d.orgao, d.data_pedido, d.status,
               d.pedido, d.resposta, d.recurso, d.decisao_recurso,
               bm25(documents_fts) AS rank
        FROM documents_fts
        JOIN documents d ON d.rowid = documents_fts.rowid
        WHERE documents_fts MATCH ?
          AND (
            (length(trim(d.pedido)) > 0 AND length(trim(d.resposta)) > 0)
            OR (length(trim(d.recurso)) > 0 AND length(trim(d.decisao_recurso)) > 0)
          )
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
                text=build_context_text(
                    {
                        "pedido": row[5] or "",
                        "resposta": row[6] or "",
                        "recurso": row[7] or "",
                        "decisao_recurso": row[8] or "",
                    }
                ),
                score=1.0 / (idx + 1),
                source="keyword",
            )
        )
    return results


def hydrate_results_with_documents(results: list[SearchResult]) -> list[SearchResult]:
    if not results or not DB_PATH.exists():
        return results
    doc_ids = [result.doc_id for result in results if result.doc_id]
    if not doc_ids:
        return results
    placeholders = ",".join("?" for _ in doc_ids)
    sql = f"""
        SELECT rowid AS doc_id, pedido, resposta, recurso, decisao_recurso
        FROM documents
        WHERE rowid IN ({placeholders})
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(sql, doc_ids).fetchall()
    text_by_doc_id = {
        str(row[0]): build_context_text(
            {
                "pedido": row[1] or "",
                "resposta": row[2] or "",
                "recurso": row[3] or "",
                "decisao_recurso": row[4] or "",
            }
        )
        for row in rows
    }
    for result in results:
        if text_by_doc_id.get(result.doc_id):
            result.text = text_by_doc_id[result.doc_id]
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
    return hydrate_results_with_documents(results)


def related_resource_search(
    protocols: list[str],
    existing_doc_ids: set[str] | None = None,
) -> list[SearchResult]:
    if not DB_PATH.exists():
        return []
    existing_doc_ids = existing_doc_ids or set()
    related: list[SearchResult] = []
    sql = """
        SELECT d.rowid AS doc_id, d.protocolo, d.orgao, d.data_pedido, d.status,
               d.pedido, d.resposta, d.recurso, d.decisao_recurso
        FROM documents d
        WHERE d.protocolo = ?
          AND length(trim(d.recurso)) > 0
          AND length(trim(d.decisao_recurso)) > 0
        ORDER BY
          CASE WHEN length(trim(d.recurso)) > 0 THEN 0 ELSE 1 END,
          d.rowid
    """
    seen_protocols = [protocol for protocol in dict.fromkeys(protocols) if protocol]
    with sqlite3.connect(DB_PATH) as conn:
        for protocol in seen_protocols:
            rows = conn.execute(sql, (protocol,)).fetchall()
            for row in rows:
                doc_id = str(row[0])
                if doc_id in existing_doc_ids:
                    continue
                text = build_context_text(
                    {
                        "pedido": row[5] or "",
                        "resposta": row[6] or "",
                        "recurso": row[7] or "",
                        "decisao_recurso": row[8] or "",
                    }
                )
                if not text:
                    continue
                related.append(
                    SearchResult(
                        doc_id=doc_id,
                        protocolo=row[1] or "",
                        orgao=row[2] or "",
                        data_pedido=row[3] or "",
                        status=row[4] or "",
                        text=text,
                        score=0.0,
                        source="related_resource",
                    )
                )
    return related


def enrich_with_related_resources(results: list[SearchResult]) -> list[SearchResult]:
    protocols = [result.protocolo for result in results if result.protocolo]
    existing_doc_ids = {result.doc_id for result in results}
    related = related_resource_search(protocols, existing_doc_ids=existing_doc_ids)
    return results + related


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
