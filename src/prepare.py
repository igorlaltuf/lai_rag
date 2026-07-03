from __future__ import annotations

import argparse
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, RAW_DIR, ensure_dirs


COLUMN_ALIASES = {
    "protocolo": ["protocolo", "protocolo_pedido", "protocolopedido", "numero_protocolo", "id_pedido", "idpedido", "num_protocolo"],
    "orgao": ["orgao", "orgao_destinatario", "orgaodestinatario", "orgao_pedido", "orgaopedido", "nome_orgao", "instituicao"],
    "data_pedido": ["data_pedido", "data", "data_registro", "dataregistro", "data_abertura"],
    "tema": ["tema", "assunto", "assunto_pedido", "assuntopedido", "subassuntopedido", "categoria", "tag"],
    "pedido": [
        "pedido",
        "texto_pedido",
        "solicitacao",
        "resumo_solicitacao",
        "resumosolicitacao",
        "detalhamento_solicitacao",
        "detalhamentosolicitacao",
        "descricao_pedido",
        "pergunta",
    ],
    "resposta": ["resposta", "texto_resposta", "resposta_pedido", "conteudo_resposta", "detalhamentodecisao", "detalhamento_decisao"],
    "status": ["status", "situacao", "decisao", "tipo_resposta", "tiporesposta"],
    "recurso": ["recurso", "texto_recurso", "descricao_recurso", "descrecurso", "detalhamentorecurso", "detalhamento_recurso"],
    "decisao_recurso": ["decisao_recurso", "resposta_recurso", "respostarecurso", "resultado_recurso", "informacaoconcedida"],
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def is_relevant_csv(name: str) -> bool:
    lowered = name.lower()
    if not lowered.endswith(".csv"):
        return False
    if "linkarquivo" in lowered or "solicitantes" in lowered:
        return False
    return "pedidos_csv" in lowered or "recursos_csv" in lowered or "recursos_reclamacoes_csv" in lowered


def is_attachment_csv(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".csv") and "linkarquivo" in lowered


def read_csv_falabr(file_obj) -> pd.DataFrame:
    try:
        return pd.read_csv(file_obj, sep=";", encoding="utf-16", dtype=str, on_bad_lines="skip")
    except UnicodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, sep=";", encoding="latin1", dtype=str, on_bad_lines="skip")


def read_any(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if is_relevant_csv(name):
                    with zf.open(name) as fh:
                        frames.append(read_csv_falabr(fh))
        return frames
    if path.suffix.lower() == ".csv":
        with path.open("rb") as fh:
            return [read_csv_falabr(fh)]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return [pd.read_excel(path, dtype=str)]
    if path.suffix.lower() == ".xml":
        return [pd.read_xml(path, dtype=str)]
    return []


def read_attachment_frames(path: Path) -> list[tuple[str, pd.DataFrame]]:
    if path.suffix.lower() == ".zip":
        frames: list[tuple[str, pd.DataFrame]] = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if is_attachment_csv(name):
                    with zf.open(name) as fh:
                        frames.append((name, read_csv_falabr(fh)))
        return frames
    if path.suffix.lower() == ".csv" and is_attachment_csv(path.name):
        with path.open("rb") as fh:
            return [(path.name, read_csv_falabr(fh))]
    return []


def pick_column(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    normalized = {normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized:
            return df[normalized[key]].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def pick_joined_columns(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    normalized = {normalize_column_name(col): col for col in df.columns}
    series = pd.Series([""] * len(df), index=df.index, dtype=str)
    used_columns: set[str] = set()
    for alias in aliases:
        key = normalize_column_name(alias)
        if key not in normalized:
            continue
        column = normalized[key]
        if column in used_columns:
            continue
        used_columns.add(column)
        values = df[column].fillna("").astype(str).str.strip()
        series = (series + " " + values).str.strip()
    return series.str.replace(r"\s+", " ", regex=True)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({field: pick_column(df, aliases) for field, aliases in COLUMN_ALIASES.items()})
    for field in ["tema", "pedido", "resposta", "recurso", "decisao_recurso"]:
        out[field] = pick_joined_columns(df, COLUMN_ALIASES[field])
    out["protocolo"] = out["protocolo"].where(out["protocolo"].str.len() > 0, out.index.astype(str))
    out["data_pedido"] = pd.to_datetime(out["data_pedido"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d").fillna("")
    text_fields = ["orgao", "tema", "pedido", "resposta", "status", "recurso", "decisao_recurso"]
    for field in text_fields:
        out[field] = out[field].str.replace(r"\s+", " ", regex=True).str.strip()
    out["document_text"] = (
        "Tema: " + out["tema"] + "\n"
        "Pedido: " + out["pedido"] + "\n"
        "Resposta: " + out["resposta"] + "\n"
        "Recurso: " + out["recurso"] + "\n"
        "Decisao do recurso: " + out["decisao_recurso"]
    ).str.strip()
    return out[out["document_text"].str.len() > 30].copy()


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith("."):
            continue
        for frame in read_any(path):
            if not frame.empty:
                frames.append(normalize_frame(frame))
    if not frames:
        raise FileNotFoundError("Nenhum arquivo de dados encontrado em data/raw. Rode src.download primeiro.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["protocolo", "document_text"])
    return df


def load_id_protocol_map(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith("."):
            continue
        for frame in read_any(path):
            normalized = {normalize_column_name(col): col for col in frame.columns}
            if "idpedido" not in normalized or "protocolopedido" not in normalized:
                continue
            id_col = normalized["idpedido"]
            protocolo_col = normalized["protocolopedido"]
            orgao_col = normalized.get("orgaodestinatario") or normalized.get("orgaopedido")
            data_col = normalized.get("dataregistro")
            mapped = pd.DataFrame(
                {
                    "id_pedido": frame[id_col].fillna("").astype(str).str.strip(),
                    "protocolo": frame[protocolo_col].fillna("").astype(str).str.strip(),
                    "orgao": frame[orgao_col].fillna("").astype(str).str.strip() if orgao_col else "",
                    "data_pedido": frame[data_col].fillna("").astype(str).str.strip() if data_col else "",
                }
            )
            frames.append(mapped)
    if not frames:
        return pd.DataFrame(columns=["id_pedido", "protocolo", "orgao", "data_pedido"])
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["id_pedido"].str.len() > 0) & (out["protocolo"].str.len() > 0)]
    out["data_pedido"] = pd.to_datetime(out["data_pedido"], errors="coerce", dayfirst=True).dt.strftime("%Y-%m-%d").fillna("")
    return out.drop_duplicates(subset=["id_pedido", "protocolo"])


def normalize_attachment_frame(name: str, df: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    normalized = {normalize_column_name(col): col for col in df.columns}
    if "idpedido" not in normalized or "urlarquivo" not in normalized:
        return pd.DataFrame()
    source_kind = "resource" if "recursos" in name.lower() else "request"
    out = pd.DataFrame(
        {
            "id_pedido": df[normalized["idpedido"]].fillna("").astype(str).str.strip(),
            "id_recurso": df[normalized["idrecurso"]].fillna("").astype(str).str.strip() if "idrecurso" in normalized else "",
            "id_anexo": (
                df[normalized["idanexorecurso"]].fillna("").astype(str).str.strip()
                if "idanexorecurso" in normalized
                else df[normalized["idanexopedido"]].fillna("").astype(str).str.strip()
                if "idanexopedido" in normalized
                else ""
            ),
            "orgao": df[normalized["orgaodestinatario"]].fillna("").astype(str).str.strip() if "orgaodestinatario" in normalized else "",
            "instancia": df[normalized["instancia"]].fillna("").astype(str).str.strip() if "instancia" in normalized else "",
            "tipo_anexo": df[normalized["tipoanexo"]].fillna("").astype(str).str.strip() if "tipoanexo" in normalized else "",
            "nome_arquivo": df[normalized["nomearquivo"]].fillna("").astype(str).str.strip() if "nomearquivo" in normalized else "",
            "url_arquivo": df[normalized["urlarquivo"]].fillna("").astype(str).str.strip(),
            "source_kind": source_kind,
        }
    )
    out = out.merge(id_map[["id_pedido", "protocolo"]], on="id_pedido", how="left")
    out["protocolo"] = out["protocolo"].fillna("")
    is_pdf = out["nome_arquivo"].str.lower().str.contains(r"\.pdf(?:$|\?)", regex=True)
    is_response = out["tipo_anexo"].isin(["Anexo Resposta", "Anexo Resposta Recurso"])
    out = out[is_pdf & is_response & out["url_arquivo"].str.len().gt(0) & out["protocolo"].str.len().gt(0)].copy()
    return out.drop_duplicates(subset=["protocolo", "id_anexo", "url_arquivo"])


def load_attachments(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    id_map = load_id_protocol_map(raw_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith("."):
            continue
        for name, frame in read_attachment_frames(path):
            normalized = normalize_attachment_frame(name, frame, id_map)
            if not normalized.empty:
                frames.append(normalized)
    if not frames:
        return pd.DataFrame(
            columns=[
                "id_pedido",
                "id_recurso",
                "id_anexo",
                "protocolo",
                "orgao",
                "instancia",
                "tipo_anexo",
                "nome_arquivo",
                "url_arquivo",
                "source_kind",
            ]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["protocolo", "id_anexo", "url_arquivo"])


def write_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH, attachments: pd.DataFrame | None = None) -> None:
    ensure_dirs()
    with sqlite3.connect(db_path) as conn:
        df.to_sql("documents", conn, if_exists="replace", index=False)
        conn.execute("DROP TABLE IF EXISTS documents_fts")
        conn.execute(
            """
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                protocolo, orgao, tema, pedido, resposta, status, recurso, decisao_recurso, document_text
            )
            """
        )
        if attachments is not None:
            attachments.to_sql("attachments", conn, if_exists="replace", index=False)
        else:
            conn.execute("DROP TABLE IF EXISTS attachments")
        conn.execute(
            """
            INSERT INTO documents_fts(rowid, protocolo, orgao, tema, pedido, resposta, status, recurso, decisao_recurso, document_text)
            SELECT rowid, protocolo, orgao, tema, pedido, resposta, status, recurso, decisao_recurso, document_text
            FROM documents
            """
        )


def run() -> pd.DataFrame:
    df = load_raw()
    attachments = load_attachments()
    write_sqlite(df, attachments=attachments)
    print(f"{len(df)} documentos normalizados em {DB_PATH}")
    print(f"{len(attachments)} anexos PDF de resposta normalizados em {DB_PATH}")
    return df


def main() -> None:
    argparse.ArgumentParser().parse_args()
    run()


if __name__ == "__main__":
    main()
