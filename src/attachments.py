from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from pypdf import PdfReader

from src.config import ATTACHMENTS_DIR, DB_PATH, ensure_dirs


@dataclass(frozen=True)
class Attachment:
    protocolo: str
    tipo_anexo: str
    nome_arquivo: str
    url_arquivo: str
    instancia: str = ""
    source_kind: str = ""


@dataclass(frozen=True)
class AttachmentExcerpt:
    attachment: Attachment
    status: str
    text_start: str = ""
    text_end: str = ""
    error: str = ""


def stable_attachment_id(attachment: Attachment) -> str:
    raw = f"{attachment.protocolo}|{attachment.nome_arquivo}|{attachment.url_arquivo}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def attachment_paths(attachment: Attachment) -> tuple[Path, Path]:
    ensure_dirs()
    stem = stable_attachment_id(attachment)
    return ATTACHMENTS_DIR / f"{stem}.pdf", ATTACHMENTS_DIR / f"{stem}.txt"


def attachment_error_path(attachment: Attachment) -> Path:
    ensure_dirs()
    return ATTACHMENTS_DIR / f"{stable_attachment_id(attachment)}.err"


def is_obsolete_cached_error(error: str) -> bool:
    normalized = " ".join((error or "").lower().split())
    obsolete_markers = [
        "url do anexo nao retornou um pdf valido",
        "url do anexo não retornou um pdf válido",
        "stream has ended unexpectedly",
        "eof marker not found",
        "invalid pdf header",
    ]
    return any(marker in normalized for marker in obsolete_markers)


def id_aws_from_url(url: str) -> str:
    values = parse_qs(urlparse(url).query).get("idAws", [])
    return values[0] if values else ""


def download_attachment_via_api(attachment: Attachment, timeout: int = 60) -> bytes:
    id_aws = id_aws_from_url(attachment.url_arquivo)
    if not id_aws:
        raise ValueError("URL do anexo nao contem idAws.")
    response = requests.post(
        "https://api-laibr.cgu.gov.br/publico/busca/pedidos/arquivo",
        data=id_aws.encode("utf-8"),
        timeout=timeout,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Content-Type": "text/plain",
            "Origin": "https://buscalai.cgu.gov.br",
            "Referer": "https://buscalai.cgu.gov.br/download-arquivo",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
        },
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("conteudo") if isinstance(data, dict) else None
    if not content:
        raise ValueError("API de anexos nao retornou conteudo.")
    import base64

    return base64.b64decode(content)


def get_attachments_for_protocols(protocols: list[str], limit_per_protocol: int = 3) -> dict[str, list[Attachment]]:
    if not DB_PATH.exists():
        return {}
    unique_protocols = [protocol for protocol in dict.fromkeys(protocols) if protocol]
    if not unique_protocols:
        return {}
    sql = """
        SELECT protocolo, tipo_anexo, nome_arquivo, url_arquivo, instancia, source_kind
        FROM attachments
        WHERE protocolo = ?
        ORDER BY
          CASE WHEN tipo_anexo = 'Anexo Resposta Recurso' THEN 0 ELSE 1 END,
          nome_arquivo
        LIMIT ?
    """
    found: dict[str, list[Attachment]] = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            for protocol in unique_protocols:
                rows = conn.execute(sql, (protocol, limit_per_protocol)).fetchall()
                found[protocol] = [
                    Attachment(
                        protocolo=row[0] or "",
                        tipo_anexo=row[1] or "",
                        nome_arquivo=row[2] or "",
                        url_arquivo=row[3] or "",
                        instancia=row[4] or "",
                        source_kind=row[5] or "",
                    )
                    for row in rows
                ]
    except sqlite3.OperationalError:
        return {}
    return found


def download_attachment(attachment: Attachment, timeout: int = 60) -> Path:
    pdf_path, _ = attachment_paths(attachment)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        if pdf_path.read_bytes()[:5] != b"%PDF-":
            pdf_path.unlink(missing_ok=True)
        else:
            return pdf_path
    content = download_attachment_via_api(attachment, timeout=timeout)
    if content[:5] != b"%PDF-":
        raise ValueError("URL do anexo nao retornou um PDF valido.")
    pdf_path.write_bytes(content)
    return pdf_path


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def get_pdf_text_cached(attachment: Attachment) -> tuple[str, str]:
    pdf_path, text_path = attachment_paths(attachment)
    if text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="ignore"), "cache"
    pdf_path = download_attachment(attachment)
    text = extract_pdf_text(pdf_path)
    text_path.write_text(text, encoding="utf-8")
    attachment_error_path(attachment).unlink(missing_ok=True)
    return text, "baixado"


def excerpt_text(text: str, edge_chars: int = 3000) -> tuple[str, str]:
    text = " ".join((text or "").split())
    if len(text) <= edge_chars * 2:
        return text, ""
    return text[:edge_chars].strip(), text[-edge_chars:].strip()


def load_attachment_excerpts(protocols: list[str], edge_chars: int = 3000, limit_per_protocol: int = 3) -> list[AttachmentExcerpt]:
    excerpts: list[AttachmentExcerpt] = []
    by_protocol = get_attachments_for_protocols(protocols, limit_per_protocol=limit_per_protocol)
    for attachments in by_protocol.values():
        for attachment in attachments:
            error_path = attachment_error_path(attachment)
            if error_path.exists():
                cached_error = error_path.read_text(encoding="utf-8", errors="ignore")
                if not is_obsolete_cached_error(cached_error):
                    excerpts.append(
                        AttachmentExcerpt(
                            attachment=attachment,
                            status="erro_cache",
                            error=cached_error,
                        )
                    )
                    continue
                error_path.unlink(missing_ok=True)
            try:
                text, status = get_pdf_text_cached(attachment)
                if not text:
                    excerpts.append(AttachmentExcerpt(attachment=attachment, status=status, error="PDF sem texto extraivel."))
                    continue
                text_start, text_end = excerpt_text(text, edge_chars=edge_chars)
                excerpts.append(
                    AttachmentExcerpt(
                        attachment=attachment,
                        status=status,
                        text_start=text_start,
                        text_end=text_end,
                    )
                )
            except Exception as exc:
                error = str(exc)
                error_path.write_text(error, encoding="utf-8")
                excerpts.append(AttachmentExcerpt(attachment=attachment, status="erro", error=error))
    return excerpts


def build_attachment_context(excerpts: list[AttachmentExcerpt]) -> str:
    blocks: list[str] = []
    for idx, excerpt in enumerate(excerpts, start=1):
        attachment = excerpt.attachment
        block = [
            f"[Anexo {idx}]",
            "Tipo da fonte: attachment_response",
            f"Protocolo: {attachment.protocolo}",
            f"Tipo do anexo: {attachment.tipo_anexo}",
            f"Instancia: {attachment.instancia}",
            f"Nome do arquivo: {attachment.nome_arquivo}",
            f"Status do anexo: {excerpt.status}",
        ]
        if excerpt.error:
            block.append(f"Erro: {excerpt.error}")
        else:
            block.append(f"Trecho inicial: {excerpt.text_start}")
            if excerpt.text_end:
                block.append(f"Trecho final: {excerpt.text_end}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)
