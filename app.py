from __future__ import annotations

import sqlite3

import chromadb
import pandas as pd
import streamlit as st

from src.attachments import AttachmentExcerpt, load_attachment_excerpts
from src.config import DB_PATH, VECTOR_DIR, load_settings
from src.costs import MODEL_PRICES_USD_PER_1M
from src.rag import answer_topic
from src.retrieval import results_to_frame
from src.safety import sanitize_user_query, was_sanitized


st.set_page_config(page_title="LAI 2026 RAG", layout="wide")


@st.cache_data(show_spinner=False)
def count_base_stats() -> dict[str, int]:
    stats = {"documents": 0, "indexable_documents": 0, "chunks": 0}
    if not DB_PATH.exists():
        return stats
    with sqlite3.connect(DB_PATH) as conn:
        stats["documents"] = int(conn.execute("SELECT count(*) FROM documents").fetchone()[0])
        stats["indexable_documents"] = int(
            conn.execute(
                """
                SELECT count(*)
                FROM documents
                WHERE
                  (length(trim(pedido)) > 0 AND length(trim(resposta)) > 0)
                  OR (length(trim(recurso)) > 0 AND length(trim(decisao_recurso)) > 0)
                """
            ).fetchone()[0]
        )
    if VECTOR_DIR.exists():
        try:
            collection = chromadb.PersistentClient(path=str(VECTOR_DIR)).get_collection("lai_2026")
            stats["chunks"] = int(collection.count())
        except Exception:
            stats["chunks"] = 0
    return stats


def format_date_br(value: str) -> str:
    if not value:
        return "Não informada"
    try:
        return pd.to_datetime(value, errors="raise").strftime("%d/%m/%Y")
    except Exception:
        return value


def has_meaningful_resource(value: str) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return False
    empty_markers = [
        "não identificado",
        "nao identificado",
        "não informado",
        "nao informado",
        "sem informação",
        "sem informacao",
        "sem recurso",
        "não houve recurso",
        "nao houve recurso",
        "não consta",
        "nao consta",
    ]
    return not any(marker in text for marker in empty_markers)


def section_from_context(text: str, label: str) -> str:
    marker = f"{label}:"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    next_positions = [
        pos
        for other in ["Pedido:", "Resposta:", "Recurso:", "Decisao do recurso:"]
        if other != marker and (pos := text.find(other, start)) != -1
    ]
    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def original_request_by_protocol(results) -> dict[str, str]:
    requests: dict[str, str] = {}
    for result in results:
        if not result.protocolo or result.protocolo in requests:
            continue
        pedido = section_from_context(result.text, "Pedido")
        if pedido:
            requests[result.protocolo] = pedido
    return requests


def resources_by_protocol(results) -> dict[str, list[dict[str, str]]]:
    resources: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for result in results:
        if not result.protocolo:
            continue
        recurso = section_from_context(result.text, "Recurso")
        decisao = section_from_context(result.text, "Decisao do recurso")
        if not recurso and not decisao:
            continue
        key = (result.protocolo, recurso, decisao)
        if key in seen:
            continue
        seen.add(key)
        resources.setdefault(result.protocolo, []).append(
            {
                "data": result.data_pedido,
                "status": result.status,
                "recurso": recurso,
                "decisao": decisao,
            }
        )
    return resources


def attachment_excerpts_by_protocol(excerpts: list[AttachmentExcerpt]) -> dict[str, list[AttachmentExcerpt]]:
    grouped: dict[str, list[AttachmentExcerpt]] = {}
    for excerpt in excerpts:
        grouped.setdefault(excerpt.attachment.protocolo, []).append(excerpt)
    return grouped


settings = load_settings()
base_stats = count_base_stats()
documents_count = base_stats["documents"]

st.title("Explorador de pedidos de informação ao governo brasileiro")
st.caption("Busca híbrida em pedidos, respostas e recursos da LAI, com sugestões geradas por LLM.")

with st.sidebar:
    st.header("Busca")
    top_k = st.slider("Documentos recuperados", 3, 20, settings.rag_top_k)
    vector_weight = st.slider("Peso semântico", 0.0, 1.0, settings.rag_vector_weight, 0.05)
    model_options = [model for model in MODEL_PRICES_USD_PER_1M if model.startswith("gpt")]
    default_idx = model_options.index(settings.generation_model) if settings.generation_model in model_options else 0
    generation_model = st.selectbox("Modelo OpenAI", model_options, index=default_idx)

if documents_count == 0:
    st.warning(
        "Base processada não encontrada. Rode: "
        "uv run python -m src.bootstrap para baixar, preparar e indexar os dados."
    )

topic = st.chat_input("Digite um tema, por exemplo: reconhecimento facial, IA em contratos, LGPD...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if topic:
    st.session_state.messages.append({"role": "user", "content": topic})
    with st.chat_message("user"):
        st.markdown(topic)
    with st.chat_message("assistant"):
        with st.spinner("Buscando pedidos semelhantes e gerando sugestões..."):
            sanitized_topic = sanitize_user_query(topic)
            if was_sanitized(topic, sanitized_topic):
                st.caption(f"Consulta sanitizada usada na busca: {sanitized_topic}")
            answer, results = answer_topic(topic, top_k=top_k, vector_weight=vector_weight, model=generation_model)
            original_requests = original_request_by_protocol(results)
            resources = resources_by_protocol(results)
            attachment_excerpts = load_attachment_excerpts([result.protocolo for result in results if result.protocolo])
            attachments = attachment_excerpts_by_protocol(attachment_excerpts)
            if answer.pedidos_encontrados and not answer.analise_por_pedido:
                st.markdown("**Pedidos encontrados**")
                for item in answer.pedidos_encontrados:
                    label = f"{item.protocolo} - {item.orgao}".strip(" -")
                    st.markdown(f"- **{label}**  \n  {item.resumo}")
            if answer.analise_por_pedido:
                st.markdown("**Pedidos encontrados**")
                for idx, item in enumerate(answer.analise_por_pedido, start=1):
                    summary = item.resumo_pedido or "Pedido sem resumo gerado."
                    metadata = (
                        f"Protocolo: {item.protocolo or 'Não informado'} | "
                        f"Órgão: {item.orgao or 'Não informado'} | "
                        f"Data: {format_date_br(item.data)}"
                    )
                    st.markdown(f"**{idx}. {summary}**")
                    with st.expander(metadata, expanded=False):
                        original_request = original_requests.get(item.protocolo, "")
                        if original_request:
                            st.markdown("**Texto original do pedido**")
                            st.write(original_request)
                        st.markdown("**Resposta do órgão**")
                        st.write(item.resumo_resposta or "Não identificada.")
                        protocol_resources = resources.get(item.protocolo, [])
                        if protocol_resources:
                            st.markdown("**Recursos**")
                            for resource_idx, resource in enumerate(protocol_resources, start=1):
                                st.markdown(f"**Instância {resource_idx}**")
                                if resource["data"] or resource["status"]:
                                    details = [
                                        f"Data: {format_date_br(resource['data'])}" if resource["data"] else "",
                                        f"Status: {resource['status']}" if resource["status"] else "",
                                    ]
                                    st.caption(" | ".join(part for part in details if part))
                                if resource["recurso"]:
                                    st.markdown("Texto do recurso")
                                    st.write(resource["recurso"])
                                if resource["decisao"]:
                                    st.markdown("Resposta/decisão do recurso")
                                    st.write(resource["decisao"])
                        elif has_meaningful_resource(item.recurso):
                            st.markdown("**Recursos**")
                            st.write(item.recurso)
                        protocol_attachments = attachments.get(item.protocolo, [])
                        if protocol_attachments:
                            st.markdown("**Anexos analisados**")
                            for attachment_idx, excerpt in enumerate(protocol_attachments, start=1):
                                attachment = excerpt.attachment
                                label = f"{attachment_idx}. {attachment.nome_arquivo or 'PDF sem nome'}"
                                st.markdown(f"**{label}**")
                                details = [
                                    attachment.tipo_anexo,
                                    attachment.instancia,
                                    f"status: {excerpt.status}",
                                ]
                                st.caption(" | ".join(part for part in details if part))
                                if excerpt.error:
                                    st.warning(excerpt.error)
                        st.markdown("**Lacunas**")
                        if item.lacunas:
                            for lacuna in item.lacunas:
                                st.write(f"- {lacuna}")
                        else:
                            st.write("Não identificadas.")
            elif answer.pedidos_encontrados:
                st.dataframe(pd.DataFrame([item.model_dump() for item in answer.pedidos_encontrados]), width="stretch")
            if answer.respostas_observadas and not answer.analise_por_pedido:
                st.markdown("**O que foi respondido**")
                for item in answer.respostas_observadas:
                    st.write(f"- {item}")
            if answer.lacunas and not answer.analise_por_pedido:
                st.markdown("**Lacunas**")
                for item in answer.lacunas:
                    st.write(f"- {item}")
            if answer.ideias_novos_pedidos:
                st.markdown("**Sugestões de novos pedidos**")
                for idea in answer.ideias_novos_pedidos:
                    with st.expander(idea.titulo, expanded=True):
                        st.write(idea.texto_sugerido)
                        st.caption(idea.justificativa)
                        if idea.fontes:
                            st.caption("Fontes: " + ", ".join(idea.fontes))
            if answer.alertas_limitacoes:
                st.info(" ".join(answer.alertas_limitacoes))
            st.caption(f"Modelo: {generation_model} | Custo estimado: {answer.estimativa_custo or 'n/d'}")
            if results:
                with st.expander("Documentos recuperados"):
                    st.dataframe(results_to_frame(results), width="stretch")
            st.session_state.messages.append({"role": "assistant", "content": answer.model_dump_json(indent=2)})
