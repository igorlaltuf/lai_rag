from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from src.config import DB_PATH, load_settings
from src.costs import MODEL_PRICES_USD_PER_1M
from src.rag import answer_topic
from src.retrieval import results_to_frame
from src.safety import sanitize_user_query, was_sanitized


st.set_page_config(page_title="LAI 2026 RAG", layout="wide")


@st.cache_data(show_spinner=False)
def count_documents() -> int:
    if not DB_PATH.exists():
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        return int(conn.execute("SELECT count(*) FROM documents").fetchone()[0])


settings = load_settings()
documents_count = count_documents()

st.title("Assistente de pedidos de LAI 2026")
st.caption("Busca hibrida em pedidos, respostas e recursos da LAI, com sugestoes geradas por LLM.")

with st.sidebar:
    st.header("Busca")
    top_k = st.slider("Documentos recuperados", 3, 20, settings.rag_top_k)
    vector_weight = st.slider("Peso semantico", 0.0, 1.0, settings.rag_vector_weight, 0.05)
    model_options = [model for model in MODEL_PRICES_USD_PER_1M if model.startswith("gpt")]
    default_idx = model_options.index(settings.generation_model) if settings.generation_model in model_options else 0
    generation_model = st.selectbox("Modelo OpenAI", model_options, index=default_idx)
    st.metric("Documentos na base", f"{documents_count:,}".replace(",", "."))

if documents_count == 0:
    st.warning(
        "Base processada nao encontrada. Rode: "
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
        with st.spinner("Buscando pedidos semelhantes e gerando sugestoes..."):
            sanitized_topic = sanitize_user_query(topic)
            if was_sanitized(topic, sanitized_topic):
                st.caption(f"Consulta sanitizada usada na busca: {sanitized_topic}")
            answer, results = answer_topic(topic, top_k=top_k, vector_weight=vector_weight, model=generation_model)
            st.subheader(answer.resumo_tema)
            if answer.pedidos_encontrados:
                st.markdown("**O que ja foi pedido**")
                st.dataframe(pd.DataFrame([item.model_dump() for item in answer.pedidos_encontrados]), use_container_width=True)
            if answer.respostas_observadas:
                st.markdown("**O que foi respondido**")
                for item in answer.respostas_observadas:
                    st.write(f"- {item}")
            if answer.lacunas:
                st.markdown("**Lacunas**")
                for item in answer.lacunas:
                    st.write(f"- {item}")
            if answer.ideias_novos_pedidos:
                st.markdown("**Ideias de novos pedidos**")
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
                    st.dataframe(results_to_frame(results), use_container_width=True)
            st.session_state.messages.append({"role": "assistant", "content": answer.model_dump_json(indent=2)})
