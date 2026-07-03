import pandas as pd

from src.index import build_content_text, build_context_text, filter_indexable_documents, make_chunks, split_by_tokens


def test_split_short_text_keeps_single_chunk():
    assert split_by_tokens("um texto curto") == ["um texto curto"]


def test_split_long_text_creates_chunks():
    text = "palavra " * 4000
    chunks = split_by_tokens(text, max_tokens=100, overlap=10)
    assert len(chunks) > 1
    assert all(chunks)


def test_filter_indexable_documents_removes_metadata_only_rows():
    df = pd.DataFrame(
        [
            {"protocolo": "1", "tema": "Meio ambiente", "pedido": "", "resposta": "", "recurso": "", "decisao_recurso": ""},
            {"protocolo": "2", "tema": "Meio ambiente", "pedido": "Solicito dados", "resposta": "", "recurso": "", "decisao_recurso": ""},
            {"protocolo": "3", "tema": "Meio ambiente", "pedido": "", "resposta": "", "recurso": "Recurso textual", "decisao_recurso": ""},
            {"protocolo": "4", "tema": "Meio ambiente", "pedido": "Solicito dados", "resposta": "Resposta enviada", "recurso": "", "decisao_recurso": ""},
            {"protocolo": "5", "tema": "Meio ambiente", "pedido": "", "resposta": "", "recurso": "Recurso textual", "decisao_recurso": "Indeferido"},
        ]
    )

    filtered = filter_indexable_documents(df)

    assert filtered["protocolo"].tolist() == ["4", "5"]


def test_make_chunks_uses_only_content_fields_for_vector_text():
    df = pd.DataFrame(
        [
            {
                "doc_id": 10,
                "protocolo": "999",
                "orgao": "Secretaria de Estado de Meio Ambiente",
                "tema": "Meio ambiente",
                "data_pedido": "2026-01-01",
                "status": "Concluida",
                "pedido": "Solicito licencas ambientais.",
                "resposta": "Foram enviadas as licencas.",
                "recurso": "",
                "decisao_recurso": "",
            }
        ]
    )

    chunks = make_chunks(df)

    assert len(chunks) == 1
    assert "Solicito licencas ambientais" in chunks[0].text
    assert "Foram enviadas" in chunks[0].text
    assert "Secretaria de Estado de Meio Ambiente" not in chunks[0].text
    assert "Tema: Meio ambiente" not in chunks[0].text


def test_build_content_text_keeps_resource_content():
    text = build_content_text({"pedido": "", "resposta": "", "recurso": "Recurso apresentado.", "decisao_recurso": "Indeferido."})

    assert "Recurso apresentado" in text
    assert "Indeferido" in text


def test_build_context_text_keeps_answer_when_request_is_long():
    text = build_context_text({"pedido": "pedido longo " * 300, "resposta": "Resposta objetiva do orgao.", "recurso": "", "decisao_recurso": ""})

    assert "Pedido:" in text
    assert "Resposta objetiva do orgao" in text
