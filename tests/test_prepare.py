import pandas as pd

from src.prepare import normalize_attachment_frame, normalize_frame


def test_normalize_frame_maps_common_columns():
    df = pd.DataFrame(
        {
            "numero_protocolo": ["123"],
            "nome_orgao": ["CGU"],
            "data_registro": ["10/02/2026"],
            "solicitacao": ["Quero dados sobre IA."],
            "texto_resposta": ["Resposta enviada."],
            "situacao": ["Acesso concedido"],
        }
    )
    out = normalize_frame(df)
    assert out.iloc[0]["protocolo"] == "123"
    assert out.iloc[0]["orgao"] == "CGU"
    assert "Quero dados" in out.iloc[0]["document_text"]


def test_normalize_attachment_frame_keeps_only_response_pdfs_with_protocol():
    df = pd.DataFrame(
        {
            "IdPedido": ["10", "10", "11", "12"],
            "OrgaoDestinatario": ["CGU", "CGU", "CGU", "CGU"],
            "IdAnexoPedido": ["1", "2", "3", "4"],
            "NomeArquivo": ["resposta.pdf", "pedido.pdf", "resposta.docx", "sem_protocolo.pdf"],
            "TipoAnexo": ["Anexo Resposta", "Anexo Pedido", "Anexo Resposta", "Anexo Resposta"],
            "UrlArquivo": ["https://example/a.pdf", "https://example/b.pdf", "https://example/c.docx", "https://example/d.pdf"],
        }
    )
    id_map = pd.DataFrame({"id_pedido": ["10", "11"], "protocolo": ["123", "456"]})

    out = normalize_attachment_frame("PedidosLinkArquivo.csv", df, id_map)

    assert out["protocolo"].tolist() == ["123"]
    assert out.iloc[0]["nome_arquivo"] == "resposta.pdf"
