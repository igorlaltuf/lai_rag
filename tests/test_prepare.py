import pandas as pd

from src.prepare import normalize_frame


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
