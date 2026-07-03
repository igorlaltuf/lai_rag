from src.rag import parse_answer


def test_parse_answer_validates_json():
    raw = """
    {
      "resumo_tema": "IA",
      "pedidos_encontrados": [],
      "respostas_observadas": [],
      "lacunas": ["faltam contratos"],
      "ideias_novos_pedidos": [],
      "fontes": [],
      "alertas_limitacoes": [],
      "estimativa_custo": "US$ 0.0010"
    }
    """
    answer = parse_answer(raw)
    assert answer.resumo_tema == "IA"
