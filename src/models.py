from __future__ import annotations

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    protocolo: str = ""
    orgao: str = ""
    data: str = ""
    trecho: str = ""


class PedidoEncontrado(BaseModel):
    protocolo: str = ""
    orgao: str = ""
    resumo: str
    status_resposta: str = ""


class IdeiaPedido(BaseModel):
    titulo: str
    texto_sugerido: str
    justificativa: str
    fontes: list[str] = Field(default_factory=list)


class AnalisePedido(BaseModel):
    protocolo: str = ""
    orgao: str = ""
    data: str = ""
    resumo_pedido: str = ""
    resumo_resposta: str = ""
    recurso: str = ""
    lacunas: list[str] = Field(default_factory=list)
    ideia_novo_pedido: str = ""


class RAGAnswer(BaseModel):
    resumo_tema: str
    pedidos_encontrados: list[PedidoEncontrado] = Field(default_factory=list)
    analise_por_pedido: list[AnalisePedido] = Field(default_factory=list)
    respostas_observadas: list[str] = Field(default_factory=list)
    lacunas: list[str] = Field(default_factory=list)
    ideias_novos_pedidos: list[IdeiaPedido] = Field(default_factory=list)
    fontes: list[SourceRef] = Field(default_factory=list)
    alertas_limitacoes: list[str] = Field(default_factory=list)
    estimativa_custo: str = ""
