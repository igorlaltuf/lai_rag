# Assistente RAG para pedidos e recursos da LAI 2026

Aplicacao Python que baixa dados publicos de pedidos/respostas/recursos da LAI, prepara um corpus de 2026, cria busca hibrida com ChromaDB + SQLite FTS5 e usa a API da OpenAI para sugerir novos pedidos de acesso a informacao.

## Requisitos

- Python 3.11
- `make`
- `uv` (o `make setup` tenta instalar se nao existir)
- chave `OPENAI_API_KEY` para embeddings e respostas completas

```bash
make setup
```

Edite `.env` e configure `OPENAI_API_KEY`.

## Pipeline

Para uma demo mais barata e rapida, com indexacao limitada:

```bash
make demo
make app
```

Por padrao, `make demo` usa `DEMO_LIMIT=5000`. Para mudar:

```bash
make demo DEMO_LIMIT=10000
```

Para preparar a base completa de 2026:

```bash
make run
make app
```

`make run` baixa os ZIPs publicos da CGU, cria o SQLite local e gera o indice vetorial completo. A geracao de embeddings usa a API da OpenAI.

Comandos equivalentes por etapa:

```bash
uv run python -m src.download
uv run python -m src.prepare
uv run python -m src.index --batch-size 32
uv run streamlit run app.py
```

Se a pagina publica do Fala.BR/CGU nao expuser links baixaveis de 2026 no momento da execucao, `src.download` cria um pequeno corpus sintetico demonstrativo. Para exigir somente download real, use `uv run python -m src.download --no-sample`.

## Modelos e custo

Modelos configuraveis no `.env`:

- `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`
- `OPENAI_GENERATION_MODEL=gpt-5.4-nano`


## Como funciona

1. `download`: tenta descobrir arquivos CSV/XML/XLS/ZIP de 2026 na pagina oficial de dados abertos da LAI.
2. `prepare`: normaliza colunas comuns, cria `data/processed/lai_2026.sqlite` e uma tabela FTS5.
3. `index`: divide documentos em chunks, gera embeddings OpenAI e persiste no ChromaDB.
4. `retrieval`: combina busca semantica e keyword com Reciprocal Rank Fusion.
5. `rag`: monta prompt com contexto recuperado e valida a resposta em JSON com Pydantic.
6. `app`: mostra chatbot, fontes e tabela de documentos recuperados.


## Testes

```bash
make test
```

Os testes cobrem chunking, normalizacao, custo e parsing do JSON sem depender de chamadas externas.

## Notebook

O notebook `notebooks/relatorio_lai_rag.ipynb` serve como relatorio executavel do projeto. Ele cobre:

- sanitizacao da consulta do usuario;
- comparacao de tres tecnicas de prompt;
- comparacao de retrieval keyword, semantico e hibrido;
- explicacao do pipeline RAG;
- discussao de encoder-only vs decoder-only, tokenizacao, custo, privacidade, latencia e controle;
- riscos, limitacoes e melhorias futuras.

Para abrir:

```bash
make notebook
```

Ou abra o arquivo diretamente em Jupyter/VS Code usando o ambiente criado por `uv sync`.
