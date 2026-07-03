.PHONY: help setup sync env demo run app test clean-data notebook

PYTHON ?= python3
UV ?= uv
DEMO_LIMIT ?= 5000
PORT ?= 8501

help:
	@echo "Comandos disponiveis:"
	@echo "  make setup      Instala uv se necessario, sincroniza dependencias e cria .env"
	@echo "  make demo       Prepara uma base menor para demo barata (DEMO_LIMIT=$(DEMO_LIMIT))"
	@echo "  make run        Prepara a base completa de 2026 e indexa tudo"
	@echo "  make app        Sobe o Streamlit em http://localhost:$(PORT)"
	@echo "  make test       Executa a suite de testes"
	@echo "  make notebook   Registra o kernel Jupyter do projeto"
	@echo "  make clean-data Remove dados baixados/processados/indexados"

setup:
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "uv nao encontrado. Instalando com o instalador oficial..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "uv encontrado: $$($(UV) --version)"; \
	fi
	$(UV) sync
	@if [ ! -f .env ]; then cp .env.example .env; echo "Arquivo .env criado. Preencha OPENAI_API_KEY."; else echo ".env ja existe."; fi

sync:
	$(UV) sync

env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Arquivo .env criado. Preencha OPENAI_API_KEY."; else echo ".env ja existe."; fi

demo:
	$(UV) run python -m src.bootstrap --limit $(DEMO_LIMIT) --batch-size 32

run:
	$(UV) run python -m src.bootstrap --batch-size 32

app:
	STREAMLIT_BROWSER_GATHER_USAGE_STATS=false $(UV) run streamlit run app.py --server.address 0.0.0.0 --server.port $(PORT) --server.headless true

test:
	$(UV) run pytest

notebook:
	$(UV) run python -m ipykernel install --user --name llm-law-rag

clean-data:
	rm -rf data/raw/* data/processed/* data/vector/*
	touch data/raw/.gitkeep data/processed/.gitkeep data/vector/.gitkeep
