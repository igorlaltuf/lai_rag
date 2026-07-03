from __future__ import annotations

import argparse

import chromadb

from src.config import DB_PATH, VECTOR_DIR, load_settings
from src.download import run as download_run
from src.index import run as index_run
from src.prepare import run as prepare_run


def vector_count() -> int:
    try:
        collection = chromadb.PersistentClient(path=str(VECTOR_DIR)).get_collection("lai_2026")
        return int(collection.count())
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara a base local para rodar a interface.")
    parser.add_argument("--skip-index", action="store_true", help="Baixa e prepara SQLite, mas nao gera embeddings.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho dos lotes de embeddings.")
    parser.add_argument("--limit", type=int, default=None, help="Limita documentos indexados para uma demo barata/rapida.")
    args = parser.parse_args()

    settings = load_settings()
    download_run()
    if not DB_PATH.exists():
        prepare_run()
    else:
        print(f"SQLite ja existe em {DB_PATH}")

    if args.skip_index:
        print("Indexacao vetorial pulada por --skip-index.")
        return
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY nao configurada. Configure .env ou rode com --skip-index.")
    count = vector_count()
    if count:
        print(f"Indice vetorial ja existe com {count} chunks. Para recriar, rode src.index sem --no-reset.")
        return
    index_run(reset=True, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
