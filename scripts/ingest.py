import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.ingestion.pipeline import ingest_directory  # noqa: E402
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.vectorstore import get_vectorstore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos en ChromaDB")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reset", action="store_true", help="Borra la colección antes de indexar")
    args = parser.parse_args()

    settings = get_settings()
    vs = get_vectorstore(settings, get_embeddings(settings))
    if args.reset:
        vs.delete_collection()
        vs = get_vectorstore(settings, get_embeddings(settings))
    n_docs, n_chunks = ingest_directory(args.directory, vs, settings)
    print(f"Indexados {n_docs} documentos / {n_chunks} chunks")


if __name__ == "__main__":
    main()
