import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.ingestion.pipeline import ingest_directory  # noqa: E402
from app.knowledge_bases.service import KnowledgeBaseService  # noqa: E402
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.vectorstore import get_vectorstore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos en ChromaDB")
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--kb",
        default="general",
        help="ID de la base de conocimiento destino (por defecto: general)",
    )
    parser.add_argument(
        "--kb-name",
        default=None,
        help="Nombre visible al registrar una nueva Knowledge Base (por defecto: valor de --kb)",
    )
    parser.add_argument(
        "--kb-description",
        default="",
        help="Descripción opcional para la Knowledge Base",
    )
    parser.add_argument("--reset", action="store_true", help="Borra la colección antes de indexar")
    args = parser.parse_args()

    settings = get_settings()
    vs = get_vectorstore(settings, get_embeddings(settings))
    if args.reset:
        vs.delete_collection()
        vs = get_vectorstore(settings, get_embeddings(settings))

    kb_svc = KnowledgeBaseService(vs, settings)
    try:
        kb_svc.get_knowledge_base(args.kb)
    except Exception:
        kb_svc.create_knowledge_base(
            name=args.kb_name or args.kb,
            description=args.kb_description,
            kb_id=args.kb,
        )

    n_docs, n_chunks = ingest_directory(
        args.directory, vs, settings, knowledge_base_id=args.kb
    )
    print(f"Indexados {n_docs} documentos / {n_chunks} chunks en KB '{args.kb}'")


if __name__ == "__main__":
    main()
