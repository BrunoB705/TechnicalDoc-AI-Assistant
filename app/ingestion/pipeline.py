from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import load_directory, load_file


def _index(docs: list[Document], vectorstore: Chroma, settings: Settings) -> tuple[int, int]:
    for d in docs:
        d.page_content = clean_text(d.page_content)
    docs = [d for d in docs if d.page_content]
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if chunks:
        vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len({d.metadata["doc_id"] for d in docs}), len(chunks)


def ingest_directory(directory: Path, vectorstore: Chroma, settings: Settings) -> tuple[int, int]:
    """Indexa un directorio. Devuelve (n_documentos, n_chunks). Idempotente por chunk_id."""
    return _index(load_directory(directory), vectorstore, settings)


def ingest_file(
    path: Path, vectorstore: Chroma, settings: Settings, root: Path | None = None
) -> tuple[int, int]:
    """Indexa un ?nico archivo. Devuelve (n_documentos, n_chunks)."""
    return _index(load_file(path, root=root), vectorstore, settings)
