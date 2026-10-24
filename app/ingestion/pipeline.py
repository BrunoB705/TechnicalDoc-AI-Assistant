from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import DEFAULT_KB_ID, load_directory, load_file

INDEX_BATCH_SIZE = 1000


def _index(docs: list[Document], vectorstore: Chroma, settings: Settings) -> tuple[int, int]:
    for d in docs:
        d.page_content = clean_text(d.page_content)
    docs = [d for d in docs if d.page_content]
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    # Chroma rechaza inserciones de más de ~5461 elementos: se indexa por lotes.
    for i in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[i : i + INDEX_BATCH_SIZE]
        vectorstore.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
    return len({d.metadata["doc_id"] for d in docs}), len(chunks)


def ingest_directory(
    directory: Path,
    vectorstore: Chroma,
    settings: Settings,
    knowledge_base_id: str = DEFAULT_KB_ID,
) -> tuple[int, int]:
    """Indexa un directorio en una KB. Devuelve (n_documentos, n_chunks). Idempotente."""
    return _index(
        load_directory(directory, knowledge_base_id=knowledge_base_id), vectorstore, settings
    )


def ingest_file(
    path: Path,
    vectorstore: Chroma,
    settings: Settings,
    root: Path | None = None,
    knowledge_base_id: str = DEFAULT_KB_ID,
) -> tuple[int, int]:
    """Indexa un único archivo en una KB. Devuelve (n_documentos, n_chunks)."""
    return _index(
        load_file(path, root=root, knowledge_base_id=knowledge_base_id), vectorstore, settings
    )
