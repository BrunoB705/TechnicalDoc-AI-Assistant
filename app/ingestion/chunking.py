from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _section(meta: dict) -> str | None:
    parts = [meta[k] for k in ("h1", "h2", "h3") if k in meta]
    return " > ".join(parts) if parts else None


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Divide documentos en chunks con metadata y chunk_id determinista."""
    recursive = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    md_splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False)

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("file_type") == "md":
            pieces = []
            for sec in md_splitter.split_text(doc.page_content):
                meta = {**doc.metadata, **sec.metadata}
                section = _section(sec.metadata)
                if section:
                    meta["section"] = section
                pieces.extend(
                    recursive.split_documents([Document(sec.page_content, metadata=meta)])
                )
        else:
            pieces = recursive.split_documents([doc])
        chunks.extend(p for p in pieces if p.page_content.strip())

    counters: dict[str, int] = {}
    for c in chunks:
        did = c.metadata["doc_id"]
        i = counters.get(did, 0)
        counters[did] = i + 1
        c.metadata["chunk_id"] = f"{did}:{i}"
        # Chroma solo acepta metadata escalar
        for k in ("h1", "h2", "h3"):
            c.metadata.pop(k, None)
    return chunks
