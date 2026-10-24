import hashlib
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
DEFAULT_KB_ID = "general"


def _doc_id(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def load_file(
    path: Path, root: Path | None = None, knowledge_base_id: str = DEFAULT_KB_ID
) -> list[Document]:
    """Carga un archivo md/txt/pdf como lista de Documents (un Document por página en PDF)."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extensión no soportada: {ext}")

    raw = path.read_bytes()
    source = path.relative_to(root).as_posix() if root else path.name
    base = {
        "source": source,
        "doc_id": _doc_id(raw),
        "file_type": ext.lstrip("."),
        "title": path.stem,
        "knowledge_base_id": knowledge_base_id,
    }

    if ext == ".pdf":
        reader = PdfReader(path)
        return [
            Document(page_content=page.extract_text() or "", metadata={**base, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    return [Document(page_content=raw.decode("utf-8", errors="replace"), metadata=base)]


def load_directory(directory: Path, knowledge_base_id: str = DEFAULT_KB_ID) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(path, root=directory, knowledge_base_id=knowledge_base_id))
    return docs
