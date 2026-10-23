import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma

from app.config import Settings
from app.exceptions import (
    DocumentNotFoundError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidDocumentError,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_file


@dataclass
class DocumentInfo:
    doc_id: str
    source: str
    title: str
    file_type: str
    chunks: int
    pages: int | None = None


class DocumentService:
    """Gestión (alta, listado, baja) de documentos indexados."""

    def __init__(self, vectorstore: Chroma, settings: Settings) -> None:
        self._vs = vectorstore
        self._settings = settings

    # ---- consultas ----
    def _group(self, metadatas: list[dict]) -> list[DocumentInfo]:
        docs: dict[str, DocumentInfo] = {}
        for m in metadatas:
            info = docs.setdefault(
                m["doc_id"],
                DocumentInfo(
                    doc_id=m["doc_id"],
                    source=m["source"],
                    title=m.get("title", m["source"]),
                    file_type=m.get("file_type", ""),
                    chunks=0,
                ),
            )
            info.chunks += 1
            if m.get("page") is not None:
                info.pages = max(info.pages or 0, int(m["page"]))
        return sorted(docs.values(), key=lambda d: d.source)

    def list_documents(self) -> list[DocumentInfo]:
        return self._group(self._vs.get(include=["metadatas"])["metadatas"] or [])

    def get_document(self, doc_id: str) -> DocumentInfo:
        res = self._vs.get(where={"doc_id": doc_id}, include=["metadatas"])
        docs = self._group(res["metadatas"] or [])
        if not docs:
            raise DocumentNotFoundError(f"No existe el documento '{doc_id}'.")
        return docs[0]

    # ---- alta ----
    def add_document(self, filename: str, content: bytes) -> DocumentInfo:
        name = Path(filename or "").name
        if not name:
            raise InvalidDocumentError("El archivo no tiene nombre.")
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidDocumentError(
                f"Extensión no soportada: '{ext}'. Permitidas: {sorted(SUPPORTED_EXTENSIONS)}."
            )
        if len(content) > self._settings.max_upload_mb * 1024 * 1024:
            raise DocumentTooLargeError(
                f"El archivo supera el máximo de {self._settings.max_upload_mb} MB."
            )

        doc_id = hashlib.sha256(content).hexdigest()[:16]
        existing = self._vs.get(where={"doc_id": doc_id}, include=["metadatas"])
        if existing["ids"]:
            src = existing["metadatas"][0]["source"]
            raise DuplicateDocumentError(f"El contenido ya está indexado como '{src}'.")

        docs_dir = self._settings.docs_dir
        docs_dir.mkdir(parents=True, exist_ok=True)
        target = docs_dir / name
        backup = target.read_bytes() if target.exists() else None
        old = self._vs.get(where={"source": name}, include=[])["ids"]

        target.write_bytes(content)
        try:
            _, n_chunks = ingest_file(target, self._vs, self._settings, root=docs_dir)
        except Exception as exc:
            self._restore(target, backup)
            raise InvalidDocumentError(f"No se pudo procesar el archivo: {exc}") from exc
        if n_chunks == 0:
            self._restore(target, backup)
            raise InvalidDocumentError("No se pudo extraer texto del archivo.")

        if old:  # mismo nombre, contenido distinto: reemplaza la versión anterior
            self._vs.delete(ids=old)
        return self.get_document(doc_id)

    @staticmethod
    def _restore(target: Path, backup: bytes | None) -> None:
        if backup is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(backup)

    # ---- baja ----
    def delete_document(self, doc_id: str) -> None:
        info = self.get_document(doc_id)
        ids = self._vs.get(where={"doc_id": doc_id}, include=[])["ids"]
        self._vs.delete(ids=ids)
        docs_dir = self._settings.docs_dir.resolve()
        path = (docs_dir / info.source).resolve()
        if path.is_file() and path.is_relative_to(docs_dir):
            path.unlink()
