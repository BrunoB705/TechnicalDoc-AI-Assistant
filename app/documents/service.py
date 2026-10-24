import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from langchain_chroma import Chroma

from app.config import Settings
from app.exceptions import (
    DocumentNotFoundError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidDocumentError,
    RAGError,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_file
from app.knowledge_bases.service import KnowledgeBaseService

UPLOAD_CHUNK_SIZE = 64 * 1024  # bytes leídos por iteración al recibir un upload


@dataclass
class DocumentInfo:
    doc_id: str
    source: str
    title: str
    file_type: str
    chunks: int
    pages: int | None = None
    knowledge_base_id: str = "general"


class DocumentService:
    """Gestión (alta, listado, baja) de documentos indexados por Knowledge Base."""

    def __init__(
        self,
        vectorstore: Chroma,
        settings: Settings,
        kb_service: KnowledgeBaseService | None = None,
    ) -> None:
        self._vs = vectorstore
        self._settings = settings
        self._kb_service = kb_service

    @property
    def max_upload_bytes(self) -> int:
        return self._settings.max_upload_mb * 1024 * 1024

    def _too_large(self) -> DocumentTooLargeError:
        return DocumentTooLargeError(
            f"El archivo supera el máximo de {self._settings.max_upload_mb} MB."
        )

    def _resolve_kb(self, knowledge_base_id: str | None) -> str:
        kb_id = knowledge_base_id or self._settings.default_kb_id
        if self._kb_service is not None:
            self._kb_service.ensure_exists(kb_id)
        return kb_id

    def _kb_dir(self, kb_id: str) -> Path:
        return self._settings.docs_dir / kb_id

    # ---- consultas ----
    def _matches_kb(self, meta: dict, kb_id: str) -> bool:
        doc_kb = meta.get("knowledge_base_id") or self._settings.default_kb_id
        return doc_kb == kb_id

    def _group(self, metadatas: list[dict], kb_id: str) -> list[DocumentInfo]:
        docs: dict[str, DocumentInfo] = {}
        for m in metadatas:
            if not m or not self._matches_kb(m, kb_id):
                continue
            info = docs.setdefault(
                m["doc_id"],
                DocumentInfo(
                    doc_id=m["doc_id"],
                    source=m["source"],
                    title=m.get("title", m["source"]),
                    file_type=m.get("file_type", ""),
                    chunks=0,
                    knowledge_base_id=kb_id,
                ),
            )
            info.chunks += 1
            if m.get("page") is not None:
                info.pages = max(info.pages or 0, int(m["page"]))
        return sorted(docs.values(), key=lambda d: d.source)

    def list_documents(self, knowledge_base_id: str | None = None) -> list[DocumentInfo]:
        kb_id = self._resolve_kb(knowledge_base_id)
        metadatas = self._vs.get(include=["metadatas"]).get("metadatas") or []
        return self._group(metadatas, kb_id)

    def get_document(self, doc_id: str, knowledge_base_id: str | None = None) -> DocumentInfo:
        kb_id = self._resolve_kb(knowledge_base_id)
        res = self._vs.get(where={"doc_id": doc_id}, include=["metadatas"])
        docs = self._group(res.get("metadatas") or [], kb_id)
        if not docs:
            raise DocumentNotFoundError(
                f"No existe el documento '{doc_id}' en la base '{kb_id}'."
            )
        return docs[0]

    def _ids_for_kb(self, where: dict, kb_id: str) -> list[str]:
        res = self._vs.get(where=where, include=["metadatas"])
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        return [
            cid
            for cid, m in zip(ids, metas, strict=False)
            if m is not None and self._matches_kb(m, kb_id)
        ]

    # ---- alta ----
    def read_upload(self, stream: BinaryIO) -> bytes:
        """Lee un upload en bloques de 64 KB y corta apenas se supera el máximo permitido."""
        limit = self.max_upload_bytes
        buf = bytearray()
        while chunk := stream.read(UPLOAD_CHUNK_SIZE):
            buf.extend(chunk)
            if len(buf) > limit:
                raise self._too_large()
        return bytes(buf)

    def add_document(
        self, filename: str, content: bytes, knowledge_base_id: str | None = None
    ) -> DocumentInfo:
        kb_id = self._resolve_kb(knowledge_base_id)
        name = Path(filename or "").name
        if not name:
            raise InvalidDocumentError("El archivo no tiene nombre.")
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidDocumentError(
                f"Extensión no soportada: '{ext}'. Permitidas: {sorted(SUPPORTED_EXTENSIONS)}."
            )
        if len(content) > self.max_upload_bytes:
            raise self._too_large()

        doc_id = hashlib.sha256(content).hexdigest()[:16]
        existing = self._vs.get(where={"doc_id": doc_id}, include=["metadatas"])
        for m in existing.get("metadatas") or []:
            if m and self._matches_kb(m, kb_id):
                src = m["source"]
                raise DuplicateDocumentError(
                    f"El contenido ya está indexado en la base '{kb_id}' como '{src}'."
                )

        kb_dir = self._kb_dir(kb_id)
        kb_dir.mkdir(parents=True, exist_ok=True)
        target = kb_dir / name
        backup = target.read_bytes() if target.exists() else None
        old_ids = self._ids_for_kb({"source": name}, kb_id)

        target.write_bytes(content)
        try:
            if kb_id == self._settings.default_kb_id:
                _, n_chunks = ingest_file(target, self._vs, self._settings, root=kb_dir)
            else:
                _, n_chunks = ingest_file(
                    target, self._vs, self._settings, root=kb_dir, knowledge_base_id=kb_id
                )
            if n_chunks == 0:
                raise InvalidDocumentError("No se pudo extraer texto del archivo.")
            if old_ids:  # mismo nombre en la misma KB, contenido distinto: reemplaza anterior
                self._vs.delete(ids=old_ids)
            return self.get_document(doc_id, knowledge_base_id=kb_id)
        except Exception as exc:
            self._rollback(doc_id, kb_id, target, backup)
            if isinstance(exc, RAGError):
                raise
            raise InvalidDocumentError(f"No se pudo procesar el archivo: {exc}") from exc

    def _rollback(
        self, doc_id: str, kb_id: str, target: Path, backup: bytes | None
    ) -> None:
        """Deshace un alta fallida en la KB actual sin afectar a otras KBs."""
        try:
            ids = self._ids_for_kb({"doc_id": doc_id}, kb_id)
            if ids:
                self._vs.delete(ids=ids)
        finally:
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(backup)

    # ---- baja ----
    def delete_document(
        self, doc_id: str, knowledge_base_id: str | None = None
    ) -> None:
        kb_id = self._resolve_kb(knowledge_base_id)
        info = self.get_document(doc_id, knowledge_base_id=kb_id)
        ids = self._ids_for_kb({"doc_id": doc_id}, kb_id)
        if ids:
            self._vs.delete(ids=ids)
        kb_dir = self._kb_dir(kb_id).resolve()
        path = (kb_dir / info.source).resolve()
        if path.is_file() and path.is_relative_to(kb_dir):
            path.unlink()
