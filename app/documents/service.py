import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from langchain_chroma import Chroma

from app.config import Settings
from app.documents.catalog import init_documents_table, sync_documents_catalog
from app.exceptions import (
    DocumentNotFoundError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidDocumentError,
    RAGError,
    ReadOnlyModeError,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_file
from app.knowledge_bases.service import KnowledgeBaseService

UPLOAD_CHUNK_SIZE = 64 * 1024  # bytes leídos por iteración al recibir un upload

_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(filename: str) -> str:
    """Sanitiza el nombre contra traversal, nombres reservados y caracteres inválidos."""
    name = Path(filename or "").name.strip()
    if not name:
        raise InvalidDocumentError("El archivo no tiene nombre.")
    stem = Path(name).stem.upper()
    if stem in _RESERVED_NAMES:
        name = f"doc_{name}"
    # Sanitizar caracteres prohibidos en sistemas Windows / POSIX
    sanitized = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name)
    return sanitized


@dataclass
class DocumentInfo:
    doc_id: str
    source: str
    title: str
    file_type: str
    chunks: int
    pages: int | None = None
    knowledge_base_id: str = "general"
    created_at: str = ""


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
        self._db_path = settings.catalog_path
        with self._connect() as conn:
            init_documents_table(conn)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

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

    def list_documents(self, knowledge_base_id: str | None = None) -> list[DocumentInfo]:
        kb_id = self._resolve_kb(knowledge_base_id)
        with self._connect() as conn:
            sync_documents_catalog(conn, self._vs, self._settings)
            rows = conn.execute(
                """
                SELECT
                    doc_id, knowledge_base_id, source, title, file_type, chunks, pages, created_at
                FROM documents
                WHERE knowledge_base_id = ?
                ORDER BY created_at DESC, lower(source) ASC
                """,
                (kb_id,),
            ).fetchall()
            return [
                DocumentInfo(
                    doc_id=r["doc_id"],
                    knowledge_base_id=r["knowledge_base_id"],
                    source=r["source"],
                    title=r["title"],
                    file_type=r["file_type"],
                    chunks=r["chunks"],
                    pages=r["pages"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def get_document(self, doc_id: str, knowledge_base_id: str | None = None) -> DocumentInfo:
        kb_id = self._resolve_kb(knowledge_base_id)
        with self._connect() as conn:
            sync_documents_catalog(conn, self._vs, self._settings)
            row = conn.execute(
                """
                SELECT
                    doc_id, knowledge_base_id, source, title, file_type, chunks, pages, created_at
                FROM documents
                WHERE doc_id = ? AND knowledge_base_id = ?
                """,
                (doc_id, kb_id),
            ).fetchone()
            if not row:
                raise DocumentNotFoundError(
                    f"No existe el documento '{doc_id}' en la base '{kb_id}'."
                )
            return DocumentInfo(
                doc_id=row["doc_id"],
                knowledge_base_id=row["knowledge_base_id"],
                source=row["source"],
                title=row["title"],
                file_type=row["file_type"],
                chunks=row["chunks"],
                pages=row["pages"],
                created_at=row["created_at"],
            )

    def _ids_for_kb(self, where: dict, kb_id: str) -> list[str]:
        res = self._vs.get(where=where, include=["metadatas"])
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        return [
            cid
            for cid, m in zip(ids, metas, strict=False)
            if m is not None and self._matches_kb(m, kb_id)
        ]

    def _check_writable(self) -> None:
        if self._settings.read_only_mode:
            raise ReadOnlyModeError(
                "El servicio está en modo de solo lectura (READ_ONLY_MODE=true)."
            )

    # ---- alta ----
    def read_upload(self, stream: BinaryIO) -> bytes:
        self._check_writable()
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
        self._check_writable()
        kb_id = self._resolve_kb(knowledge_base_id)
        name = sanitize_filename(filename)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise InvalidDocumentError(
                f"Extensión no soportada: '{ext}'. Permitidas: {sorted(SUPPORTED_EXTENSIONS)}."
            )
        if len(content) > self.max_upload_bytes:
            raise self._too_large()

        doc_id = hashlib.sha256(content).hexdigest()[:16]
        with self._connect() as conn:
            sync_documents_catalog(conn, self._vs, self._settings)
            dup = conn.execute(
                "SELECT source FROM documents WHERE doc_id = ? AND knowledge_base_id = ?",
                (doc_id, kb_id),
            ).fetchone()
            if dup:
                raise DuplicateDocumentError(
                    f"El contenido ya está indexado en la base '{kb_id}' como '{dup['source']}'."
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
                with self._connect() as conn:
                    conn.execute(
                        "DELETE FROM documents WHERE source = ? AND knowledge_base_id = ?",
                        (name, kb_id),
                    )
                    conn.commit()

            # Extraer páginas si es PDF
            pages = None
            if ext == ".pdf":
                res = self._vs.get(where={"doc_id": doc_id}, include=["metadatas"])
                p_list = [
                    int(m["page"])
                    for m in (res.get("metadatas") or [])
                    if m and m.get("page") is not None
                ]
                if p_list:
                    pages = max(p_list)

            now_str = datetime.now(UTC).isoformat()
            doc_info = DocumentInfo(
                doc_id=doc_id,
                source=name,
                title=Path(name).stem,
                file_type=ext.lstrip("."),
                chunks=n_chunks,
                pages=pages,
                knowledge_base_id=kb_id,
                created_at=now_str,
            )
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (doc_id, knowledge_base_id, source, title, file_type, chunks, pages, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_info.doc_id,
                        doc_info.knowledge_base_id,
                        doc_info.source,
                        doc_info.title,
                        doc_info.file_type,
                        doc_info.chunks,
                        doc_info.pages,
                        doc_info.created_at,
                    ),
                )
                conn.commit()
            return doc_info
        except Exception as exc:
            self._rollback(doc_id, kb_id, target, backup)
            if isinstance(exc, RAGError):
                raise
            raise InvalidDocumentError(f"No se pudo procesar el archivo: {exc}") from exc

    def _rollback(self, doc_id: str, kb_id: str, target: Path, backup: bytes | None) -> None:
        """Deshace un alta fallida en la KB actual sin afectar a otras KBs."""
        try:
            ids = self._ids_for_kb({"doc_id": doc_id}, kb_id)
            if ids:
                self._vs.delete(ids=ids)
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM documents WHERE doc_id = ? AND knowledge_base_id = ?",
                    (doc_id, kb_id),
                )
                conn.commit()
        finally:
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(backup)

    # ---- baja ----
    def delete_document(self, doc_id: str, knowledge_base_id: str | None = None) -> None:
        self._check_writable()
        kb_id = self._resolve_kb(knowledge_base_id)
        info = self.get_document(doc_id, knowledge_base_id=kb_id)
        ids = self._ids_for_kb({"doc_id": doc_id}, kb_id)
        if ids:
            self._vs.delete(ids=ids)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM documents WHERE doc_id = ? AND knowledge_base_id = ?",
                (doc_id, kb_id),
            )
            conn.commit()
        kb_dir = self._kb_dir(kb_id).resolve()
        path = (kb_dir / info.source).resolve()
        if path.is_file() and path.is_relative_to(kb_dir):
            path.unlink()
