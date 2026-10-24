import re
import shutil
import sqlite3
import unicodedata
from datetime import UTC, datetime

from langchain_chroma import Chroma

from app.config import Settings
from app.exceptions import (
    DuplicateKnowledgeBaseError,
    InvalidKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    ReadOnlyModeError,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.knowledge_bases.models import KnowledgeBase

DEFAULT_KB_NAME = "General"
DEFAULT_KB_DESCRIPTION = "Base de conocimiento general por defecto."


def slugify_kb_id(name: str) -> str:
    """Genera un identificador slug estable a partir del nombre visible."""
    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")
    return slug[:64]


class KnowledgeBaseService:
    """Catálogo persistente de Knowledge Bases en SQLite + sincronización con ChromaDB."""

    def __init__(self, vectorstore: Chroma, settings: Settings) -> None:
        self._vs = vectorstore
        self._settings = settings
        self._db_path = settings.catalog_path
        self._init_catalog()
        self.migrate_legacy_data()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_catalog(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )
            default_id = self._settings.default_kb_id
            row = conn.execute(
                "SELECT id FROM knowledge_bases WHERE id = ?", (default_id,)
            ).fetchone()
            if not row:
                now = datetime.now(UTC).isoformat()
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_bases (id, name, description, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (default_id, DEFAULT_KB_NAME, DEFAULT_KB_DESCRIPTION, now),
                )
            conn.commit()

    def migrate_legacy_data(self) -> None:
        """Asigna archivos y vectores preexistentes sin KB a la base por defecto ('general')."""
        default_id = self._settings.default_kb_id

        # 1. Migrar archivos sueltos en la raíz de docs_dir hacia docs_dir/general/
        docs_dir = self._settings.docs_dir
        if docs_dir.is_dir():
            target_dir = docs_dir / default_id
            for item in list(docs_dir.iterdir()):
                if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dest = target_dir / item.name
                    if not dest.exists():
                        shutil.move(str(item), str(dest))

        # 2. Migrar chunks en ChromaDB que aún no tengan knowledge_base_id
        res = self._vs.get(include=["metadatas"])
        ids = res.get("ids") or []
        metadatas = res.get("metadatas") or []
        to_update_ids: list[str] = []
        to_update_metas: list[dict] = []
        for cid, meta in zip(ids, metadatas, strict=False):
            if meta is not None and not meta.get("knowledge_base_id"):
                new_meta = {**meta, "knowledge_base_id": default_id}
                to_update_ids.append(cid)
                to_update_metas.append(new_meta)
        if to_update_ids:
            self._vs._collection.update(ids=to_update_ids, metadatas=to_update_metas)

    def _stats_by_kb(self) -> dict[str, tuple[int, int]]:
        """Devuelve {kb_id: (n_documents, n_chunks)} a partir de las metadatas en ChromaDB."""
        default_id = self._settings.default_kb_id
        metadatas = self._vs.get(include=["metadatas"]).get("metadatas") or []
        doc_sets: dict[str, set[str]] = {}
        chunk_counts: dict[str, int] = {}
        for m in metadatas:
            if not m:
                continue
            kb_id = m.get("knowledge_base_id") or default_id
            chunk_counts[kb_id] = chunk_counts.get(kb_id, 0) + 1
            if "doc_id" in m:
                doc_sets.setdefault(kb_id, set()).add(m["doc_id"])
        all_kbs = set(chunk_counts) | set(doc_sets)
        return {k: (len(doc_sets.get(k, set())), chunk_counts.get(k, 0)) for k in all_kbs}

    def list_knowledge_bases(self) -> list[KnowledgeBase]:
        stats = self._stats_by_kb()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, created_at FROM knowledge_bases "
                "ORDER BY CASE WHEN id = ? THEN 0 ELSE 1 END, created_at ASC",
                (self._settings.default_kb_id,),
            ).fetchall()
        result: list[KnowledgeBase] = []
        for r in rows:
            n_docs, n_chunks = stats.get(r["id"], (0, 0))
            result.append(
                KnowledgeBase(
                    id=r["id"],
                    name=r["name"],
                    description=r["description"],
                    created_at=r["created_at"],
                    documents=n_docs,
                    chunks=n_chunks,
                )
            )
        return result

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, description, created_at FROM knowledge_bases WHERE id = ?",
                (kb_id,),
            ).fetchone()
        if not row:
            raise KnowledgeBaseNotFoundError(f"No existe la base de conocimiento '{kb_id}'.")
        stats = self._stats_by_kb()
        n_docs, n_chunks = stats.get(kb_id, (0, 0))
        return KnowledgeBase(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            documents=n_docs,
            chunks=n_chunks,
        )

    def ensure_exists(self, kb_id: str) -> KnowledgeBase:
        return self.get_knowledge_base(kb_id)

    def _check_writable(self) -> None:
        if self._settings.read_only_mode:
            raise ReadOnlyModeError(
                "El servicio está en modo de solo lectura (READ_ONLY_MODE=true)."
            )

    def create_knowledge_base(
        self, name: str, description: str = "", kb_id: str | None = None
    ) -> KnowledgeBase:
        self._check_writable()
        clean_name = name.strip()
        if not clean_name:
            raise InvalidKnowledgeBaseError("El nombre de la base no puede estar vacío.")
        resolved_id = slugify_kb_id(kb_id if kb_id is not None else clean_name)
        if not resolved_id:
            raise InvalidKnowledgeBaseError(
                "No se pudo generar un identificador válido para la base de conocimiento."
            )

        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, name FROM knowledge_bases WHERE id = ? OR lower(name) = lower(?)",
                (resolved_id, clean_name),
            ).fetchone()
            if existing:
                raise DuplicateKnowledgeBaseError(
                    f"Ya existe una base con id '{resolved_id}' o nombre '{clean_name}'."
                )
            conn.execute(
                "INSERT INTO knowledge_bases (id, name, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (resolved_id, clean_name, description.strip(), now),
            )
            conn.commit()

        (self._settings.docs_dir / resolved_id).mkdir(parents=True, exist_ok=True)
        return KnowledgeBase(
            id=resolved_id,
            name=clean_name,
            description=description.strip(),
            created_at=now,
            documents=0,
            chunks=0,
        )

    def delete_knowledge_base(self, kb_id: str) -> None:
        self._check_writable()
        if kb_id == self._settings.default_kb_id:
            raise InvalidKnowledgeBaseError(f"No se puede eliminar la base por defecto '{kb_id}'.")
        self.get_knowledge_base(kb_id)  # verifica existencia (404 si no existe)

        # 1. Eliminar vectores asociados en ChromaDB
        existing = self._vs.get(where={"knowledge_base_id": kb_id}, include=[])
        if existing["ids"]:
            self._vs.delete(ids=existing["ids"])

        # 2. Eliminar directorio físico de documentos de esta KB
        kb_dir = (self._settings.docs_dir / kb_id).resolve()
        docs_root = self._settings.docs_dir.resolve()
        if kb_dir.is_dir() and kb_dir.is_relative_to(docs_root):
            shutil.rmtree(kb_dir, ignore_errors=True)

        # 3. Eliminar registro del catálogo SQLite
        with self._connect() as conn:
            conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
            conn.commit()
