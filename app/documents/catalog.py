import sqlite3
from datetime import UTC, datetime

from langchain_chroma import Chroma

from app.config import Settings


def init_documents_table(conn: sqlite3.Connection) -> None:
    """Crea la tabla documents y sus índices si no existen."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            file_type TEXT NOT NULL,
            chunks INTEGER NOT NULL DEFAULT 0,
            pages INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(knowledge_base_id)")


def sync_documents_catalog(
    conn: sqlite3.Connection,
    vectorstore: Chroma,
    settings: Settings,
    force: bool = False,
) -> None:
    """Sincroniza el catálogo de SQLite con ChromaDB si difieren en cantidad de chunks."""
    init_documents_table(conn)

    chroma_count = vectorstore._collection.count()
    row = conn.execute("SELECT COALESCE(SUM(chunks), 0) FROM documents").fetchone()
    sqlite_chunk_count = row[0] if row else 0

    if not force and chroma_count == sqlite_chunk_count:
        return

    res = vectorstore.get(include=["metadatas"])
    metadatas = res.get("metadatas") or []

    docs: dict[str, dict] = {}
    default_kb = settings.default_kb_id

    for m in metadatas:
        if not m or "doc_id" not in m:
            continue
        doc_id = m["doc_id"]
        kb_id = m.get("knowledge_base_id") or default_kb
        source = m.get("source", "")
        if doc_id not in docs:
            created_at = m.get("created_at") or ""
            if not created_at and source:
                kb_dir = settings.docs_dir / kb_id
                fpath = kb_dir / source
                if not fpath.is_file():
                    fpath = settings.docs_dir / source
                if fpath.is_file():
                    try:
                        mtime = fpath.stat().st_mtime
                        created_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
                    except Exception:
                        created_at = ""
            if not created_at:
                created_at = datetime.now(UTC).isoformat()

            docs[doc_id] = {
                "doc_id": doc_id,
                "knowledge_base_id": kb_id,
                "source": source,
                "title": m.get("title", source),
                "file_type": m.get("file_type", ""),
                "chunks": 0,
                "pages": None,
                "created_at": created_at,
            }
        info = docs[doc_id]
        info["chunks"] += 1
        if m.get("page") is not None:
            info["pages"] = max(info["pages"] or 0, int(m["page"]))

    conn.execute("DELETE FROM documents")
    for doc in docs.values():
        conn.execute(
            """
            INSERT INTO documents
            (doc_id, knowledge_base_id, source, title, file_type, chunks, pages, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc["doc_id"],
                doc["knowledge_base_id"],
                doc["source"],
                doc["title"],
                doc["file_type"],
                doc["chunks"],
                doc["pages"],
                doc["created_at"],
            ),
        )
    conn.commit()
