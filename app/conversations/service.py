import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime

from app.config import Settings
from app.conversations.models import (
    ConversationDetail,
    ConversationSummary,
    MessageRecord,
)
from app.exceptions import (
    ConversationNotFoundError,
    InvalidConversationError,
    ReadOnlyModeError,
)

DEFAULT_CONVERSATION_TITLE = "Nueva conversación"


class ConversationService:
    """Servicio de persistencia y gestión de conversaciones en SQLite."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._db_path = settings.catalog_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    last_kb_id TEXT NOT NULL DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    grounded INTEGER NOT NULL DEFAULT 1,
                    reason TEXT,
                    message TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    retrieved_json TEXT NOT NULL DEFAULT '[]',
                    knowledge_base_id TEXT NOT NULL DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                    ON messages(conversation_id, id ASC);
                """
            )
            conn.commit()

    def _check_writable(self) -> None:
        if self._settings.read_only_mode:
            raise ReadOnlyModeError(
                "El servicio está en modo de solo lectura (READ_ONLY_MODE=true)."
            )

    def create_conversation(
        self, title: str | None = None, kb_id: str | None = None
    ) -> ConversationSummary:
        self._check_writable()
        conv_id = uuid.uuid4().hex
        clean_title = (title.strip() if title and title.strip() else DEFAULT_CONVERSATION_TITLE)[
            :120
        ]
        resolved_kb_id = kb_id or self._settings.default_kb_id
        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, last_kb_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, clean_title, resolved_kb_id, now, now),
            )
            conn.commit()

        return ConversationSummary(
            id=conv_id,
            title=clean_title,
            last_kb_id=resolved_kb_id,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def list_conversations(self) -> list[ConversationSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.last_kb_id, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.created_at DESC
                """
            ).fetchall()

        return [
            ConversationSummary(
                id=r["id"],
                title=r["title"],
                last_kb_id=r["last_kb_id"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )
            for r in rows
        ]

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, last_kb_id, created_at, updated_at
                FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            ).fetchone()
            if not row:
                raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'.")

            msg_rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, grounded, reason, message,
                       sources_json, retrieved_json, knowledge_base_id, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        messages = [
            MessageRecord(
                id=m["id"],
                conversation_id=m["conversation_id"],
                role=m["role"],
                content=m["content"],
                grounded=bool(m["grounded"]),
                reason=m["reason"],
                message=m["message"],
                sources=json.loads(m["sources_json"]) if m["sources_json"] else [],
                retrieved=json.loads(m["retrieved_json"]) if m["retrieved_json"] else [],
                knowledge_base_id=m["knowledge_base_id"],
                created_at=m["created_at"],
            )
            for m in msg_rows
        ]

        return ConversationDetail(
            id=row["id"],
            title=row["title"],
            last_kb_id=row["last_kb_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=len(messages),
            messages=messages,
        )

    def rename_conversation(self, conversation_id: str, title: str) -> ConversationSummary:
        self._check_writable()
        clean_title = title.strip()
        if not clean_title:
            raise InvalidConversationError("El título de la conversación no puede estar vacío.")
        clean_title = clean_title[:120]
        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, last_kb_id, created_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'.")

            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, now, conversation_id),
            )
            count_row = conn.execute(
                "SELECT COUNT(id) AS cnt FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            msg_count = count_row["cnt"] if count_row else 0
            conn.commit()

        return ConversationSummary(
            id=conversation_id,
            title=clean_title,
            last_kb_id=row["last_kb_id"],
            created_at=row["created_at"],
            updated_at=now,
            message_count=msg_count,
        )

    def delete_conversation(self, conversation_id: str) -> None:
        self._check_writable()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'.")
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

    def record_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str | None,
        grounded: bool,
        reason: str | None = None,
        message: str | None = None,
        sources: list[dict] | None = None,
        retrieved: list[dict] | None = None,
        kb_id: str = "general",
    ) -> None:
        if self._settings.read_only_mode:
            return

        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, last_kb_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'.")

            if row["last_kb_id"] != kb_id:
                raise InvalidConversationError(
                    f"La conversación '{conversation_id}' pertenece a la base "
                    f"'{row['last_kb_id']}', no a '{kb_id}'."
                )

            # Insertar mensaje de usuario
            conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, grounded, reason, message,
                    sources_json, retrieved_json, knowledge_base_id, created_at
                ) VALUES (?, 'user', ?, 1, NULL, NULL, '[]', '[]', ?, ?)
                """,
                (conversation_id, question, kb_id, now),
            )

            # Insertar mensaje del asistente
            assistant_content = (
                answer if (answer is not None and answer.strip()) else (message or "")
            )
            conn.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, grounded, reason, message,
                    sources_json, retrieved_json, knowledge_base_id, created_at
                ) VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    assistant_content,
                    int(bool(grounded)),
                    reason,
                    message,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(retrieved or [], ensure_ascii=False),
                    kb_id,
                    now,
                ),
            )

            # Auto-titulación si la conversación mantiene el título por defecto
            new_title = row["title"]
            if row["title"] == DEFAULT_CONVERSATION_TITLE:
                auto_title = re.sub(r"\s+", " ", question).strip()[:60]
                if auto_title:
                    new_title = auto_title

            conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_title, now, conversation_id),
            )
            conn.commit()

    def get_recent_history(
        self,
        conversation_id: str,
        max_exchanges: int | None = None,
        kb_id: str | None = None,
    ) -> list[tuple[str, str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                raise ConversationNotFoundError(f"No existe la conversación '{conversation_id}'.")

            limit_exchanges = (
                max_exchanges
                if max_exchanges is not None
                else self._settings.conversation_window_exchanges
            )
            if limit_exchanges <= 0:
                return []

            limit_messages = limit_exchanges * 2
            where_clause = "WHERE conversation_id = ?"
            params: list[str | int] = [conversation_id]
            if kb_id is not None:
                where_clause += " AND knowledge_base_id = ?"
                params.append(kb_id)
            params.append(limit_messages)

            msg_rows = conn.execute(
                f"""
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM messages
                    {where_clause}
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                params,
            ).fetchall()

        return [(r["role"], r["content"]) for r in msg_rows]
