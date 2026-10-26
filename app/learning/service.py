import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from graphlib import CycleError, TopologicalSorter
from typing import Any

import httpx
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.conversations.service import ConversationService
from app.exceptions import (
    InvalidLearningActionError,
    LearningSessionNotFoundError,
    ReadOnlyModeError,
)
from app.rag.service import RAGService, SourceHit, format_context


class LearningService:
    """State machine and SQLite persistence for the interactive Learn experience."""

    def __init__(
        self, settings: Settings, rag: RAGService, conversations: ConversationService
    ) -> None:
        self.settings = settings
        self.rag = rag
        self.conversations = conversations
        self.db_path = settings.catalog_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS learning_sessions_v2 (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL UNIQUE,
                    knowledge_base_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    state TEXT NOT NULL,
                    previous_state TEXT,
                    strict_mode INTEGER NOT NULL DEFAULT 0,
                    current_node_index INTEGER NOT NULL DEFAULT 0,
                    score REAL,
                    plan_json TEXT,
                    content TEXT NOT NULL DEFAULT '',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    feedback_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS learning_nodes_v2 (
                    id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    PRIMARY KEY (session_id, id),
                    UNIQUE (session_id, position),
                    FOREIGN KEY (session_id) REFERENCES learning_sessions_v2(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS learning_attempts_v2 (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    node_id TEXT,
                    kind TEXT NOT NULL,
                    question TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    correct_option_id TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    answer TEXT,
                    correct INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES learning_sessions_v2(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_learning_attempts_v2_session
                    ON learning_attempts_v2(session_id, created_at);
                """
            )

    def _check_writable(self) -> None:
        if self.settings.read_only_mode:
            raise ReadOnlyModeError("Aprender no puede guardar progreso con READ_ONLY_MODE activo.")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _parse_json(text: str | None, default: Any) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return default

    def _row(self, session_id: str) -> sqlite3.Row:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM learning_sessions_v2 WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise LearningSessionNotFoundError(f"No existe la sesión '{session_id}'.")
        return row

    def _get_pending_attempt(self, session_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM learning_attempts_v2 WHERE session_id = ? AND correct IS NULL "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()

    def get_session(self, session_id: str) -> dict[str, Any]:
        row = self._row(session_id)
        with self._connect() as conn:
            node_rows = conn.execute(
                "SELECT id, position, data_json, status FROM learning_nodes_v2 "
                "WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        nodes = []
        for item in node_rows:
            data = self._parse_json(item["data_json"], {})
            data.update(id=item["id"], status=item["status"], position=item["position"])
            nodes.append(data)
        attempt = self._get_pending_attempt(session_id)
        quiz = None
        if attempt:
            quiz = {
                "id": attempt["id"],
                "kind": attempt["kind"],
                "node_id": attempt["node_id"],
                "question": attempt["question"],
                "options": self._parse_json(attempt["options_json"], []),
            }
        plan = self._parse_json(row["plan_json"], None)
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "knowledge_base_id": row["knowledge_base_id"],
            "goal": row["goal"],
            "state": row["state"],
            "previous_state": row["previous_state"],
            "strict_mode": bool(row["strict_mode"]),
            "current_node_index": row["current_node_index"],
            "score": row["score"],
            "content": row["content"],
            "plan": plan,
            "nodes": nodes,
            "quiz": quiz,
            "feedback": self._parse_json(row["feedback_json"], None),
            "sources": self._parse_json(row["sources_json"], []),
        }

    def get_by_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM learning_sessions_v2 WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
        return self.get_session(row["id"]) if row else None

    def _save_fields(self, session_id: str, **fields: Any) -> None:
        allowed = {
            "goal",
            "state",
            "previous_state",
            "current_node_index",
            "score",
            "plan_json",
            "content",
            "sources_json",
            "feedback_json",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = self._now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE learning_sessions_v2 SET {assignments} WHERE id = ?",
                (*updates.values(), session_id),
            )
            conn.commit()

    def _invoke_json(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        # JSON examples in system instructions are literal braces, not template variables.
        literal_system = system.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages(
            [("system", literal_system), ("human", "{payload}")]
        )
        response = (prompt | self.rag.llm).invoke(
            {"payload": json.dumps(payload, ensure_ascii=False)}
        )
        raw = str(response.content).strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        try:
            value = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise InvalidLearningActionError(
                "El modelo devolvió contenido inválido; el progreso no se avanzó. Intentá de nuevo."
            ) from exc
        if not isinstance(value, dict):
            raise InvalidLearningActionError("La respuesta pedagógica debe ser un objeto JSON.")
        return value

    @staticmethod
    def _validate_quiz(value: dict[str, Any]) -> dict[str, Any]:
        question = str(value.get("question", "")).strip()
        options = value.get("options")
        answer = str(value.get("correct_option_id", "")).strip()
        explanation = str(value.get("explanation", "")).strip()
        if not question or not isinstance(options, list) or len(options) < 2 or not explanation:
            raise InvalidLearningActionError("El modelo no generó un quiz completo.")
        normalized = []
        ids = set()
        for option in options:
            if not isinstance(option, dict):
                raise InvalidLearningActionError(
                    "Las opciones del quiz tienen un formato inválido."
                )
            option_id = str(option.get("id", "")).strip()
            text = str(option.get("text", "")).strip()
            if not option_id or not text or option_id in ids:
                raise InvalidLearningActionError("Las opciones del quiz deben tener IDs únicos.")
            ids.add(option_id)
            normalized.append({"id": option_id, "text": text})
        if answer not in ids:
            raise InvalidLearningActionError("La clave de respuesta no coincide con las opciones.")
        return {
            "question": question,
            "options": normalized,
            "correct_option_id": answer,
            "explanation": explanation,
        }

    @staticmethod
    def _validate_evidence_citations(text: str, sources: list[dict]) -> None:
        cited = set(re.findall(r"\[(\d+|W\d+)\]", text))
        valid = {str(source["id"]) for source in sources}
        if not cited or not cited.issubset(valid):
            raise InvalidLearningActionError(
                "La respuesta debe citar evidencia recuperada con identificadores válidos."
            )

    def _retrieve_goal(self, goal: str, kb_id: str) -> tuple[list[SourceHit], str]:
        step_back = self.rag._step_back_question(goal) if self.settings.enable_step_back else None
        hits, sources, _ = self.rag._retrieve(goal, kb_id=kb_id, step_back_query=step_back)
        return sources, format_context(hits)

    def start_session(self, conversation_id: str, goal: str, strict_mode: bool = False) -> dict:
        self._check_writable()
        conv = self.conversations.get_conversation(conversation_id)
        if self.get_by_conversation(conversation_id) is not None:
            raise InvalidLearningActionError(
                "Esta conversación ya tiene una sesión de Aprender; retomá la sesión existente."
            )
        kb_id = conv.last_kb_id or self.settings.default_kb_id
        _, kb_name, is_empty = self.rag._resolve_kb(kb_id)
        sources, context = self._retrieve_goal(goal, kb_id) if not is_empty else ([], "")
        state = "diagnosing"
        content = ""
        quiz: dict[str, Any] | None = None
        if not sources:
            state = "blocked" if strict_mode else "awaiting_web_consent"
            content = f"No encontré evidencia suficiente en la KB «{kb_name}». " + (
                "En Modo Estricto no consultaré fuentes externas; agregá material a la KB "
                "o desactivalo."
                if strict_mode
                else "Podés autorizar una búsqueda web para complementar la KB."
            )
        else:
            generated = self._invoke_json(
                "Sos un tutor. Generá UNA pregunta diagnóstica de opción múltiple basada solo "
                "en el contexto. No asumas que el alumno sabe el tema. Devolvé JSON con question, "
                "options [{id,text}] (3 o 4 opciones), correct_option_id y explanation. Las "
                "opciones deben ser afirmaciones breves y paralelas. La explicación debe citar "
                "la evidencia pertinente con su identificador exacto, por ejemplo [1].",
                {"goal": goal, "context": context},
            )
            quiz = self._validate_quiz(generated)
            self._validate_evidence_citations(
                quiz["explanation"], [source.__dict__ for source in sources]
            )
        session_id = uuid.uuid4().hex
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO learning_sessions_v2 (id, conversation_id, knowledge_base_id, goal, "
                "state, strict_mode, content, sources_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    conversation_id,
                    kb_id,
                    goal.strip(),
                    state,
                    int(strict_mode),
                    content,
                    json.dumps([s.__dict__ for s in sources], ensure_ascii=False),
                    now,
                    now,
                ),
            )
            if quiz:
                self._insert_attempt(conn, session_id, None, "diagnostic", quiz)
            conn.commit()
        self._record_chat(
            conversation_id, f"Aprender: {goal}", content or quiz["question"], sources, kb_id
        )
        return self.get_session(session_id)

    @staticmethod
    def _insert_attempt(
        conn: sqlite3.Connection,
        session_id: str,
        node_id: str | None,
        kind: str,
        quiz: dict[str, Any],
    ) -> str:
        attempt_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO learning_attempts_v2 (id, session_id, node_id, kind, question, "
            "options_json, correct_option_id, explanation, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                session_id,
                node_id,
                kind,
                quiz["question"],
                json.dumps(quiz["options"], ensure_ascii=False),
                quiz["correct_option_id"],
                quiz["explanation"],
                LearningService._now(),
            ),
        )
        return attempt_id

    def _record_chat(
        self, conversation_id: str, question: str, answer: str, sources: list, kb_id: str
    ) -> None:
        self.conversations.record_turn(
            conversation_id,
            question,
            answer,
            grounded=bool(sources),
            reason=None if sources else "below_threshold",
            sources=[s.__dict__ if hasattr(s, "__dict__") else s for s in sources],
            retrieved=[s.__dict__ if hasattr(s, "__dict__") else s for s in sources],
            kb_id=kb_id,
            pedagogical_mode="learn",
        )

    def _plan(self, session_id: str, row: sqlite3.Row, diagnostic: dict[str, Any]) -> dict:
        sources = self._parse_json(row["sources_json"], [])
        context = "\n\n".join(f"[{s['id']}] {s['source']}: {s['snippet']}" for s in sources)
        result = self._invoke_json(
            "Diseñá un plan de aprendizaje pequeño, basado solo en la evidencia dada. "
            "Devolvé JSON: {approach:string,nodes:[{id:string,title:string,rationale:string,"
            "prerequisites:[id],source_ids:[integer|string]}]}. Usá 2 a 6 nodos ordenados; "
            "cada source_id debe existir en la evidencia; no inventes premisas. "
            "El plan es una propuesta, no enseñes.",
            {"goal": row["goal"], "diagnostic": diagnostic, "evidence": context},
        )
        nodes = result.get("nodes")
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 8:
            raise InvalidLearningActionError("No se pudo validar el mapa de conceptos.")
        valid_source_ids = {str(s["id"]) for s in sources}
        ids: set[str] = set()
        normalized = []
        for pos, node in enumerate(nodes):
            if not isinstance(node, dict):
                raise InvalidLearningActionError("Un concepto del mapa tiene formato inválido.")
            node_id = str(node.get("id", f"n{pos + 1}")).strip()
            title = str(node.get("title", "")).strip()
            prereqs = node.get("prerequisites", [])
            source_ids = node.get("source_ids", [])
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]+", node_id)
                or node_id in ids
                or not title
                or not isinstance(prereqs, list)
            ):
                raise InvalidLearningActionError("El mapa contiene nodos inválidos o repetidos.")
            ids.add(node_id)
            if not isinstance(source_ids, list) or any(
                str(i) not in valid_source_ids for i in source_ids
            ):
                raise InvalidLearningActionError("El mapa cita una fuente que no fue recuperada.")
            normalized.append(
                {
                    "id": node_id,
                    "title": title,
                    "rationale": str(node.get("rationale", "")),
                    "prerequisites": [str(p) for p in prereqs],
                    "source_ids": source_ids,
                }
            )
        try:
            sorter = TopologicalSorter({n["id"]: set(n["prerequisites"]) for n in normalized})
            topological_order = tuple(sorter.static_order())
        except (CycleError, KeyError) as exc:
            raise InvalidLearningActionError(
                "El mapa tiene ciclos o prerrequisitos inexistentes."
            ) from exc
        normalized_by_id = {node["id"]: node for node in normalized}
        normalized = [normalized_by_id[node_id] for node_id in topological_order]
        plan = {"approach": str(result.get("approach", "")), "nodes": normalized}
        with self._connect() as conn:
            conn.execute("DELETE FROM learning_nodes_v2 WHERE session_id = ?", (session_id,))
            for pos, node in enumerate(normalized):
                conn.execute(
                    "INSERT INTO learning_nodes_v2 (id, session_id, position, title, data_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        node["id"],
                        session_id,
                        pos,
                        node["title"],
                        json.dumps(node, ensure_ascii=False),
                    ),
                )
            conn.execute(
                "UPDATE learning_sessions_v2 SET state='plan_pending', plan_json=?, content=?, "
                "updated_at=? WHERE id=?",
                (json.dumps(plan, ensure_ascii=False), plan["approach"], self._now(), session_id),
            )
            conn.commit()
        return plan

    def _teach_node(
        self, session_id: str, index: int, kind: str = "node", preserve_feedback: bool = False
    ) -> None:
        session = self.get_session(session_id)
        if index >= len(session["nodes"]):
            self._start_final_assessment(session_id)
            return
        node = session["nodes"][index]
        sources = session["sources"]
        evidence = "\n\n".join(
            f"[{s['id']}] {s['source']}: {s['snippet']}"
            for s in sources
            if not node["source_ids"] or s["id"] in node["source_ids"]
        )
        allowed_sources = [
            source
            for source in sources
            if not node["source_ids"] or source["id"] in node["source_ids"]
        ]
        citation_ids = ", ".join(f"[{source['id']}]" for source in allowed_sources)
        tool_hint, _ = self.rag._detect_and_run_tools(node["title"] + " " + session["goal"])
        prompt = (
            "Enseñá un solo concepto: motivá por qué se necesita, derivá los pasos, conectalo "
            "con prerrequisitos y terminá con una pregunta de opción múltiple. Basate solo en "
            "la evidencia indicada. En content y explanation incluye una cita con uno de estos "
            f"IDs exactos: {citation_ids}. Cita fuentes KB como [n] y web como [Wn]. "
            "Devolvé JSON {content,question,options:[{id,text}],correct_option_id,explanation}. "
            "No reveles la clave en content."
        )
        payload = {
            "goal": session["goal"],
            "node": node,
            "evidence": evidence,
            "tool_result": tool_hint,
        }
        generated = self._invoke_json(prompt, payload)
        quiz = self._validate_quiz(generated)
        content = str(generated.get("content", "")).strip()
        if not content:
            raise InvalidLearningActionError(
                "El modelo no generó una explicación para el concepto."
            )
        try:
            self._validate_evidence_citations(content, allowed_sources)
            self._validate_evidence_citations(quiz["explanation"], allowed_sources)
        except InvalidLearningActionError:
            retry_prompt = (
                prompt
                + " Es obligatorio citar al menos una de las etiquetas exactas de evidencia en "
                "cada campo content y explanation; no omitas las etiquetas ni inventes otras."
            )
            generated = self._invoke_json(retry_prompt, payload)
            quiz = self._validate_quiz(generated)
            content = str(generated.get("content", "")).strip()
            self._validate_evidence_citations(content, allowed_sources)
            self._validate_evidence_citations(quiz["explanation"], allowed_sources)
        with self._connect() as conn:
            feedback_sql = "" if preserve_feedback else ", feedback_json=NULL"
            conn.execute(
                "UPDATE learning_sessions_v2 SET state=?, current_node_index=?, content=?"
                f"{feedback_sql}, updated_at=? WHERE id=?",
                (
                    "final_assessment" if kind == "final" else "teaching",
                    index,
                    content,
                    self._now(),
                    session_id,
                ),
            )
            self._insert_attempt(conn, session_id, node["id"], kind, quiz)
            conn.commit()

    def _start_final_assessment(self, session_id: str) -> None:
        self._teach_node(session_id, 0, kind="final")

    def _reset_attempt(self, session_id: str, attempt_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE learning_attempts_v2 SET answer=NULL, correct=NULL WHERE id=?",
                (attempt_id,),
            )
            conn.execute(
                "UPDATE learning_sessions_v2 SET feedback_json=NULL, updated_at=? WHERE id=?",
                (self._now(), session_id),
            )
            conn.commit()

    def _advance_final(self, session_id: str) -> None:
        session = self.get_session(session_id)
        idx = session["current_node_index"] + 1
        if idx < len(session["nodes"]):
            self._teach_node(session_id, idx, kind="final")
            return
        latest_by_node: dict[str, bool] = {}
        with self._connect() as conn:
            attempts = conn.execute(
                "SELECT node_id, correct, created_at FROM learning_attempts_v2 WHERE session_id=? "
                "AND kind='final' AND correct IS NOT NULL ORDER BY created_at",
                (session_id,),
            ).fetchall()
        for attempt in attempts:
            latest_by_node[attempt["node_id"]] = bool(attempt["correct"])
        score = sum(latest_by_node.values()) / max(len(session["nodes"]), 1)
        state = "completed" if score >= 0.8 else "review"
        with self._connect() as conn:
            conn.execute(
                "UPDATE learning_sessions_v2 SET state=?, score=?, content=?, updated_at=? "
                "WHERE id=?",
                (
                    state,
                    score,
                    "Evaluación aprobada."
                    if state == "completed"
                    else "Repasemos los conceptos que todavía cuestan.",
                    self._now(),
                    session_id,
                ),
            )
            for node in session["nodes"]:
                conn.execute(
                    "UPDATE learning_nodes_v2 SET status=? WHERE session_id=? AND id=?",
                    (
                        "mastered" if latest_by_node.get(node["id"]) else "review",
                        session_id,
                        node["id"],
                    ),
                )
            conn.commit()

    def act(self, session_id: str, action: str, answer: str | None = None) -> dict:
        self._check_writable()
        session = self.get_session(session_id)
        state = session["state"]
        answer = (answer or "").strip()
        if action == "pause" and state not in {"paused", "completed", "blocked"}:
            self._save_fields(
                session_id,
                previous_state=state,
                state="paused",
                content="Sesión pausada. Podés retomarla cuando quieras.",
            )
        elif action == "resume" and state == "paused":
            previous = session["previous_state"] or "plan_pending"
            self._save_fields(session_id, state=previous, previous_state=None)
        elif action == "approve_web" and state == "awaiting_web_consent":
            self._check_writable()
            api_key = (
                self.settings.tavily_api_key.get_secret_value()
                if self.settings.tavily_api_key
                else ""
            )
            if not api_key:
                self._save_fields(
                    session_id,
                    state="blocked",
                    content=(
                        "La búsqueda web no está configurada. Agregá material a la KB "
                        "para continuar."
                    ),
                )
            else:
                try:
                    response = httpx.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": api_key,
                            "query": session["goal"],
                            "search_depth": "basic",
                            "max_results": 5,
                        },
                        timeout=self.settings.llm_timeout,
                    )
                    response.raise_for_status()
                    results = response.json().get("results", [])
                except (httpx.HTTPError, ValueError):
                    self._save_fields(
                        session_id,
                        state="blocked",
                        content=(
                            "La búsqueda web falló. Podés agregar material a la KB y continuar."
                        ),
                    )
                    results = []
                web_sources = [
                    {
                        "id": f"W{i + 1}",
                        "kind": "web",
                        "source": item.get("title", "Fuente web"),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", "")[:1200],
                        "score": float(item.get("score", 0.0)),
                        "knowledge_base_id": session["knowledge_base_id"],
                    }
                    for i, item in enumerate(results)
                    if item.get("url") and item.get("content")
                ]
                if not web_sources:
                    self._save_fields(
                        session_id,
                        state="blocked",
                        content="La búsqueda autorizada no encontró evidencia utilizable.",
                    )
                else:
                    merged = session["sources"] + web_sources
                    generated = self._invoke_json(
                        "Generá UNA pregunta de opción múltiple para diagnosticar conocimientos "
                        "del objetivo usando la evidencia provista. Devuelve JSON question, "
                        "options [{id,text}], correct_option_id, explanation.",
                        {"goal": session["goal"], "evidence": web_sources},
                    )
                    quiz = self._validate_quiz(generated)
                    self._validate_evidence_citations(quiz["explanation"], web_sources)
                    with self._connect() as conn:
                        conn.execute(
                            "UPDATE learning_sessions_v2 SET sources_json=?, state='diagnosing', "
                            "content=?, updated_at=? WHERE id=?",
                            (
                                json.dumps(merged, ensure_ascii=False),
                                "Fuentes web recuperadas. Respondé la pregunta diagnóstica.",
                                self._now(),
                                session_id,
                            ),
                        )
                        self._insert_attempt(conn, session_id, None, "diagnostic", quiz)
                        conn.commit()
        elif action == "decline_web" and state == "awaiting_web_consent":
            self._save_fields(
                session_id,
                state="blocked",
                content=(
                    "Sesión en pausa: agregá material a la KB para aprender este tema "
                    "con fuentes verificables."
                ),
            )
        elif action == "answer_diagnostic" and state == "diagnosing":
            attempt = self._get_pending_attempt(session_id)
            row = self._row(session_id)
            if attempt:
                self._grade(session_id, attempt, answer)
                row = self._row(session_id)
            elif not row["feedback_json"]:
                raise InvalidLearningActionError("No hay una pregunta diagnóstica pendiente.")
            diagnostic = self._parse_json(row["feedback_json"], {})
            try:
                self._plan(session_id, row, diagnostic)
            except Exception:
                # A failed plan generation must leave the diagnostic answer retryable.
                if attempt:
                    with self._connect() as conn:
                        conn.execute(
                            "UPDATE learning_attempts_v2 SET answer=NULL, correct=NULL WHERE id=?",
                            (attempt["id"],),
                        )
                        conn.execute(
                            "UPDATE learning_sessions_v2 SET feedback_json=NULL WHERE id=?",
                            (session_id,),
                        )
                        conn.commit()
                raise
        elif action == "approve_plan" and state == "plan_pending":
            self._teach_node(session_id, 0)
        elif action == "revise_plan" and state == "plan_pending":
            if not answer:
                raise InvalidLearningActionError("Indicá qué querés cambiar del plan.")
            self._save_fields(
                session_id,
                goal=f"{session['goal']}\nAjuste solicitado: {answer}",
                state="diagnosing",
            )
            row = self._row(session_id)
            self._plan(session_id, row, {"requested_adjustment": answer})
        elif action == "answer_quiz" and state in {"teaching", "final_assessment"}:
            attempt = self._get_pending_attempt(session_id)
            feedback = self._grade(session_id, attempt, answer)
            if attempt["kind"] == "final":
                try:
                    self._advance_final(session_id)
                except Exception:
                    self._reset_attempt(session_id, attempt["id"])
                    raise
            elif feedback["correct"]:
                self._save_fields(session_id, state="awaiting_advance", content=feedback["message"])
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE learning_nodes_v2 SET status='mastered' "
                        "WHERE session_id=? AND id=?",
                        (session_id, attempt["node_id"]),
                    )
                    conn.commit()
            else:
                try:
                    self._teach_node(
                        session_id, session["current_node_index"], preserve_feedback=True
                    )
                except Exception:
                    self._reset_attempt(session_id, attempt["id"])
                    raise
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE learning_nodes_v2 SET status='review' WHERE session_id=? AND id=?",
                        (session_id, attempt["node_id"]),
                    )
                    conn.commit()
        elif action == "advance" and state in {"awaiting_advance", "review"}:
            if state == "review":
                self._start_final_assessment(session_id)
            else:
                self._teach_node(session_id, session["current_node_index"] + 1)
        elif action in {"hint", "check", "solve"} and state in {"teaching", "checking"}:
            if action == "check" and not answer:
                raise InvalidLearningActionError("Compartí tu procedimiento para poder corregirlo.")
            node = session["nodes"][session["current_node_index"]]
            prompt = {
                "hint": "Dá una pista progresiva breve sin revelar resultado ni clave del quiz.",
                "check": (
                    "Corregí el procedimiento del estudiante: separá lo correcto y el error, "
                    "citando evidencia."
                ),
                "solve": (
                    "Resolvé el ejercicio paso a paso citando reglas; usa el resultado "
                    "determinista provisto."
                ),
            }[action]
            verification, _ = self.rag._detect_and_run_tools(answer or session["goal"])
            allowed_sources = [
                source
                for source in session["sources"]
                if not node["source_ids"] or source["id"] in node["source_ids"]
            ]
            citation_ids = ", ".join(f"[{source['id']}]" for source in allowed_sources)
            prompt_text = (
                f"{prompt} Devolvé JSON {{content:string}}. No inventes citas. "
                f"Incluí una cita textual válida con uno de estos IDs exactos: {citation_ids}."
            )
            payload = {
                "goal": session["goal"],
                "node": node,
                "answer": answer,
                "sources": allowed_sources,
                "tool_result": verification,
            }
            value = self._invoke_json(prompt_text, payload)
            content = str(value.get("content", "")).strip()
            try:
                self._validate_evidence_citations(content, allowed_sources)
            except InvalidLearningActionError:
                value = self._invoke_json(
                    prompt_text
                    + " Es obligatorio citar una de las etiquetas exactas dentro de content; "
                    "no omitas la cita ni inventes otro ID.",
                    payload,
                )
                content = str(value.get("content", "")).strip()
            self._validate_evidence_citations(content, allowed_sources)
            self._save_fields(session_id, content=content)
        elif action == "finish" and state not in {"completed", "blocked"}:
            self._save_fields(
                session_id, state="completed", content="Sesión cerrada por el estudiante."
            )
        else:
            raise InvalidLearningActionError(
                f"La acción '{action}' no corresponde al estado '{state}'."
            )

        updated = self.get_session(session_id)
        try:
            visible = updated["content"] or (updated["quiz"] or {}).get("question", "")
            self._record_chat(
                updated["conversation_id"],
                f"{action}: {answer}"[:1000],
                visible,
                updated["sources"],
                updated["knowledge_base_id"],
            )
        except Exception:
            # Session state is authoritative; a transcript hiccup must not roll it back.
            pass
        return updated

    def _grade(self, session_id: str, attempt: sqlite3.Row | None, answer: str) -> dict[str, Any]:
        if attempt is None:
            raise InvalidLearningActionError("No hay una pregunta pendiente para responder.")
        options = self._parse_json(attempt["options_json"], [])
        normalized = answer.strip().upper()
        if normalized not in {str(option["id"]).upper() for option in options}:
            raise InvalidLearningActionError(
                "Respondé con el identificador de una de las opciones."
            )
        correct = normalized == attempt["correct_option_id"].upper()
        feedback = {
            "correct": correct,
            "selected_option_id": answer,
            "correct_option_id": attempt["correct_option_id"],
            "explanation": attempt["explanation"],
            "message": ("¡Correcto! " if correct else "Todavía no. ") + attempt["explanation"],
        }
        with self._connect() as conn:
            conn.execute(
                "UPDATE learning_attempts_v2 SET answer=?, correct=? "
                "WHERE id=? AND correct IS NULL",
                (answer, int(correct), attempt["id"]),
            )
            conn.execute(
                "UPDATE learning_sessions_v2 SET feedback_json=?, updated_at=? WHERE id=?",
                (json.dumps(feedback, ensure_ascii=False), self._now(), session_id),
            )
            conn.commit()
        return feedback
