import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from app.config import Settings
from app.conversations.service import ConversationService
from app.exceptions import EmptyIndexError, InvalidConversationError, LLMUnavailableError
from app.knowledge_bases.service import KnowledgeBaseService
from app.rag.citations import extract_citations, strip_invalid_citations
from app.rag.prompts import (
    CITATION_REMINDER,
    CONDENSE_PROMPT,
    CONVERSATIONAL_PROMPT,
    NO_EVIDENCE,
    PROMPT,
    STEP_BACK_PROMPT,
    build_professor_prompt,
    format_history,
)
from app.rag.tools.registry import ToolRegistry, get_default_registry

logger = logging.getLogger("rag")

_LLM_CONNECTION_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.ConnectError,
    httpx.TimeoutException,
    ConnectionError,
    TimeoutError,
)
_LLM_UNAVAILABLE_MSG = "El modelo de lenguaje no está disponible o agotó el tiempo de espera."

MESSAGES = {
    "below_threshold": "No se encontró evidencia suficientemente relevante en la documentación.",
    "empty_knowledge_base": "La base seleccionada no contiene documentos cargados.",
    "model_refused": "La documentación disponible no contiene información para responder esto.",
    "no_citations": (
        "No se pudo verificar la respuesta contra las fuentes recuperadas (faltan citas)."
    ),
    "missing_premise_strict": (
        "La base de conocimiento no contiene la convención o regla necesaria para resolver "
        "este problema y el Modo Estricto está activo."
    ),
}


@dataclass
class SourceHit:
    id: int
    source: str
    section: str | None
    page: int | None
    snippet: str
    score: float
    knowledge_base_id: str = "general"


@dataclass
class QueryResult:
    answer: str | None
    grounded: bool
    sources: list[SourceHit] = field(default_factory=list)
    retrieved: list[SourceHit] = field(default_factory=list)
    reason: str | None = None
    message: str | None = None
    knowledge_base_id: str = "general"
    conversation_id: str | None = None
    pedagogical_mode: str = "auto"
    strict_mode: bool = False
    verification_data: dict[str, Any] | None = None


def _to_hit(idx: int, doc: Document, score: float) -> SourceHit:
    m = doc.metadata
    return SourceHit(
        id=idx,
        source=m.get("source", "desconocido"),
        section=m.get("section"),
        page=m.get("page"),
        snippet=doc.page_content[:300],
        score=round(float(score), 4),
        knowledge_base_id=m.get("knowledge_base_id", "general"),
    )


def format_context(hits: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, _) in enumerate(hits, start=1):
        m = doc.metadata
        loc = " · ".join(str(x) for x in (m.get("source"), m.get("section"), m.get("page")) if x)
        blocks.append(f"[{i}] ({loc})\n{doc.page_content}")
    return "\n\n".join(blocks)


class RAGService:
    def __init__(
        self,
        vectorstore: Chroma,
        llm: BaseChatModel,
        settings: Settings,
        kb_service: KnowledgeBaseService | None = None,
        conversation_service: ConversationService | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.vs = vectorstore
        self.llm = llm
        self.settings = settings
        self.kb_service = kb_service
        self.conversation_service = conversation_service or ConversationService(settings)
        self.tool_registry = tool_registry or get_default_registry()

    def _condense_question(self, history: list[tuple[str, str]], question: str) -> str:
        if not history:
            return question
        try:
            msg = (CONDENSE_PROMPT | self.llm).invoke(
                {"history": format_history(history), "question": question}
            )
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "llm_error", "error": type(exc).__name__}))
            raise LLMUnavailableError(_LLM_UNAVAILABLE_MSG) from exc
        return str(msg.content).strip() or question

    async def _acondense_question(self, history: list[tuple[str, str]], question: str) -> str:
        if not history:
            return question
        try:
            msg = await (CONDENSE_PROMPT | self.llm).ainvoke(
                {"history": format_history(history), "question": question}
            )
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "llm_error", "error": type(exc).__name__}))
            raise LLMUnavailableError(_LLM_UNAVAILABLE_MSG) from exc
        return str(msg.content).strip() or question

    def _step_back_question(self, question: str) -> str:
        try:
            msg = (STEP_BACK_PROMPT | self.llm).invoke({"question": question})
            text = str(msg.content).strip()
            return text or question
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "step_back_error", "error": type(exc).__name__}))
            return question

    async def _astep_back_question(self, question: str) -> str:
        try:
            msg = await (STEP_BACK_PROMPT | self.llm).ainvoke({"question": question})
            text = str(msg.content).strip()
            return text or question
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "step_back_error", "error": type(exc).__name__}))
            return question

    def _detect_and_run_tools(self, question: str) -> tuple[str | None, dict[str, Any]]:
        """Detecta patrones computables comunes (IEEE 754, Ca2, bases, calculadora)
        y ejecuta herramientas deterministas de apoyo.
        """
        low = question.lower()
        tool_data: dict[str, Any] = {}
        hints: list[str] = []

        # 1. Detección de IEEE 754
        if any(w in low for w in ("ieee", "flotante", "precisión", "precision")):
            m_num = re.search(r"(-?\d+(?:\.\d+)?)", question)
            if m_num:
                val = float(m_num.group(1))
                fmt = "binary32"
                if any(w in low for w in ("media", "16", "binary16", "half")):
                    fmt = "binary16"
                elif any(w in low for w in ("doble", "64", "binary64", "double")):
                    fmt = "binary64"
                res = self.tool_registry.execute("ieee754", value=val, format_name=fmt)
                if res.success:
                    hints.append(f"Cálculo IEEE 754 para {val} ({fmt}):\n{res.output}")
                    tool_data["ieee754"] = res.data

        # 2. Detección de Complemento a Dos / Enteros
        if any(w in low for w in ("complemento", "ca2", "ca1", "signo y magnitud")):
            m_num = re.search(r"(-?\d+)", question)
            m_bits = re.search(r"(\d+)\s*bits?", low)
            bits = int(m_bits.group(1)) if m_bits else 8
            if m_num:
                val = int(m_num.group(1))
                res = self.tool_registry.execute("integer_representation", value=val, bits=bits)
                if res.success:
                    hints.append(f"Representación entera para {val} ({bits} bits):\n{res.output}")
                    tool_data["integer_representation"] = res.data

        # 3. Operaciones aritméticas / módulo
        if any(w in low for w in ("mod", "%", "módulo", "modulo", "resto")):
            m_mod = re.search(r"(\d+)\s*(?:mod|%|módulo|modulo)\s*(\d+)", low)
            if m_mod:
                expr = f"{m_mod.group(1)} % {m_mod.group(2)}"
                res = self.tool_registry.execute("calculator", expression=expr)
                if res.success:
                    hints.append(f"Operación módulo calculada: {expr} = {res.output}")
                    tool_data["calculator"] = res.data

        combined = "\n---\n".join(hints) if hints else None
        return combined, tool_data

    def _generate(
        self,
        context: str,
        question: str,
        history: list[tuple[str, str]] | None = None,
        pedagogical_mode: str = "auto",
        strict_mode: bool = False,
        verification_hint: str | None = None,
    ) -> str:
        try:
            if self.settings.enable_professor:
                prompt_template = build_professor_prompt(
                    mode=pedagogical_mode,
                    strict_mode=strict_mode,
                    verification_hint=verification_hint,
                    has_history=bool(history),
                )
            elif history:
                prompt_template = CONVERSATIONAL_PROMPT
            else:
                prompt_template = PROMPT

            chain_input: dict[str, Any] = {"context": context, "question": question}
            if history:
                chain_input["history"] = format_history(history)

            msg = (prompt_template | self.llm).invoke(chain_input)
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "llm_error", "error": type(exc).__name__}))
            raise LLMUnavailableError(_LLM_UNAVAILABLE_MSG) from exc
        return str(msg.content).strip()

    async def _agenerate(
        self,
        context: str,
        question: str,
        history: list[tuple[str, str]] | None = None,
        pedagogical_mode: str = "auto",
        strict_mode: bool = False,
        verification_hint: str | None = None,
    ) -> str:
        try:
            if self.settings.enable_professor:
                prompt_template = build_professor_prompt(
                    mode=pedagogical_mode,
                    strict_mode=strict_mode,
                    verification_hint=verification_hint,
                    has_history=bool(history),
                )
            elif history:
                prompt_template = CONVERSATIONAL_PROMPT
            else:
                prompt_template = PROMPT

            chain_input: dict[str, Any] = {"context": context, "question": question}
            if history:
                chain_input["history"] = format_history(history)

            msg = await (prompt_template | self.llm).ainvoke(chain_input)
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "llm_error", "error": type(exc).__name__}))
            raise LLMUnavailableError(_LLM_UNAVAILABLE_MSG) from exc
        return str(msg.content).strip()

    def _validate_and_resolve_conv_kb(
        self, conversation_id: str | None, requested_kb_id: str | None
    ) -> tuple[str, list[tuple[str, str]]]:
        """Valida que la conversación exista y que pertenezca a la base solicitada.

        Si requested_kb_id es None pero conversation_id fue provisto, infiere la base asociada
        a la conversación. Recupera el historial filtrado por esa base.
        """
        if not conversation_id or self.conversation_service is None:
            return requested_kb_id or self.settings.default_kb_id, []

        conv = self.conversation_service.get_conversation(conversation_id)
        conv_kb = conv.last_kb_id or self.settings.default_kb_id

        if requested_kb_id is not None and requested_kb_id != conv_kb:
            raise InvalidConversationError(
                f"La conversación '{conversation_id}' pertenece a la base '{conv_kb}', "
                f"no a la base '{requested_kb_id}' solicitada."
            )

        resolved_kb = conv_kb
        history = self.conversation_service.get_recent_history(
            conversation_id, self.settings.conversation_window_exchanges, kb_id=resolved_kb
        )
        return resolved_kb, history

    def _resolve_history(
        self, conversation_id: str | None, kb_id: str | None = None
    ) -> list[tuple[str, str]]:
        if conversation_id and self.conversation_service is not None:
            return self.conversation_service.get_recent_history(
                conversation_id, self.settings.conversation_window_exchanges, kb_id=kb_id
            )
        return []

    def _save_turn(
        self,
        conversation_id: str | None,
        question: str,
        result: QueryResult,
        kb_id: str,
    ) -> None:
        if conversation_id and self.conversation_service is not None:
            self.conversation_service.record_turn(
                conversation_id=conversation_id,
                question=question,
                answer=result.answer,
                grounded=result.grounded,
                reason=result.reason,
                message=result.message,
                sources=[s.__dict__ for s in result.sources],
                retrieved=[s.__dict__ for s in result.retrieved],
                kb_id=kb_id,
                pedagogical_mode=result.pedagogical_mode,
                strict_mode=result.strict_mode,
            )

    def _resolve_kb(self, knowledge_base_id: str | None) -> tuple[str, str, bool]:
        """Valida la KB y devuelve (kb_id, kb_name, is_empty_kb)."""
        kb_id = knowledge_base_id or self.settings.default_kb_id
        if self.kb_service is not None:
            kb = self.kb_service.get_knowledge_base(kb_id)
            return kb.id, kb.name, kb.chunks == 0
        if self.vs._collection.count() == 0:
            raise EmptyIndexError("No hay documentos indexados. Ejecutá la ingesta primero.")
        return kb_id, kb_id, False

    def _retrieve(
        self,
        question: str,
        top_k: int | None = None,
        kb_id: str = "general",
        step_back_query: str | None = None,
    ) -> tuple[list[tuple[Document, float]], list[SourceHit], int]:
        if self.vs._collection.count() == 0:
            raise EmptyIndexError("No hay documentos indexados. Ejecutá la ingesta primero.")

        k = top_k or self.settings.top_k
        t0 = time.perf_counter()
        found = self.vs.similarity_search_with_relevance_scores(
            question, k=k, filter={"knowledge_base_id": kb_id}
        )

        if (
            step_back_query
            and step_back_query.strip()
            and step_back_query.strip().lower() != question.strip().lower()
        ):
            sb_found = self.vs.similarity_search_with_relevance_scores(
                step_back_query, k=k, filter={"knowledge_base_id": kb_id}
            )
            by_id: dict[str, tuple[Document, float]] = {}
            for doc, score in found + sb_found:
                doc_key = doc.metadata.get("chunk_id") or doc.page_content
                if doc_key not in by_id or score > by_id[doc_key][1]:
                    by_id[doc_key] = (doc, score)
            found = sorted(by_id.values(), key=lambda x: x[1], reverse=True)[:k]

        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        hits = [(d, s) for d, s in found if s >= self.settings.min_relevance]
        retrieved = [_to_hit(i, d, s) for i, (d, s) in enumerate(hits, start=1)]
        return hits, retrieved, retrieval_ms

    def _postprocess_generation(
        self,
        text: str,
        question: str,
        retrieved: list[SourceHit],
        hits: list[tuple[Document, float]],
        retrieval_ms: int,
        t1: float,
        kb_id: str,
        conversation_id: str | None = None,
        pedagogical_mode: str = "auto",
        strict_mode: bool = False,
        tool_data: dict[str, Any] | None = None,
    ) -> QueryResult:
        generation_ms = int((time.perf_counter() - t1) * 1000)

        if NO_EVIDENCE in text:
            reason = "missing_premise_strict" if strict_mode else "model_refused"
            msg = MESSAGES.get(reason, MESSAGES["model_refused"])
            self._log(
                question,
                retrieval_ms,
                generation_ms,
                len(retrieved),
                len(hits),
                reason,
                kb_id=kb_id,
            )
            return QueryResult(
                None,
                False,
                retrieved=retrieved,
                reason=reason,
                message=msg,
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
                pedagogical_mode=pedagogical_mode,
                strict_mode=strict_mode,
                verification_data=tool_data,
            )

        valid_ids = {h.id for h in retrieved}
        cited = [n for n in extract_citations(text) if n in valid_ids]
        text = strip_invalid_citations(text, valid_ids)
        if not cited:
            self._log(
                question,
                retrieval_ms,
                generation_ms,
                len(retrieved),
                len(hits),
                "no_citations",
                kb_id=kb_id,
            )
            return QueryResult(
                text,
                False,
                retrieved=retrieved,
                reason="no_citations",
                message=MESSAGES["no_citations"],
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
                pedagogical_mode=pedagogical_mode,
                strict_mode=strict_mode,
                verification_data=tool_data,
            )

        sources = [h for h in retrieved if h.id in cited]
        self._log(
            question,
            retrieval_ms,
            generation_ms,
            len(retrieved),
            len(hits),
            "ok",
            len(sources),
            kb_id=kb_id,
        )
        return QueryResult(
            text,
            True,
            sources=sources,
            retrieved=retrieved,
            knowledge_base_id=kb_id,
            conversation_id=conversation_id,
            pedagogical_mode=pedagogical_mode,
            strict_mode=strict_mode,
            verification_data=tool_data,
        )

    def _finalize_generation(
        self,
        text: str,
        context: str,
        question: str,
        retrieved: list[SourceHit],
        hits: list[tuple[Document, float]],
        retrieval_ms: int,
        t1: float,
        kb_id: str,
        history: list[tuple[str, str]] | None = None,
        conversation_id: str | None = None,
        pedagogical_mode: str = "auto",
        strict_mode: bool = False,
        verification_hint: str | None = None,
        tool_data: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Política unificada de validación de citas, reintento guiado, abstención y logging."""
        if NO_EVIDENCE not in text and not extract_citations(text):
            # el modelo olvidó citar: un reintento con recordatorio
            text = self._generate(
                context,
                question + CITATION_REMINDER,
                history=history,
                pedagogical_mode=pedagogical_mode,
                strict_mode=strict_mode,
                verification_hint=verification_hint,
            )
        return self._postprocess_generation(
            text,
            question,
            retrieved,
            hits,
            retrieval_ms,
            t1,
            kb_id,
            conversation_id,
            pedagogical_mode=pedagogical_mode,
            strict_mode=strict_mode,
            tool_data=tool_data,
        )

    async def _afinalize_generation(
        self,
        text: str,
        context: str,
        question: str,
        retrieved: list[SourceHit],
        hits: list[tuple[Document, float]],
        retrieval_ms: int,
        t1: float,
        kb_id: str,
        history: list[tuple[str, str]] | None = None,
        conversation_id: str | None = None,
        pedagogical_mode: str = "auto",
        strict_mode: bool = False,
        verification_hint: str | None = None,
        tool_data: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Versión asíncrona no bloqueante de finalización para streaming SSE."""
        if NO_EVIDENCE not in text and not extract_citations(text):
            # el modelo olvidó citar: un reintento con recordatorio
            text = await self._agenerate(
                context,
                question + CITATION_REMINDER,
                history=history,
                pedagogical_mode=pedagogical_mode,
                strict_mode=strict_mode,
                verification_hint=verification_hint,
            )
        return self._postprocess_generation(
            text,
            question,
            retrieved,
            hits,
            retrieval_ms,
            t1,
            kb_id,
            conversation_id,
            pedagogical_mode=pedagogical_mode,
            strict_mode=strict_mode,
            tool_data=tool_data,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
        conversation_id: str | None = None,
        pedagogical_mode: str | None = None,
        strict_mode: bool | None = None,
    ) -> QueryResult:
        resolved_mode = (
            pedagogical_mode
            if pedagogical_mode is not None
            else self.settings.default_pedagogical_mode
        )
        resolved_strict = (
            strict_mode
            if strict_mode is not None
            else self.settings.default_strict_mode
        )

        resolved_kb, history = self._validate_and_resolve_conv_kb(
            conversation_id, knowledge_base_id
        )
        kb_id, kb_name, is_empty_kb = self._resolve_kb(resolved_kb)
        if is_empty_kb:
            msg = f"La base '{kb_name}' no contiene documentos cargados."
            self._log(question, 0, 0, 0, 0, "empty_knowledge_base", kb_id=kb_id)
            result = QueryResult(
                None,
                False,
                reason="empty_knowledge_base",
                message=msg,
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
            )
            self._save_turn(conversation_id, question, result, kb_id)
            return result

        search_query = self._condense_question(history, question)
        step_back_query = None
        if self.settings.enable_step_back:
            step_back_query = self._step_back_question(search_query)

        hits, retrieved, retrieval_ms = self._retrieve(
            search_query, top_k, kb_id=kb_id, step_back_query=step_back_query
        )
        if not hits:
            msg = (
                f"No se encontró evidencia suficiente en la base '{kb_name}'."
                if self.kb_service is not None
                else MESSAGES["below_threshold"]
            )
            self._log(question, retrieval_ms, 0, len(retrieved), 0, "below_threshold", kb_id=kb_id)
            result = QueryResult(
                None,
                False,
                reason="below_threshold",
                message=msg,
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
            )
            self._save_turn(conversation_id, question, result, kb_id)
            return result

        verification_hint, tool_data = self._detect_and_run_tools(search_query)
        context = format_context(hits)
        t1 = time.perf_counter()
        text = self._generate(
            context,
            question,
            history=history,
            pedagogical_mode=resolved_mode,
            strict_mode=resolved_strict,
            verification_hint=verification_hint,
        )
        result = self._finalize_generation(
            text,
            context,
            question,
            retrieved,
            hits,
            retrieval_ms,
            t1,
            kb_id,
            history=history,
            conversation_id=conversation_id,
            pedagogical_mode=resolved_mode,
            strict_mode=resolved_strict,
            verification_hint=verification_hint,
            tool_data=tool_data,
        )
        self._save_turn(conversation_id, question, result, kb_id)
        return result

    @staticmethod
    def _end_event(
        reason: str | None,
        answer: str | None = None,
        sources: list[SourceHit] | None = None,
        cited: list[int] | None = None,
        message: str | None = None,
        conversation_id: str | None = None,
        pedagogical_mode: str = "auto",
        strict_mode: bool = False,
        verification_data: dict[str, Any] | None = None,
    ) -> dict:
        """Evento final del stream. Mismo contrato para éxito y para toda abstención."""
        sources = sources or []
        default_msg = MESSAGES.get(reason) if reason else None
        resolved_msg = message if message is not None else default_msg
        data: dict[str, Any] = {
            "grounded": reason is None,
            "reason": reason,
            "message": resolved_msg,
            "answer": answer,
            "sources": [s.__dict__ for s in sources],
            "cited_ids": cited or [],
        }
        if pedagogical_mode and pedagogical_mode != "auto":
            data["pedagogical_mode"] = pedagogical_mode
            data["strict_mode"] = strict_mode
        elif strict_mode:
            data["strict_mode"] = strict_mode
        if verification_data:
            data["verification_data"] = verification_data
        if conversation_id is not None:
            data["conversation_id"] = conversation_id
        return {
            "event": "end",
            "data": data,
        }

    async def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
        conversation_id: str | None = None,
        pedagogical_mode: str | None = None,
        strict_mode: bool | None = None,
    ) -> AsyncIterator[dict]:
        """Transmite la respuesta token a token como eventos Server-Sent Events (SSE).

        Contrato: `retrieved` (opcional) -> `token`* -> `end` (siempre, con grounded/reason/
        message/answer/sources/cited_ids/conversation_id). `error` queda reservado a fallos
        reales (LLM caído).
        """
        resolved_mode = (
            pedagogical_mode
            if pedagogical_mode is not None
            else self.settings.default_pedagogical_mode
        )
        resolved_strict = (
            strict_mode
            if strict_mode is not None
            else self.settings.default_strict_mode
        )

        resolved_kb, history = self._validate_and_resolve_conv_kb(
            conversation_id, knowledge_base_id
        )
        kb_id, kb_name, is_empty_kb = self._resolve_kb(resolved_kb)
        if is_empty_kb:
            msg = f"La base '{kb_name}' no contiene documentos cargados."
            self._log(question, 0, 0, 0, 0, "empty_knowledge_base", kb_id=kb_id)
            empty_res = QueryResult(
                None,
                False,
                reason="empty_knowledge_base",
                message=msg,
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
            )
            self._save_turn(conversation_id, question, empty_res, kb_id)
            yield self._end_event(
                "empty_knowledge_base",
                message=msg,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
            )
            return

        try:
            search_query = await self._acondense_question(history, question)
        except LLMUnavailableError as exc:
            self._log(question, 0, 0, 0, 0, "llm_unavailable", kb_id=kb_id)
            yield {"event": "error", "data": {"error": "llm_unavailable", "detail": str(exc)}}
            return

        step_back_query = None
        if self.settings.enable_step_back:
            step_back_query = await self._astep_back_question(search_query)

        hits, retrieved, retrieval_ms = self._retrieve(
            search_query, top_k, kb_id=kb_id, step_back_query=step_back_query
        )
        if not hits:
            msg = (
                f"No se encontró evidencia suficiente en la base '{kb_name}'."
                if self.kb_service is not None
                else MESSAGES["below_threshold"]
            )
            self._log(question, retrieval_ms, 0, len(retrieved), 0, "below_threshold", kb_id=kb_id)
            below_res = QueryResult(
                None,
                False,
                reason="below_threshold",
                message=msg,
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
            )
            self._save_turn(conversation_id, question, below_res, kb_id)
            yield self._end_event(
                "below_threshold",
                message=msg,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
            )
            return

        # Notificar fuentes recuperadas inicialmente
        yield {
            "event": "retrieved",
            "data": {
                "retrieved": [h.__dict__ for h in retrieved],
                "retrieval_ms": retrieval_ms,
            },
        }

        verification_hint, tool_data = self._detect_and_run_tools(search_query)
        context = format_context(hits)

        if self.settings.enable_professor:
            prompt_template = build_professor_prompt(
                mode=resolved_mode,
                strict_mode=resolved_strict,
                verification_hint=verification_hint,
                has_history=bool(history),
            )
        elif history:
            prompt_template = CONVERSATIONAL_PROMPT
        else:
            prompt_template = PROMPT

        chain = prompt_template | self.llm
        chain_input: dict[str, Any] = {"context": context, "question": question}
        if history:
            chain_input["history"] = format_history(history)

        full_text = []
        t1 = time.perf_counter()

        try:
            async for chunk in chain.astream(chain_input):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    full_text.append(content)
                    yield {"event": "token", "data": {"token": content}}
        except _LLM_CONNECTION_ERRORS as exc:
            logger.warning(json.dumps({"event": "llm_error", "error": type(exc).__name__}))
            self._log(
                question,
                retrieval_ms,
                int((time.perf_counter() - t1) * 1000),
                len(retrieved),
                len(hits),
                "llm_unavailable",
                kb_id=kb_id,
            )
            yield {
                "event": "error",
                "data": {"error": "llm_unavailable", "detail": _LLM_UNAVAILABLE_MSG},
            }
            return

        accumulated = "".join(full_text).strip()
        try:
            result = await self._afinalize_generation(
                accumulated,
                context,
                question,
                retrieved,
                hits,
                retrieval_ms,
                t1,
                kb_id,
                history=history,
                conversation_id=conversation_id,
                pedagogical_mode=resolved_mode,
                strict_mode=resolved_strict,
                verification_hint=verification_hint,
                tool_data=tool_data,
            )
        except LLMUnavailableError as exc:
            self._log(
                question,
                retrieval_ms,
                int((time.perf_counter() - t1) * 1000),
                len(retrieved),
                len(hits),
                "llm_unavailable",
                kb_id=kb_id,
            )
            yield {"event": "error", "data": {"error": "llm_unavailable", "detail": str(exc)}}
            return

        self._save_turn(conversation_id, question, result, kb_id)
        cited_ids = [s.id for s in result.sources]
        yield self._end_event(
            result.reason,
            answer=result.answer,
            sources=result.sources,
            cited=cited_ids,
            message=result.message,
            conversation_id=conversation_id,
            pedagogical_mode=resolved_mode,
            strict_mode=resolved_strict,
            verification_data=tool_data,
        )

    @staticmethod
    def _log(
        question,
        retrieval_ms,
        generation_ms,
        found,
        kept,
        outcome,
        cited=0,
        kb_id: str = "general",
    ) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "query",
                    "knowledge_base_id": kb_id,
                    "question_len": len(question),
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "found": found,
                    "above_threshold": kept,
                    "cited": cited,
                    "outcome": outcome,
                }
            )
        )
