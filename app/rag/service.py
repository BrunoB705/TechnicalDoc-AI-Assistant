import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

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
    format_history,
)

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
    ):
        self.vs = vectorstore
        self.llm = llm
        self.settings = settings
        self.kb_service = kb_service
        self.conversation_service = conversation_service or ConversationService(settings)

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

    def _generate(
        self,
        context: str,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        try:
            if history:
                msg = (CONVERSATIONAL_PROMPT | self.llm).invoke(
                    {
                        "history": format_history(history),
                        "context": context,
                        "question": question,
                    }
                )
            else:
                msg = (PROMPT | self.llm).invoke({"context": context, "question": question})
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
        self, question: str, top_k: int | None = None, kb_id: str = "general"
    ) -> tuple[list[tuple[Document, float]], list[SourceHit], int]:
        if self.vs._collection.count() == 0:
            raise EmptyIndexError("No hay documentos indexados. Ejecutá la ingesta primero.")

        k = top_k or self.settings.top_k
        t0 = time.perf_counter()
        found = self.vs.similarity_search_with_relevance_scores(
            question, k=k, filter={"knowledge_base_id": kb_id}
        )
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        hits = [(d, s) for d, s in found if s >= self.settings.min_relevance]
        retrieved = [_to_hit(i, d, s) for i, (d, s) in enumerate(hits, start=1)]
        return hits, retrieved, retrieval_ms

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
    ) -> QueryResult:
        """Política unificada de validación de citas, reintento guiado, abstención y logging."""
        if NO_EVIDENCE not in text and not extract_citations(text):
            # el modelo olvidó citar: un reintento con recordatorio
            text = self._generate(context, question + CITATION_REMINDER, history=history)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        if NO_EVIDENCE in text:
            self._log(
                question,
                retrieval_ms,
                generation_ms,
                len(retrieved),
                len(hits),
                "model_refused",
                kb_id=kb_id,
            )
            return QueryResult(
                None,
                False,
                retrieved=retrieved,
                reason="model_refused",
                message=MESSAGES["model_refused"],
                knowledge_base_id=kb_id,
                conversation_id=conversation_id,
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
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
        conversation_id: str | None = None,
    ) -> QueryResult:
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
            )
            self._save_turn(conversation_id, question, result, kb_id)
            return result

        search_query = self._condense_question(history, question)
        hits, retrieved, retrieval_ms = self._retrieve(search_query, top_k, kb_id=kb_id)
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
            )
            self._save_turn(conversation_id, question, result, kb_id)
            return result

        context = format_context(hits)
        t1 = time.perf_counter()
        text = self._generate(context, question, history=history)
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
    ) -> dict:
        """Evento final del stream. Mismo contrato para éxito y para toda abstención."""
        sources = sources or []
        default_msg = MESSAGES.get(reason) if reason else None
        resolved_msg = message if message is not None else default_msg
        data = {
            "grounded": reason is None,
            "reason": reason,
            "message": resolved_msg,
            "answer": answer,
            "sources": [s.__dict__ for s in sources],
            "cited_ids": cited or [],
        }
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
    ) -> AsyncIterator[dict]:
        """Transmite la respuesta token a token como eventos Server-Sent Events (SSE).

        Contrato: `retrieved` (opcional) -> `token`* -> `end` (siempre, con grounded/reason/
        message/answer/sources/cited_ids/conversation_id). `error` queda reservado a fallos
        reales (LLM caído).
        """
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
            )
            self._save_turn(conversation_id, question, empty_res, kb_id)
            yield self._end_event(
                "empty_knowledge_base", message=msg, conversation_id=conversation_id
            )
            return

        try:
            search_query = self._condense_question(history, question)
        except LLMUnavailableError as exc:
            self._log(question, 0, 0, 0, 0, "llm_unavailable", kb_id=kb_id)
            yield {"event": "error", "data": {"error": "llm_unavailable", "detail": str(exc)}}
            return

        hits, retrieved, retrieval_ms = self._retrieve(search_query, top_k, kb_id=kb_id)
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
            )
            self._save_turn(conversation_id, question, below_res, kb_id)
            yield self._end_event("below_threshold", message=msg, conversation_id=conversation_id)
            return

        # Notificar fuentes recuperadas inicialmente
        yield {
            "event": "retrieved",
            "data": {
                "retrieved": [h.__dict__ for h in retrieved],
                "retrieval_ms": retrieval_ms,
            },
        }

        context = format_context(hits)
        if history:
            chain = CONVERSATIONAL_PROMPT | self.llm
            chain_input = {
                "history": format_history(history),
                "context": context,
                "question": question,
            }
        else:
            chain = PROMPT | self.llm
            chain_input = {"context": context, "question": question}

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
            result = self._finalize_generation(
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
