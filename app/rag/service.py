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
from app.exceptions import EmptyIndexError, LLMUnavailableError
from app.knowledge_bases.service import KnowledgeBaseService
from app.rag.citations import extract_citations, strip_invalid_citations
from app.rag.prompts import NO_EVIDENCE, PROMPT

logger = logging.getLogger("rag")

_LLM_CONNECTION_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)

MESSAGES = {
    "empty_knowledge_base": "La base seleccionada no contiene documentos cargados.",
    "below_threshold": "No encontré evidencia suficiente en los documentos para responder.",
    "model_refused": "Los documentos recuperados no alcanzan para responder la pregunta.",
    "no_citations": "La respuesta no cita ninguna fuente, por lo que no pudo verificarse.",
}

CITATION_REMINDER = (
    "\n\nRecordatorio: citá cada afirmación con el número de la fuente entre corchetes, "
    "por ejemplo [1]."
)


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
    sources: list[SourceHit] = field(default_factory=list)  # solo las realmente citadas
    retrieved: list[SourceHit] = field(default_factory=list)  # todas las recuperadas
    reason: str | None = None  # por qué no hay respuesta fundamentada
    message: str | None = None  # mensaje explicativo cuando grounded=False
    knowledge_base_id: str = "general"


def _to_hit(i: int, doc: Document, score: float) -> SourceHit:
    m = doc.metadata
    return SourceHit(
        id=i,
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
    ):
        self.vs = vectorstore
        self.llm = llm
        self.settings = settings
        self.kb_service = kb_service

    def _generate(self, context: str, question: str) -> str:
        try:
            msg = (PROMPT | self.llm).invoke({"context": context, "question": question})
        except _LLM_CONNECTION_ERRORS as exc:
            raise LLMUnavailableError(f"El modelo de lenguaje no responde: {exc}") from exc
        return str(msg.content).strip()

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

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
    ) -> QueryResult:
        kb_id, kb_name, is_empty_kb = self._resolve_kb(knowledge_base_id)
        if is_empty_kb:
            msg = f"La base '{kb_name}' no contiene documentos cargados."
            self._log(question, 0, 0, 0, 0, "empty_knowledge_base", kb_id=kb_id)
            return QueryResult(
                None,
                False,
                reason="empty_knowledge_base",
                message=msg,
                knowledge_base_id=kb_id,
            )

        hits, retrieved, retrieval_ms = self._retrieve(question, top_k, kb_id=kb_id)
        if not hits:
            msg = (
                f"No se encontró evidencia suficiente en la base '{kb_name}'."
                if self.kb_service is not None
                else MESSAGES["below_threshold"]
            )
            self._log(
                question, retrieval_ms, 0, len(retrieved), 0, "below_threshold", kb_id=kb_id
            )
            return QueryResult(
                None,
                False,
                reason="below_threshold",
                message=msg,
                knowledge_base_id=kb_id,
            )

        context = format_context(hits)
        t1 = time.perf_counter()
        text = self._generate(context, question)
        if NO_EVIDENCE not in text and not extract_citations(text):
            # el modelo olvidó citar: un reintento con recordatorio
            text = self._generate(context, question + CITATION_REMINDER)
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
        )

    @staticmethod
    def _end_event(
        reason: str | None,
        answer: str | None = None,
        sources: list[SourceHit] | None = None,
        cited: list[int] | None = None,
        message: str | None = None,
    ) -> dict:
        """Evento final del stream. Mismo contrato para éxito y para toda abstención."""
        sources = sources or []
        default_msg = MESSAGES.get(reason) if reason else None
        resolved_msg = message if message is not None else default_msg
        return {
            "event": "end",
            "data": {
                "grounded": reason is None,
                "reason": reason,
                "message": resolved_msg,
                "answer": answer,
                "sources": [s.__dict__ for s in sources],
                "cited_ids": cited or [],
            },
        }

    async def stream_answer(
        self,
        question: str,
        top_k: int | None = None,
        knowledge_base_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """Transmite la respuesta token a token como eventos Server-Sent Events (SSE).

        Contrato: `retrieved` (opcional) -> `token`* -> `end` (siempre, con grounded/reason/
        message/answer/sources/cited_ids). `error` queda reservado a fallos reales (LLM caído).
        """
        kb_id, kb_name, is_empty_kb = self._resolve_kb(knowledge_base_id)
        if is_empty_kb:
            msg = f"La base '{kb_name}' no contiene documentos cargados."
            self._log(question, 0, 0, 0, 0, "empty_knowledge_base", kb_id=kb_id)
            yield self._end_event("empty_knowledge_base", message=msg)
            return

        hits, retrieved, retrieval_ms = self._retrieve(question, top_k, kb_id=kb_id)
        if not hits:
            msg = (
                f"No se encontró evidencia suficiente en la base '{kb_name}'."
                if self.kb_service is not None
                else MESSAGES["below_threshold"]
            )
            self._log(
                question, retrieval_ms, 0, len(retrieved), 0, "below_threshold", kb_id=kb_id
            )
            yield self._end_event("below_threshold", message=msg)
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
        chain = PROMPT | self.llm
        full_text = []

        try:
            async for chunk in chain.astream({"context": context, "question": question}):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    full_text.append(content)
                    yield {"event": "token", "data": {"token": content}}
        except _LLM_CONNECTION_ERRORS as exc:
            yield {"event": "error", "data": {"error": "llm_unavailable", "detail": str(exc)}}
            return

        accumulated = "".join(full_text).strip()
        if NO_EVIDENCE in accumulated:
            yield self._end_event("model_refused")
            return

        valid_ids = {h.id for h in retrieved}
        cited = [n for n in extract_citations(accumulated) if n in valid_ids]
        answer = strip_invalid_citations(accumulated, valid_ids)
        if not cited:
            yield self._end_event("no_citations", answer=answer)
            return

        sources = [h for h in retrieved if h.id in cited]
        yield self._end_event(None, answer=answer, sources=sources, cited=cited)

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
