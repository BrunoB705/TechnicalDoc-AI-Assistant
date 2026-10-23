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


@dataclass
class QueryResult:
    answer: str | None
    grounded: bool
    sources: list[SourceHit] = field(default_factory=list)  # solo las realmente citadas
    retrieved: list[SourceHit] = field(default_factory=list)  # todas las recuperadas
    reason: str | None = None  # por qué no hay respuesta fundamentada


def _to_hit(i: int, doc: Document, score: float) -> SourceHit:
    m = doc.metadata
    return SourceHit(
        id=i,
        source=m.get("source", "desconocido"),
        section=m.get("section"),
        page=m.get("page"),
        snippet=doc.page_content[:300],
        score=round(float(score), 4),
    )


def format_context(hits: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, _) in enumerate(hits, start=1):
        m = doc.metadata
        loc = " · ".join(str(x) for x in (m.get("source"), m.get("section"), m.get("page")) if x)
        blocks.append(f"[{i}] ({loc})\n{doc.page_content}")
    return "\n\n".join(blocks)


class RAGService:
    def __init__(self, vectorstore: Chroma, llm: BaseChatModel, settings: Settings):
        self.vs = vectorstore
        self.llm = llm
        self.settings = settings

    def _generate(self, context: str, question: str) -> str:
        try:
            msg = (PROMPT | self.llm).invoke({"context": context, "question": question})
        except _LLM_CONNECTION_ERRORS as exc:
            raise LLMUnavailableError(f"El modelo de lenguaje no responde: {exc}") from exc
        return str(msg.content).strip()

    def _retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[tuple[Document, float]], list[SourceHit], int]:
        if self.vs._collection.count() == 0:
            raise EmptyIndexError("No hay documentos indexados. Ejecutá la ingesta primero.")

        k = top_k or self.settings.top_k
        t0 = time.perf_counter()
        found = self.vs.similarity_search_with_relevance_scores(question, k=k)
        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        hits = [(d, s) for d, s in found if s >= self.settings.min_relevance]
        retrieved = [_to_hit(i, d, s) for i, (d, s) in enumerate(hits, start=1)]
        return hits, retrieved, retrieval_ms

    def answer(self, question: str, top_k: int | None = None) -> QueryResult:
        hits, retrieved, retrieval_ms = self._retrieve(question, top_k)
        if not hits:
            self._log(question, retrieval_ms, 0, len(retrieved), 0, "below_threshold")
            return QueryResult(None, False, reason="below_threshold")

        context = format_context(hits)
        t1 = time.perf_counter()
        text = self._generate(context, question)
        if NO_EVIDENCE not in text and not extract_citations(text):
            # el modelo olvidó citar: un reintento con recordatorio
            text = self._generate(context, question + CITATION_REMINDER)
        generation_ms = int((time.perf_counter() - t1) * 1000)

        if NO_EVIDENCE in text:
            self._log(
                question, retrieval_ms, generation_ms, len(retrieved), len(hits), "model_refused"
            )
            return QueryResult(None, False, retrieved=retrieved, reason="model_refused")

        valid_ids = {h.id for h in retrieved}
        cited = [n for n in extract_citations(text) if n in valid_ids]
        text = strip_invalid_citations(text, valid_ids)
        if not cited:
            self._log(
                question, retrieval_ms, generation_ms, len(retrieved), len(hits), "no_citations"
            )
            return QueryResult(text, False, retrieved=retrieved, reason="no_citations")

        sources = [h for h in retrieved if h.id in cited]
        self._log(
            question, retrieval_ms, generation_ms, len(retrieved), len(hits), "ok", len(sources)
        )
        return QueryResult(text, True, sources=sources, retrieved=retrieved)

    async def stream_answer(self, question: str, top_k: int | None = None) -> AsyncIterator[dict]:
        """Transmite la respuesta token a token como eventos Server-Sent Events (SSE)."""
        hits, retrieved, retrieval_ms = self._retrieve(question, top_k)
        if not hits:
            yield {
                "event": "error",
                "data": {
                    "grounded": False,
                    "reason": "below_threshold",
                    "message": "No encontré evidencia suficiente en los documentos para responder.",
                },
            }
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
            yield {
                "event": "end",
                "data": {
                    "grounded": False,
                    "reason": "model_refused",
                    "sources": [],
                    "message": "Los documentos recuperados no alcanzan para responder la pregunta.",
                },
            }
            return

        valid_ids = {h.id for h in retrieved}
        cited = [n for n in extract_citations(accumulated) if n in valid_ids]
        sources = [h.__dict__ for h in retrieved if h.id in cited]

        yield {
            "event": "end",
            "data": {
                "grounded": len(sources) > 0,
                "sources": sources,
                "cited_ids": cited,
            },
        }

    @staticmethod
    def _log(question, retrieval_ms, generation_ms, found, kept, outcome, cited=0) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "query",
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
