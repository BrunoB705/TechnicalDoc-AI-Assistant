from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.rag.service import RAGService


def test_step_back_question_sync():
    fake_llm = FakeListChatModel(
        responses=["¿Cuáles son las especificaciones de punto flotante precisión simple IEEE 754?"]
    )
    vs = MagicMock(spec=Chroma)
    settings = Settings(enable_step_back=True)
    rag = RAGService(vs, fake_llm, settings)

    sb = rag._step_back_question("Calculame el número 137 en precisión simple")
    assert "IEEE 754" in sb


@pytest.mark.anyio
async def test_step_back_question_async():
    fake_llm = FakeListChatModel(
        responses=["¿Cuáles son las especificaciones de punto flotante precisión simple IEEE 754?"]
    )
    vs = MagicMock(spec=Chroma)
    settings = Settings(enable_step_back=True)
    rag = RAGService(vs, fake_llm, settings)

    sb = await rag._astep_back_question("Calculame el número 137 en precisión simple")
    assert "IEEE 754" in sb


def test_step_back_fallback_on_error():
    class BrokenLLM(FakeListChatModel):
        def _call(self, *args, **kwargs):
            import openai

            raise openai.APIConnectionError(request=MagicMock())

    vs = MagicMock(spec=Chroma)
    settings = Settings(enable_step_back=True)
    rag = RAGService(vs, BrokenLLM(responses=[]), settings)

    original = "Calculame el número 137"
    sb = rag._step_back_question(original)
    assert sb == original


def test_retrieve_fuses_and_deduplicates_step_back_results():
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 2

    doc1 = Document(
        page_content="Norma IEEE 754 con sesgo 127.",
        metadata={"chunk_id": "chunk_ieee", "source": "ieee.md", "knowledge_base_id": "general"},
    )
    doc2 = Document(
        page_content="Ejemplo de cálculo 137.",
        metadata={"chunk_id": "chunk_calc", "source": "calc.md", "knowledge_base_id": "general"},
    )

    def fake_search(query, k, filter):
        if "137" in query:
            return [(doc2, 0.70), (doc1, 0.60)]
        # step-back query
        return [(doc1, 0.95)]

    vs.similarity_search_with_relevance_scores.side_effect = fake_search

    settings = Settings(min_relevance=0.50, top_k=2)
    rag = RAGService(vs, MagicMock(), settings)

    hits, retrieved, _ = rag._retrieve(
        "Calculame 137",
        step_back_query="Norma IEEE 754 especificaciones",
        kb_id="general",
    )

    assert len(hits) == 2
    # doc1 debe tener el score más alto (0.95) producto de la fusión
    assert hits[0][0].metadata["chunk_id"] == "chunk_ieee"
    assert hits[0][1] == 0.95
    assert hits[1][0].metadata["chunk_id"] == "chunk_calc"
    assert hits[1][1] == 0.70


def test_answer_invokes_step_back_when_enabled():
    fake_llm = FakeListChatModel(
        responses=[
            "¿Cuáles son las reglas de IEEE 754?",  # llamada de step-back
            "Aplicando la norma [1], 137 en binario es 10001001.",  # llamada de generación
        ]
    )
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1
    doc = Document(
        page_content="Norma IEEE 754 de precisión simple [1].",
        metadata={"chunk_id": "chunk_1", "source": "norma.md", "knowledge_base_id": "general"},
    )
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.90)]

    settings = Settings(enable_step_back=True, min_relevance=0.80)
    rag = RAGService(vs, fake_llm, settings)

    res = rag.answer("Calculame el número 137 en punto flotante")
    assert res.grounded is True
    assert "10001001" in res.answer
    assert len(res.sources) == 1


@pytest.mark.anyio
async def test_stream_answer_invokes_step_back_when_enabled():
    fake_llm = FakeListChatModel(
        responses=[
            "¿Cuáles son las reglas de IEEE 754?",  # llamada de step-back
            "Aplicando la norma [1], 137 en binario es 10001001.",  # llamada de streaming
        ]
    )
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1
    doc = Document(
        page_content="Norma IEEE 754 de precisión simple [1].",
        metadata={"chunk_id": "chunk_1", "source": "norma.md", "knowledge_base_id": "general"},
    )
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.90)]

    settings = Settings(enable_step_back=True, min_relevance=0.80)
    rag = RAGService(vs, fake_llm, settings)

    events = [ev async for ev in rag.stream_answer("Calculame 137")]
    end_event = next(e for e in events if e["event"] == "end")
    assert end_event["data"]["grounded"] is True
    assert "10001001" in end_event["data"]["answer"]
