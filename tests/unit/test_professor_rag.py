from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.rag.service import RAGService


def _build_rag(responses: list[str], settings: Settings | None = None) -> RAGService:
    fake_llm = FakeListChatModel(responses=responses)
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1
    doc = Document(
        page_content="IEEE 754: 1 bit signo, 8 exponente con sesgo 127 y 23 mantisa.",
        metadata={"source": "ieee.md", "knowledge_base_id": "general"},
    )
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.95)]
    return RAGService(vs, fake_llm, settings or Settings())


def test_answer_solve_mode_triggers_tools_and_verification_data():
    rag = _build_rag(
        responses=[
            "Para 137 [1], en binario es 10001001. Normalizado 1.0001001 * 2^7. "
            "Exponente sesgado: 7 + 127 = 134 (10000110). Mantisa: 00010010000000000000000."
        ]
    )
    res = rag.answer(
        "Calculame el número 137 en punto flotante IEEE 754 de precisión simple",
        pedagogical_mode="solve",
    )
    assert res.grounded is True
    assert res.pedagogical_mode == "solve"
    assert res.strict_mode is False
    assert res.verification_data is not None
    assert "ieee754" in res.verification_data
    assert res.verification_data["ieee754"]["biased_exponent"] == 134
    assert res.verification_data["ieee754"]["hex"] == "0x43090000"


def test_answer_strict_mode_abstains_with_missing_premise_strict():
    # En modo estricto, si el LLM emite NO_EVIDENCE, la razón debe ser missing_premise_strict
    rag = _build_rag(responses=["NO_EVIDENCE"])
    res = rag.answer(
        "Calculame 137 en formato de 16 bits no especificado",
        strict_mode=True,
    )
    assert res.grounded is False
    assert res.reason == "missing_premise_strict"
    assert "Modo Estricto" in (res.message or "")


def test_answer_hint_mode_preserved():
    rag = _build_rag(
        responses=[
            "Pista inicial [1]: descomponé 137 en suma de potencias de 2 (128 + 8 + 1). "
            "¿Qué exponente de 2 corresponde a 128?"
        ]
    )
    res = rag.answer(
        "Dame una pista para calcular 137 en IEEE 754",
        pedagogical_mode="hint",
    )
    assert res.grounded is True
    assert res.pedagogical_mode == "hint"
    assert "Pista inicial" in (res.answer or "")


def test_answer_check_mode_diagnosis():
    rag = _build_rag(
        responses=[
            "Revisión de tu desarrollo [1]: El procedimiento para normalizar es correcto, "
            "pero cometiste un error conceptual: olvidaste sumarle el sesgo de 127 al exponente."
        ]
    )
    res = rag.answer(
        "Para 137 puse E=7 sin sesgo, ¿está bien?",
        pedagogical_mode="check",
    )
    assert res.grounded is True
    assert res.pedagogical_mode == "check"


@pytest.mark.anyio
async def test_stream_answer_emits_professor_metadata_in_end_event():
    rag = _build_rag(
        responses=["Desarrollo paso a paso [1]: S=0, E=10000110, M=00010010000000000000000."]
    )
    events = []
    async for event in rag.stream_answer(
        "Calculame 137 en punto flotante simple",
        pedagogical_mode="solve",
        strict_mode=False,
    ):
        events.append(event)

    end_events = [e for e in events if e["event"] == "end"]
    assert len(end_events) == 1
    end_data = end_events[0]["data"]
    assert end_data["grounded"] is True
    assert end_data["pedagogical_mode"] == "solve"
    assert end_data["strict_mode"] is False
    assert "verification_data" in end_data
    assert "ieee754" in end_data["verification_data"]


def test_detect_and_run_tools_sanitizes_ieee754_question():
    rag = _build_rag(responses=["ok"])
    hint, data = rag._detect_and_run_tools("En IEEE 754 de 32 bits representa el 42")
    assert data is not None
    assert "ieee754" in data
    # Debe haber calculado para 42, no para 754
    assert data["ieee754"]["value"] == 42.0
    assert data["ieee754"]["hex"] == "0x42280000"


def test_detect_and_run_tools_sanitizes_ca2_question():
    rag = _build_rag(responses=["ok"])
    hint, data = rag._detect_and_run_tools("En Ca2 de 8 bits representa el número -5")
    assert data is not None
    assert "integer_representation" in data
    # Debe haber calculado para -5, no para 2
    assert data["integer_representation"]["value"] == -5
    assert data["integer_representation"]["twos_complement"] == "11111011"


def test_detect_and_run_tools_runs_base_converter():
    rag = _build_rag(responses=["ok"])
    hint, data = rag._detect_and_run_tools("Convierte el número 255 a binario en 8 bits")
    assert data is not None
    assert "base_converter" in data
    assert data["base_converter"]["decimal"] == 255
    assert data["base_converter"]["binary"] == "11111111"
