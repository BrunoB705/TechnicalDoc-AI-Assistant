from pathlib import Path
from unittest.mock import MagicMock

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings
from app.evaluation.metrics import (
    check_hint_progression,
    check_numerical_correctness,
    check_student_error_diagnosis,
)
from app.evaluation.runner import load_dataset
from app.rag.service import RAGService

PROFESSOR_DATASET_PATH = Path("eval/professor_dataset.jsonl")
PROFESSOR_TEST_DATASET_PATH = Path("eval/test_professor_dataset.jsonl")


def test_professor_datasets_integrity_and_disjointness():
    cases = load_dataset(PROFESSOR_DATASET_PATH)
    assert len(cases) == 30
    case_ids = [c.id for c in cases]
    assert len(set(case_ids)) == len(case_ids)

    test_cases = load_dataset(PROFESSOR_TEST_DATASET_PATH)
    assert len(test_cases) >= 12
    test_ids = [c.id for c in test_cases]
    assert len(set(test_ids)) == len(test_ids)

    # Disjuntos en IDs y en preguntas
    assert set(case_ids).isdisjoint(set(test_ids))
    dev_q = {c.question.strip().lower() for c in cases}
    test_q = {c.question.strip().lower() for c in test_cases}
    assert dev_q.isdisjoint(test_q)

    # Validación de categorías pedagógicas
    expected_categories = {"solve", "hint", "check", "explain", "missing_premise"}
    assert {c.category for c in cases} == expected_categories
    assert {c.category for c in test_cases} == expected_categories


def test_check_numerical_correctness():
    # Coincidencia con espaciado binario y valores esperados
    generated = """El número 137 en IEEE 754 precisión simple se compone de:
- Signo: 0
- Exponente: 1000 0110_2 (sesgo 127 + 7 = 134)
- Mantisa: 00010010000000000000000_2
Hexadecimal: 0x43090000"""

    expected = {
        "sign": "0",
        "exponent": "10000110",
        "bias": "127",
        "fraction": "0001001",
    }
    assert check_numerical_correctness(generated, expected) is True

    # Falla si falta un campo numérico crítico
    wrong_expected = {"exponent": "10111111"}
    assert check_numerical_correctness(generated, wrong_expected) is False


def test_check_hint_progression():
    # En modo pistas, no debe revelar la solución final prohibida
    good_hint = (
        "Primero convertí 137 a binario sumando potencias de 2 (128 + 8 + 1). "
        "¿Qué potencias obtenés?"
    )
    assert check_hint_progression(good_hint, ["10000110", "0x43090000"]) is True

    premature_solution = "El resultado final en binario es 0 10000110 00010010000000000000000."
    assert check_hint_progression(premature_solution, ["10000110"]) is False


def test_check_student_error_diagnosis():
    correction = (
        "Excelente inicio, pero cometiste un error conceptual: "
        "olvidaste sumarle el sesgo de 127 al exponente 7."
    )
    assert check_student_error_diagnosis(correction, ["sesgo", "bias"]) is True
    assert check_student_error_diagnosis(correction, ["overflow", "desbordamiento"]) is False


def test_reproduce_ieee754_diagnosis_step_back_retrieval():
    """Reproduce el caso diagnóstico: una búsqueda con datos particulares (137)

    falla o degrada si no se abstrae con step-back a los conceptos de la norma.
    """
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1

    doc_rule = Document(
        page_content=(
            "IEEE 754 precisión simple: 1 bit signo, 8 bits exponente con sesgo 127 y 23 mantisa."
        ),
        metadata={
            "chunk_id": "chunk_rule",
            "source": "representacion_y_alu.md",
            "knowledge_base_id": "arquitectura",
        },
    )

    def mock_search(query: str, k: int, filter: dict):
        # Búsqueda literal con número 137 simula score bajo por falta de coincidencia textual
        if "137" in query:
            return [(doc_rule, 0.65)]
        # Búsqueda conceptual de fondo (step-back) recupera con alta relevancia
        return [(doc_rule, 0.94)]

    vs.similarity_search_with_relevance_scores.side_effect = mock_search

    # Caso 1: Con step-back deshabilitado y min_relevance=0.80, cae en below_threshold
    settings_no_sb = Settings(enable_step_back=False, min_relevance=0.80)
    rag_no_sb = RAGService(vs, MagicMock(), settings_no_sb)
    hits, _, _ = rag_no_sb._retrieve(
        "Calculame el número 137 en precisión simple", kb_id="arquitectura"
    )
    assert len(hits) == 0

    # Caso 2: Con step-back habilitado, se fusionan los resultados conceptuales (score 0.94 >= 0.80)
    settings_sb = Settings(enable_step_back=True, min_relevance=0.80)
    rag_sb = RAGService(vs, MagicMock(), settings_sb)
    hits_sb, _, _ = rag_sb._retrieve(
        "Calculame el número 137 en precisión simple",
        step_back_query="IEEE 754 formato precisión simple exponente sesgo mantisa",
        kb_id="arquitectura",
    )
    assert len(hits_sb) == 1
    assert hits_sb[0][1] == 0.94


def test_evaluate_dataset_computes_pedagogical_metrics_with_llm():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from app.evaluation.metrics import EvalCase
    from app.evaluation.runner import evaluate_dataset

    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1
    doc = Document(
        page_content="IEEE 754: 1 bit signo, 8 exponente con sesgo 127 y 23 mantisa.",
        metadata={"chunk_id": "c1", "source": "norma.md", "knowledge_base_id": "arquitectura"},
    )
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.95)]

    cases = [
        EvalCase(
            id="p_solve",
            knowledge_base_id="arquitectura",
            category="solve",
            should_abstain=False,
            question="Calculame 137 en simple",
            expected_source="norma.md",
            expected_answer="S=0 E=10000110 M=0001001",
            pedagogical_mode="solve",
            expected_calculation={"sign": "0", "exponent": "10000110"},
        ),
        EvalCase(
            id="p_hint",
            knowledge_base_id="arquitectura",
            category="hint",
            should_abstain=False,
            question="Dame una pista para 137",
            expected_source="norma.md",
            expected_answer="Pensá en potencias de dos",
            pedagogical_mode="hint",
            forbidden_in_hints=["10000110", "0x43090000"],
        ),
        EvalCase(
            id="p_check",
            knowledge_base_id="arquitectura",
            category="check",
            should_abstain=False,
            question="¿Está bien si puse E=7?",
            expected_source="norma.md",
            expected_answer="sesgo, bias",
            pedagogical_mode="check",
            student_attempt="E=7",
        ),
        EvalCase(
            id="p_strict",
            knowledge_base_id="arquitectura",
            category="missing_premise",
            should_abstain=True,
            question="Calculame en formato de 16 bits",
            strict_mode=True,
        ),
    ]

    llm = FakeListChatModel(
        responses=[
            "Paso a paso [1]: S=0, E=10000110, M=0001001.",
            "Pista guiada [1]: Descomponé 137 en potencias de dos (128 + 8 + 1).",
            "Diagnóstico [1]: Cometiste un error con el sesgo del exponente.",
            "NO_EVIDENCE",
        ]
    )

    settings = Settings(min_relevance=0.80, top_k=3)
    summary, outcomes = evaluate_dataset(cases, vs, settings, llm=llm)

    assert summary.numerical_correctness_rate == 1.0
    assert summary.hint_progression_rate == 1.0
    assert summary.student_diagnosis_rate == 1.0
    assert summary.strict_abstention_accuracy == 1.0
    assert outcomes[0].numerical_correct is True
    assert outcomes[1].hint_valid is True
    assert outcomes[2].diagnosis_correct is True
    assert outcomes[3].abstained is True
