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
