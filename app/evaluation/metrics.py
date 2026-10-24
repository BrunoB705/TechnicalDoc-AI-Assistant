import re
from dataclasses import dataclass
from typing import Literal

Category = Literal["positive", "negative", "cross_kb"]

_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "al",
    "a",
    "en",
    "con",
    "por",
    "para",
    "sin",
    "sobre",
    "entre",
    "hacia",
    "hasta",
    "como",
    "que",
    "se",
    "su",
    "sus",
    "es",
    "son",
    "o",
    "u",
    "y",
    "e",
    "ni",
    "no",
    "si",
    "cada",
    "cuando",
    "donde",
    "muy",
    "mas",
    "más",
    "pero",
}

_SPANISH_SUFFIXES = (
    "amente",
    "ciones",
    "cion",
    "ción",
    "ados",
    "adas",
    "idos",
    "idas",
    "ando",
    "iendo",
    "ivas",
    "ivos",
    "iva",
    "ivo",
    "ado",
    "ada",
    "ido",
    "ida",
    "ar",
    "er",
    "ir",
    "as",
    "es",
    "os",
    "a",
    "e",
    "o",
)


def _stem_word(word: str) -> str:
    """Lematización ligera de sufijos comunes en español."""
    w = word.lower().strip()
    for suff in _SPANISH_SUFFIXES:
        if len(w) > len(suff) + 3 and w.endswith(suff):
            return w[: -len(suff)]
    return w


@dataclass
class EvalCase:
    id: str
    knowledge_base_id: str
    category: str
    should_abstain: bool
    question: str
    expected_source: str | None = None
    expected_answer: str | None = None
    pedagogical_mode: str | None = None
    student_attempt: str | None = None
    expected_calculation: dict[str, object] | None = None
    forbidden_in_hints: list[str] | None = None
    strict_mode: bool = False


@dataclass
class CaseOutcome:
    case: EvalCase
    retrieved_sources: list[str]
    retrieved_kbs: list[str]
    top_score: float | None
    abstained: bool
    retrieval_ms: float
    generation_ms: float = 0.0
    rank: int | None = None  # posición 1-based del primer chunk del documento esperado
    leaked_cross_kb: bool = False
    generated_answer: str | None = None
    cited_sources: list[str] | None = None
    answer_correct: bool | None = None
    citations_valid: bool | None = None
    abstention_reason: str | None = None
    abstention_message: str | None = None
    numerical_correct: bool | None = None
    hint_valid: bool | None = None
    diagnosis_correct: bool | None = None


@dataclass
class EvalSummary:
    total_cases: int
    positive_cases: int
    abstention_cases: int
    recall_at_k: float
    mrr: float
    refusal_accuracy: float
    negative_refusal_rate: float
    cross_kb_refusal_rate: float
    false_abstention_rate: float
    isolation_rate: float
    avg_retrieval_ms: float
    p95_retrieval_ms: float
    avg_generation_ms: float
    composite_score: float
    evaluated_with_llm: bool = False
    answer_correctness: float | None = None
    citation_validity_rate: float | None = None
    end_to_end_score: float | None = None
    numerical_correctness_rate: float | None = None
    hint_progression_rate: float | None = None
    student_diagnosis_rate: float | None = None
    strict_abstention_accuracy: float | None = None


def compute_rank(retrieved_sources: list[str], expected_source: str | None) -> int | None:
    """Devuelve la posición 1-based de la primera aparición de expected_source, o None."""
    if not expected_source:
        return None
    for idx, src in enumerate(retrieved_sources, start=1):
        if src == expected_source or src.endswith(f"/{expected_source}"):
            return idx
    return None


def hit_at_k(
    retrieved_sources: list[str], expected_source: str | None, k: int | None = None
) -> bool:
    """True si expected_source aparece dentro de los primeros k fragmentos recuperados."""
    sliced = retrieved_sources[:k] if k is not None else retrieved_sources
    return compute_rank(sliced, expected_source) is not None


def reciprocal_rank(retrieved_sources: list[str], expected_source: str | None) -> float:
    """1 / rank si el documento esperado fue recuperado, o 0.0 en caso contrario."""
    r = compute_rank(retrieved_sources, expected_source)
    return (1.0 / r) if r is not None else 0.0


def _extract_stemmed_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ_.-]+", text.lower())
    stems = set()
    for w in words:
        if len(w) >= 2 and w not in _STOPWORDS:
            stems.add(_stem_word(w))
    return stems


def check_answer_correctness(generated: str | None, expected: str | None) -> bool:
    """Comprueba de forma determinista que la respuesta generada cubra términos clave esperados."""
    if not expected:
        return generated is None
    if not generated:
        return False
    expected_stems = _extract_stemmed_keywords(expected)
    if not expected_stems:
        return True
    gen_stems = _extract_stemmed_keywords(generated)
    overlap = len(expected_stems & gen_stems) / len(expected_stems)
    return overlap >= 0.35


def check_citation_validity(
    cited_sources: list[str],
    expected_source: str | None,
) -> bool:
    """Comprueba que existan citas válidas y que al menos una apunte al documento esperado."""
    if not cited_sources:
        return False
    if not expected_source:
        return True
    return compute_rank(cited_sources, expected_source) is not None


def check_numerical_correctness(
    generated: str | None,
    expected_calculation: dict[str, object] | None,
) -> bool:
    """Verifica que los campos numéricos y de bits calculados aparezcan en el texto generado."""
    if not expected_calculation:
        return True
    if not generated:
        return False
    gen_norm = re.sub(r"\s+", "", generated.lower())
    for val in expected_calculation.values():
        val_str = str(val).lower()
        val_clean = re.sub(r"\s+", "", val_str)
        if val_clean not in gen_norm and val_str not in generated.lower():
            return False
    return True


def check_hint_progression(
    generated: str | None,
    forbidden_in_hints: list[str] | None,
) -> bool:
    """Comprueba que una respuesta en modo pistas no revele la solución final prohibida."""
    if not forbidden_in_hints:
        return True
    if not generated:
        return True
    gen_norm = re.sub(r"\s+", "", generated.lower())
    for forbidden in forbidden_in_hints:
        clean = re.sub(r"\s+", "", str(forbidden).lower())
        if clean and (clean in gen_norm or str(forbidden).lower() in generated.lower()):
            return False
    return True


def check_student_error_diagnosis(
    generated: str | None,
    expected_diagnosis_terms: list[str] | None,
) -> bool:
    """Comprueba que la corrección identifique los términos clave del error del alumno."""
    if not expected_diagnosis_terms:
        return True
    if not generated:
        return False
    gen_low = generated.lower()
    return any(term.lower() in gen_low for term in expected_diagnosis_terms)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def aggregate_outcomes(outcomes: list[CaseOutcome]) -> EvalSummary:
    if not outcomes:
        return EvalSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    positives = [o for o in outcomes if not o.case.should_abstain]
    abstentions = [o for o in outcomes if o.case.should_abstain]
    negatives = [o for o in outcomes if o.case.category == "negative"]
    cross_kbs = [o for o in outcomes if o.case.category == "cross_kb"]

    hits = [1.0 for o in positives if not o.abstained and o.rank is not None]
    rr_values = [
        (1.0 / o.rank) if (not o.abstained and o.rank is not None) else 0.0 for o in positives
    ]
    false_abstentions = [1.0 for o in positives if o.abstained]

    recall_val = sum(hits) / len(positives) if positives else 0.0
    mrr_val = sum(rr_values) / len(positives) if positives else 0.0
    false_abst_val = sum(false_abstentions) / len(positives) if positives else 0.0

    refusal_acc = (
        sum(1.0 for o in abstentions if o.abstained) / len(abstentions) if abstentions else 1.0
    )
    neg_refusal = sum(1.0 for o in negatives if o.abstained) / len(negatives) if negatives else 1.0
    cross_refusal = (
        sum(1.0 for o in cross_kbs if o.abstained) / len(cross_kbs) if cross_kbs else 1.0
    )

    isolation = sum(1.0 for o in outcomes if not o.leaked_cross_kb) / len(outcomes)

    ret_times = [o.retrieval_ms for o in outcomes]
    gen_times = [o.generation_ms for o in outcomes]
    avg_ret = sum(ret_times) / len(ret_times)
    p95_ret = _percentile(ret_times, 95.0)
    avg_gen = sum(gen_times) / len(gen_times)

    evaluated_llm = any(
        o.answer_correct is not None or o.citations_valid is not None for o in outcomes
    )
    ans_corr: float | None = None
    cit_val: float | None = None
    end_to_end: float | None = None
    if evaluated_llm and positives:
        ans_corr = round(sum(1.0 for o in positives if o.answer_correct) / len(positives), 4)
        cit_val = round(sum(1.0 for o in positives if o.citations_valid) / len(positives), 4)
        # Índice End-to-End integrando recuperación, abstención, exactitud y fundamentación
        end_to_end = round(
            (
                0.30 * recall_val
                + 0.20 * mrr_val
                + 0.20 * refusal_acc
                + 0.15 * ans_corr
                + 0.15 * cit_val
            )
            * isolation,
            4,
        )

    # Puntaje compuesto de recuperación y abstención:
    # 45% Recall@k + 25% MRR + 30% Refusal Accuracy, penalizado si falla el aislamiento
    composite = (0.45 * recall_val + 0.25 * mrr_val + 0.30 * refusal_acc) * isolation

    num_outcomes = [o for o in outcomes if o.numerical_correct is not None]
    num_rate = (
        round(sum(1.0 for o in num_outcomes if o.numerical_correct) / len(num_outcomes), 4)
        if num_outcomes
        else None
    )

    hint_outcomes = [o for o in outcomes if o.hint_valid is not None]
    hint_rate = (
        round(sum(1.0 for o in hint_outcomes if o.hint_valid) / len(hint_outcomes), 4)
        if hint_outcomes
        else None
    )

    diag_outcomes = [o for o in outcomes if o.diagnosis_correct is not None]
    diag_rate = (
        round(sum(1.0 for o in diag_outcomes if o.diagnosis_correct) / len(diag_outcomes), 4)
        if diag_outcomes
        else None
    )

    strict_cases = [o for o in outcomes if o.case.strict_mode and o.case.should_abstain]
    strict_acc = (
        round(sum(1.0 for o in strict_cases if o.abstained) / len(strict_cases), 4)
        if strict_cases
        else None
    )

    return EvalSummary(
        total_cases=len(outcomes),
        positive_cases=len(positives),
        abstention_cases=len(abstentions),
        recall_at_k=round(recall_val, 4),
        mrr=round(mrr_val, 4),
        refusal_accuracy=round(refusal_acc, 4),
        negative_refusal_rate=round(neg_refusal, 4),
        cross_kb_refusal_rate=round(cross_refusal, 4),
        false_abstention_rate=round(false_abst_val, 4),
        isolation_rate=round(isolation, 4),
        avg_retrieval_ms=round(avg_ret, 2),
        p95_retrieval_ms=round(p95_ret, 2),
        avg_generation_ms=round(avg_gen, 2),
        composite_score=round(composite, 4),
        evaluated_with_llm=evaluated_llm,
        answer_correctness=ans_corr,
        citation_validity_rate=cit_val,
        end_to_end_score=end_to_end,
        numerical_correctness_rate=num_rate,
        hint_progression_rate=hint_rate,
        student_diagnosis_rate=diag_rate,
        strict_abstention_accuracy=strict_acc,
    )
