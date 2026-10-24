from dataclasses import dataclass
from typing import Literal

Category = Literal["positive", "negative", "cross_kb"]


@dataclass
class EvalCase:
    id: str
    knowledge_base_id: str
    category: Category
    should_abstain: bool
    question: str
    expected_source: str | None = None
    expected_answer: str | None = None


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

    # Recall@k y MRR sobre preguntas positivas (solo cuentan como acierto si no se abstuvo)
    hits = [
        1.0
        for o in positives
        if not o.abstained and o.rank is not None
    ]
    rr_values = [
        (1.0 / o.rank) if (not o.abstained and o.rank is not None) else 0.0
        for o in positives
    ]
    false_abstentions = [1.0 for o in positives if o.abstained]

    recall_val = sum(hits) / len(positives) if positives else 0.0
    mrr_val = sum(rr_values) / len(positives) if positives else 0.0
    false_abst_val = sum(false_abstentions) / len(positives) if positives else 0.0

    refusal_acc = (
        sum(1.0 for o in abstentions if o.abstained) / len(abstentions)
        if abstentions
        else 1.0
    )
    neg_refusal = (
        sum(1.0 for o in negatives if o.abstained) / len(negatives)
        if negatives
        else 1.0
    )
    cross_refusal = (
        sum(1.0 for o in cross_kbs if o.abstained) / len(cross_kbs)
        if cross_kbs
        else 1.0
    )

    isolation = sum(1.0 for o in outcomes if not o.leaked_cross_kb) / len(outcomes)

    ret_times = [o.retrieval_ms for o in outcomes]
    gen_times = [o.generation_ms for o in outcomes]
    avg_ret = sum(ret_times) / len(ret_times)
    p95_ret = _percentile(ret_times, 95.0)
    avg_gen = sum(gen_times) / len(gen_times)

    # Puntaje compuesto ponderado para calibración:
    # 45% Recall@k + 25% MRR + 30% Refusal Accuracy, penalizado si falla el aislamiento
    composite = (0.45 * recall_val + 0.25 * mrr_val + 0.30 * refusal_acc) * isolation

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
    )
