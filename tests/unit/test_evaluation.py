from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.evaluation.metrics import (
    CaseOutcome,
    EvalCase,
    aggregate_outcomes,
    compute_rank,
    hit_at_k,
    reciprocal_rank,
)
from app.evaluation.runner import (
    build_eval_index,
    calibrate_grid,
    evaluate_dataset,
    format_markdown_report,
    load_dataset,
    serialize_report,
)

DATASET_PATH = Path("eval/dataset.jsonl")
CORPUS_PATH = Path("eval/corpus")


class DomainKeywordEmbeddings(Embeddings):
    """Embeddings sintéticos deterministas sin red que distinguen los dominios del corpus."""

    GROUPS = (
        (
            "cache", "caché", "l1", "l2", "l3", "lru",
            "write-through", "write-back", "correspondencia",
        ),
        (
            "complemento", "ieee", "754", "exponente", "mantisa",
            "overflow", "alu", "signo", "banderas",
        ),
        (
            "pipeline", "risc", "raw", "forwarding",
            "load-use", "predictor", "etapas", "saltos",
        ),
        (
            "fastapi", "puerto", "8000", "question", "top_k",
            "llm_timeout", "stream", "citas", "cita",
        ),
        (
            "e5-small", "sha-256", "doc_id", "64 kb",
            "upload_chunk_size", "5461", "sqlite", "embeddings",
        ),
        (
            "python 3.12", "2 gb", "openai_base_url",
            "ingest.py", "10 minutos", "logger", "uvicorn",
        ),
    )

    def _encode(self, text: str) -> list[float]:
        low = text.lower()
        vec = [float(sum(1 for kw in grp if kw in low)) for grp in self.GROUPS]
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            return [x / norm for x in vec] + [0.0]
        # Vector ortogonal para preguntas negativas fuera de dominio
        return [0.0 for _ in self.GROUPS] + [1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def test_dataset_jsonl_has_40_to_50_cases_with_all_categories():
    cases = load_dataset(DATASET_PATH)
    assert 40 <= len(cases) <= 50
    ids = [c.id for c in cases]
    assert len(set(ids)) == len(ids)

    categories = {c.category for c in cases}
    assert categories == {"positive", "negative", "cross_kb"}
    kbs = {c.knowledge_base_id for c in cases}
    assert kbs == {"arquitectura", "backend", "operaciones"}

    for c in cases:
        if c.category == "positive":
            assert c.should_abstain is False
            assert c.expected_source and c.expected_answer
        else:
            assert c.should_abstain is True
            assert c.expected_source is None


def test_hit_at_k_and_mrr_calculations():
    sources = ["otro.md", "memorias_y_cache.md", "tercero.md"]
    assert compute_rank(sources, "memorias_y_cache.md") == 2
    assert hit_at_k(sources, "memorias_y_cache.md", k=1) is False
    assert hit_at_k(sources, "memorias_y_cache.md", k=2) is True
    assert reciprocal_rank(sources, "memorias_y_cache.md") == 0.5
    assert reciprocal_rank(sources, "inexistente.md") == 0.0


def test_aggregate_outcomes_computes_recall_mrr_refusal_and_isolation():
    c_pos1 = EvalCase("1", "arquitectura", "positive", False, "q1", "a.md", "ans")
    c_pos2 = EvalCase("2", "arquitectura", "positive", False, "q2", "b.md", "ans")
    c_neg = EvalCase("3", "arquitectura", "negative", True, "q3")
    c_cross = EvalCase("4", "backend", "cross_kb", True, "q4")

    outcomes = [
        CaseOutcome(c_pos1, ["a.md"], ["arquitectura"], 0.85, False, 5.0, rank=1),
        CaseOutcome(c_pos2, ["x.md", "b.md"], ["arquitectura"], 0.75, False, 7.0, rank=2),
        CaseOutcome(c_neg, [], ["arquitectura"], 0.20, True, 4.0, rank=None),
        CaseOutcome(c_cross, [], ["backend"], 0.25, True, 6.0, rank=None),
    ]
    summary = aggregate_outcomes(outcomes)
    assert summary.total_cases == 4
    assert summary.recall_at_k == 1.0
    assert summary.mrr == 0.75  # (1/1 + 1/2) / 2
    assert summary.refusal_accuracy == 1.0
    assert summary.negative_refusal_rate == 1.0
    assert summary.cross_kb_refusal_rate == 1.0
    assert summary.isolation_rate == 1.0


def test_evaluate_dataset_and_calibration_grid_offline(tmp_path):
    cases = load_dataset(DATASET_PATH)
    emb = DomainKeywordEmbeddings()
    vs, settings, kb_svc = build_eval_index(
        CORPUS_PATH, emb, tmp_path / "idx", chunk_size=1000, chunk_overlap=150
    )
    llm = FakeListChatModel(responses=["Respuesta con cita [1]."] * len(cases))
    summary, outcomes = evaluate_dataset(
        cases,
        vs,
        settings,
        kb_service=kb_svc,
        llm=llm,
        top_k=4,
        min_relevance=0.25,
    )
    assert summary.total_cases == len(cases)
    assert summary.isolation_rate == 1.0
    assert summary.recall_at_k >= 0.85
    assert summary.refusal_accuracy >= 0.70

    grid = calibrate_grid(
        cases[:12],
        CORPUS_PATH,
        emb,
        chunk_sizes=(500, 1000),
        top_ks=(3, 4),
        thresholds=(0.20, 0.50),
    )
    assert len(grid) == 8
    assert grid[0].summary.composite_score >= grid[-1].summary.composite_score

    md = format_markdown_report(summary, outcomes, calibration=grid)
    assert "Recall@k" in md and "Calibración Empírica" in md
    payload = serialize_report(summary, outcomes, grid)
    assert payload["summary"]["total_cases"] == len(cases)
    assert len(payload["calibration_top"]) == 8
