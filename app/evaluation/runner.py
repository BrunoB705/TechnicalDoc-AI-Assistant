import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings
from app.evaluation.metrics import (
    CaseOutcome,
    EvalCase,
    EvalSummary,
    aggregate_outcomes,
    check_answer_correctness,
    check_citation_validity,
    check_hint_progression,
    check_numerical_correctness,
    check_student_error_diagnosis,
    compute_rank,
)
from app.ingestion.pipeline import ingest_directory
from app.knowledge_bases.service import KnowledgeBaseService
from app.rag.service import RAGService


@dataclass
class CalibrationEntry:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance: float
    summary: EvalSummary


def load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        cases.append(EvalCase(**item))
    return cases


def build_eval_index(
    corpus_dir: Path,
    embeddings: Embeddings,
    work_dir: Path,
    chunk_size: int = 650,
    chunk_overlap: int = 100,
) -> tuple[Chroma, Settings, KnowledgeBaseService]:
    """Indexa cada subcarpeta de corpus_dir como una Knowledge Base independiente."""
    settings = Settings(
        chroma_dir=work_dir / "chroma",
        docs_dir=work_dir / "docs",
        catalog_path=work_dir / "catalog.sqlite3",
        collection_name=f"eval_{chunk_size}_{chunk_overlap}",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        openai_api_key=None,
    )
    vs = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    kb_svc = KnowledgeBaseService(vs, settings)
    for sub in sorted(corpus_dir.iterdir()):
        if sub.is_dir():
            kb_id = sub.name
            try:
                kb_svc.get_knowledge_base(kb_id)
            except Exception:
                kb_svc.create_knowledge_base(name=kb_id.capitalize(), kb_id=kb_id)
            ingest_directory(sub, vs, settings, knowledge_base_id=kb_id)
    return vs, settings, kb_svc


def evaluate_dataset(
    cases: list[EvalCase],
    vectorstore: Chroma,
    settings: Settings,
    kb_service: KnowledgeBaseService | None = None,
    llm: BaseChatModel | None = None,
    top_k: int | None = None,
    min_relevance: float | None = None,
) -> tuple[EvalSummary, list[CaseOutcome]]:
    """Ejecuta los casos del dataset midiendo recuperación, abstención y calidad de generación."""
    k = top_k if top_k is not None else settings.top_k
    threshold = min_relevance if min_relevance is not None else settings.min_relevance
    eval_settings = settings.model_copy(update={"top_k": k, "min_relevance": threshold})

    outcomes: list[CaseOutcome] = []
    for case in cases:
        t0 = time.perf_counter()
        found = vectorstore.similarity_search_with_relevance_scores(
            case.question,
            k=k,
            filter={"knowledge_base_id": case.knowledge_base_id},
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        kept = [(doc, score) for doc, score in found if score >= threshold]
        retrieved_sources = [doc.metadata.get("source", "") for doc, _ in kept]
        retrieved_kbs = [doc.metadata.get("knowledge_base_id", "general") for doc, _ in found]
        top_score = float(found[0][1]) if found else None

        generation_ms = 0.0
        abstained = len(kept) == 0
        abstention_reason: str | None = "below_threshold" if abstained else None
        abstention_message: str | None = (
            "No se encontró evidencia suficientemente relevante en la documentación."
            if abstained
            else None
        )
        generated_answer: str | None = None
        cited_sources: list[str] | None = None
        answer_correct: bool | None = None
        citations_valid: bool | None = None
        numerical_correct: bool | None = None
        hint_valid: bool | None = None
        diagnosis_correct: bool | None = None

        if llm is not None:
            if not abstained:
                rag = RAGService(vectorstore, llm, eval_settings, kb_service=kb_service)
                t1 = time.perf_counter()
                res = rag.answer(
                    case.question,
                    top_k=k,
                    knowledge_base_id=case.knowledge_base_id,
                    pedagogical_mode=case.pedagogical_mode,
                    strict_mode=case.strict_mode,
                )
                generation_ms = (time.perf_counter() - t1) * 1000.0
                abstained = not res.grounded
                abstention_reason = res.reason
                abstention_message = res.message
                generated_answer = res.answer
                cited_sources = [s.source for s in res.sources]
            else:
                cited_sources = []

            if not case.should_abstain:
                answer_correct = (not abstained) and check_answer_correctness(
                    generated_answer, case.expected_answer
                )
                citations_valid = (not abstained) and check_citation_validity(
                    cited_sources or [], case.expected_source
                )
                if case.expected_calculation:
                    numerical_correct = (not abstained) and check_numerical_correctness(
                        generated_answer, case.expected_calculation
                    )
                if case.forbidden_in_hints:
                    hint_valid = check_hint_progression(generated_answer, case.forbidden_in_hints)
                if case.pedagogical_mode == "check":
                    diag_terms = []
                    if case.expected_calculation and "error_type" in case.expected_calculation:
                        diag_terms.append(str(case.expected_calculation["error_type"]))
                    if case.expected_answer:
                        diag_terms.extend(
                            [
                                t.strip()
                                for t in case.expected_answer.split(",")
                                if len(t.strip()) < 35
                            ]
                        )
                    if not diag_terms and case.expected_answer:
                        diag_terms = [case.expected_answer]
                    diagnosis_correct = (not abstained) and check_student_error_diagnosis(
                        generated_answer, diag_terms
                    )
            else:
                answer_correct = abstained
                citations_valid = abstained

        leaked = any(kb != case.knowledge_base_id for kb in retrieved_kbs)
        rank = compute_rank(retrieved_sources, case.expected_source)

        outcomes.append(
            CaseOutcome(
                case=case,
                retrieved_sources=retrieved_sources,
                retrieved_kbs=retrieved_kbs,
                top_score=round(top_score, 4) if top_score is not None else None,
                abstained=abstained,
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                rank=rank,
                leaked_cross_kb=leaked,
                generated_answer=generated_answer,
                cited_sources=cited_sources,
                answer_correct=answer_correct,
                citations_valid=citations_valid,
                abstention_reason=abstention_reason,
                abstention_message=abstention_message,
                numerical_correct=numerical_correct,
                hint_valid=hint_valid,
                diagnosis_correct=diagnosis_correct,
            )
        )

    return aggregate_outcomes(outcomes), outcomes


def _outcomes_from_cached(
    cases: list[EvalCase],
    cached_hits: list[tuple[list[tuple], float]],
    top_k: int,
    threshold: float,
) -> list[CaseOutcome]:
    outcomes: list[CaseOutcome] = []
    for case, (found_all, retrieval_ms) in zip(cases, cached_hits, strict=False):
        found = found_all[:top_k]
        kept = [(doc, score) for doc, score in found if score >= threshold]
        retrieved_sources = [doc.metadata.get("source", "") for doc, _ in kept]
        retrieved_kbs = [doc.metadata.get("knowledge_base_id", "general") for doc, _ in found]
        top_score = float(found[0][1]) if found else None
        abstained = len(kept) == 0
        leaked = any(kb != case.knowledge_base_id for kb in retrieved_kbs)
        rank = compute_rank(retrieved_sources, case.expected_source)
        outcomes.append(
            CaseOutcome(
                case=case,
                retrieved_sources=retrieved_sources,
                retrieved_kbs=retrieved_kbs,
                top_score=round(top_score, 4) if top_score is not None else None,
                abstained=abstained,
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=0.0,
                rank=rank,
                leaked_cross_kb=leaked,
                abstention_reason="below_threshold" if abstained else None,
            )
        )
    return outcomes


def calibrate_grid(
    cases: list[EvalCase],
    corpus_dir: Path,
    embeddings: Embeddings,
    chunk_sizes: tuple[int, ...] = (500, 650, 1000),
    top_ks: tuple[int, ...] = (3, 4, 6),
    thresholds: tuple[float, ...] = (0.35, 0.75, 0.80, 0.82, 0.837, 0.85),
) -> list[CalibrationEntry]:
    """Compara configuraciones de chunk_size, top_k y min_relevance y las ordena por score."""
    results: list[CalibrationEntry] = []
    max_k = max(top_ks)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base_dir = Path(tmp)
        for csize in chunk_sizes:
            overlap = max(50, int(csize * 0.15))
            vs, _, _ = build_eval_index(
                corpus_dir,
                embeddings,
                base_dir / f"cs_{csize}",
                chunk_size=csize,
                chunk_overlap=overlap,
            )
            cached: list[tuple[list[tuple], float]] = []
            for case in cases:
                t0 = time.perf_counter()
                found = vs.similarity_search_with_relevance_scores(
                    case.question,
                    k=max_k,
                    filter={"knowledge_base_id": case.knowledge_base_id},
                )
                ms = (time.perf_counter() - t0) * 1000.0
                cached.append((found, ms))

            for k in top_ks:
                for thr in thresholds:
                    outcomes = _outcomes_from_cached(cases, cached, top_k=k, threshold=thr)
                    summary = aggregate_outcomes(outcomes)
                    results.append(
                        CalibrationEntry(
                            chunk_size=csize,
                            chunk_overlap=overlap,
                            top_k=k,
                            min_relevance=thr,
                            summary=summary,
                        )
                    )
    return sorted(
        results,
        key=lambda r: (
            r.summary.composite_score,
            r.summary.recall_at_k,
            r.summary.refusal_accuracy,
            r.summary.mrr,
        ),
        reverse=True,
    )


def format_markdown_report(
    baseline: EvalSummary,
    outcomes: list[CaseOutcome],
    calibration: list[CalibrationEntry] | None = None,
    config_params: dict | None = None,
    test_summary: EvalSummary | None = None,
    test_outcomes: list[CaseOutcome] | None = None,
) -> str:
    cfg = config_params or {}
    mode_label = (
        f"End-to-End con LLM ({cfg.get('llm_model', 'local')})"
        if baseline.evaluated_with_llm
        else "Recuperación Vectorial, Aislamiento y Abstención por Umbral (sin LLM)"
    )
    lines = [
        "# Informe de Evaluación Cuantitativa y Calibración RAG",
        "",
        "## 1. Configuración Evaluada",
        f"- **Modo de evaluación**: `{mode_label}`",
        f"- **Casos conjunto de calibración/desarrollo**: `{baseline.total_cases}` "
        f"(`{baseline.positive_cases}` positivos, `{baseline.abstention_cases}` de abstención)",
        f"- **Parámetros activos**: `chunk_size={cfg.get('chunk_size', 650)}`, "
        f"`top_k={cfg.get('top_k', 4)}`, `min_relevance={cfg.get('min_relevance', 0.80)}`",
        "",
        "## 2. Métricas de Recuperación y Aislamiento (Conjunto de Desarrollo)",
        "",
        "| Métrica | Valor | Descripción |",
        "|---|---|---|",
        f"| **Recall@k (Hit@k)** | `{baseline.recall_at_k:.2%}` | "
        "% de preguntas positivas con el documento esperado en top-k |",
        f"| **MRR** | `{baseline.mrr:.4f}` | Mean Reciprocal Rank del documento fuente esperado |",
        f"| **Refusal Accuracy** | `{baseline.refusal_accuracy:.2%}` | "
        "Tasa global de abstención correcta (negativas + cross-KB) |",
        f"| **Abstención Negativas** | `{baseline.negative_refusal_rate:.2%}` | "
        "Rechazo en preguntas fuera de dominio |",
        f"| **Abstención Cross-KB** | `{baseline.cross_kb_refusal_rate:.2%}` | "
        "Rechazo cuando la evidencia reside en otra Knowledge Base |",
        f"| **Falsas Abstenciones** | `{baseline.false_abstention_rate:.2%}` | "
        "Preguntas con evidencia que fueron rechazadas por el umbral |",
        f"| **Aislamiento entre KBs** | `{baseline.isolation_rate:.2%}` | "
        "Consultas sin ningún chunk filtrado desde una KB ajena |",
        f"| **Latencia Recuperación (media / p95)** | "
        f"`{baseline.avg_retrieval_ms:.2f} ms` / `{baseline.p95_retrieval_ms:.2f} ms` | "
        "Búsqueda vectorial filtrada en ChromaDB |",
        f"| **Puntaje Compuesto Recuperación** | `{baseline.composite_score:.4f}` | "
        "45% Recall@k + 25% MRR + 30% Refusal Accuracy |",
        "",
    ]

    if baseline.evaluated_with_llm:
        lines.extend(
            [
                "## 2.1 Métricas de Generación y Citas (con LLM)",
                "",
                "| Métrica de Generación | Valor | Descripción |",
                "|---|---|---|",
                f"| **Exactitud de Respuesta (Stemmed Overlap)** | "
                f"`{(baseline.answer_correctness or 0.0):.2%}` | "
                "Cobertura de hechos esperados en respuestas fundamentadas |",
                f"| **Validez de Citas (`[n]`)** | "
                f"`{(baseline.citation_validity_rate or 0.0):.2%}` | "
                "Respuestas con citas válidas que apuntan al documento esperado |",
                f"| **Latencia Media de Generación** | `{baseline.avg_generation_ms:.2f} ms` | "
                "Tiempo promedio de síntesis y validación en el LLM |",
                f"| **Índice End-to-End Ponderado** | `{(baseline.end_to_end_score or 0.0):.4f}` | "
                "30% Recall + 20% MRR + 20% Refusal + 15% Correctness + 15% Citations |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "> **Nota metodológica**: Esta corrida se ejecutó en modo determinista sin LLM "
                "(solo recuperación y umbral vectorial). Para evaluar también la generación y "
                "validez de citas del modelo de lenguaje, ejecutá "
                "`uv run python scripts/evaluate.py --with-llm`.",
                "",
            ]
        )

    if any(
        x is not None
        for x in (
            baseline.numerical_correctness_rate,
            baseline.hint_progression_rate,
            baseline.student_diagnosis_rate,
            baseline.strict_abstention_accuracy,
        )
    ):
        lines.extend(
            [
                "## 2.3 Métricas Pedagógicas del Modo Profesor",
                "",
                "| Métrica Pedagógica | Valor | Descripción |",
                "|---|---|---|",
            ]
        )
        if baseline.numerical_correctness_rate is not None:
            lines.append(
                f"| **Exactitud Numérica / Simbólica** | "
                f"`{baseline.numerical_correctness_rate:.2%}` | "
                "Precisión de cálculos, campos IEEE 754 y representaciones de bits |"
            )
        if baseline.hint_progression_rate is not None:
            lines.append(
                f"| **Calidad de Pistas Progresivas** | "
                f"`{baseline.hint_progression_rate:.2%}` | "
                "Pistas que orientan sin revelar de antemano la respuesta final prohibida |"
            )
        if baseline.student_diagnosis_rate is not None:
            lines.append(
                f"| **Diagnóstico de Errores del Estudiante** | "
                f"`{baseline.student_diagnosis_rate:.2%}` | "
                "Detección del paso incorrecto en resoluciones enviadas por el estudiante |"
            )
        if baseline.strict_abstention_accuracy is not None:
            lines.append(
                f"| **Abstención en Modo Estricto** | "
                f"`{baseline.strict_abstention_accuracy:.2%}` | "
                "Abstención fiel cuando faltan premisas en la base de conocimiento |"
            )
        lines.append("")

    if test_summary is not None:
        lines.extend(
            [
                "## 2.2 Validación en Conjunto de Prueba Independiente (Held-Out Test Set)",
                "",
                f"Evaluado sobre `{test_summary.total_cases}` preguntas no utilizadas "
                f"para seleccionar el umbral (`{test_summary.positive_cases}` positivas, "
                f"`{test_summary.abstention_cases}` de abstención):",
                "",
                "| Métrica (Held-Out Test Set) | Valor |",
                "|---|---|",
                f"| **Recall@k** | `{test_summary.recall_at_k:.2%}` |",
                f"| **MRR** | `{test_summary.mrr:.4f}` |",
                f"| **Refusal Accuracy** | `{test_summary.refusal_accuracy:.2%}` |",
                f"| **Aislamiento entre KBs** | `{test_summary.isolation_rate:.2%}` |",
                f"| **Puntaje Compuesto Recuperación** | `{test_summary.composite_score:.4f}` |",
            ]
        )
        if test_summary.evaluated_with_llm:
            lines.extend(
                [
                    f"| **Exactitud de Respuesta** | "
                    f"`{(test_summary.answer_correctness or 0.0):.2%}` |",
                    f"| **Validez de Citas (`[n]`)** | "
                    f"`{(test_summary.citation_validity_rate or 0.0):.2%}` |",
                    f"| **Índice End-to-End Ponderado** | "
                    f"`{(test_summary.end_to_end_score or 0.0):.4f}` |",
                ]
            )
        lines.append("")

    if calibration:
        lines.extend(
            [
                "## 3. Calibración Empírica (`chunk_size` × `top_k` × `min_relevance`)",
                "",
                "| Rank | `chunk_size` | `top_k` | `min_relevance` | Recall@k | MRR | "
                "Refusal Acc. | Isolation | Score |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for idx, entry in enumerate(calibration[:12], start=1):
            s = entry.summary
            lines.append(
                f"| {idx} | `{entry.chunk_size}` | `{entry.top_k}` | `{entry.min_relevance:.3f}` | "
                f"`{s.recall_at_k:.2%}` | `{s.mrr:.4f}` | `{s.refusal_accuracy:.2%}` | "
                f"`{s.isolation_rate:.2%}` | **{s.composite_score:.4f}** |"
            )
        lines.append("")

    all_outcomes = list(outcomes) + (list(test_outcomes) if test_outcomes else [])
    failures = [
        o
        for o in all_outcomes
        if (not o.case.should_abstain and (o.abstained or o.rank is None))
        or (o.case.should_abstain and not o.abstained)
        or (o.answer_correct is False)
        or (o.citations_valid is False)
        or o.leaked_cross_kb
    ]
    lines.append("## 4. Desglose de Desvíos y Diagnóstico")
    lines.append("")
    if not failures:
        lines.append("- Todos los casos cumplieron estrictamente el comportamiento esperado.")
    else:
        for o in failures:
            lines.append(
                f"- **Caso `{o.case.id}`** (`{o.case.category}`, KB=`{o.case.knowledge_base_id}`, "
                f"top_score=`{o.top_score}`):\n"
                f"  - **Pregunta**: {o.case.question}\n"
                f"  - **Abstenido**: `{o.abstained}` (Motivo: `{o.abstention_reason}`)\n"
                f"  - **Respuesta esperada**: {o.case.expected_answer or 'N/A'}\n"
                f"  - **Respuesta generada**: "
                f"{o.generated_answer or o.abstention_message or 'N/A'}\n"
                f"  - **Fuentes citadas**: `{o.cited_sources}` "
                f"(Esperada: `{o.case.expected_source}`)\n"
                f"  - **Correcto**: `{o.answer_correct}`, **Citas válidas**: `{o.citations_valid}`"
            )
    lines.append("")
    return "\n".join(lines)


def _serialize_outcome(o: CaseOutcome) -> dict:
    return {
        "id": o.case.id,
        "knowledge_base_id": o.case.knowledge_base_id,
        "category": o.case.category,
        "question": o.case.question,
        "should_abstain": o.case.should_abstain,
        "expected_source": o.case.expected_source,
        "expected_answer": o.case.expected_answer,
        "retrieved_sources": o.retrieved_sources,
        "retrieved_kbs": o.retrieved_kbs,
        "top_score": o.top_score,
        "abstained": o.abstained,
        "abstention_reason": o.abstention_reason,
        "abstention_message": o.abstention_message,
        "generated_answer": o.generated_answer,
        "cited_sources": o.cited_sources,
        "answer_correct": o.answer_correct,
        "citations_valid": o.citations_valid,
        "rank": o.rank,
        "leaked_cross_kb": o.leaked_cross_kb,
        "retrieval_ms": o.retrieval_ms,
        "generation_ms": o.generation_ms,
    }


def serialize_report(
    summary: EvalSummary,
    outcomes: list[CaseOutcome],
    calibration: list[CalibrationEntry] | None = None,
    test_summary: EvalSummary | None = None,
    test_outcomes: list[CaseOutcome] | None = None,
    config_params: dict | None = None,
) -> dict:
    return {
        "config": config_params or {},
        "summary": asdict(summary),
        "test_summary": asdict(test_summary) if test_summary is not None else None,
        "calibration_top": [
            {
                "chunk_size": c.chunk_size,
                "chunk_overlap": c.chunk_overlap,
                "top_k": c.top_k,
                "min_relevance": c.min_relevance,
                **asdict(c.summary),
            }
            for c in (calibration or [])[:15]
        ],
        "cases": [_serialize_outcome(o) for o in outcomes],
        "test_cases": [_serialize_outcome(o) for o in (test_outcomes or [])],
    }
