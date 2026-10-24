import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.evaluation.runner import (  # noqa: E402
    build_eval_index,
    calibrate_grid,
    evaluate_dataset,
    format_markdown_report,
    load_dataset,
    serialize_report,
)
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.llm import get_llm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta evaluación cuantitativa RAG y calibración empírica"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/dataset.jsonl"),
        help="Ruta al archivo JSONL de desarrollo/calibración (por defecto: eval/dataset.jsonl)",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=Path("eval/test_dataset.jsonl"),
        help="Ruta al conjunto de prueba independiente (held-out test set)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("eval/corpus"),
        help="Directorio raíz con subcarpetas por Knowledge Base",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("eval/reports"),
        help="Directorio de salida para informes JSON y Markdown",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="chunk_size para la corrida base (por defecto usa Settings)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top_k para la corrida base (por defecto usa Settings)",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=None,
        help="min_relevance para la corrida base (por defecto usa Settings)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Compara grilla de chunk_size (500 vs 1000), top_k (3, 4, 6) y min_relevance",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Ejecuta también el LLM para medir exactitud de respuesta y validez de citas [n]",
    )
    args = parser.parse_args()

    settings = get_settings()
    embeddings = get_embeddings(settings, lazy=False)
    cases = load_dataset(args.dataset)
    test_cases = (
        load_dataset(args.test_dataset) if args.test_dataset and args.test_dataset.exists() else []
    )
    llm = get_llm(settings) if args.with_llm else None

    csize = args.chunk_size or settings.chunk_size
    coverlap = max(50, int(csize * 0.15)) if args.chunk_size else settings.chunk_overlap
    top_k = args.top_k or settings.top_k
    min_rel = args.min_relevance if args.min_relevance is not None else settings.min_relevance

    calibration = None
    if args.calibrate:
        calibration = calibrate_grid(cases, args.corpus, embeddings)
        best = calibration[0]
        if args.min_relevance is None and args.chunk_size is None and args.top_k is None:
            csize = best.chunk_size
            coverlap = best.chunk_overlap
            top_k = best.top_k
            min_rel = best.min_relevance

    test_summary = None
    test_outcomes = []
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            vs, eval_settings, kb_svc = build_eval_index(
                args.corpus,
                embeddings,
                Path(tmp),
                chunk_size=csize,
                chunk_overlap=coverlap,
            )
            summary, outcomes = evaluate_dataset(
                cases,
                vs,
                eval_settings,
                kb_service=kb_svc,
                llm=llm,
                top_k=top_k,
                min_relevance=min_rel,
            )
            if test_cases:
                test_summary, test_outcomes = evaluate_dataset(
                    test_cases,
                    vs,
                    eval_settings,
                    kb_service=kb_svc,
                    llm=llm,
                    top_k=top_k,
                    min_relevance=min_rel,
                )
    except Exception as exc:
        if args.with_llm and "LLMUnavailableError" in type(exc).__name__:
            provider = settings.llm_provider
            url = settings.openai_base_url if provider == "openai" else settings.ollama_base_url
            target_url = url or "OpenAI Cloud"
            print(
                f"\n[ERROR] No se pudo conectar con el servidor LLM ({provider}) en {target_url}.\n"
                f"Detalle: {exc}\n\n"
                "Para usar '--with-llm':\n"
                "  1. Si usás LM Studio: abrilo y hacé clic en 'Start Server' en http://localhost:1234.\n"
                "  2. Si usás Ollama: ejecutá 'ollama serve' en http://localhost:11434.\n"
                "  3. Si usás OpenAI en la nube: configurá OPENAI_API_KEY en tu .env.\n\n"
                "Nota: Podés ejecutar la evaluación de recuperación, umbrales y aislamiento con:\n"
                "      uv run python scripts/evaluate.py\n",
                file=sys.stderr,
            )
            sys.exit(1)
        raise

    active_cfg = {
        "chunk_size": csize,
        "chunk_overlap": coverlap,
        "top_k": top_k,
        "min_relevance": min_rel,
        "llm_provider": settings.llm_provider,
        "llm_model": (
            settings.openai_model if settings.llm_provider == "openai" else settings.ollama_model
        ),
        "embedding_model": settings.embedding_model,
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / "latest.json"
    cal_md_path = args.reports_dir / "calibration_report.md"
    latest_md_path = args.reports_dir / "latest_report.md"

    payload = serialize_report(
        summary,
        outcomes,
        calibration,
        test_summary=test_summary,
        test_outcomes=test_outcomes,
        config_params=active_cfg,
    )
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = format_markdown_report(
        summary,
        outcomes,
        calibration=calibration,
        config_params=active_cfg,
        test_summary=test_summary,
        test_outcomes=test_outcomes,
    )
    cal_md_path.write_text(md, encoding="utf-8")
    latest_md_path.write_text(md, encoding="utf-8")

    print(
        f"[Dev Set: {summary.total_cases} casos] "
        f"Recall@{top_k}={summary.recall_at_k:.2%} | "
        f"MRR={summary.mrr:.4f} | "
        f"RefusalAcc={summary.refusal_accuracy:.2%} | "
        f"Isolation={summary.isolation_rate:.2%} | "
        f"Score={summary.composite_score:.4f}"
    )
    if test_summary is not None:
        print(
            f"[Held-Out Test Set: {test_summary.total_cases} casos] "
            f"Recall@{top_k}={test_summary.recall_at_k:.2%} | "
            f"MRR={test_summary.mrr:.4f} | "
            f"RefusalAcc={test_summary.refusal_accuracy:.2%} | "
            f"Isolation={test_summary.isolation_rate:.2%} | "
            f"Score={test_summary.composite_score:.4f}"
        )
    if calibration:
        top = calibration[0]
        print(
            f"Mejor configuración calibrada: chunk_size={top.chunk_size}, "
            f"top_k={top.top_k}, min_relevance={top.min_relevance:.3f} "
            f"(Score={top.summary.composite_score:.4f})"
        )


if __name__ == "__main__":
    main()
