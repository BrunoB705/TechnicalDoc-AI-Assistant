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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta evaluación cuantitativa RAG y calibración empírica"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/dataset.jsonl"),
        help="Ruta al archivo JSONL con el ground truth (por defecto: eval/dataset.jsonl)",
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
    args = parser.parse_args()

    settings = get_settings()
    embeddings = get_embeddings(settings, lazy=False)
    cases = load_dataset(args.dataset)

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
            top_k=top_k,
            min_relevance=min_rel,
        )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / "latest.json"
    md_path = args.reports_dir / "calibration_report.md"

    payload = serialize_report(summary, outcomes, calibration)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = format_markdown_report(
        summary,
        outcomes,
        calibration=calibration,
        config_params={"chunk_size": csize, "top_k": top_k, "min_relevance": min_rel},
    )
    md_path.write_text(md, encoding="utf-8")

    print(
        f"Evaluados {summary.total_cases} casos | "
        f"Recall@{top_k}={summary.recall_at_k:.2%} | "
        f"MRR={summary.mrr:.4f} | "
        f"RefusalAcc={summary.refusal_accuracy:.2%} | "
        f"Isolation={summary.isolation_rate:.2%} | "
        f"Score={summary.composite_score:.4f}"
    )
    if calibration:
        top = calibration[0]
        print(
            f"Mejor configuración calibrada: chunk_size={top.chunk_size}, "
            f"top_k={top.top_k}, min_relevance={top.min_relevance:.2f} "
            f"(Score={top.summary.composite_score:.4f})"
        )


if __name__ == "__main__":
    main()
