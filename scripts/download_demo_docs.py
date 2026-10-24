import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.ingestion.pipeline import ingest_directory  # noqa: E402
from app.knowledge_bases.service import KnowledgeBaseService  # noqa: E402
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.vectorstore import get_vectorstore  # noqa: E402


@dataclass(frozen=True)
class DemoDocSpec:
    kb_id: str
    kb_name: str
    kb_description: str
    filename: str
    url: str
    fallback_content: str


DEMO_SPECS: tuple[DemoDocSpec, ...] = (
    DemoDocSpec(
        kb_id="python-peps",
        kb_name="Python PEPs",
        kb_description="Propuestas de mejora de Python (PEP 8, PEP 20, PEP 484).",
        filename="pep-0008-style-guide.md",
        url="https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst",
        fallback_content=(
            "# PEP 8 — Guía de Estilo para Código Python\n\n"
            "## Indentación y Longitud de Línea\n"
            "- Utilice **4 espacios** por nivel de indentación. Nunca mezcle tabs y espacios.\n"
            "- Limite las líneas a **79 caracteres** para código (o hasta 99/100 si el equipo "
            "lo acuerda) y **72 caracteres** para docstrings y comentarios.\n\n"
            "## Convenciones de Nombres\n"
            "- **Funciones, variables y módulos**: `snake_case` en minúsculas con guiones bajos.\n"
            "- **Clases y Excepciones**: `CapWords` (`PascalCase`).\n"
            "- **Constantes**: `UPPER_CASE_WITH_UNDERSCORES` a nivel de módulo.\n\n"
            "## Importaciones\n"
            "Las importaciones deben agruparse en tres bloques separados por línea en blanco:\n"
            "1. Biblioteca estándar de Python.\n"
            "2. Bibliotecas de terceros relacionadas.\n"
            "3. Importaciones propias de la aplicación o biblioteca local.\n"
        ),
    ),
    DemoDocSpec(
        kb_id="python-peps",
        kb_name="Python PEPs",
        kb_description="Propuestas de mejora de Python (PEP 8, PEP 20, PEP 484).",
        filename="pep-0484-type-hints.md",
        url="https://raw.githubusercontent.com/python/peps/main/peps/pep-0484.rst",
        fallback_content=(
            "# PEP 484 — Type Hints (Anotaciones de Tipos en Python)\n\n"
            "## Objetivo y Filosofía\n"
            "PEP 484 introduce una sintaxis estándar para anotaciones de tipos en funciones, "
            "métodos y variables utilizando el módulo `typing`. Python continúa siendo un "
            "lenguaje de tipado dinámico: los chequeos de tipos son opcionales y se realizan "
            "mediante analizadores estáticos fuera de tiempo de ejecución.\n\n"
            "## Constructores Fundamentales\n"
            "- `Union[X, Y]` (o `X | Y` desde Python 3.10+): acepta valores de tipo X o Y.\n"
            "- `Optional[X]` (o `X | None`): equivalente a una unión con `None`.\n"
            "- `Callable[[Arg1, Arg2], ReturnType]`: describe funciones y callbacks tipados.\n"
            "- `TypeVar`: permite definir funciones y clases genéricas parametrizadas.\n"
        ),
    ),
    DemoDocSpec(
        kb_id="fastapi-guide",
        kb_name="FastAPI Guide",
        kb_description="Documentación técnica de FastAPI: dependencias, concurrencia y streaming.",
        filename="fastapi-dependencies-and-async.md",
        url="https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/async.md",
        fallback_content=(
            "# FastAPI — Inyección de Dependencias y Concurrencia Asíncrona\n\n"
            "## Sistema de Inyección de Dependencias (`Depends`)\n"
            "FastAPI incluye un motor declarativo de inyección de dependencias con `Depends()`. "
            "Permite compartir conexiones, configuraciones cacheadas y servicios de dominio, "
            "además de sustituirlos en pruebas automatizadas usando "
            "`app.dependency_overrides[dep] = mock_fn`.\n\n"
            "## Rutas `async def` vs `def`\n"
            "- Use `async def` cuando invoque corrutinas nativas con `await` (por ejemplo, "
            "streaming de tokens desde un LLM con `astream` o clientes HTTP asíncronos).\n"
            "- Use `def` estándar cuando la ruta ejecute operaciones bloqueantes de CPU o E/S "
            "síncrona (FastAPI las ejecuta en un *threadpool* externo sin bloquear el loop).\n"
        ),
    ),
)


def materialize_demo_docs(
    output_dir: Path,
    offline: bool = False,
    timeout: float = 5.0,
    include_eval_corpus: Path | None = Path("eval/corpus"),
) -> dict[str, list[Path]]:
    """Descarga o genera el corpus público en output_dir/{kb_id}/."""
    created: dict[str, list[Path]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in DEMO_SPECS:
        kb_dir = output_dir / spec.kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        target = kb_dir / spec.filename

        text: str | None = None
        if not offline:
            try:
                resp = httpx.get(spec.url, timeout=timeout, follow_redirects=True)
                if resp.status_code == 200 and len(resp.text.strip()) >= 100:
                    text = resp.text
            except Exception:
                text = None

        if text is None:
            text = spec.fallback_content

        target.write_text(text, encoding="utf-8")
        created.setdefault(spec.kb_id, []).append(target)

    if include_eval_corpus and include_eval_corpus.is_dir():
        for sub in sorted(include_eval_corpus.iterdir()):
            if not sub.is_dir():
                continue
            kb_dir = output_dir / sub.name
            kb_dir.mkdir(parents=True, exist_ok=True)
            for doc in sorted(sub.glob("*.md")):
                dest = kb_dir / doc.name
                shutil.copyfile(doc, dest)
                created.setdefault(sub.name, []).append(dest)

    return created


def ingest_demo_corpus(demo_dir: Path) -> dict[str, tuple[int, int]]:
    """Registra cada subcarpeta de demo_dir en el catálogo e indexa sus documentos."""
    settings = get_settings()
    vs = get_vectorstore(settings, get_embeddings(settings))
    kb_svc = KnowledgeBaseService(vs, settings)
    spec_meta = {s.kb_id: (s.kb_name, s.kb_description) for s in DEMO_SPECS}

    stats: dict[str, tuple[int, int]] = {}
    for sub in sorted(demo_dir.iterdir()):
        if not sub.is_dir():
            continue
        kb_id = sub.name
        name, desc = spec_meta.get(
            kb_id, (kb_id.replace("-", " ").title(), f"Corpus de demostración ({kb_id}).")
        )
        try:
            kb_svc.get_knowledge_base(kb_id)
        except Exception:
            kb_svc.create_knowledge_base(name=name, description=desc, kb_id=kb_id)

        kb_docs_dir = settings.docs_dir / kb_id
        kb_docs_dir.mkdir(parents=True, exist_ok=True)
        for f in sub.iterdir():
            if f.is_file():
                shutil.copyfile(f, kb_docs_dir / f.name)

        n_docs, n_chunks = ingest_directory(kb_docs_dir, vs, settings, knowledge_base_id=kb_id)
        stats[kb_id] = (n_docs, n_chunks)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga o genera el corpus público de demostración organizado en KBs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo_docs"),
        help="Directorio destino (por defecto: data/demo_docs)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usar directamente los documentos públicos embebidos sin HTTP",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Crear las Knowledge Bases e indexar el corpus de demostración en ChromaDB",
    )
    args = parser.parse_args()

    created = materialize_demo_docs(args.output, offline=args.offline)
    total_files = sum(len(v) for v in created.values())
    print(f"Corpus demo listo en '{args.output}' ({len(created)} KBs / {total_files} archivos)")

    if args.ingest:
        stats = ingest_demo_corpus(args.output)
        for kb_id, (n_docs, n_chunks) in stats.items():
            print(f"  - KB '{kb_id}': {n_docs} documentos / {n_chunks} chunks indexados")


if __name__ == "__main__":
    main()
