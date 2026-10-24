import subprocess
import sys
from pathlib import Path

from app.config import Settings
from app.rag.embeddings import get_embeddings
from scripts.download_demo_docs import DEMO_SPECS, materialize_demo_docs


def test_settings_defaults_and_env_overrides(monkeypatch):
    # Limpiar posibles variables de entorno para verificar los defaults calibrados del código
    for key in (
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "MIN_RELEVANCE",
        "READ_ONLY_MODE",
        "RATE_LIMIT_PER_MINUTE",
    ):
        monkeypatch.delenv(key, raising=False)

    s_default = Settings(_env_file=None)
    assert s_default.chunk_size == 650
    assert s_default.chunk_overlap == 100
    assert s_default.top_k == 4
    assert s_default.min_relevance == 0.80
    assert s_default.read_only_mode is False
    assert s_default.rate_limit_per_minute == 60

    monkeypatch.setenv("CHUNK_SIZE", "750")
    monkeypatch.setenv("MIN_RELEVANCE", "0.837")
    monkeypatch.setenv("READ_ONLY_MODE", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "15")

    s_override = Settings(_env_file=None)
    assert s_override.chunk_size == 750
    assert s_override.min_relevance == 0.837
    assert s_override.read_only_mode is True
    assert s_override.rate_limit_per_minute == 15


def test_get_embeddings_reuses_singleton_instance():
    s = Settings(_env_file=None)
    emb1 = get_embeddings(s, lazy=True)
    emb2 = get_embeddings(s, lazy=True)
    assert emb1 is emb2


def test_ingest_cli_accepts_kb_name_and_description():
    proc = subprocess.run(
        [sys.executable, "scripts/ingest.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--kb" in proc.stdout
    assert "--kb-name" in proc.stdout
    assert "--kb-description" in proc.stdout
    assert "--reset" in proc.stdout


def test_materialize_demo_docs_offline_creates_kb_folders(tmp_path):
    out = tmp_path / "demo_docs"
    created = materialize_demo_docs(out, offline=True)
    assert {"python-peps", "fastapi-guide", "arquitectura", "backend", "operaciones"} <= set(
        created
    )
    for spec in DEMO_SPECS:
        target = out / spec.kb_id / spec.filename
        assert target.is_file()
        assert len(target.read_text(encoding="utf-8")) > 100


def test_materialize_demo_docs_uses_http_when_available_and_falls_back_on_error(
    tmp_path, monkeypatch
):
    class FakeResp:
        status_code = 200
        text = "# Remote PEP\n\n" + ("Contenido descargado de prueba. " * 10)

    calls = []

    def fake_get(url, timeout=5.0, follow_redirects=True):
        calls.append(url)
        if "fastapi" in url:
            raise OSError("network unreachable")
        return FakeResp()

    monkeypatch.setattr("scripts.download_demo_docs.httpx.get", fake_get)
    out = tmp_path / "demo_http"
    created = materialize_demo_docs(out, offline=False, include_eval_corpus=None)
    assert len(calls) == len(DEMO_SPECS)
    assert "Remote PEP" in (out / "python-peps" / "pep-0008-style-guide.md").read_text(
        encoding="utf-8"
    )
    # El que falló por red cae limpiamente al fallback embebido
    assert "FastAPI" in (out / "fastapi-guide" / "fastapi-dependencies-and-async.md").read_text(
        encoding="utf-8"
    )
    assert set(created) == {"python-peps", "fastapi-guide"}


def test_dockerfile_and_ci_workflow_are_present_and_valid():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'VOLUME ["/app/data"]' in dockerfile
    assert "EXPOSE 8000" in dockerfile

    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv run ruff check ." in ci
    assert "uv run pytest -q -W ignore" in ci
