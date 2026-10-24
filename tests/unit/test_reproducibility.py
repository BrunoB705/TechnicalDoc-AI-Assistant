from pathlib import Path

from scripts.download_demo_docs import DEMO_SPECS, materialize_demo_docs


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
