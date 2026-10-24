import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.api.routes.query import reset_rate_limiter
from app.config import get_settings
from app.dependencies import (
    get_document_service,
    get_knowledge_base_service,
    get_rag_service,
)
from app.documents.service import DocumentService
from app.knowledge_bases.service import KnowledgeBaseService
from app.main import create_app
from app.rag.service import RAGService
from tests.integration.test_api import use_llm
from tests.integration.test_robustness import parse_sse


@pytest.fixture
def readonly_client(vectorstore, settings, tmp_path):
    from app.ingestion.pipeline import ingest_directory
    from tests.conftest import FIXTURES

    reset_rate_limiter()
    s = settings.model_copy(
        update={
            "docs_dir": tmp_path / "ro_docs",
            "catalog_path": tmp_path / "ro_catalog.sqlite3",
            "read_only_mode": True,
        }
    )
    ingest_directory(FIXTURES, vectorstore, s)
    kb_svc = KnowledgeBaseService(vectorstore, s)
    doc_svc = DocumentService(vectorstore, s, kb_service=kb_svc)
    llm = FakeListChatModel(responses=["El puerto por defecto es 8000 [1]."])
    rag_svc = RAGService(vectorstore, llm, s, kb_service=kb_svc)

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: s
    app.dependency_overrides[get_knowledge_base_service] = lambda: kb_svc
    app.dependency_overrides[get_document_service] = lambda: doc_svc
    app.dependency_overrides[get_rag_service] = lambda: rag_svc
    return TestClient(app)


def test_health_live_and_ready_endpoints(docs_client):
    live = docs_client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "ok", "version": "1.0.0", "read_only_mode": False}

    ready = docs_client.get("/health/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["knowledge_bases"] >= 1
    assert body["total_chunks"] >= 1
    assert body["read_only_mode"] is False


def test_read_only_mode_blocks_all_mutations_with_403(readonly_client):
    # 1. Crear KB bloqueado
    r_create = readonly_client.post("/knowledge-bases", json={"name": "Nueva"})
    assert r_create.status_code == 403
    assert r_create.json()["error"] == "read_only_mode"

    # 2. Eliminar KB bloqueado
    r_del_kb = readonly_client.delete("/knowledge-bases/general")
    assert r_del_kb.status_code == 403
    assert r_del_kb.json()["error"] == "read_only_mode"

    # 3. Subir documento en KB bloqueado
    r_up_kb = readonly_client.post(
        "/knowledge-bases/general/documents",
        files={"file": ("x.md", b"# Test\nContenido", "text/markdown")},
    )
    assert r_up_kb.status_code == 403
    assert r_up_kb.json()["error"] == "read_only_mode"

    # 4. Subir documento en ruta legada bloqueado
    r_up_leg = readonly_client.post(
        "/documents",
        files={"file": ("x.md", b"# Test\nContenido", "text/markdown")},
    )
    assert r_up_leg.status_code == 403
    assert r_up_leg.json()["error"] == "read_only_mode"

    # 5. Eliminar documento bloqueado
    docs = readonly_client.get("/knowledge-bases/general/documents").json()["documents"]
    assert len(docs) >= 1
    doc_id = docs[0]["doc_id"]

    r_del_doc = readonly_client.delete(f"/knowledge-bases/general/documents/{doc_id}")
    assert r_del_doc.status_code == 403
    assert r_del_doc.json()["error"] == "read_only_mode"


def test_read_only_mode_allows_reads_and_queries(readonly_client):
    assert readonly_client.get("/health/live").json()["read_only_mode"] is True
    assert readonly_client.get("/health/ready").json()["read_only_mode"] is True
    assert readonly_client.get("/knowledge-bases").status_code == 200
    assert readonly_client.get("/knowledge-bases/general/documents").status_code == 200

    q = readonly_client.post("/query", json={"question": "¿Cuál es el puerto por defecto?"})
    assert q.status_code == 200
    assert q.json()["grounded"] is True


def test_rate_limiter_blocks_excess_queries_with_429(client, vectorstore, settings):
    reset_rate_limiter()
    use_llm(
        client,
        vectorstore,
        settings,
        FakeListChatModel(responses=["Respuesta [1]."]),
        rate_limit_per_minute=2,
    )
    assert client.post("/query", json={"question": "consulta 1"}).status_code == 200
    assert client.post("/query", json={"question": "consulta 2"}).status_code == 200
    r_blocked = client.post("/query", json={"question": "consulta 3"})
    assert r_blocked.status_code == 429
    assert r_blocked.json()["error"] == "rate_limit_exceeded"
    reset_rate_limiter()


def test_stream_applies_same_citation_retry_policy_as_sync_query(client, vectorstore, settings):
    reset_rate_limiter()
    llm = FakeListChatModel(responses=["Respuesta inicial sin citas.", "Respuesta corregida [2]."])
    use_llm(client, vectorstore, settings, llm)
    r = client.post("/query/stream", json={"question": "pregunta de prueba"})
    assert r.status_code == 200
    end = parse_sse(r.text)[-1][1]
    assert end["grounded"] is True
    assert end["cited_ids"] == [2]
    assert [s["id"] for s in end["sources"]] == [2]


def test_llm_error_does_not_leak_internal_details(client, vectorstore, settings):
    class LeakyLLM(FakeListChatModel):
        def _call(self, *a, **k):
            raise ConnectionError("secret_internal_url=http://10.0.0.42:1234/v1 key=sk-secret")

    use_llm(client, vectorstore, settings, LeakyLLM(responses=["x"]))
    r = client.post("/query", json={"question": "pregunta de prueba"})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "10.0.0.42" not in detail and "sk-secret" not in detail
