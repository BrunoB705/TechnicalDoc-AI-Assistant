import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.dependencies import get_rag_service
from app.rag.service import RAGService


def use_llm(client, vectorstore, settings, llm, **overrides):
    s = settings.model_copy(update=overrides)
    client.app.dependency_overrides[get_rag_service] = lambda: RAGService(vectorstore, llm, s)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_query_returns_answer_with_cited_sources_only(client):
    r = client.post("/query", json={"question": "¿Cuál es el puerto por defecto?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert "8000" in body["answer"]
    assert [s["id"] for s in body["sources"]] == [1]  # el LLM falso solo cita [1]
    assert len(body["retrieved"]) == 3  # pero se recuperaron 3


def test_query_validation(client):
    assert client.post("/query", json={"question": "a"}).status_code == 422


def test_no_evidence_from_llm(client, vectorstore, settings):
    use_llm(client, vectorstore, settings, FakeListChatModel(responses=["NO_EVIDENCE"]))
    body = client.post("/query", json={"question": "¿Capital de Francia?"}).json()
    assert body["grounded"] is False and body["answer"] is None and body["message"]


def test_no_evidence_by_threshold_skips_llm(client, vectorstore, settings):
    class Boom(FakeListChatModel):
        def _call(self, *a, **k):
            raise AssertionError("El LLM no debería invocarse")

    use_llm(client, vectorstore, settings, Boom(responses=["x"]), min_relevance=5.0)
    body = client.post("/query", json={"question": "cualquier cosa"}).json()
    assert body["grounded"] is False


def test_invalid_citations_are_removed(client, vectorstore, settings):
    llm = FakeListChatModel(responses=["Dato real [1] y dato inventado [9]."])
    use_llm(client, vectorstore, settings, llm)
    body = client.post("/query", json={"question": "pregunta de prueba"}).json()
    assert body["grounded"] is True
    assert "[9]" not in body["answer"] and "[1]" in body["answer"]


def test_missing_citations_retry_once(client, vectorstore, settings):
    llm = FakeListChatModel(responses=["Respuesta sin citas.", "Ahora sí [2]."])
    use_llm(client, vectorstore, settings, llm)
    body = client.post("/query", json={"question": "pregunta de prueba"}).json()
    assert body["grounded"] is True
    assert [s["id"] for s in body["sources"]] == [2]


def test_no_citations_after_retry_is_not_grounded(client, vectorstore, settings):
    llm = FakeListChatModel(responses=["Sin citas.", "Sigue sin citas."])
    use_llm(client, vectorstore, settings, llm)
    body = client.post("/query", json={"question": "pregunta de prueba"}).json()
    assert body["grounded"] is False
    assert body["sources"] == [] and body["message"]


def test_llm_down_returns_503(client, vectorstore, settings):
    class Down(FakeListChatModel):
        def _call(self, *a, **k):
            raise ConnectionError("connection refused")

    use_llm(client, vectorstore, settings, Down(responses=["x"]))
    r = client.post("/query", json={"question": "pregunta de prueba"})
    assert r.status_code == 503
    assert r.json()["error"] == "llm_unavailable"


def test_empty_index_returns_409(client, settings, fake_llm, tmp_path):
    empty = Chroma(
        collection_name="empty",
        embedding_function=DeterministicFakeEmbedding(size=32),
        persist_directory=str(tmp_path / "empty"),
    )
    client.app.dependency_overrides[get_rag_service] = lambda: RAGService(empty, fake_llm, settings)
    r = client.post("/query", json={"question": "pregunta de prueba"})
    assert r.status_code == 409
    assert r.json()["error"] == "empty_index"


def test_query_stream_success(client):
    r = client.post("/query/stream", json={"question": "¿Cuál es el puerto por defecto?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    assert "event: token" in text or "event: end" in text
    assert "8000" in text


@pytest.mark.parametrize("path", ["/health"])
def test_unknown_routes_404(client, path):
    assert client.get("/nope").status_code == 404
