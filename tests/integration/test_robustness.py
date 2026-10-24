import hashlib
import io

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.documents.service import UPLOAD_CHUNK_SIZE, DocumentService
from app.exceptions import DocumentTooLargeError
from tests.integration.test_api import use_llm

MB = 1024 * 1024


def upload(client, name, content, ctype="text/plain"):
    return client.post("/documents", files={"file": (name, content, ctype)})


def doc_id_of(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def parse_sse(text: str) -> list[tuple[str, dict]]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ---------- ingesta: archivos problemáticos ----------


def test_empty_file_is_rejected(docs_client):
    r = upload(docs_client, "vacio.md", b"")
    assert r.status_code == 422
    assert not (docs_client.docs_dir / "vacio.md").exists()


def test_corrupt_pdf_is_rejected_and_leaves_no_trace(docs_client, vectorstore):
    before = vectorstore.get()["ids"]
    r = upload(docs_client, "roto.pdf", b"esto no es un pdf", "application/pdf")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_document"
    assert not (docs_client.docs_dir / "roto.pdf").exists()
    assert vectorstore.get()["ids"] == before


def test_size_limit_is_exact(docs_client):
    ok = upload(docs_client, "justo.txt", b"a" * MB)
    assert ok.status_code == 201
    over = upload(docs_client, "pasado.txt", b"b" * (MB + 1))
    assert over.status_code == 413
    assert not (docs_client.docs_dir / "pasado.txt").exists()


# ---------- upload en streaming ----------


class CountingStream(io.RawIOBase):
    """Stream infinito que registra cuántos bytes se le pidieron leer."""

    def __init__(self):
        self.served = 0

    def readable(self):
        return True

    def read(self, size=-1):
        assert size > 0, "debe leerse en bloques acotados, no con read() completo"
        assert size <= UPLOAD_CHUNK_SIZE
        self.served += size
        return b"x" * size


def test_read_upload_stops_early_without_reading_whole_stream(vectorstore, settings):
    service = DocumentService(vectorstore, settings.model_copy(update={"max_upload_mb": 1}))
    stream = CountingStream()
    with pytest.raises(DocumentTooLargeError):
        service.read_upload(stream)
    assert stream.served <= MB + UPLOAD_CHUNK_SIZE


def test_read_upload_returns_full_content_under_limit(vectorstore, settings):
    service = DocumentService(vectorstore, settings)
    data = b"z" * (UPLOAD_CHUNK_SIZE * 2 + 123)
    assert service.read_upload(io.BytesIO(data)) == data


# ---------- atomicidad ----------


def _failing_ingest(vectorstore):
    """Simula un fallo a mitad de ingesta: inserta un chunk y luego explota."""

    def boom(path, vs, settings, root=None):
        did = doc_id_of(path.read_bytes())
        vectorstore.add_documents(
            [Document(page_content="parcial", metadata={"doc_id": did, "source": path.name})],
            ids=[f"{did}:0"],
        )
        raise RuntimeError("fallo de embeddings")

    return boom


def test_failed_ingest_rolls_back_vectors_and_file(docs_client, vectorstore, monkeypatch):
    content = b"contenido que fallara al indexar"
    monkeypatch.setattr("app.documents.service.ingest_file", _failing_ingest(vectorstore))
    r = upload(docs_client, "falla.txt", content)
    assert r.status_code == 422
    assert vectorstore.get(where={"doc_id": doc_id_of(content)})["ids"] == []
    assert not (docs_client.docs_dir / "falla.txt").exists()


def test_failed_replacement_keeps_previous_version(docs_client, vectorstore, monkeypatch):
    first = upload(docs_client, "v.txt", b"version original").json()
    monkeypatch.setattr("app.documents.service.ingest_file", _failing_ingest(vectorstore))
    r = upload(docs_client, "v.txt", b"version nueva que falla")
    assert r.status_code == 422
    assert docs_client.get(f"/documents/{first['doc_id']}").status_code == 200
    assert (docs_client.docs_dir / "v.txt").read_bytes() == b"version original"
    assert vectorstore.get(where={"doc_id": doc_id_of(b"version nueva que falla")})["ids"] == []


# ---------- paridad del contrato SSE ----------

END_KEYS = {"grounded", "reason", "message", "answer", "sources", "cited_ids"}


def test_stream_success_ends_with_full_contract(client):
    events = parse_sse(client.post("/query/stream", json={"question": "puerto por defecto"}).text)
    names = [n for n, _ in events]
    assert names[0] == "retrieved" and names[-1] == "end" and "error" not in names
    end = events[-1][1]
    assert set(end) == END_KEYS
    assert end["grounded"] is True and end["reason"] is None and end["message"] is None
    assert "8000" in end["answer"] and end["cited_ids"] == [1]
    assert [s["id"] for s in end["sources"]] == [1]


def test_stream_below_threshold_is_an_end_event_not_error(client, vectorstore, settings):
    use_llm(client, vectorstore, settings, FakeListChatModel(responses=["x"]), min_relevance=5.0)
    events = parse_sse(client.post("/query/stream", json={"question": "cualquier cosa"}).text)
    assert [n for n, _ in events] == ["end"]
    end = events[0][1]
    assert set(end) == END_KEYS
    assert end["grounded"] is False and end["reason"] == "below_threshold"
    assert end["message"] and end["sources"] == [] and end["answer"] is None


def test_stream_model_refused_uses_same_contract(client, vectorstore, settings):
    use_llm(client, vectorstore, settings, FakeListChatModel(responses=["NO_EVIDENCE"]))
    events = parse_sse(client.post("/query/stream", json={"question": "pregunta de prueba"}).text)
    end = events[-1][1]
    assert events[-1][0] == "end" and set(end) == END_KEYS
    assert end["grounded"] is False and end["reason"] == "model_refused" and end["message"]


def test_stream_without_citations_is_not_grounded(client, vectorstore, settings):
    use_llm(client, vectorstore, settings, FakeListChatModel(responses=["Sin citas."]))
    events = parse_sse(client.post("/query/stream", json={"question": "pregunta de prueba"}).text)
    end = events[-1][1]
    assert set(end) == END_KEYS
    assert end["grounded"] is False and end["reason"] == "no_citations" and end["message"]
    assert end["sources"] == []


def test_stream_strips_invalid_citations_from_final_answer(client, vectorstore, settings):
    llm = FakeListChatModel(responses=["Dato real [1] y dato inventado [9]."])
    use_llm(client, vectorstore, settings, llm)
    r = client.post("/query/stream", json={"question": "pregunta de prueba"})
    end = parse_sse(r.text)[-1][1]
    assert end["grounded"] is True
    assert "[9]" not in end["answer"] and "[1]" in end["answer"]
