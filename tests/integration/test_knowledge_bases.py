import json

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.dependencies import (
    get_document_service,
    get_knowledge_base_service,
    get_rag_service,
)
from app.documents.service import DocumentService
from app.knowledge_bases.service import KnowledgeBaseService
from app.main import create_app
from app.rag.service import RAGService


class KeywordEmbeddings(Embeddings):
    """Embeddings deterministas basados en palabras clave para evaluar relevancia real sin red."""

    KEYWORDS = ("python", "recursion", "decorador", "cache", "memoria", "pipeline", "red")

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        vec = [1.0 if kw in low else 0.0 for kw in self.KEYWORDS]
        if not any(vec):
            vec.append(1.0)
        else:
            vec.append(0.0)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def kb_env(settings, tmp_path):
    s = settings.model_copy(
        update={
            "chroma_dir": tmp_path / "kb_chroma",
            "docs_dir": tmp_path / "kb_docs",
            "catalog_path": tmp_path / "kb_catalog.sqlite3",
            "min_relevance": 0.5,
        }
    )
    vs = Chroma(
        collection_name="kb_test",
        embedding_function=KeywordEmbeddings(),
        persist_directory=str(s.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    kb_svc = KnowledgeBaseService(vs, s)
    doc_svc = DocumentService(vs, s, kb_service=kb_svc)
    llm = FakeListChatModel(responses=["Respuesta fundamentada [1]."])
    rag_svc = RAGService(vs, llm, s, kb_service=kb_svc)

    app = create_app()
    app.dependency_overrides[get_knowledge_base_service] = lambda: kb_svc
    app.dependency_overrides[get_document_service] = lambda: doc_svc
    app.dependency_overrides[get_rag_service] = lambda: rag_svc
    return TestClient(app), vs, s, kb_svc, doc_svc


def upload_to_kb(client: TestClient, kb_id: str, name: str, content: bytes):
    return client.post(
        f"/knowledge-bases/{kb_id}/documents",
        files={"file": (name, content, "text/markdown")},
    )


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ---------- Escenarios A a H de Aislamiento ----------


def test_case_a_strict_isolation_between_knowledge_bases(kb_env):
    """Caso A: Consultar Programación nunca recupera chunks de Arquitectura."""
    client, *_ = kb_env
    kb_prog = client.post("/knowledge-bases", json={"name": "Programación"}).json()["id"]
    kb_arq = client.post("/knowledge-bases", json={"name": "Arquitectura"}).json()["id"]

    assert (
        upload_to_kb(
            client, kb_prog, "python.md", b"# Python\n\nUso de recursion y decorador en python."
        ).status_code
        == 201
    )
    assert (
        upload_to_kb(
            client,
            kb_arq,
            "arq.md",
            b"# Hardware\n\nLa memoria cache acelera el pipeline y python.",
        ).status_code
        == 201
    )

    r = client.post(
        "/query",
        json={"question": "¿Cómo se usa recursion en python?", "knowledge_base_id": kb_prog},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["knowledge_base_id"] == kb_prog
    assert {s["source"] for s in body["retrieved"]} == {"python.md"}
    assert all(s["knowledge_base_id"] == kb_prog for s in body["retrieved"])


def test_case_b_and_c_abstention_on_wrong_kb_and_success_on_right_kb(kb_env):
    """Casos B y C: Abstención en Programación y éxito en Arquitectura."""
    client, *_ = kb_env
    kb_prog = client.post("/knowledge-bases", json={"name": "Programación"}).json()["id"]
    kb_arq = client.post("/knowledge-bases", json={"name": "Arquitectura"}).json()["id"]

    upload_to_kb(client, kb_prog, "python.md", b"Funciones y decorador en python.")
    upload_to_kb(client, kb_arq, "cache.md", b"La memoria cache L1 reduce latencia.")

    # Caso B: Preguntar por caché en Programación -> abstención below_threshold
    r_wrong = client.post(
        "/query",
        json={"question": "¿Qué hace la memoria cache?", "knowledge_base_id": kb_prog},
    )
    assert r_wrong.status_code == 200
    b_wrong = r_wrong.json()
    assert b_wrong["grounded"] is False
    assert b_wrong["reason"] == "below_threshold"
    assert "Programación" in b_wrong["message"]
    assert b_wrong["sources"] == []

    # Caso C: Misma pregunta en Arquitectura -> éxito con citas válidas
    r_right = client.post(
        "/query",
        json={"question": "¿Qué hace la memoria cache?", "knowledge_base_id": kb_arq},
    )
    assert r_right.status_code == 200
    b_right = r_right.json()
    assert b_right["grounded"] is True
    assert [s["source"] for s in b_right["sources"]] == ["cache.md"]


def test_case_d_unknown_knowledge_base_returns_404(kb_env):
    """Caso D: Consulta o subida en un kb_id inexistente retorna 404 sin hacer búsqueda global."""
    client, *_ = kb_env
    r = client.post(
        "/query",
        json={"question": "¿Qué hace la memoria cache?", "knowledge_base_id": "kb-fantasma"},
    )
    assert r.status_code == 404
    assert r.json()["error"] == "knowledge_base_not_found"

    r_stream = client.post(
        "/query/stream",
        json={"question": "¿Qué hace la memoria cache?", "knowledge_base_id": "kb-fantasma"},
    )
    assert r_stream.status_code == 404
    assert r_stream.json()["error"] == "knowledge_base_not_found"

    assert client.get("/knowledge-bases/kb-fantasma/documents").status_code == 404


def test_case_e_empty_knowledge_base_abstains_cleanly(kb_env):
    """Caso E: Consulta en una KB recién creada y vacía devuelve abstención limpia sin 500."""
    client, *_ = kb_env
    kb_empty = client.post("/knowledge-bases", json={"name": "Redes"}).json()["id"]

    r = client.post(
        "/query",
        json={"question": "¿Qué es una red?", "knowledge_base_id": kb_empty},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is False
    assert body["reason"] == "empty_knowledge_base"
    assert "Redes" in body["message"]

    events = parse_sse(
        client.post(
            "/query/stream",
            json={"question": "¿Qué es una red?", "knowledge_base_id": kb_empty},
        ).text
    )
    assert len(events) == 1 and events[0][0] == "end"
    assert events[0][1]["grounded"] is False
    assert events[0][1]["reason"] == "empty_knowledge_base"


def test_case_f_persistence_and_legacy_migration(kb_env):
    """Caso F: El catálogo persiste y los datos legados migran a 'general'."""
    client, vs, settings, *_ = kb_env
    kb_id = client.post(
        "/knowledge-bases",
        json={"name": "Sistemas Operativos", "description": "Procesos y memoria"},
    ).json()["id"]
    upload_to_kb(client, kb_id, "proc.md", b"Planificacion de memoria y procesos.")

    # Simulamos además un archivo y vector legado sin knowledge_base_id
    legacy_file = settings.docs_dir / "legado.txt"
    legacy_file.write_bytes(b"documento legado")
    vs.add_documents(
        [Document(page_content="legado", metadata={"doc_id": "leg123", "source": "legado.txt"})],
        ids=["leg123:0"],
    )

    # Nueva instancia de servicio apuntando al mismo SQLite y Chroma
    reopened = KnowledgeBaseService(vs, settings)
    kbs = {k.id: k for k in reopened.list_knowledge_bases()}
    assert kb_id in kbs and kbs[kb_id].documents == 1
    assert "general" in kbs and kbs["general"].documents >= 1
    assert (settings.docs_dir / "general" / "legado.txt").exists()


def test_case_g_atomic_document_And_kb_deletion_does_not_affect_others(kb_env):
    """Caso G: Mismo archivo en dos KBs coexiste y borrarlo en una no afecta a la otra."""
    client, *_ = kb_env
    kb1 = client.post("/knowledge-bases", json={"name": "Base Uno"}).json()["id"]
    kb2 = client.post("/knowledge-bases", json={"name": "Base Dos"}).json()["id"]

    same_content = b"Contenido compartido sobre memoria cache."
    d1 = upload_to_kb(client, kb1, "nota.md", same_content).json()
    d2 = upload_to_kb(client, kb2, "nota.md", same_content).json()
    assert d1["doc_id"] == d2["doc_id"]

    # Borrar el documento en kb1 deja intacto el de kb2
    assert client.delete(f"/knowledge-bases/{kb1}/documents/{d1['doc_id']}").status_code == 204
    assert client.get(f"/knowledge-bases/{kb1}/documents").json()["total"] == 0
    assert client.get(f"/knowledge-bases/{kb2}/documents").json()["total"] == 1

    # Borrar kb2 por completo elimina sus vectores y su carpeta
    assert client.delete(f"/knowledge-bases/{kb2}").status_code == 204
    assert client.get(f"/knowledge-bases/{kb2}").status_code == 404


def test_case_h_streaming_sse_respects_kb_isolation(kb_env):
    """Caso H: /query/stream filtra estrictamente por knowledge_base_id."""
    client, *_ = kb_env
    kb_prog = client.post("/knowledge-bases", json={"name": "Programación"}).json()["id"]
    kb_arq = client.post("/knowledge-bases", json={"name": "Arquitectura"}).json()["id"]

    upload_to_kb(client, kb_prog, "py.md", b"En python existe recursion.")
    upload_to_kb(client, kb_arq, "arq.md", b"En hardware existe memoria cache.")

    events = parse_sse(
        client.post(
            "/query/stream",
            json={"question": "Explica recursion en python", "knowledge_base_id": kb_prog},
        ).text
    )
    retrieved_ev = next(d for name, d in events if name == "retrieved")
    assert [r["source"] for r in retrieved_ev["retrieved"]] == ["py.md"]
    end_ev = events[-1][1]
    assert end_ev["grounded"] is True
    assert [s["source"] for s in end_ev["sources"]] == ["py.md"]


def test_kb_validation_and_default_protection(kb_env):
    """No permite duplicar nombres de KB ni borrar la base por defecto 'general'."""
    client, *_ = kb_env
    assert client.post("/knowledge-bases", json={"name": "Física"}).status_code == 201
    dup = client.post("/knowledge-bases", json={"name": "física"})
    assert dup.status_code == 409
    assert dup.json()["error"] == "duplicate_knowledge_base"

    del_default = client.delete("/knowledge-bases/general")
    assert del_default.status_code == 422
    assert del_default.json()["error"] == "invalid_knowledge_base"
