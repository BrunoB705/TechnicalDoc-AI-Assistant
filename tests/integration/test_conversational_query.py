import json

from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.conversations.service import ConversationService
from app.dependencies import (
    get_conversation_service,
    get_document_service,
    get_knowledge_base_service,
    get_rag_service,
)
from app.documents.service import DocumentService
from app.knowledge_bases.service import KnowledgeBaseService
from app.main import create_app
from app.rag.service import RAGService


class TopicEmbeddings(Embeddings):
    TOPICS = ("binario", "hexadecimal", "despliegue", "docker")

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        vec = [1.0 if kw in low else 0.0 for kw in self.TOPICS]
        vec.append(0.0 if any(vec) else 1.0)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_multi_turn_conversation_and_kb_switch_across_turns(settings, tmp_path):
    s = settings.model_copy(
        update={
            "chroma_dir": tmp_path / "conv_chroma",
            "docs_dir": tmp_path / "conv_docs",
            "catalog_path": tmp_path / "conv_catalog.sqlite3",
            "min_relevance": 0.5,
        }
    )
    vs = Chroma(
        collection_name="conv_test",
        embedding_function=TopicEmbeddings(),
        persist_directory=str(s.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    kb_svc = KnowledgeBaseService(vs, s)
    doc_svc = DocumentService(vs, s, kb_service=kb_svc)
    conv_svc = ConversationService(s)
    llm = FakeListChatModel(
        responses=[
            "En binario se usan 0 y 1 [1].",  # Turno 1 respuesta
            "¿Cómo se representan los números en base hexadecimal?",  # Turno 2 condensación
            "En hexadecimal se usan 0-9 y A-F [1].",  # Turno 2 respuesta
            "¿Cómo se realiza el despliegue en docker?",  # Turno 3 condensación (cambio de KB)
            "El despliegue en docker usa contenedores [1].",  # Turno 3 respuesta
        ]
    )
    rag_svc = RAGService(vs, llm, s, kb_service=kb_svc, conversation_service=conv_svc)

    app = create_app()
    app.dependency_overrides[get_knowledge_base_service] = lambda: kb_svc
    app.dependency_overrides[get_document_service] = lambda: doc_svc
    app.dependency_overrides[get_conversation_service] = lambda: conv_svc
    app.dependency_overrides[get_rag_service] = lambda: rag_svc
    client = TestClient(app)

    # Subir docs a 'general' y 'ops'
    kb_ops = client.post("/knowledge-bases", json={"name": "Ops"}).json()["id"]
    client.post(
        "/knowledge-bases/general/documents",
        files={
            "file": (
                "sistemas.md",
                b"# Numeros\nRepresentacion en binario y hexadecimal.",
                "text/markdown",
            )
        },
    )
    client.post(
        f"/knowledge-bases/{kb_ops}/documents",
        files={
            "file": (
                "deploy.md",
                b"# Ops\nEl despliegue en docker usa contenedores.",
                "text/markdown",
            )
        },
    )

    # Crear conversación
    conv = client.post("/conversations", json={}).json()
    conv_id = conv["id"]

    # Turno 1 en 'general'
    r1 = client.post(
        "/query",
        json={
            "question": "¿Cómo se representan los números en binario?",
            "knowledge_base_id": "general",
            "conversation_id": conv_id,
        },
    )
    assert r1.status_code == 200
    assert r1.json()["grounded"] is True
    assert r1.json()["conversation_id"] == conv_id

    # Turno 2 en 'general' (repregunta por stream)
    r2 = client.post(
        "/query/stream",
        json={
            "question": "¿Y en base hexadecimal?",
            "knowledge_base_id": "general",
            "conversation_id": conv_id,
        },
    )
    assert r2.status_code == 200
    ev2 = parse_sse(r2.text)
    end2 = ev2[-1][1]
    assert end2["grounded"] is True
    assert end2["conversation_id"] == conv_id

    # Turno 3 cambiando de KB a 'ops' en la misma conversación
    r3 = client.post(
        "/query",
        json={
            "question": "¿Y cómo se despliega eso en docker?",
            "knowledge_base_id": kb_ops,
            "conversation_id": conv_id,
        },
    )
    assert r3.status_code == 200
    assert r3.json()["grounded"] is True
    assert r3.json()["knowledge_base_id"] == kb_ops

    # Verificar persistencia completa de la conversación
    detail = client.get(f"/conversations/{conv_id}").json()
    assert detail["last_kb_id"] == kb_ops
    assert detail["message_count"] == 6
    assert detail["messages"][4]["knowledge_base_id"] == kb_ops


def test_query_with_nonexistent_conversation_returns_404(client):
    r = client.post(
        "/query",
        json={"question": "¿Cuál es el puerto?", "conversation_id": "no-existe-id"},
    )
    assert r.status_code == 404
    assert r.json()["error"] == "conversation_not_found"

    rs = client.post(
        "/query/stream",
        json={"question": "¿Cuál es el puerto?", "conversation_id": "no-existe-id"},
    )
    assert rs.status_code == 404
