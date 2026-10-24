import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.conversations.service import ConversationService
from app.dependencies import get_conversation_service
from app.main import create_app


@pytest.fixture
def conv_client(tmp_path):
    settings = Settings(catalog_path=tmp_path / "catalog.sqlite3")
    conv_svc = ConversationService(settings)
    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: conv_svc
    return TestClient(app), conv_svc


def test_crud_conversations(conv_client):
    client, conv_svc = conv_client

    # 1. Crear
    res = client.post(
        "/conversations",
        json={"title": "Primer Chat", "knowledge_base_id": "general"},
    )
    assert res.status_code == 201
    created = res.json()
    conv_id = created["id"]
    assert created["title"] == "Primer Chat"
    assert created["last_kb_id"] == "general"
    assert created["message_count"] == 0

    # 2. Listar
    list_res = client.get("/conversations")
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 1
    assert list_res.json()["conversations"][0]["id"] == conv_id

    # 3. Registrar un turno y obtener detalle con mensajes y fuentes
    conv_svc.record_turn(
        conversation_id=conv_id,
        question="¿Puerto por defecto?",
        answer="El puerto es 8000 [1].",
        grounded=True,
        sources=[
            {
                "id": 1,
                "source": "deploy.md",
                "section": "Variables",
                "page": None,
                "snippet": "PORT=8000",
                "score": 0.91,
                "knowledge_base_id": "general",
            }
        ],
        retrieved=[],
        kb_id="general",
    )

    get_res = client.get(f"/conversations/{conv_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["message_count"] == 2
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["sources"][0]["source"] == "deploy.md"

    # 4. Renombrar
    patch_res = client.patch(f"/conversations/{conv_id}", json={"title": "Chat Renombrado"})
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "Chat Renombrado"

    # 5. Borrar
    del_res = client.delete(f"/conversations/{conv_id}")
    assert del_res.status_code == 204
    assert client.get(f"/conversations/{conv_id}").status_code == 404


def test_conversation_not_found_and_validation_errors(conv_client):
    client, _ = conv_client

    assert client.get("/conversations/no-existe").status_code == 404
    assert client.delete("/conversations/no-existe").status_code == 404
    assert client.patch("/conversations/no-existe", json={"title": "Hola"}).status_code == 404

    created = client.post("/conversations", json={}).json()
    conv_id = created["id"]
    bad_patch = client.patch(f"/conversations/{conv_id}", json={"title": "   "})
    assert bad_patch.status_code == 422


def test_conversations_read_only_mode(tmp_path):
    settings = Settings(catalog_path=tmp_path / "ro.sqlite3", read_only_mode=True)
    conv_svc = ConversationService(settings)
    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: conv_svc
    client = TestClient(app)

    r = client.post("/conversations", json={"title": "Bloqueado"})
    assert r.status_code == 403
    assert r.json()["error"] == "read_only_mode"
