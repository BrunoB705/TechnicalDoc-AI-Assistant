import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.conversations.service import ConversationService
from app.dependencies import get_conversation_service, get_learning_service
from app.learning.service import LearningService
from app.main import create_app
from app.rag.service import RAGService, SourceHit


def test_learning_api_gates_teaching_and_restores_session(tmp_path):
    settings = Settings(catalog_path=tmp_path / "catalog.sqlite3", enable_step_back=False)
    conversations = ConversationService(settings)
    conversation = conversations.create_conversation(kb_id="architecture")
    source = SourceHit(1, "rules.md", None, None, "binary16 fields", 0.95, "architecture")
    rag = RAGService(
        MagicMock(),
        FakeListChatModel(
            responses=[
                json.dumps(
                    {
                        "question": "¿Qué conocés?",
                        "options": [{"id": "A", "text": "Signo"}, {"id": "B", "text": "Sesgo"}],
                        "correct_option_id": "A",
                        "explanation": "El signo ocupa un bit [1].",
                    }
                )
            ]
        ),
        settings,
        conversation_service=conversations,
    )
    rag._resolve_kb = MagicMock(return_value=("architecture", "Architecture", False))
    rag._retrieve = MagicMock(
        return_value=(
            [(Document(page_content=source.snippet, metadata={"source": source.source}), 0.95)],
            [source],
            1,
        )
    )
    learning = LearningService(settings, rag, conversations)
    learning._retrieve_goal = MagicMock(return_value=([source], "[1] rules.md: binary16 fields"))

    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: conversations
    app.dependency_overrides[get_learning_service] = lambda: learning
    client = TestClient(app)

    started = client.post(
        "/learning-sessions",
        json={"conversation_id": conversation.id, "goal": "Aprender binary16"},
    )
    assert started.status_code == 201
    session = started.json()
    assert session["state"] == "diagnosing"
    assert "correct_option_id" not in session["quiz"]

    premature = client.post(
        f"/learning-sessions/{session['id']}/actions",
        json={"action": "approve_plan"},
    )
    assert premature.status_code == 422
    assert client.get(f"/learning-sessions/{session['id']}").json()["state"] == "diagnosing"
    restored = client.get(f"/learning-sessions/by-conversation/{conversation.id}")
    assert restored.status_code == 200
    assert restored.json()["id"] == session["id"]
