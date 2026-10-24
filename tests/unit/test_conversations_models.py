from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.conversations.models import ConversationDetail, ConversationSummary, MessageRecord
from app.exceptions import ConversationNotFoundError, InvalidConversationError
from app.main import create_app


def test_conversation_summary_instantiation():
    now = datetime.now(UTC).isoformat()
    c = ConversationSummary(
        id="conv-123",
        title="Charla sobre binario",
        last_kb_id="general",
        created_at=now,
        updated_at=now,
        message_count=2,
    )
    assert c.id == "conv-123"
    assert c.title == "Charla sobre binario"
    assert c.last_kb_id == "general"
    assert c.message_count == 2


def test_conversation_detail_with_messages():
    now = datetime.now(UTC).isoformat()
    msg = MessageRecord(
        id=1,
        conversation_id="conv-123",
        role="user",
        content="Hola",
        grounded=True,
        reason=None,
        message=None,
        sources=[],
        retrieved=[],
        knowledge_base_id="general",
        created_at=now,
    )
    detail = ConversationDetail(
        id="conv-123",
        title="Charla",
        last_kb_id="general",
        created_at=now,
        updated_at=now,
        message_count=1,
        messages=[msg],
    )
    assert len(detail.messages) == 1
    assert detail.messages[0].role == "user"
    assert detail.messages[0].content == "Hola"


def test_conversation_exceptions_registered_in_app():
    assert issubclass(ConversationNotFoundError, Exception)
    assert issubclass(InvalidConversationError, Exception)

    app: FastAPI = create_app()

    @app.get("/_test_conv_404")
    def _raise_404():
        raise ConversationNotFoundError("No existe la conversación.")

    @app.get("/_test_conv_422")
    def _raise_422():
        raise InvalidConversationError("Título inválido.")

    tc = TestClient(app)
    r404 = tc.get("/_test_conv_404")
    assert r404.status_code == 404
    assert r404.json() == {
        "error": "conversation_not_found",
        "detail": "No existe la conversación.",
    }

    r422 = tc.get("/_test_conv_422")
    assert r422.status_code == 422
    assert r422.json() == {"error": "invalid_conversation", "detail": "Título inválido."}
