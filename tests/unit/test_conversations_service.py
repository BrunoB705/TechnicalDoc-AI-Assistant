import pytest

from app.config import Settings
from app.conversations.service import DEFAULT_CONVERSATION_TITLE, ConversationService
from app.exceptions import (
    ConversationNotFoundError,
    InvalidConversationError,
    ReadOnlyModeError,
)


@pytest.fixture
def conv_service(tmp_path):
    settings = Settings(
        catalog_path=tmp_path / "catalog.sqlite3",
        conversation_window_exchanges=6,
    )
    return ConversationService(settings)


def test_create_and_list_conversations(conv_service):
    c1 = conv_service.create_conversation(title="Test Chat", kb_id="general")
    assert c1.title == "Test Chat"
    assert c1.last_kb_id == "general"
    assert c1.message_count == 0

    listed = conv_service.list_conversations()
    assert len(listed) == 1
    assert listed[0].id == c1.id


def test_record_turn_auto_titling_and_history(conv_service):
    c = conv_service.create_conversation(kb_id="arquitectura")
    assert c.title == DEFAULT_CONVERSATION_TITLE

    conv_service.record_turn(
        conversation_id=c.id,
        question="¿Cómo funcionan los números binarios?",
        answer="En binario se usan 0 y 1 [1].",
        grounded=True,
        reason=None,
        message=None,
        sources=[{"id": 1, "source": "math.md"}],
        retrieved=[{"id": 1, "source": "math.md"}],
        kb_id="arquitectura",
    )

    detail = conv_service.get_conversation(c.id)
    assert detail.title == "¿Cómo funcionan los números binarios?"
    assert detail.last_kb_id == "arquitectura"
    assert detail.knowledge_base_id == "arquitectura"
    assert detail.message_count == 2
    assert len(detail.messages) == 2
    assert detail.messages[0].role == "user"
    assert detail.messages[1].role == "assistant"
    assert detail.messages[1].sources == [{"id": 1, "source": "math.md"}]
    assert detail.messages[1].knowledge_base_id == "arquitectura"

    history = conv_service.get_recent_history(c.id, max_exchanges=6)
    assert history == [
        ("user", "¿Cómo funcionan los números binarios?"),
        ("assistant", "En binario se usan 0 y 1 [1]."),
    ]


def test_record_turn_rejects_cross_kb(conv_service):
    c = conv_service.create_conversation(title="Chat Ops", kb_id="ops")
    with pytest.raises(
        InvalidConversationError, match="pertenece a la base 'ops', no a 'general'"
    ):
        conv_service.record_turn(
            conversation_id=c.id,
            question="¿Pregunta cruzada?",
            answer="Respuesta",
            grounded=True,
            kb_id="general",
        )


def test_get_recent_history_filters_by_kb_id(conv_service):
    c = conv_service.create_conversation(title="Chat General", kb_id="general")
    conv_service.record_turn(
        conversation_id=c.id,
        question="Pregunta 1",
        answer="Respuesta 1",
        grounded=True,
        kb_id="general",
    )
    hist_gen = conv_service.get_recent_history(c.id, kb_id="general")
    assert len(hist_gen) == 2
    hist_ops = conv_service.get_recent_history(c.id, kb_id="ops")
    assert len(hist_ops) == 0


def test_sliding_window_limits_to_max_exchanges(conv_service):
    c = conv_service.create_conversation(title="Larga")
    for i in range(8):
        conv_service.record_turn(
            conversation_id=c.id,
            question=f"Pregunta {i}",
            answer=f"Respuesta {i} [1].",
            grounded=True,
            kb_id="general",
        )

    history = conv_service.get_recent_history(c.id, max_exchanges=6)
    assert len(history) == 12
    assert history[0] == ("user", "Pregunta 2")
    assert history[-1] == ("assistant", "Respuesta 7 [1].")


def test_ungrounded_turn_preserves_message_in_history(conv_service):
    c = conv_service.create_conversation()
    conv_service.record_turn(
        conversation_id=c.id,
        question="¿Hay datos de Marte?",
        answer=None,
        grounded=False,
        reason="below_threshold",
        message="No se encontró evidencia suficientemente relevante en la documentación.",
        kb_id="general",
    )
    history = conv_service.get_recent_history(c.id)
    assert len(history) == 2
    assert history[1] == (
        "assistant",
        "No se encontró evidencia suficientemente relevante en la documentación.",
    )


def test_rename_and_validation(conv_service):
    c = conv_service.create_conversation(title="Inicial")
    updated = conv_service.rename_conversation(c.id, "  Nuevo Nombre  ")
    assert updated.title == "Nuevo Nombre"

    with pytest.raises(InvalidConversationError):
        conv_service.rename_conversation(c.id, "   ")


def test_not_found_errors(conv_service):
    with pytest.raises(ConversationNotFoundError):
        conv_service.get_conversation("inexistente")
    with pytest.raises(ConversationNotFoundError):
        conv_service.get_recent_history("inexistente")
    with pytest.raises(ConversationNotFoundError):
        conv_service.rename_conversation("inexistente", "Nuevo")
    with pytest.raises(ConversationNotFoundError):
        conv_service.delete_conversation("inexistente")


def test_delete_conversation_cascades(conv_service):
    c = conv_service.create_conversation(title="Para borrar")
    conv_service.record_turn(
        conversation_id=c.id,
        question="Hola",
        answer="Hola [1].",
        grounded=True,
        kb_id="general",
    )
    conv_service.delete_conversation(c.id)

    with pytest.raises(ConversationNotFoundError):
        conv_service.get_conversation(c.id)


def test_read_only_mode_blocks_writes(tmp_path):
    s = Settings(catalog_path=tmp_path / "ro.sqlite3", read_only_mode=True)
    ro_service = ConversationService(s)
    with pytest.raises(ReadOnlyModeError):
        ro_service.create_conversation(title="Fallará")
