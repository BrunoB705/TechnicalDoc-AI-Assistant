from unittest.mock import MagicMock

from langchain_chroma import Chroma
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.conversations.service import ConversationService
from app.rag.service import RAGService


def test_first_turn_does_not_call_condensation(tmp_path):
    fake_llm = FakeListChatModel(responses=["En binario se usan 0 y 1 [1]."])
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1
    doc = MagicMock()
    doc.page_content = "Los números en binario se representan con 0 y 1."
    doc.metadata = {"source": "binario.md", "knowledge_base_id": "general"}
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]

    settings = Settings(catalog_path=tmp_path / "cat.sqlite3")
    conv_svc = ConversationService(settings)
    c = conv_svc.create_conversation()

    rag = RAGService(vs, fake_llm, settings, conversation_service=conv_svc)
    res = rag.answer("¿Cómo se representan los números en binario?", conversation_id=c.id)

    assert res.grounded is True
    assert res.conversation_id == c.id
    vs.similarity_search_with_relevance_scores.assert_called_once_with(
        "¿Cómo se representan los números en binario?",
        k=settings.top_k,
        filter={"knowledge_base_id": "general"},
    )
    history = conv_svc.get_recent_history(c.id)
    assert len(history) == 2


def test_second_turn_condenses_question_and_retrieves_with_standalone_query(tmp_path):
    fake_llm = FakeListChatModel(
        responses=[
            "¿Cómo se representan los números en base hexadecimal?",
            "En hexadecimal se usan dígitos del 0 al 9 y letras A a F [1].",
        ]
    )
    vs = MagicMock(spec=Chroma)
    vs._collection.count.return_value = 1
    doc = MagicMock()
    doc.page_content = "En base hexadecimal se usan 0-9 y A-F."
    doc.metadata = {"source": "hex.md", "knowledge_base_id": "general"}
    vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.92)]

    settings = Settings(catalog_path=tmp_path / "cat.sqlite3")
    conv_svc = ConversationService(settings)
    c = conv_svc.create_conversation()
    conv_svc.record_turn(
        c.id,
        "¿Cómo se representan los números en binario?",
        "En binario se usan 0 y 1 [1].",
        True,
        kb_id="general",
    )

    rag = RAGService(vs, fake_llm, settings, conversation_service=conv_svc)
    res = rag.answer("¿Y en base hexadecimal?", conversation_id=c.id)

    assert res.grounded is True
    vs.similarity_search_with_relevance_scores.assert_called_once_with(
        "¿Cómo se representan los números en base hexadecimal?",
        k=settings.top_k,
        filter={"knowledge_base_id": "general"},
    )
    history = conv_svc.get_recent_history(c.id)
    assert len(history) == 4
