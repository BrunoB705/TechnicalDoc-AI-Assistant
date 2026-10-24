import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.conversations.service import ConversationService
from app.dependencies import get_conversation_service, get_rag_service
from app.rag.service import RAGService
from tests.integration.test_api import use_llm


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_query_api_accepts_pedagogical_mode_and_returns_metadata(
    client, vectorstore, settings
):
    fake_llm = FakeListChatModel(
        responses=["Resolución paso a paso [1]: S=0, E=10000110, M=00010010000000000000000."]
    )
    use_llm(client, vectorstore, settings, fake_llm)

    payload = {
        "question": "Calculame 137 en punto flotante simple",
        "pedagogical_mode": "solve",
        "strict_mode": False,
    }
    res = client.post("/query", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["grounded"] is True
    assert body["pedagogical_mode"] == "solve"
    assert body["strict_mode"] is False
    assert body["verification_data"] is not None
    assert "ieee754" in body["verification_data"]


def test_query_api_validates_pedagogical_mode(client):
    res = client.post(
        "/query",
        json={
            "question": "¿Cuál es el formato?",
            "pedagogical_mode": "invalido",
        },
    )
    assert res.status_code == 422


def test_query_stream_api_propagates_professor_metadata_in_end_event(
    client, vectorstore, settings
):
    fake_llm = FakeListChatModel(
        responses=["Pista conceptual [1]: Primero pasá 137 a binario."]
    )
    use_llm(client, vectorstore, settings, fake_llm)

    payload = {
        "question": "Dame una pista para representar 137",
        "pedagogical_mode": "hint",
        "strict_mode": True,
    }
    res = client.post("/query/stream", json=payload)
    assert res.status_code == 200
    events = parse_sse(res.text)
    end_events = [data for evt, data in events if evt == "end"]
    assert len(end_events) == 1
    end = end_events[0]
    assert end["grounded"] is True
    assert end["pedagogical_mode"] == "hint"
    assert end["strict_mode"] is True


def test_conversation_persists_pedagogical_mode_and_strict_mode(
    client, vectorstore, settings, tmp_path
):
    conv_svc = ConversationService(settings)
    client.app.dependency_overrides[get_conversation_service] = lambda: conv_svc
    fake_llm = FakeListChatModel(
        responses=["Diagnóstico [1]: El exponente sin sesgo es incorrecto."]
    )
    client.app.dependency_overrides[get_rag_service] = lambda: RAGService(
        vectorstore, fake_llm, settings, conversation_service=conv_svc
    )

    conv_res = client.post("/conversations", json={"title": "Tutoría IEEE 754"})
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    query_payload = {
        "question": "¿Está bien si pongo E=7 para 137?",
        "conversation_id": conv_id,
        "pedagogical_mode": "check",
        "strict_mode": True,
    }
    res = client.post("/query", json=query_payload)
    assert res.status_code == 200

    detail_res = client.get(f"/conversations/{conv_id}")
    assert detail_res.status_code == 200
    messages = detail_res.json()["messages"]
    assert len(messages) == 2
    user_msg, assistant_msg = messages[0], messages[1]

    assert user_msg["role"] == "user"
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["pedagogical_mode"] == "check"
    assert assistant_msg["strict_mode"] is True
