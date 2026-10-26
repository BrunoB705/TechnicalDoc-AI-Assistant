import json
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.conversations.service import ConversationService
from app.exceptions import InvalidLearningActionError
from app.learning.service import LearningService
from app.rag.service import RAGService, SourceHit


def _json(**value):
    return json.dumps(value)


def _service(tmp_path, responses):
    settings = Settings(catalog_path=tmp_path / "catalog.sqlite3", enable_step_back=False)
    conversations = ConversationService(settings)
    conversation = conversations.create_conversation(kb_id="architecture")
    rag = RAGService(
        MagicMock(),
        FakeListChatModel(responses=responses),
        settings,
        conversation_service=conversations,
    )
    source = SourceHit(
        1,
        "rules.md",
        None,
        None,
        "binary16 has 1 sign, 5 exponent and 10 fraction bits",
        0.95,
        "architecture",
    )
    rag._resolve_kb = MagicMock(return_value=("architecture", "Architecture", False))
    rag._retrieve = MagicMock(
        return_value=(
            [(Document(page_content=source.snippet, metadata={"source": "rules.md"}), 0.95)],
            [source],
            1,
        )
    )
    service = LearningService(settings, rag, conversations)
    service._retrieve_goal = MagicMock(return_value=([source], "[1] rules.md: " + source.snippet))
    return service, conversations, conversation


def _lesson(correct="A"):
    return _json(
        content="El formato distribuye los bits entre signo, exponente y fracción [1].",
        question="¿Qué campo codifica el signo?",
        options=[{"id": "A", "text": "El bit de signo"}, {"id": "B", "text": "El sesgo"}],
        correct_option_id=correct,
        explanation="El primer bit codifica el signo [1].",
    )


def test_learning_requires_plan_approval_and_persists_progress(tmp_path):
    responses = [
        _json(
            question="¿Qué campo conocés?",
            options=[{"id": "A", "text": "Signo"}, {"id": "B", "text": "Sesgo"}],
            correct_option_id="A",
            explanation="El signo ocupa un bit [1].",
        ),
        _json(
            approach="Partiremos de los campos y construiremos el formato.",
            nodes=[
                {
                    "id": "n1",
                    "title": "Campos de binary16",
                    "rationale": "Necesitamos conocer el ancho de cada campo.",
                    "prerequisites": [],
                    "source_ids": [1],
                }
            ],
        ),
        _lesson("A"),
        _lesson("B"),
    ]
    service, conversations, conv = _service(tmp_path, responses)
    session = service.start_session(conv.id, "Aprender binary16")
    assert session["state"] == "diagnosing"
    assert "correct_option_id" not in session["quiz"]
    with pytest.raises(InvalidLearningActionError):
        service.act(session["id"], "approve_plan")

    session = service.act(session["id"], "answer_diagnostic", "A")
    assert session["state"] == "plan_pending"
    session = service.act(session["id"], "approve_plan")
    assert session["state"] == "teaching"
    assert session["quiz"]["kind"] == "node"
    session = service.act(session["id"], "answer_quiz", "A")
    assert session["state"] == "awaiting_advance"
    session = service.act(session["id"], "advance")
    assert session["state"] == "final_assessment"
    session = service.act(session["id"], "answer_quiz", "B")
    assert session["state"] == "completed"
    assert session["score"] == 1.0

    restored = LearningService(service.settings, service.rag, conversations)
    assert restored.get_session(session["id"])["state"] == "completed"
    assert restored.get_by_conversation(conv.id)["id"] == session["id"]


def test_wrong_answer_reteaches_same_concept(tmp_path):
    responses = [
        _json(
            question="¿Qué campo conocés?",
            options=[{"id": "A", "text": "Signo"}, {"id": "B", "text": "Sesgo"}],
            correct_option_id="A",
            explanation="El signo ocupa un bit [1].",
        ),
        _json(
            approach="Partiremos de los campos.",
            nodes=[
                {
                    "id": "n1",
                    "title": "Campos",
                    "rationale": "Fundamentos",
                    "prerequisites": [],
                    "source_ids": [1],
                }
            ],
        ),
        _lesson("A"),
        _lesson("A"),
    ]
    service, _, conv = _service(tmp_path, responses)
    session = service.start_session(conv.id, "Aprender campos")
    session = service.act(session["id"], "answer_diagnostic", "A")
    session = service.act(session["id"], "approve_plan")
    session = service.act(session["id"], "answer_quiz", "B")
    assert session["state"] == "teaching"
    assert session["current_node_index"] == 0
    assert session["nodes"][0]["status"] == "review"
    assert session["quiz"] is not None


def test_learning_rejects_invalid_map_sources_without_advancing(tmp_path):
    responses = [
        _json(
            question="¿Qué campo conocés?",
            options=[{"id": "A", "text": "Signo"}, {"id": "B", "text": "Sesgo"}],
            correct_option_id="A",
            explanation="El signo ocupa un bit [1].",
        ),
        _json(
            approach="Plan inválido",
            nodes=[
                {
                    "id": "n1",
                    "title": "Campos",
                    "rationale": "",
                    "prerequisites": [],
                    "source_ids": [999],
                }
            ],
        ),
    ]
    service, _, conv = _service(tmp_path, responses)
    session = service.start_session(conv.id, "Aprender campos")
    with pytest.raises(InvalidLearningActionError):
        service.act(session["id"], "answer_diagnostic", "A")
    assert service.get_session(session["id"])["state"] == "diagnosing"


def test_solve_retries_when_model_omits_valid_citation(tmp_path):
    responses = [
        _json(
            question="¿Qué campo conocés?",
            options=[{"id": "A", "text": "Signo"}, {"id": "B", "text": "Sesgo"}],
            correct_option_id="A",
            explanation="El signo ocupa un bit [1].",
        ),
        _json(
            approach="Partiremos de los campos.",
            nodes=[
                {
                    "id": "n1",
                    "title": "Campos",
                    "rationale": "Fundamentos",
                    "prerequisites": [],
                    "source_ids": [1],
                }
            ],
        ),
        _lesson("A"),
        _json(content="Los sistemas representan valores con símbolos."),
        _json(content="Los sistemas representan valores con símbolos [1]."),
    ]
    service, _, conversation = _service(tmp_path, responses)
    session = service.start_session(conversation.id, "Aprender campos")
    session = service.act(session["id"], "answer_diagnostic", "A")
    session = service.act(session["id"], "approve_plan")

    session = service.act(session["id"], "solve")

    assert session["state"] == "teaching"
    assert session["content"].endswith("[1].")
