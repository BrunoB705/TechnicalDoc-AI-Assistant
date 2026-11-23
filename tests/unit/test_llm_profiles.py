from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.exceptions import LLMUnavailableError
from app.rag.llm import get_llm, is_opencode_configured
from app.rag.service import RAGService


class _FakeCollection:
    def count(self):
        return 1


class _FakeVS:
    _collection = _FakeCollection()

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None):
        return [
            (
                Document(
                    page_content="El puerto por defecto es 8000.",
                    metadata={"source": "deploy.md", "knowledge_base_id": "general"},
                ),
                0.92,
            )
        ]


def test_get_llm_opencode_requires_api_key(tmp_path: Path):
    settings = Settings(
        _env_file=None, chroma_dir=tmp_path / "chroma", docs_dir=tmp_path / "docs"
    )
    assert is_opencode_configured(settings) is False
    with pytest.raises(LLMUnavailableError):
        get_llm(settings, profile="opencode")


def test_get_llm_opencode_builds_openai_client(tmp_path: Path):
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        docs_dir=tmp_path / "docs",
        opencode_api_key="oc-secret-key",
        opencode_base_url="https://opencode.ai/zen/go/v1",
        opencode_model="deepseek-v4.1-flash",
    )
    assert is_opencode_configured(settings) is True
    llm = get_llm(settings, profile="opencode")
    assert llm.model_name == "deepseek-v4.1-flash"
    assert llm.openai_api_base == "https://opencode.ai/zen/go/v1"
    assert llm.default_headers["x-opencode-session"] == "technicaldoc-ai-assistant"


def test_rag_service_switches_llm_profile(tmp_path: Path):
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        docs_dir=tmp_path / "docs",
        catalog_path=tmp_path / "catalog.sqlite3",
        enable_step_back=False,
        enable_professor=False,
    )
    local_llm = FakeListChatModel(responses=["Respuesta local [1]"])
    opencode_llm = FakeListChatModel(responses=["Respuesta opencode [1]"])

    svc = RAGService(
        _FakeVS(),
        local_llm,
        settings,
        llm_resolver=lambda p: opencode_llm if p == "opencode" else local_llm,
    )

    r_local = svc.answer("¿Cuál es el puerto?", llm_profile="local")
    assert r_local.grounded is True
    assert "Respuesta local" in r_local.answer

    r_oc = svc.answer("¿Cuál es el puerto?", llm_profile="opencode")
    assert r_oc.grounded is True
    assert "Respuesta opencode" in r_oc.answer


def test_ui_and_profiles_endpoint_expose_llm_profile_controls(client):
    profiles = client.get("/llm/profiles")
    assert profiles.status_code == 200
    data = profiles.json()
    assert "opencode_configured" in data
    assert data["default"] == "local"

    html = client.get("/").text
    assert 'data-llm="local"' in html
    assert 'data-llm="opencode"' in html

    js = client.get("/static/app.js").text
    assert "llm_profile: selectedLlmProfile" in js
    assert "/llm/profiles" in js
