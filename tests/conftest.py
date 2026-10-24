from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.dependencies import (
    get_document_service,
    get_knowledge_base_service,
    get_rag_service,
)
from app.documents.service import DocumentService
from app.knowledge_bases.service import KnowledgeBaseService
from app.main import create_app
from app.rag.service import RAGService

FIXTURES = Path(__file__).parent / "fixtures" / "docs"


def make_pdf(text: str) -> bytes:
    """PDF mínimo de una página (pypdf reconstruye el xref si hace falta)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return out


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_dir=tmp_path / "chroma",
        docs_dir=tmp_path / "docs",
        catalog_path=tmp_path / "catalog.sqlite3",
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
        min_relevance=-10.0,  # los embeddings falsos no tienen semántica
        openai_api_key=None,
    )


@pytest.fixture
def vectorstore(settings) -> Chroma:
    return Chroma(
        collection_name="test",
        embedding_function=DeterministicFakeEmbedding(size=32),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["El puerto por defecto es 8000 [1]."])


@pytest.fixture
def client(vectorstore, fake_llm, settings):
    from app.ingestion.pipeline import ingest_directory

    ingest_directory(FIXTURES, vectorstore, settings)
    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: RAGService(vectorstore, fake_llm, settings)
    return TestClient(app)


@pytest.fixture
def docs_client(client, vectorstore, settings, tmp_path):
    s = settings.model_copy(update={"docs_dir": tmp_path / "docs", "max_upload_mb": 1})
    kb_svc = KnowledgeBaseService(vectorstore, s)
    doc_svc = DocumentService(vectorstore, s, kb_service=kb_svc)
    client.app.dependency_overrides[get_knowledge_base_service] = lambda: kb_svc
    client.app.dependency_overrides[get_document_service] = lambda: doc_svc
    client.docs_dir = s.docs_dir / s.default_kb_id
    return client
