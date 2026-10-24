from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.documents.service import DocumentService
from app.knowledge_bases.service import KnowledgeBaseService
from app.rag.embeddings import get_embeddings
from app.rag.llm import get_llm
from app.rag.service import RAGService
from app.rag.vectorstore import get_vectorstore


@lru_cache
def get_shared_embeddings() -> Embeddings:
    settings = get_settings()
    return get_embeddings(settings)


@lru_cache
def get_shared_vectorstore() -> Chroma:
    settings = get_settings()
    return get_vectorstore(settings, get_shared_embeddings())


@lru_cache
def get_knowledge_base_service() -> KnowledgeBaseService:
    settings = get_settings()
    return KnowledgeBaseService(get_shared_vectorstore(), settings)


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    kb_svc = get_knowledge_base_service()
    return RAGService(
        get_shared_vectorstore(), get_llm(settings), settings, kb_service=kb_svc
    )


@lru_cache
def get_document_service() -> DocumentService:
    settings = get_settings()
    kb_svc = get_knowledge_base_service()
    return DocumentService(get_shared_vectorstore(), settings, kb_service=kb_svc)
