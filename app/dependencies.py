from functools import lru_cache

from app.config import get_settings
from app.documents.service import DocumentService
from app.rag.embeddings import get_embeddings
from app.rag.llm import get_llm
from app.rag.service import RAGService
from app.rag.vectorstore import get_vectorstore


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()
    vs = get_vectorstore(settings, get_embeddings(settings))
    return RAGService(vs, get_llm(settings), settings)


@lru_cache
def get_document_service() -> DocumentService:
    settings = get_settings()
    return DocumentService(get_vectorstore(settings, get_embeddings(settings)), settings)
