import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import conversations, documents, health, knowledge_bases, query, ui
from app.exceptions import (
    ConversationNotFoundError,
    DocumentNotFoundError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    DuplicateKnowledgeBaseError,
    EmptyIndexError,
    InvalidConversationError,
    InvalidDocumentError,
    InvalidKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    LLMUnavailableError,
    RateLimitExceededError,
    ReadOnlyModeError,
)


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = FastAPI(title="TechnicalDoc AI Assistant", version="1.0.0")
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(knowledge_bases.router)
    app.include_router(documents.router)
    app.include_router(conversations.router)
    app.include_router(ui.router)
    app.mount("/static", StaticFiles(directory=ui.STATIC_DIR), name="static")

    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(_: Request, exc: LLMUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"error": "llm_unavailable", "detail": str(exc)}
        )

    @app.exception_handler(EmptyIndexError)
    async def _empty_index(_: Request, exc: EmptyIndexError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": "empty_index", "detail": str(exc)})

    def _register(exc_type: type[Exception], status: int, code: str) -> None:
        @app.exception_handler(exc_type)
        async def _handler(_: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status, content={"error": code, "detail": str(exc)})

    _register(DocumentNotFoundError, 404, "document_not_found")
    _register(DuplicateDocumentError, 409, "duplicate_document")
    _register(InvalidDocumentError, 422, "invalid_document")
    _register(DocumentTooLargeError, 413, "document_too_large")
    _register(KnowledgeBaseNotFoundError, 404, "knowledge_base_not_found")
    _register(DuplicateKnowledgeBaseError, 409, "duplicate_knowledge_base")
    _register(InvalidKnowledgeBaseError, 422, "invalid_knowledge_base")
    _register(ConversationNotFoundError, 404, "conversation_not_found")
    _register(InvalidConversationError, 422, "invalid_conversation")
    _register(ReadOnlyModeError, 403, "read_only_mode")
    _register(RateLimitExceededError, 429, "rate_limit_exceeded")

    return app


app = create_app()
