import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents, health, query, ui
from app.exceptions import (
    DocumentNotFoundError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    EmptyIndexError,
    InvalidDocumentError,
    LLMUnavailableError,
)


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = FastAPI(title="TechnicalDoc AI Assistant", version="0.2.0")
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(documents.router)
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

    return app


app = create_app()
