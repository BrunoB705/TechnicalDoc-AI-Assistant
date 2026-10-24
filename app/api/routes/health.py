import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.dependencies import get_knowledge_base_service
from app.knowledge_bases.service import KnowledgeBaseService

logger = logging.getLogger("rag")
router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/live")
def health_live(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "version": "1.0.0",
        "read_only_mode": settings.read_only_mode,
    }


@router.get("/health/ready")
def health_ready(
    settings: Settings = Depends(get_settings),
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> JSONResponse:
    try:
        kbs = kb_service.list_knowledge_bases()
        total_chunks = sum(k.chunks for k in kbs)
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "knowledge_bases": len(kbs),
                "total_chunks": total_chunks,
                "read_only_mode": settings.read_only_mode,
            },
        )
    except Exception as exc:
        logger.error("Readiness check failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "detail": "El almacén de datos no está disponible temporalmente.",
            },
        )
