import json
import time
from collections import defaultdict, deque
from dataclasses import asdict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.dependencies import get_rag_service
from app.exceptions import RateLimitExceededError
from app.rag.service import MESSAGES, RAGService

router = APIRouter()

_WINDOW_SECONDS = 60.0
_request_timestamps: dict[str, deque[float]] = defaultdict(deque)


def reset_rate_limiter() -> None:
    _request_timestamps.clear()


def check_rate_limit(request: Request, service: RAGService = Depends(get_rag_service)) -> None:
    limit = service.settings.rate_limit_per_minute
    if limit <= 0:
        return
    client_ip = request.client.host if request.client else "local"
    now = time.monotonic()
    bucket = _request_timestamps[client_ip]
    while bucket and (now - bucket[0]) > _WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= limit:
        raise RateLimitExceededError(
            f"Límite de {limit} consultas por minuto superado. Intentá de nuevo en unos segundos."
        )
    bucket.append(now)


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(check_rate_limit)],
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def query(req: QueryRequest, service: RAGService = Depends(get_rag_service)) -> QueryResponse:
    start = time.perf_counter()
    result = service.answer(
        req.question,
        req.top_k,
        knowledge_base_id=req.knowledge_base_id,
        conversation_id=req.conversation_id,
    )
    msg = None if result.grounded else (result.message or MESSAGES.get(result.reason or ""))
    return QueryResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[asdict(s) for s in result.sources],
        retrieved=[asdict(s) for s in result.retrieved],
        latency_ms=int((time.perf_counter() - start) * 1000),
        message=msg,
        reason=result.reason,
        knowledge_base_id=result.knowledge_base_id,
        conversation_id=result.conversation_id,
    )


@router.post(
    "/query/stream",
    dependencies=[Depends(check_rate_limit)],
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def query_stream(
    req: QueryRequest,
    request: Request,
    service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    """Endpoint de streaming por Server-Sent Events (SSE). Emite tokens a medida que se generan."""
    # Validamos que la conversación pertenezca a la KB (422/404), la KB exista (404)
    # o que el índice esté disponible (409) antes de abrir SSE
    resolved_kb, _ = service._validate_and_resolve_conv_kb(
        req.conversation_id, req.knowledge_base_id
    )
    service._resolve_kb(resolved_kb)

    async def event_generator():
        async for item in service.stream_answer(
            req.question,
            req.top_k,
            knowledge_base_id=req.knowledge_base_id,
            conversation_id=req.conversation_id,
        ):
            if await request.is_disconnected():
                break
            event = item.get("event", "message")
            data = json.dumps(item.get("data", {}), ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
