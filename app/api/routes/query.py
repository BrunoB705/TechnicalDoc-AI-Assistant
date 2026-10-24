import json
import time
from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.dependencies import get_rag_service
from app.rag.service import MESSAGES, RAGService

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def query(req: QueryRequest, service: RAGService = Depends(get_rag_service)) -> QueryResponse:
    start = time.perf_counter()
    result = service.answer(
        req.question, req.top_k, knowledge_base_id=req.knowledge_base_id
    )
    msg = (
        None
        if result.grounded
        else (result.message or MESSAGES.get(result.reason or ""))
    )
    return QueryResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[asdict(s) for s in result.sources],
        retrieved=[asdict(s) for s in result.retrieved],
        latency_ms=int((time.perf_counter() - start) * 1000),
        message=msg,
        reason=result.reason,
        knowledge_base_id=result.knowledge_base_id,
    )


@router.post(
    "/query/stream",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def query_stream(
    req: QueryRequest, service: RAGService = Depends(get_rag_service)
) -> StreamingResponse:
    """Endpoint de streaming por Server-Sent Events (SSE). Emite tokens a medida que se generan."""
    # Validamos que la KB exista (404) o que el índice esté disponible (409) antes de abrir SSE
    service._resolve_kb(req.knowledge_base_id)

    async def event_generator():
        async for item in service.stream_answer(
            req.question, req.top_k, knowledge_base_id=req.knowledge_base_id
        ):
            event = item.get("event", "message")
            data = json.dumps(item.get("data", {}), ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
