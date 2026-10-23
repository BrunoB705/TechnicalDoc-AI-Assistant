import json
import time
from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.dependencies import get_rag_service
from app.rag.service import RAGService

router = APIRouter()

MESSAGES = {
    "below_threshold": "No encontré evidencia suficiente en los documentos para responder.",
    "model_refused": "Los documentos recuperados no alcanzan para responder la pregunta.",
    "no_citations": "La respuesta no cita ninguna fuente, por lo que no pudo verificarse.",
}


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def query(req: QueryRequest, service: RAGService = Depends(get_rag_service)) -> QueryResponse:
    start = time.perf_counter()
    result = service.answer(req.question, req.top_k)
    return QueryResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[asdict(s) for s in result.sources],
        retrieved=[asdict(s) for s in result.retrieved],
        latency_ms=int((time.perf_counter() - start) * 1000),
        message=None if result.grounded else MESSAGES.get(result.reason or ""),
    )


@router.post("/query/stream")
async def query_stream(
    req: QueryRequest, service: RAGService = Depends(get_rag_service)
) -> StreamingResponse:
    """Endpoint de streaming por Server-Sent Events (SSE). Emite tokens a medida que se generan."""

    async def event_generator():
        async for item in service.stream_answer(req.question, req.top_k):
            event = item.get("event", "message")
            data = json.dumps(item.get("data", {}), ensure_ascii=False)
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
