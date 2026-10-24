from dataclasses import asdict

from fastapi import APIRouter, Depends, Response

from app.api.schemas import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    ErrorResponse,
    UpdateConversationRequest,
)
from app.conversations.service import ConversationService
from app.dependencies import get_conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])

_ERRORS = {
    403: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


@router.get("", response_model=ConversationListResponse)
def list_conversations(
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationListResponse:
    items = [ConversationSummaryResponse(**asdict(c)) for c in service.list_conversations()]
    return ConversationListResponse(total=len(items), conversations=items)


@router.post(
    "",
    response_model=ConversationSummaryResponse,
    status_code=201,
    responses={403: _ERRORS[403], 422: _ERRORS[422]},
)
def create_conversation(
    req: CreateConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationSummaryResponse:
    summary = service.create_conversation(title=req.title, kb_id=req.knowledge_base_id)
    return ConversationSummaryResponse(**asdict(summary))


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    responses={404: _ERRORS[404]},
)
def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetailResponse:
    detail = service.get_conversation(conversation_id)
    return ConversationDetailResponse(**asdict(detail))


@router.patch(
    "/{conversation_id}",
    response_model=ConversationSummaryResponse,
    responses={403: _ERRORS[403], 404: _ERRORS[404], 422: _ERRORS[422]},
)
def rename_conversation(
    conversation_id: str,
    req: UpdateConversationRequest,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationSummaryResponse:
    summary = service.rename_conversation(conversation_id, title=req.title)
    return ConversationSummaryResponse(**asdict(summary))


@router.delete(
    "/{conversation_id}",
    status_code=204,
    responses={403: _ERRORS[403], 404: _ERRORS[404]},
)
def delete_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    service.delete_conversation(conversation_id)
    return Response(status_code=204)
