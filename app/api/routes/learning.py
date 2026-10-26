from fastapi import APIRouter, Depends

from app.api.schemas import (
    ErrorResponse,
    LearningActionRequest,
    LearningSessionResponse,
    LearningStartRequest,
)
from app.dependencies import get_learning_service
from app.exceptions import LearningSessionNotFoundError
from app.learning.service import LearningService

router = APIRouter(prefix="/learning-sessions", tags=["learning"])


@router.post(
    "",
    response_model=LearningSessionResponse,
    status_code=201,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def start_session(
    request: LearningStartRequest,
    service: LearningService = Depends(get_learning_service),
) -> LearningSessionResponse:
    return LearningSessionResponse(
        **service.start_session(request.conversation_id, request.goal, request.strict_mode)
    )


@router.get("/by-conversation/{conversation_id}", response_model=LearningSessionResponse)
def get_session_for_conversation(
    conversation_id: str,
    service: LearningService = Depends(get_learning_service),
) -> LearningSessionResponse:
    session = service.get_by_conversation(conversation_id)
    if session is None:
        raise LearningSessionNotFoundError("Esta conversación no tiene una sesión de Aprender.")
    return LearningSessionResponse(**session)


@router.get(
    "/{session_id}",
    response_model=LearningSessionResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_session(
    session_id: str,
    service: LearningService = Depends(get_learning_service),
) -> LearningSessionResponse:
    return LearningSessionResponse(**service.get_session(session_id))


@router.post(
    "/{session_id}/actions",
    response_model=LearningSessionResponse,
    responses={
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def act(
    session_id: str,
    request: LearningActionRequest,
    service: LearningService = Depends(get_learning_service),
) -> LearningSessionResponse:
    return LearningSessionResponse(**service.act(session_id, request.action, request.answer))
