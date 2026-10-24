from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Response, UploadFile

from app.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
)
from app.dependencies import get_document_service, get_knowledge_base_service
from app.documents.service import DocumentService
from app.knowledge_bases.service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])

_ERRORS = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


@router.get("", response_model=KnowledgeBaseListResponse)
def list_knowledge_bases(
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseListResponse:
    kbs = [KnowledgeBaseResponse(**asdict(k)) for k in service.list_knowledge_bases()]
    return KnowledgeBaseListResponse(total=len(kbs), knowledge_bases=kbs)


@router.post(
    "",
    response_model=KnowledgeBaseResponse,
    status_code=201,
    responses={409: _ERRORS[409], 422: _ERRORS[422]},
)
def create_knowledge_base(
    req: KnowledgeBaseCreateRequest,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseResponse:
    kb = service.create_knowledge_base(req.name, req.description, kb_id=req.id)
    return KnowledgeBaseResponse(**asdict(kb))


@router.get(
    "/{kb_id}",
    response_model=KnowledgeBaseResponse,
    responses={404: _ERRORS[404]},
)
def get_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(**asdict(service.get_knowledge_base(kb_id)))


@router.delete(
    "/{kb_id}",
    status_code=204,
    responses={404: _ERRORS[404], 422: _ERRORS[422]},
)
def delete_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> Response:
    service.delete_knowledge_base(kb_id)
    return Response(status_code=204)


# ---- Documentos dentro de una Knowledge Base ----


@router.get(
    "/{kb_id}/documents",
    response_model=DocumentListResponse,
    responses={404: _ERRORS[404]},
)
def list_kb_documents(
    kb_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    docs = [DocumentResponse(**asdict(d)) for d in service.list_documents(knowledge_base_id=kb_id)]
    return DocumentListResponse(total=len(docs), documents=docs)


@router.post(
    "/{kb_id}/documents",
    response_model=DocumentResponse,
    status_code=201,
    responses=_ERRORS,
)
def upload_kb_document(
    kb_id: str,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    content = service.read_upload(file.file)
    info = service.add_document(file.filename or "", content, knowledge_base_id=kb_id)
    return DocumentResponse(**asdict(info))


@router.get(
    "/{kb_id}/documents/{doc_id}",
    response_model=DocumentResponse,
    responses={404: _ERRORS[404]},
)
def get_kb_document(
    kb_id: str,
    doc_id: str,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    info = service.get_document(doc_id, knowledge_base_id=kb_id)
    return DocumentResponse(**asdict(info))


@router.delete(
    "/{kb_id}/documents/{doc_id}",
    status_code=204,
    responses={404: _ERRORS[404]},
)
def delete_kb_document(
    kb_id: str,
    doc_id: str,
    service: DocumentService = Depends(get_document_service),
) -> Response:
    service.delete_document(doc_id, knowledge_base_id=kb_id)
    return Response(status_code=204)
