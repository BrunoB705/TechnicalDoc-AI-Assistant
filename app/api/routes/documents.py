from dataclasses import asdict

from fastapi import APIRouter, Depends, File, Response, UploadFile

from app.api.schemas import DocumentListResponse, DocumentResponse, ErrorResponse
from app.dependencies import get_document_service
from app.documents.service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])

_ERRORS = {
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


@router.post("", response_model=DocumentResponse, status_code=201, responses=_ERRORS)
def upload_document(
    file: UploadFile = File(...), service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    info = service.add_document(file.filename or "", file.file.read())
    return DocumentResponse(**asdict(info))


@router.get("", response_model=DocumentListResponse)
def list_documents(
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    docs = [DocumentResponse(**asdict(d)) for d in service.list_documents()]
    return DocumentListResponse(total=len(docs), documents=docs)


@router.get("/{doc_id}", response_model=DocumentResponse, responses={404: _ERRORS[404]})
def get_document(
    doc_id: str, service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    return DocumentResponse(**asdict(service.get_document(doc_id)))


@router.delete("/{doc_id}", status_code=204, responses={404: _ERRORS[404]})
def delete_document(
    doc_id: str, service: DocumentService = Depends(get_document_service)
) -> Response:
    service.delete_document(doc_id)
    return Response(status_code=204)
