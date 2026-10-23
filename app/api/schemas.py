from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    id: int
    source: str
    section: str | None = None
    page: int | None = None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str | None
    grounded: bool
    sources: list[Source]  # solo las fuentes citadas en la respuesta
    retrieved: list[Source] = []  # todas las recuperadas sobre el umbral
    latency_ms: int
    message: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str


class DocumentResponse(BaseModel):
    doc_id: str
    source: str
    title: str
    file_type: str
    chunks: int
    pages: int | None = None


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]
