from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    knowledge_base_id: str | None = Field(default=None, min_length=1, max_length=64)


class Source(BaseModel):
    id: int
    source: str
    section: str | None = None
    page: int | None = None
    snippet: str
    score: float
    knowledge_base_id: str = "general"


class QueryResponse(BaseModel):
    answer: str | None
    grounded: bool
    sources: list[Source]  # solo las fuentes citadas en la respuesta
    retrieved: list[Source] = []  # todas las recuperadas sobre el umbral
    latency_ms: int
    message: str | None = None
    reason: str | None = None
    knowledge_base_id: str = "general"


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
    knowledge_base_id: str = "general"


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)
    id: str | None = Field(default=None, min_length=1, max_length=64)


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    documents: int
    chunks: int


class KnowledgeBaseListResponse(BaseModel):
    total: int
    knowledge_bases: list[KnowledgeBaseResponse]


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    knowledge_base_id: str | None = Field(default=None, min_length=1, max_length=64)


class UpdateConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("El título de la conversación no puede estar vacío.")
        return clean


class MessageResponse(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    grounded: bool = True
    reason: str | None = None
    message: str | None = None
    sources: list[Source] = []
    retrieved: list[Source] = []
    knowledge_base_id: str = "general"
    created_at: str


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str
    last_kb_id: str = "general"
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationListResponse(BaseModel):
    total: int
    conversations: list[ConversationSummaryResponse]


class ConversationDetailResponse(ConversationSummaryResponse):
    messages: list[MessageResponse] = []
