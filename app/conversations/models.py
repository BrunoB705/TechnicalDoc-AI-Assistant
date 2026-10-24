from dataclasses import dataclass, field


@dataclass
class MessageRecord:
    id: int
    conversation_id: str
    role: str
    content: str
    grounded: bool = True
    reason: str | None = None
    message: str | None = None
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    knowledge_base_id: str = "general"
    pedagogical_mode: str | None = None
    strict_mode: bool | None = None
    created_at: str = ""


@dataclass
class ConversationSummary:
    id: str
    title: str
    last_kb_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    knowledge_base_id: str = ""

    def __post_init__(self) -> None:
        if not self.knowledge_base_id:
            self.knowledge_base_id = self.last_kb_id


@dataclass
class ConversationDetail:
    id: str
    title: str
    last_kb_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    messages: list[MessageRecord] = field(default_factory=list)
    knowledge_base_id: str = ""

    def __post_init__(self) -> None:
        if not self.knowledge_base_id:
            self.knowledge_base_id = self.last_kb_id
