from dataclasses import dataclass


@dataclass
class KnowledgeBase:
    id: str
    name: str
    description: str
    created_at: str
    documents: int = 0
    chunks: int = 0
