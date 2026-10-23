from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings


class PrefixedEmbeddings(Embeddings):
    """Agrega los prefijos `query: ` / `passage: ` que requieren los modelos E5."""

    def __init__(self, inner: Embeddings, query_prefix: str = "", passage_prefix: str = ""):
        self.inner = inner
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def get_embeddings(settings: Settings) -> Embeddings:
    base = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
    if "e5" in settings.embedding_model.lower():
        return PrefixedEmbeddings(base, "query: ", "passage: ")
    return base
