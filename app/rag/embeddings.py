from collections.abc import Callable
from functools import lru_cache

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


class LazyEmbeddings(Embeddings):
    """Carga el modelo de embeddings en memoria solo cuando se necesita generar vectores."""

    def __init__(self, factory: Callable[[], Embeddings]):
        self._factory = factory
        self._inner: Embeddings | None = None

    def _get_inner(self) -> Embeddings:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_inner().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_inner().embed_query(text)


@lru_cache(maxsize=8)
def _load_hf_embeddings_by_name(model_name: str) -> Embeddings:
    try:
        base = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        base = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    if "e5" in model_name.lower():
        return PrefixedEmbeddings(base, "query: ", "passage: ")
    return base


def _load_hf_embeddings(settings: Settings) -> Embeddings:
    return _load_hf_embeddings_by_name(settings.embedding_model)


@lru_cache(maxsize=8)
def _get_cached_embeddings(model_name: str, lazy: bool = True) -> Embeddings:
    if not lazy:
        return _load_hf_embeddings_by_name(model_name)
    return LazyEmbeddings(lambda: _load_hf_embeddings_by_name(model_name))


def get_embeddings(settings: Settings, lazy: bool = True) -> Embeddings:
    """Devuelve una instancia compartida (singleton por modelo) de Embeddings."""
    return _get_cached_embeddings(settings.embedding_model, lazy=lazy)
