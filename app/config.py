from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "ollama"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    llm_timeout: float = 120.0  # segundos

    embedding_model: str = "intfloat/multilingual-e5-small"
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "techdocs"
    docs_dir: Path = Path("data/docs")
    max_upload_mb: int = 20

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    min_relevance: float = 0.35  # se calibra en la fase de evaluación


@lru_cache
def get_settings() -> Settings:
    return Settings()
