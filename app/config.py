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
    max_output_tokens: int = 1024

    embedding_model: str = "intfloat/multilingual-e5-small"
    chroma_dir: Path = Path("data/chroma")
    collection_name: str = "techdocs"
    docs_dir: Path = Path("data/docs")
    catalog_path: Path = Path("data/catalog.sqlite3")
    default_kb_id: str = "general"
    max_upload_mb: int = 20
    read_only_mode: bool = False
    rate_limit_per_minute: int = 60

    chunk_size: int = 500
    chunk_overlap: int = 75
    top_k: int = 4
    min_relevance: float = 0.80  # calibrado empíricamente para intfloat/multilingual-e5-small


@lru_cache
def get_settings() -> Settings:
    return Settings()
