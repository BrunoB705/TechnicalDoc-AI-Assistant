from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.config import Settings
from app.exceptions import LLMUnavailableError


def _build_openai_chat(
    settings: Settings,
    *,
    model: str,
    api_key: SecretStr | str,
    base_url: str | None,
    default_headers: dict[str, str] | None = None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "api_key": api_key,
        "timeout": settings.llm_timeout,
        "max_tokens": settings.max_output_tokens,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if default_headers:
        kwargs["default_headers"] = default_headers
    return ChatOpenAI(**kwargs)


def is_opencode_configured(settings: Settings) -> bool:
    return bool(settings.opencode_api_key and settings.opencode_api_key.get_secret_value().strip())


def get_llm(
    settings: Settings, profile: Literal["local", "opencode"] | None = None
) -> BaseChatModel:
    resolved_profile = profile or settings.default_llm_profile

    if resolved_profile == "opencode":
        if not is_opencode_configured(settings) or settings.opencode_api_key is None:
            raise LLMUnavailableError(
                "OpenCode Go no está configurado: definí OPENCODE_API_KEY en .env."
            )
        return _build_openai_chat(
            settings,
            model=settings.opencode_model,
            api_key=settings.opencode_api_key,
            base_url=settings.opencode_base_url,
            default_headers={
                "x-opencode-session": "technicaldoc-ai-assistant",
                "User-Agent": "technicaldoc-ai-assistant/1.0",
            },
        )

    if settings.llm_provider == "openai":
        return _build_openai_chat(
            settings,
            model=settings.openai_model,
            api_key=settings.openai_api_key or "lm-studio",
            base_url=settings.openai_base_url,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=settings.max_output_tokens,
        client_kwargs={"timeout": settings.llm_timeout},
    )
