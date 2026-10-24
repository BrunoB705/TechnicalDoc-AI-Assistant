from langchain_core.language_models import BaseChatModel

from app.config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": settings.openai_model,
            "temperature": 0,
            "api_key": settings.openai_api_key or "lm-studio",
            "timeout": settings.llm_timeout,
            "max_tokens": settings.max_output_tokens,
            "max_retries": 0,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=settings.max_output_tokens,
        client_kwargs={"timeout": settings.llm_timeout},
    )
