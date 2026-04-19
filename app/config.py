from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str
    port: int = 8000
    log_level: str = "INFO"

    # Primary LLM provider
    openai_api_key: str

    # Deprecated: migrated from Gemini to OpenAI. Safe to leave in environment
    # but no longer used by the service.
    gemini_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
