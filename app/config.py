from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str
    port: int = 8000
    log_level: str = "INFO"
    gemini_api_key: str  # Google Gemini API key for LLM category analysis

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
