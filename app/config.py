"""Application settings, loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

STORE_NAME = "Nimbus Gear"
ASSISTANT_NAME = "Nimbus"
SUPPORT_EMAIL = "support@nimbusgear.example"
SUPPORT_PHONE = "1-800-555-0142"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM. Leave the key empty to run in demo mode (no external calls).
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 2048
    llm_timeout_seconds: int = 30

    # Retrieval.
    embedding_backend: Literal["auto", "onnx", "hash"] = "auto"
    knowledge_base_dir: Path = BASE_DIR / "data" / "knowledge_base"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    chunk_max_chars: int = 900
    retrieval_top_k: int = 4
    # Minimum cosine similarity for a chunk to count as relevant.
    # Leave unset to use the default for the active embedding backend.
    min_relevance: float | None = None

    # Conversation.
    max_history_messages: int = 12
    max_sessions: int = 1000

    # Server.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def llm_enabled(self) -> bool:
        key = self.gemini_api_key.strip()
        return bool(key) and not key.lower().startswith("your")


@lru_cache
def get_settings() -> Settings:
    return Settings()
