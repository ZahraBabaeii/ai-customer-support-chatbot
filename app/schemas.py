"""Request and response models for the HTTP API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

MAX_MESSAGE_CHARS = 1000

SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{8,64}$")]


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    session_id: SessionId | None = None

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be empty")
        return value


class SourceOut(BaseModel):
    title: str
    section: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    mode: str
    grounded: bool
    sources: list[SourceOut]


class HealthResponse(BaseModel):
    status: str
    mode: str
    model: str | None
    embedding_backend: str
    indexed_chunks: int
