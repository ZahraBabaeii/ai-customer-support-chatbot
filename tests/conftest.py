"""Shared fixtures.

Tests use the offline ``hash`` embedder and an in-memory Chroma collection, so
they need neither network access nor an API key.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import BASE_DIR
from app.llm import LLMError, Message
from app.main import Components, create_app
from app.rag.chunker import load_knowledge_base
from app.rag.embeddings import HashEmbedder
from app.rag.pipeline import ChatService
from app.rag.vector_store import VectorStore
from app.sessions import SessionStore

MIN_RELEVANCE = HashEmbedder.default_min_relevance


class FakeLLM:
    """Records every call and returns a canned reply (or raises)."""

    model = "fake-model"

    def __init__(self, reply: str = "FAKE ANSWER", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, list[Message]]] = []

    def generate(self, system_prompt: str, messages: list[Message]) -> str:
        self.calls.append((system_prompt, messages))
        if self.fail:
            raise LLMError("simulated outage")
        return self.reply


@pytest.fixture(scope="session")
def chunks():
    return load_knowledge_base(BASE_DIR / "data" / "knowledge_base")


@pytest.fixture(scope="session")
def store(chunks):
    vector_store = VectorStore(HashEmbedder(), None, f"test_{uuid.uuid4().hex}")
    vector_store.sync(chunks)
    return vector_store


@pytest.fixture
def demo_service(store):
    return ChatService(store, llm=None, top_k=4, min_relevance=MIN_RELEVANCE)


@pytest.fixture
def client(store, demo_service):
    components = Components(demo_service, store, SessionStore(), "hash")
    with TestClient(create_app(components)) as test_client:
        yield test_client
