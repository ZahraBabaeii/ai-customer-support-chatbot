"""FastAPI application: wires settings, the RAG pipeline, and the HTTP routes."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.llm import GeminiClient, LLMClient
from app.logging_config import configure_logging
from app.rag.chunker import load_knowledge_base
from app.rag.embeddings import Embedder, create_embedder
from app.rag.pipeline import ChatService
from app.rag.vector_store import VectorStore
from app.schemas import ChatRequest, ChatResponse, HealthResponse, SessionId
from app.sessions import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class Components:
    service: ChatService
    store: VectorStore
    sessions: SessionStore
    embedding_backend: str


def build_components(
    settings: Settings,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    store: VectorStore | None = None,
) -> Components:
    """Build the pipeline. Arguments let tests inject fakes."""
    embedder = embedder or create_embedder(settings.embedding_backend)
    store = store or VectorStore(embedder, settings.chroma_dir)
    store.sync(load_knowledge_base(settings.knowledge_base_dir, settings.chunk_max_chars))

    if llm is None and settings.llm_enabled:
        llm = GeminiClient(
            api_key=settings.gemini_api_key.strip(),
            model=settings.gemini_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    min_relevance = (
        settings.min_relevance
        if settings.min_relevance is not None
        else embedder.default_min_relevance
    )
    service = ChatService(store, llm, settings.retrieval_top_k, min_relevance)
    sessions = SessionStore(settings.max_history_messages, settings.max_sessions)
    logger.info(
        "Ready: mode=%s model=%s embeddings=%s chunks=%d",
        service.mode, service.model, embedder.name, store.count(),
    )
    return Components(service, store, sessions, embedder.name)


def create_app(components: Components | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if components is None:
            settings = get_settings()
            configure_logging(settings.log_level)
            app.state.components = build_components(settings)
        else:
            app.state.components = components
        yield

    app = FastAPI(
        title="Nimbus Gear Support API",
        description="RAG-powered customer support chatbot backend.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s -> %d (%.0f ms)",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong on our side. Please try again."},
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        c: Components = request.app.state.components
        return HealthResponse(
            status="ok",
            mode=c.service.mode,
            model=c.service.model,
            embedding_backend=c.embedding_backend,
            indexed_chunks=c.store.count(),
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        c: Components = request.app.state.components
        session_id = body.session_id or uuid.uuid4().hex
        result = c.service.answer(body.message, c.sessions.history(session_id))
        c.sessions.append_turn(session_id, body.message, result.answer)
        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            mode=result.mode,
            grounded=result.grounded,
            sources=[asdict(s) for s in result.sources],
        )

    @app.delete("/sessions/{session_id}")
    def reset_session(session_id: SessionId, request: Request) -> dict[str, bool]:
        c: Components = request.app.state.components
        return {"cleared": c.sessions.clear(session_id)}

    return app


app = create_app()
