"""The RAG pipeline: contextualise the question, retrieve, then answer."""

import logging
import re
from dataclasses import dataclass, field

from app.llm import LLMClient, LLMError, Message
from app.rag import demo
from app.rag.prompts import REWRITE_PROMPT, SYSTEM_PROMPT, build_user_turn
from app.rag.vector_store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

_FOLLOW_UP = re.compile(
    r"\b(it|its|it's|they|them|their|this|these|those|same|that one|which one)\b"
    r"|^(and|what about|how about|or)\b",
    re.I,
)


@dataclass(frozen=True)
class Source:
    title: str
    section: str
    source: str
    score: float


@dataclass(frozen=True)
class ChatResult:
    answer: str
    mode: str  # "llm", "demo", or "fallback" (LLM configured but the call failed)
    grounded: bool  # True if relevant knowledge-base passages were found
    sources: list[Source] = field(default_factory=list)


def is_follow_up(message: str) -> bool:
    return len(message.split()) <= 3 or bool(_FOLLOW_UP.search(message))


class ChatService:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient | None = None,
        top_k: int = 4,
        min_relevance: float = 0.31,
    ) -> None:
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_relevance = min_relevance

    @property
    def mode(self) -> str:
        return "llm" if self._llm else "demo"

    @property
    def model(self) -> str | None:
        return self._llm.model if self._llm else None

    def search_query(self, message: str, history: list[Message]) -> str:
        """Turn a follow-up like "how much is it?" into a standalone query."""
        previous = [m.content for m in history if m.role == "user"]
        if not previous or not is_follow_up(message):
            return message
        if self._llm:
            try:
                rewritten = self._llm.generate(
                    REWRITE_PROMPT, [*history, Message("user", message)]
                )
                if rewritten and len(rewritten) <= 300:
                    return rewritten
            except LLMError:
                logger.warning("Query rewrite failed; using heuristic", exc_info=True)
        return f"{previous[-1]} {message}"

    def retrieve(self, query: str) -> list[SearchHit]:
        hits = self._store.search(query, self._top_k)
        return [hit for hit in hits if hit.score >= self._min_relevance]

    def answer(self, message: str, history: list[Message] | None = None) -> ChatResult:
        history = history or []

        if not self._llm:
            small_talk = demo.small_talk_reply(message)
            if small_talk:
                return ChatResult(answer=small_talk, mode="demo", grounded=False)

        query = self.search_query(message, history)
        hits = self.retrieve(query)
        sources = _to_sources(hits)
        quoted = _to_sources(demo.select_passages(hits))
        logger.info(
            "query=%r hits=%d top_score=%s", query, len(hits), hits[0].score if hits else None
        )

        if self._llm:
            try:
                text = self._llm.generate(
                    SYSTEM_PROMPT, [*history, Message("user", build_user_turn(message, hits))]
                )
                return ChatResult(text, mode="llm", grounded=bool(hits), sources=sources)
            except LLMError:
                logger.error("LLM call failed; using extractive fallback", exc_info=True)
                return ChatResult(demo.extractive_answer(hits), "fallback", bool(hits), quoted)

        return ChatResult(demo.extractive_answer(hits), "demo", bool(hits), quoted)


def _to_sources(hits: list[SearchHit]) -> list[Source]:
    return [
        Source(
            title=hit.chunk.title,
            section=hit.chunk.section,
            source=hit.chunk.source,
            score=hit.score,
        )
        for hit in hits
    ]
