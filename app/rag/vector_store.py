"""ChromaDB-backed vector store for knowledge-base chunks."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.rag.chunker import Chunk
from app.rag.embeddings import Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity, higher is more relevant


def _fingerprint(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(chunk.embed_text.encode())
    return digest.hexdigest()


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        persist_dir: Path | None = None,
        collection_name: str = "knowledge_base",
    ) -> None:
        self._embedder = embedder
        # Separate collection per backend: their vectors are not comparable.
        self._name = f"{collection_name}_{embedder.name}"
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = None

    def sync(self, chunks: list[Chunk]) -> bool:
        """Make the collection match ``chunks``. Returns True if it was rebuilt."""
        fingerprint = _fingerprint(chunks)
        self._embedder.fit([c.embed_text for c in chunks])
        try:
            existing = self._client.get_collection(self._name, embedding_function=None)
        except Exception:
            existing = None

        if existing is not None:
            metadata = existing.metadata or {}
            if metadata.get("fingerprint") == fingerprint and existing.count() == len(chunks):
                self._collection = existing
                logger.info("Vector index is up to date (%d chunks)", len(chunks))
                return False
            self._client.delete_collection(self._name)

        collection = self._client.create_collection(
            self._name,
            metadata={"hnsw:space": "cosine", "fingerprint": fingerprint},
            embedding_function=None,
        )
        collection.add(
            ids=[c.id for c in chunks],
            embeddings=self._embedder.embed([c.embed_text for c in chunks]),
            documents=[c.body for c in chunks],
            metadatas=[
                {"source": c.source, "title": c.title, "section": c.section} for c in chunks
            ],
        )
        self._collection = collection
        logger.info("Indexed %d chunks into '%s'", len(chunks), self._name)
        return True

    def count(self) -> int:
        return self._collection.count() if self._collection is not None else 0

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if self._collection is None:
            raise RuntimeError("Vector store has not been synced yet")
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        for chunk_id, body, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = Chunk(
                id=chunk_id,
                source=meta["source"],
                title=meta["title"],
                section=meta["section"],
                body=body,
            )
            hits.append(SearchHit(chunk=chunk, score=round(1.0 - float(distance), 4)))
        return hits
