"""Rebuild the vector index from data/knowledge_base:  python -m scripts.ingest

The backend does this automatically at startup when the documents change;
this script is for doing it ahead of time or inspecting the result.
"""

from app.config import get_settings
from app.logging_config import configure_logging
from app.rag.chunker import load_knowledge_base
from app.rag.embeddings import create_embedder
from app.rag.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    chunks = load_knowledge_base(settings.knowledge_base_dir, settings.chunk_max_chars)
    embedder = create_embedder(settings.embedding_backend)
    store = VectorStore(embedder, settings.chroma_dir)
    rebuilt = store.sync(chunks)
    state = "rebuilt" if rebuilt else "already up to date"
    print(f"{len(chunks)} chunks, embeddings={embedder.name}, index {state}.")


if __name__ == "__main__":
    main()
