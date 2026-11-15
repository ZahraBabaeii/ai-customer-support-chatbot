import uuid

import pytest

from app.rag.chunker import chunk_markdown
from app.rag.embeddings import HashEmbedder, tokenize
from app.rag.vector_store import VectorStore


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("What are the shipping costs for returns?") == ["ship", "cost", "return"]


def test_hash_embeddings_are_normalised_and_deterministic():
    embedder = HashEmbedder()
    a, b = embedder.embed(["refund policy", "refund policy"])
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("query", "expected_source", "expected_text"),
    [
        ("What is your return window?", "returns_and_refunds.md", "30 days"),
        ("How much is overnight shipping?", "shipping_policy.md", "$24.95"),
        ("Aero Pro headphones battery life", "products.md", "40 hours"),
        ("Which payment methods do you accept?", "faq.md", "PayPal"),
        ("How do I make a warranty claim?", "warranty.md", "order number"),
        ("What are your support hours?", "company_overview.md", "8:00 AM"),
    ],
)
def test_search_finds_the_right_passage(store, query, expected_source, expected_text):
    top = store.search(query, top_k=3)[0]
    assert top.chunk.source == expected_source
    assert expected_text in top.chunk.body


def test_scores_are_sorted_descending(store):
    hits = store.search("international shipping duties", top_k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_query_scores_low(store):
    hits = store.search("capital of France", top_k=1)
    assert hits[0].score < HashEmbedder.default_min_relevance


def test_sync_reuses_unchanged_index_and_rebuilds_on_change():
    name = f"test_{uuid.uuid4().hex}"
    chunks = chunk_markdown("# Doc\n\n## A\n\nalpha\n\n## B\n\nbeta", "doc.md")
    vector_store = VectorStore(HashEmbedder(), None, name)
    assert vector_store.sync(chunks) is True
    assert vector_store.count() == 2

    again = VectorStore(HashEmbedder(), None, name)
    assert again.sync(chunks) is False

    changed = chunk_markdown("# Doc\n\n## A\n\nalpha\n\n## B\n\ngamma", "doc.md")
    assert again.sync(changed) is True
    assert again.search("gamma", 1)[0].chunk.body == "gamma"
