from app.rag.chunker import chunk_markdown

DOC = """# Store Guide

Intro paragraph.

## Shipping

Ships in 2 days.

## Products

### Widget

- Price: $10

### Gadget

- Price: $20
"""


def test_splits_on_headings_and_tracks_section_path():
    chunks = chunk_markdown(DOC, "guide.md")
    assert [c.section for c in chunks] == [
        "",
        "Shipping",
        "Products > Widget",
        "Products > Gadget",
    ]
    assert all(c.title == "Store Guide" and c.source == "guide.md" for c in chunks)
    assert chunks[2].body == "- Price: $10"


def test_embed_text_includes_heading_path():
    widget = chunk_markdown(DOC, "guide.md")[2]
    assert widget.embed_text.startswith("Store Guide > Products > Widget\n")


def test_long_sections_are_split_without_losing_text():
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    doc = "# T\n\n## Long\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_markdown(doc, "long.md", max_chars=400)
    assert len(chunks) > 1
    assert all(len(c.body) <= 400 for c in chunks)
    joined = " ".join(c.body for c in chunks)
    assert all(f"Paragraph {i} " in joined for i in range(10))


def test_chunk_ids_are_unique_and_stable():
    first = chunk_markdown(DOC, "guide.md")
    second = chunk_markdown(DOC, "guide.md")
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_real_knowledge_base_loads(chunks):
    assert len(chunks) > 40
    sources = {c.source for c in chunks}
    assert {"products.md", "faq.md", "shipping_policy.md", "returns_and_refunds.md"} <= sources
    assert all(c.body.strip() for c in chunks)
