"""Split Markdown knowledge-base documents into retrievable chunks.

Documents are split along their heading structure, so each chunk is one
self-contained topic (one product, one policy section, one FAQ entry).
Sections longer than ``max_chars`` are split further at paragraph boundaries.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # file name
    title: str  # document title (H1)
    section: str  # heading path below the title, e.g. "Audio > Aero Pro"
    body: str  # the text shown to the user / LLM

    @property
    def embed_text(self) -> str:
        """Text that gets embedded: the heading path gives the body context."""
        heading = " > ".join(p for p in (self.title, self.section) if p)
        return f"{heading}\n{self.body}"


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split text at blank lines (then single lines) into pieces <= max_chars."""
    if len(text) <= max_chars:
        return [text]

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    units: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            units.append(block)
        else:
            units.extend(line for line in block.splitlines() if line.strip())

    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = unit
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(text: str, source: str, max_chars: int = 900) -> list[Chunk]:
    title = ""
    path: dict[int, str] = {}  # heading level -> heading text
    sections: list[tuple[str, list[str]]] = []
    lines: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in lines):
            section = " > ".join(path[level] for level in sorted(path) if level > 1)
            sections.append((section, lines.copy()))
        lines.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            lines.append(line)
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2)
        if level == 1:
            title = heading
        path[level] = heading
        for deeper in [lv for lv in path if lv > level]:
            del path[deeper]
    flush()

    chunks: list[Chunk] = []
    for section, section_lines in sections:
        body = "\n".join(section_lines).strip()
        for part, piece in enumerate(_split_long(body, max_chars)):
            digest = hashlib.sha1(f"{source}|{section}|{part}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(id=digest, source=source, title=title, section=section, body=piece)
            )
    return chunks


def load_knowledge_base(directory: Path, max_chars: int = 900) -> list[Chunk]:
    """Chunk every Markdown file in ``directory``."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {directory}")
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name, max_chars))
    if not chunks:
        raise ValueError(f"No Markdown content found in {directory}")
    return chunks
