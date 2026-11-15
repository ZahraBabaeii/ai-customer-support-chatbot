"""Text embedding backends.

``onnx`` is the real semantic model (all-MiniLM-L6-v2, run locally through
ONNX Runtime; downloaded once, ~80 MB, no API key needed).

``hash`` is a dependency-free lexical fallback (hashed bag of words). It is
used by the test suite and when the ONNX model cannot be loaded, so the app
still starts offline. It matches words, not meaning.
"""

import hashlib
import logging
import math
import re
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    default_min_relevance: float

    def fit(self, corpus: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OnnxEmbedder:
    name = "onnx"
    default_min_relevance = 0.31

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()
        self._fn(["warm up"])  # triggers the one-time model download

    def fit(self, corpus: list[str]) -> None:
        """Pretrained model: nothing to learn from the corpus."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._fn(texts)]


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from had has have how i if in is it
    its me my of on or our so than that the their them there these they this to was we
    were what when where which who why will with you your about any get much many
    long take tell please would could should""".split()
)


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
    elif len(token) > 5 and token.endswith("ly"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _terms(text: str) -> list[str]:
    """Words plus adjacent word pairs, so "Aero Pro" can match as a phrase."""
    words = tokenize(text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class HashEmbedder:
    """TF-IDF over hashed tokens. ``fit`` learns word rarity from the corpus."""

    name = "hash"
    default_min_relevance = 0.12
    _HEADING_WEIGHT = 3  # the first line of a chunk is its heading path

    def __init__(self, dimensions: int = 2048) -> None:
        self._dimensions = dimensions
        self._idf: dict[str, float] = {}
        self._max_idf = 1.0

    def fit(self, corpus: list[str]) -> None:
        document_frequency: dict[str, int] = {}
        for text in corpus:
            for token in set(_terms(text)):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = len(corpus)
        self._idf = {
            token: math.log((total + 1) / (df + 0.5)) for token, df in document_frequency.items()
        }
        self._max_idf = math.log((total + 1) / 0.5)

    def _embed_one(self, text: str) -> list[float]:
        # The extra last dimension collects terms the corpus has never seen:
        # they lower the similarity of off-topic queries without matching anything.
        vector = [0.0] * (self._dimensions + 1)
        heading, _, body = text.partition("\n")
        counts: dict[str, int] = {}
        for token in _terms(heading):
            counts[token] = counts.get(token, 0) + (self._HEADING_WEIGHT if body else 1)
        for token in _terms(body):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            if self._idf and token not in self._idf:
                if "_" not in token:  # unseen single words only, not unseen pairs
                    vector[-1] += self._max_idf
                continue
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            weight = (1 + math.log(count)) * self._idf.get(token, self._max_idf)
            vector[index] += sign * weight
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector] if norm else vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


def create_embedder(backend: str = "auto") -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    try:
        return OnnxEmbedder()
    except Exception:
        if backend == "onnx":
            raise
        logger.warning(
            "Could not load the ONNX embedding model (first run needs internet to "
            "download it). Falling back to lexical 'hash' embeddings.",
            exc_info=True,
        )
        return HashEmbedder()
