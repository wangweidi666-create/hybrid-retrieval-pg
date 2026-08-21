"""Embedding backends.

``HashingEmbedder`` is deterministic, offline and dependency-free, so tests and
a fresh clone work without an API key or a model download. It is a real
bag-of-words hashing vectoriser, not a stub: similar text genuinely produces
similar vectors. It is simply far weaker than a trained encoder.

``OpenAICompatibleEmbedder`` talks to any ``/v1/embeddings`` endpoint, which
covers hosted providers and a self-hosted vLLM or TEI server alike.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_WORD = re.compile(r"[\w']+", re.UNICODE)


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class HashingEmbedder:
    """Deterministic bag-of-words hashing vectoriser with L2 normalisation."""

    dimensions: int = 256

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # Signed buckets keep unrelated collisions from always reinforcing.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


@dataclass
class OpenAICompatibleEmbedder:
    """Any OpenAI-compatible ``/v1/embeddings`` endpoint."""

    base_url: str
    model: str
    dimensions: int = 1536
    api_key: str = "not-needed"
    timeout: float = 30.0
    batch_size: int = 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/embeddings",
                data=json.dumps({"model": self.model, "input": batch}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
            # The API is not required to preserve input order; index says where
            # each vector belongs.
            ordered = sorted(body["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in ordered)
        return vectors
