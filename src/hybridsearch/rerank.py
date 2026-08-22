"""Reranking the fused candidate list.

Fusion decides which fifty documents are worth a second look. The reranker
decides the order of the five that reach the model. It is the expensive stage,
which is exactly why it runs over a short list rather than over the corpus.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .embeddings import Embedder, HashingEmbedder, cosine_similarity
from .types import FusedHit


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> list[FusedHit]: ...


@dataclass
class IdentityReranker:
    """Keeps fusion order. The baseline every other reranker must beat."""

    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> list[FusedHit]:
        return list(hits[:top_k])


@dataclass
class EmbeddingReranker:
    """Offline reranker scoring query-document cosine similarity directly.

    A real cross-encoder attends over the query and document jointly and beats
    this comfortably. This exists so the pipeline is exercisable end to end
    with no model server running.
    """

    embedder: Embedder = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = HashingEmbedder()

    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> list[FusedHit]:
        if not hits:
            return []
        vectors = self.embedder.embed([query] + [h.text for h in hits])
        query_vector, doc_vectors = vectors[0], vectors[1:]
        scored = [
            FusedHit(h.doc_id, cosine_similarity(query_vector, v), h.ranks, h.text, h.metadata)
            for h, v in zip(hits, doc_vectors, strict=True)
        ]
        scored.sort(key=lambda h: (-h.score, h.doc_id))
        return scored[:top_k]


@dataclass
class CrossEncoderReranker:
    """HTTP cross-encoder, e.g. a TEI or vLLM reranking endpoint.

    Expects ``POST {base_url}/rerank`` with ``{"query": ..., "documents": [...]}``
    returning ``{"results": [{"index": int, "relevance_score": float}, ...]}``.
    """

    base_url: str
    model: str = ""
    api_key: str = "not-needed"
    timeout: float = 30.0

    def rerank(self, query: str, hits: Sequence[FusedHit], top_k: int) -> list[FusedHit]:
        if not hits:
            return []
        payload = {"query": query, "documents": [h.text for h in hits], "top_n": top_k}
        if self.model:
            payload["model"] = self.model
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/rerank",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        out = []
        for row in body["results"][:top_k]:
            hit = hits[row["index"]]
            out.append(
                FusedHit(
                    hit.doc_id,
                    float(row["relevance_score"]),
                    hit.ranks,
                    hit.text,
                    hit.metadata,
                )
            )
        return out
