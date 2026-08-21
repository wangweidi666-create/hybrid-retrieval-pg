"""Hybrid retrieval over Postgres: BM25 and pgvector, fused by RRF, then reranked."""

from .embeddings import Embedder, HashingEmbedder, OpenAICompatibleEmbedder, cosine_similarity
from .fusion import reciprocal_rank_fusion, weighted_score_fusion
from .pipeline import HybridRetriever
from .rerank import CrossEncoderReranker, EmbeddingReranker, IdentityReranker, Reranker
from .store import Document, DocumentStore, InMemoryStore, PgVectorStore
from .types import FusedHit, Hit

__version__ = "0.1.0"

__all__ = [
    "CrossEncoderReranker",
    "Document",
    "DocumentStore",
    "Embedder",
    "EmbeddingReranker",
    "FusedHit",
    "HashingEmbedder",
    "Hit",
    "HybridRetriever",
    "IdentityReranker",
    "InMemoryStore",
    "OpenAICompatibleEmbedder",
    "PgVectorStore",
    "Reranker",
    "cosine_similarity",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
]
