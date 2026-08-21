"""The retrieval pipeline: two retrievers, fuse, rerank, return."""

from __future__ import annotations

from dataclasses import dataclass

from .fusion import DEFAULT_K, reciprocal_rank_fusion
from .rerank import IdentityReranker, Reranker
from .store import DocumentStore
from .types import FusedHit


@dataclass
class HybridRetriever:
    """Lexical and dense retrieval, fused by rank, then reranked.

    ``candidate_limit`` is what each retriever returns before fusion; ``top_k``
    is what survives reranking. Keeping them apart is the point of the design.
    Fusion over a wide candidate pool is cheap, reranking it is not, and the
    generator only ever sees ``top_k``. Widening the pool buys recall for
    almost nothing; widening ``top_k`` costs reranker calls and context budget.
    """

    store: DocumentStore
    reranker: Reranker = None  # type: ignore[assignment]
    candidate_limit: int = 50
    top_k: int = 5
    rrf_k: int = DEFAULT_K
    lexical_weight: float = 1.0
    vector_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.reranker is None:
            self.reranker = IdentityReranker()
        if self.candidate_limit < self.top_k:
            raise ValueError(
                f"candidate_limit ({self.candidate_limit}) must be >= top_k ({self.top_k})"
            )

    def retrieve(self, query: str, top_k: int | None = None) -> list[FusedHit]:
        if not query.strip():
            return []
        k = top_k or self.top_k
        runs = {
            "lexical": self.store.lexical_search(query, self.candidate_limit),
            "vector": self.store.vector_search(query, self.candidate_limit),
        }
        fused = reciprocal_rank_fusion(
            runs,
            k=self.rrf_k,
            weights={"lexical": self.lexical_weight, "vector": self.vector_weight},
        )
        return self.reranker.rerank(query, fused, k)

    def explain(self, query: str) -> dict:
        """Per-stage output for debugging a query that returned the wrong thing.

        Which stage lost the document tells you which stage to fix: absent from
        both retrievers is an indexing or chunking problem, present in fusion
        but gone after reranking is a reranker problem.
        """
        lexical = self.store.lexical_search(query, self.candidate_limit)
        vector = self.store.vector_search(query, self.candidate_limit)
        fused = reciprocal_rank_fusion(
            {"lexical": lexical, "vector": vector},
            k=self.rrf_k,
            weights={"lexical": self.lexical_weight, "vector": self.vector_weight},
        )
        return {
            "query": query,
            "lexical": [(h.doc_id, round(h.score, 4)) for h in lexical[:10]],
            "vector": [(h.doc_id, round(h.score, 4)) for h in vector[:10]],
            "fused": [(h.doc_id, round(h.score, 6), h.found_by) for h in fused[:10]],
            "final": [h.doc_id for h in self.reranker.rerank(query, fused, self.top_k)],
        }
