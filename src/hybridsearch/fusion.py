"""Combining several ranked lists into one.

Reciprocal Rank Fusion is the default because it needs no score normalisation
and no per-corpus tuning. Given rank r (1-based) in each list, a document
scores sum(1 / (k + r)) across the lists that returned it.

The k constant damps the top of each list. With k=60 -- the value from the
original Cormack et al. paper -- rank 1 contributes 1/61 and rank 2 gives
1/62, so a single retriever cannot dominate the fused list on its own, but a
document both retrievers rank highly rises above either one's runner-up.
That behaviour is exactly what a hybrid setup is for: lexical search finds
the exact statute number, vector search finds the paraphrase, and the
documents both agree on belong on top.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .types import FusedHit, Hit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    runs: Mapping[str, Sequence[Hit]],
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse named ranked lists. Returns all fused documents, best first."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    weights = weights or {}
    for name, weight in weights.items():
        if weight < 0:
            raise ValueError(f"weight for {name!r} must be non-negative, got {weight}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    payload: dict[str, Hit] = {}

    for run_name, hits in runs.items():
        weight = weights.get(run_name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight / (k + rank)
            ranks.setdefault(hit.doc_id, {})[run_name] = rank
            # Keep the first non-empty text we see; retrievers that return IDs
            # only should not blank out one that returned the passage.
            if hit.doc_id not in payload or (hit.text and not payload[hit.doc_id].text):
                payload[hit.doc_id] = hit

    fused = [
        FusedHit(
            doc_id=doc_id,
            score=score,
            ranks=ranks[doc_id],
            text=payload[doc_id].text,
            metadata=payload[doc_id].metadata,
        )
        for doc_id, score in scores.items()
    ]
    # Ties broken by doc_id so the output is stable across runs -- otherwise a
    # retrieval eval reports phantom regressions from dict ordering alone.
    fused.sort(key=lambda h: (-h.score, h.doc_id))
    return fused


def normalise_scores(hits: Sequence[Hit]) -> list[Hit]:
    """Min-max scale a single retriever's scores into [0, 1]."""
    if not hits:
        return []
    values = [h.score for h in hits]
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [Hit(h.doc_id, 1.0, h.text, h.metadata) for h in hits]
    return [
        Hit(h.doc_id, (h.score - low) / (high - low), h.text, h.metadata) for h in hits
    ]


def weighted_score_fusion(
    runs: Mapping[str, Sequence[Hit]],
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse on normalised scores instead of ranks.

    Kept for comparison. It beats RRF when both retrievers are well calibrated
    and you have a gold set to tune the weights on; it loses badly when one
    retriever's score distribution shifts, which for a lexical scorer happens
    every time the corpus grows.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    payload: dict[str, Hit] = {}

    for run_name, hits in runs.items():
        weight = weights.get(run_name, 1.0)
        for rank, hit in enumerate(normalise_scores(hits), start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight * hit.score
            ranks.setdefault(hit.doc_id, {})[run_name] = rank
            if hit.doc_id not in payload or (hit.text and not payload[hit.doc_id].text):
                payload[hit.doc_id] = hit

    fused = [
        FusedHit(doc_id, score, ranks[doc_id], payload[doc_id].text, payload[doc_id].metadata)
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda h: (-h.score, h.doc_id))
    return fused
