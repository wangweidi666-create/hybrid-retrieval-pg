"""Shared shapes: a scored hit and a ranked result list."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hit:
    """One document returned by one retriever, with that retriever's own score.

    Scores from different retrievers are not comparable -- a BM25 score of 12.4
    and a cosine similarity of 0.81 live on different scales with different
    distributions. That incomparability is the whole reason the fusion step
    below works on ranks rather than on scores.
    """

    doc_id: str
    score: float
    text: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FusedHit:
    """A document after fusion, carrying where each retriever placed it."""

    doc_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    text: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def found_by(self) -> list[str]:
        return sorted(self.ranks)
