"""Compare retrieval configurations on a gold set.

The table this produces is the argument for hybrid retrieval. Vector-only and
lexical-only each fail on a different class of query -- paraphrase versus exact
phrase -- and the fused run is not a compromise between them but better than
both, because the fusion step rewards the documents they agree on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .pipeline import HybridRetriever
from .rerank import EmbeddingReranker, IdentityReranker
from .store import Document, DocumentStore, InMemoryStore


@dataclass(frozen=True)
class GoldQuery:
    query: str
    relevant_ids: list[str]


def load_corpus(path: str | Path) -> list[Document]:
    documents = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            documents.append(Document(row["doc_id"], row["text"], row.get("metadata", {})))
    return documents


def load_queries(path: str | Path) -> list[GoldQuery]:
    queries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            queries.append(GoldQuery(row["query"], [str(i) for i in row["relevant_ids"]]))
    return queries


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set) / len(relevant_set)


CONFIGURATIONS: dict[str, dict] = {
    "lexical only": {"lexical_weight": 1.0, "vector_weight": 0.0, "rerank": False},
    "vector only": {"lexical_weight": 0.0, "vector_weight": 1.0, "rerank": False},
    "hybrid (RRF)": {"lexical_weight": 1.0, "vector_weight": 1.0, "rerank": False},
    "hybrid + rerank": {"lexical_weight": 1.0, "vector_weight": 1.0, "rerank": True},
}


def run(store: DocumentStore, queries: Sequence[GoldQuery], k: int = 5) -> dict[str, float]:
    results: dict[str, float] = {}
    for name, config in CONFIGURATIONS.items():
        retriever = HybridRetriever(
            store=store,
            reranker=EmbeddingReranker() if config["rerank"] else IdentityReranker(),
            lexical_weight=config["lexical_weight"],
            vector_weight=config["vector_weight"],
            top_k=k,
        )
        scores = [
            recall_at_k([h.doc_id for h in retriever.retrieve(q.query, top_k=k)], q.relevant_ids, k)
            for q in queries
        ]
        results[name] = round(sum(scores) / len(scores), 4) if scores else 0.0
    return results


def render(results: dict[str, float], k: int) -> str:
    width = max(len(name) for name in results)
    lines = [f"| {'configuration':<{width}} | recall@{k} |", f"| {'-' * width} | {'-' * 8} |"]
    for name, score in results.items():
        lines.append(f"| {name:<{width}} |   {score:.4f} |")
    return "\n".join(lines)


def demo(corpus_path: str | Path, queries_path: str | Path, k: int = 5) -> dict[str, float]:
    store = InMemoryStore()
    store.index(load_corpus(corpus_path))
    return run(store, load_queries(queries_path), k=k)
