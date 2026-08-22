"""Document stores: the pgvector-backed one, and an in-memory one for tests."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .embeddings import Embedder, HashingEmbedder, cosine_similarity
from .types import Hit

_WORD = re.compile(r"[\w']+", re.UNICODE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentStore(Protocol):
    def index(self, documents: Sequence[Document]) -> int: ...
    def vector_search(self, query: str, limit: int) -> list[Hit]: ...
    def lexical_search(self, query: str, limit: int) -> list[Hit]: ...


@dataclass
class InMemoryStore:
    """BM25 plus cosine similarity over an in-process corpus.

    Same interface as the Postgres store, so the pipeline and its tests never
    need a database running. BM25 here is the real Robertson/Sparck-Jones
    formulation, not a token-overlap stand-in -- if it were fake, the fusion
    tests would be testing nothing.
    """

    embedder: Embedder = None  # type: ignore[assignment]
    k1: float = 1.2
    b: float = 0.75
    _docs: dict[str, Document] = field(default_factory=dict, repr=False)
    _tokens: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _vectors: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = HashingEmbedder()

    def index(self, documents: Sequence[Document]) -> int:
        if not documents:
            return 0
        vectors = self.embedder.embed([d.text for d in documents])
        for document, vector in zip(documents, vectors, strict=True):
            self._docs[document.doc_id] = document
            self._tokens[document.doc_id] = _WORD.findall(document.text.lower())
            self._vectors[document.doc_id] = vector
        return len(documents)

    @property
    def _avg_length(self) -> float:
        if not self._tokens:
            return 0.0
        return sum(len(t) for t in self._tokens.values()) / len(self._tokens)

    def _idf(self, term: str) -> float:
        n = len(self._tokens)
        df = sum(1 for tokens in self._tokens.values() if term in tokens)
        # Add-half smoothing keeps the idf of a term in every document at a
        # small positive value rather than letting it go negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def lexical_search(self, query: str, limit: int) -> list[Hit]:
        terms = _WORD.findall(query.lower())
        if not terms or not self._tokens:
            return []
        avg_length = self._avg_length
        scored: list[Hit] = []
        for doc_id, tokens in self._tokens.items():
            counts = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / avg_length)
                score += self._idf(term) * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append(
                    Hit(doc_id, score, self._docs[doc_id].text, self._docs[doc_id].metadata)
                )
        scored.sort(key=lambda h: (-h.score, h.doc_id))
        return scored[:limit]

    def vector_search(self, query: str, limit: int) -> list[Hit]:
        if not self._vectors:
            return []
        query_vector = self.embedder.embed([query])[0]
        scored = [
            Hit(doc_id, cosine_similarity(query_vector, vector), self._docs[doc_id].text,
                self._docs[doc_id].metadata)
            for doc_id, vector in self._vectors.items()
        ]
        scored.sort(key=lambda h: (-h.score, h.doc_id))
        return scored[:limit]


@dataclass
class PgVectorStore:
    """Postgres with pgvector for dense retrieval and tsvector/GIN for lexical.

    One database rather than Postgres plus a dedicated vector service: it keeps
    the two retrievers transactionally consistent, so a freshly written
    document is either visible to both or to neither. Split them and you get a
    window where lexical search finds a document whose embedding has not landed
    yet, which surfaces as intermittent, unreproducible recall bugs.

    Requires ``psycopg`` (``pip install '.[postgres]'``). See ``sql/schema.sql``.
    """

    dsn: str
    embedder: Embedder
    table: str = "documents"
    text_search_config: str = "english"

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "PgVectorStore needs psycopg: pip install '.[postgres]'"
            ) from exc
        return psycopg.connect(self.dsn)

    def index(self, documents: Sequence[Document]) -> int:
        if not documents:
            return 0
        import json as _json

        vectors = self.embedder.embed([d.text for d in documents])
        rows = [
            (
                d.doc_id,
                d.text,
                _json.dumps(d.metadata),
                "[" + ",".join(f"{v:.8f}" for v in vec) + "]",
            )
            for d, vec in zip(documents, vectors, strict=True)
        ]
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {self.table} (doc_id, content, metadata, embedding)
                VALUES (%s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (doc_id) DO UPDATE
                   SET content = EXCLUDED.content,
                       metadata = EXCLUDED.metadata,
                       embedding = EXCLUDED.embedding
                """,
                rows,
            )
            connection.commit()
        return len(rows)

    def vector_search(self, query: str, limit: int) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT doc_id, content, metadata, 1 - (embedding <=> %s::vector) AS score
                  FROM {self.table}
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s
                """,
                (literal, literal, limit),
            )
            return [Hit(r[0], float(r[3]), r[1], r[2] or {}) for r in cursor.fetchall()]

    def lexical_search(self, query: str, limit: int) -> list[Hit]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT doc_id, content, metadata,
                       ts_rank_cd(content_tsv, websearch_to_tsquery(%s, %s)) AS score
                  FROM {self.table}
                 WHERE content_tsv @@ websearch_to_tsquery(%s, %s)
                 ORDER BY score DESC
                 LIMIT %s
                """,
                (self.text_search_config, query, self.text_search_config, query, limit),
            )
            return [Hit(r[0], float(r[3]), r[1], r[2] or {}) for r in cursor.fetchall()]
