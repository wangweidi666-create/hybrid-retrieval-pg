# hybrid-retrieval-pg

Hybrid retrieval on a single Postgres: BM25 over `tsvector` and dense search
over `pgvector`, fused with Reciprocal Rank Fusion, then reranked.

Core logic has no dependencies and runs offline. Postgres is one backend behind
a `DocumentStore` interface, not a hard requirement to run the tests.

> Open-sourced reference implementation of an approach I built for a production
> legal-AI product. Written from scratch against a synthetic corpus — no
> employer data or internal code is included.

## Why one database

Postgres holds the text, the metadata and the vectors. Not Postgres plus a
dedicated vector service.

The reason is consistency, not convenience. With one table, a freshly written
document is visible to both retrievers or to neither. Split them and you open a
window where lexical search finds a document whose embedding has not landed
yet — which does not fail loudly, it surfaces as intermittent recall bugs that
nobody can reproduce.

```sql
CREATE TABLE documents (
    doc_id      TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding   VECTOR(256),
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON documents USING GIN (content_tsv);
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

# In-memory store, no database needed
python -m hybridsearch search "who pays if the appeal is dropped" \
  --corpus examples/corpus.jsonl --rerank
```

With Postgres:

```bash
docker compose up -d
export HYBRID_DSN="postgresql://hybrid:hybrid@localhost:5432/hybrid"
python -m hybridsearch index examples/corpus.jsonl --dsn "$HYBRID_DSN"
python -m hybridsearch search "who pays if the appeal is dropped"
```

## How fusion works

RRF needs no score normalisation and no per-corpus tuning. A document at rank
*r* in a list scores `1 / (k + r)`, summed across the lists that returned it.

Score-level fusion is the obvious alternative and it is a trap. A BM25 score of
12.4 and a cosine similarity of 0.81 are not on the same scale, and the BM25
distribution shifts every time the corpus grows. Ranks are immune to both.

With `k=60`, rank 1 contributes `1/61` and rank 2 contributes `1/62` — close
enough that no single retriever runs away with the list, far enough that a
document *both* retrievers rank highly beats either one's runner-up. That is
the whole point: lexical finds the exact article number, dense finds the
paraphrase, and what they agree on goes on top.

`weighted_score_fusion` is included for comparison. It wins when both
retrievers are calibrated and you have a gold set to tune on; it loses badly
the moment a distribution shifts.

## Candidate pool vs top_k

```python
HybridRetriever(store=store, candidate_limit=50, top_k=5)
```

`candidate_limit` is what each retriever returns before fusion. `top_k` is what
survives reranking and reaches the model. They are separate on purpose: fusing
a wide pool is nearly free, reranking it is not, and context budget is spent
only on `top_k`. Widening the pool buys recall cheaply; widening `top_k` costs
reranker calls and tokens.

## Debugging a bad result

```bash
python -m hybridsearch explain "emergency measures during the case" --corpus examples/corpus.jsonl
```

Prints each stage — lexical hits, vector hits, the fused list with which
retriever found what, and the final ranking. Which stage lost the document
tells you which stage to fix: missing from both retrievers is an indexing or
chunking problem; present after fusion but gone after reranking is a reranker
problem.

## Benchmark — and an honest result

```bash
python -m hybridsearch benchmark examples/corpus.jsonl examples/queries.jsonl -k 5
```

```
| configuration   | recall@5 |
| --------------- | -------- |
| lexical only    |   0.8333 |
| vector only     |   0.7500 |
| hybrid (RRF)    |   0.7500 |
| hybrid + rerank |   0.7500 |
```

**Hybrid does not win here, and the numbers are printed as measured.**

The default `HashingEmbedder` is a bag-of-words hashing vectoriser. It is
genuinely deterministic and genuinely offline, which is what makes a fresh
clone runnable — but it has no semantics. Look at the three queries that miss:

| query | gold passage says |
| --- | --- |
| "can I file online" | "certified **electronic portal**" |
| "getting the hearing moved up urgently" | "**expedited** on grounds of **urgency**" |
| "emergency measures during the case" | "**interim relief**" |

Zero lexical overlap in all three. A bag-of-words vector is just lexical
matching with extra steps, so the dense path is blind to exactly the queries it
was supposed to rescue — and adding a blind retriever to the fusion only
injects noise. A weight sweep confirms it: lexical-heavy ratios claw back to
`0.8333` and never exceed it.

This is the argument for having a benchmark at all. "Hybrid beats vector-only"
is a claim about your encoder and your corpus, not a law. Point the harness at
a real encoder and rerun the same command:

```bash
python -m hybridsearch benchmark examples/corpus.jsonl examples/queries.jsonl \
  --embed-base-url http://localhost:8000/v1 --embed-model <model> --dimensions 1024
```

## Pluggable pieces

| interface | offline default | production |
| --- | --- | --- |
| `Embedder` | `HashingEmbedder` | `OpenAICompatibleEmbedder` (hosted API or self-hosted vLLM/TEI) |
| `Reranker` | `EmbeddingReranker` | `CrossEncoderReranker` (HTTP `/rerank`) |
| `DocumentStore` | `InMemoryStore` (real BM25, not a stub) | `PgVectorStore` |

`InMemoryStore` implements the actual Robertson/Spärck-Jones BM25 formulation.
If it were a token-overlap stand-in the fusion tests would be testing nothing.

## Tests

```bash
pytest -q     # 42 tests, no database, no network
```

## License

MIT. See [LICENSE](LICENSE).
