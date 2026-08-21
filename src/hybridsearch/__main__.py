"""CLI: index, search, explain, benchmark."""

from __future__ import annotations

import argparse
import json
import os

from .benchmark import load_corpus, load_queries, render, run
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .pipeline import HybridRetriever
from .rerank import EmbeddingReranker, IdentityReranker
from .store import InMemoryStore, PgVectorStore


def _embedder(args):
    if getattr(args, "embed_base_url", None):
        return OpenAICompatibleEmbedder(
            base_url=args.embed_base_url, model=args.embed_model, dimensions=args.dimensions
        )
    return HashingEmbedder(dimensions=args.dimensions)


def _store(args):
    dsn = getattr(args, "dsn", None) or os.environ.get("HYBRID_DSN")
    if dsn:
        return PgVectorStore(dsn=dsn, embedder=_embedder(args))
    store = InMemoryStore(embedder=_embedder(args))
    if getattr(args, "corpus", None):
        store.index(load_corpus(args.corpus))
    return store


def _cmd_index(args) -> int:
    store = PgVectorStore(dsn=args.dsn, embedder=_embedder(args))
    written = store.index(load_corpus(args.corpus))
    print(f"indexed {written} documents")
    return 0


def _cmd_search(args) -> int:
    retriever = HybridRetriever(
        store=_store(args),
        reranker=EmbeddingReranker(_embedder(args)) if args.rerank else IdentityReranker(),
        top_k=args.top_k,
    )
    for rank, hit in enumerate(retriever.retrieve(args.query), start=1):
        found = ",".join(hit.found_by) or "-"
        print(f"{rank:>2}. {hit.doc_id:<12} {hit.score:.6f}  [{found}]  {hit.text[:80]}")
    return 0


def _cmd_explain(args) -> int:
    print(json.dumps(HybridRetriever(store=_store(args)).explain(args.query), indent=2))
    return 0


def _cmd_benchmark(args) -> int:
    store = _store(args)
    print(render(run(store, load_queries(args.queries), k=args.k), k=args.k))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybridsearch", description=__doc__)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--embed-base-url", help="OpenAI-compatible /v1 endpoint")
    parser.add_argument("--embed-model", default="text-embedding-3-small")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="load a JSONL corpus into Postgres")
    index.add_argument("corpus")
    index.add_argument("--dsn", required=True)
    index.set_defaults(func=_cmd_index)

    search = sub.add_parser("search", help="run a hybrid query")
    search.add_argument("query")
    search.add_argument("--corpus", help="JSONL corpus for the in-memory store")
    search.add_argument("--dsn", help="use Postgres instead (or set HYBRID_DSN)")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--rerank", action="store_true")
    search.set_defaults(func=_cmd_search)

    explain = sub.add_parser("explain", help="show every retrieval stage for one query")
    explain.add_argument("query")
    explain.add_argument("--corpus")
    explain.add_argument("--dsn")
    explain.set_defaults(func=_cmd_explain)

    bench = sub.add_parser("benchmark", help="compare configurations on a gold set")
    bench.add_argument("corpus")
    bench.add_argument("queries")
    bench.add_argument("--dsn")
    bench.add_argument("-k", type=int, default=5)
    bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
