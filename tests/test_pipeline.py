import pytest

from hybridsearch import EmbeddingReranker, HybridRetriever, IdentityReranker
from hybridsearch.types import FusedHit


def test_hybrid_retrieval_finds_the_expected_passage(store):
    retriever = HybridRetriever(store=store, top_k=3)
    assert "doc-455" in [h.doc_id for h in retriever.retrieve("who pays the court fees on withdrawal")]


def test_top_k_bounds_the_result_list(store):
    assert len(HybridRetriever(store=store, top_k=2).retrieve("appeal")) == 2


def test_top_k_can_be_overridden_per_call(store):
    retriever = HybridRetriever(store=store, top_k=2)
    assert len(retriever.retrieve("appeal", top_k=5)) == 5


def test_a_blank_query_returns_nothing_rather_than_the_whole_corpus(store):
    assert HybridRetriever(store=store).retrieve("   ") == []


def test_candidate_limit_below_top_k_is_rejected(store):
    with pytest.raises(ValueError, match="candidate_limit"):
        HybridRetriever(store=store, candidate_limit=2, top_k=10)


def test_hybrid_beats_vector_alone_on_an_exact_phrase_query(store):
    """The case hybrid retrieval exists for.

    A distinctive literal phrase is what lexical search is good at and what a
    weak dense encoder blurs away. Hybrid should not lose to vector-only here.
    """
    query = "suspended from the first to the thirty-first of August"
    vector_only = HybridRetriever(store=store, vector_weight=1.0, lexical_weight=0.0, top_k=3)
    hybrid = HybridRetriever(store=store, top_k=3)
    hybrid_rank = [h.doc_id for h in hybrid.retrieve(query)].index("doc-771")
    vector_ids = [h.doc_id for h in vector_only.retrieve(query)]
    vector_rank = vector_ids.index("doc-771") if "doc-771" in vector_ids else 99
    assert hybrid_rank <= vector_rank


def test_explain_shows_every_stage(store):
    explained = HybridRetriever(store=store).explain("appeal withdrawn fees")
    assert set(explained) == {"query", "lexical", "vector", "fused", "final"}
    assert explained["fused"], "fusion produced no candidates"
    # Every final document must have come through fusion.
    assert set(explained["final"]) <= {row[0] for row in explained["fused"]}


def test_explain_reports_which_retriever_found_each_document(store):
    fused = HybridRetriever(store=store).explain("court fees")["fused"]
    assert all(found_by for _, _, found_by in fused)


class TestRerankers:
    def test_identity_reranker_preserves_fusion_order(self):
        hits = [FusedHit("a", 0.3), FusedHit("b", 0.2)]
        assert [h.doc_id for h in IdentityReranker().rerank("q", hits, 2)] == ["a", "b"]

    def test_identity_reranker_truncates_to_top_k(self):
        hits = [FusedHit(str(i), 1.0) for i in range(10)]
        assert len(IdentityReranker().rerank("q", hits, 3)) == 3

    def test_embedding_reranker_promotes_the_relevant_passage(self):
        hits = [
            FusedHit("irrelevant", 0.9, text="Espresso machines require descaling every month."),
            FusedHit("relevant", 0.1, text="An appeal must be filed within sixty days of notification."),
        ]
        reranked = EmbeddingReranker().rerank("when must an appeal be filed", hits, 2)
        assert reranked[0].doc_id == "relevant"

    def test_rerankers_handle_an_empty_candidate_list(self):
        assert EmbeddingReranker().rerank("q", [], 5) == []
        assert IdentityReranker().rerank("q", [], 5) == []
