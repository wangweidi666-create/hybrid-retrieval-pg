import pytest

from hybridsearch import Document, HashingEmbedder, InMemoryStore, cosine_similarity


def test_bm25_ranks_the_document_containing_the_query_terms_first(store):
    hits = store.lexical_search("appeal withdrawn court fees", limit=3)
    assert hits[0].doc_id == "doc-455"


def test_bm25_returns_nothing_for_terms_absent_from_the_corpus(store):
    assert store.lexical_search("zebra parachute", limit=5) == []


def test_bm25_scores_are_positive_and_descending(store):
    hits = store.lexical_search("appeal", limit=5)
    assert all(h.score > 0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_vector_search_returns_the_whole_corpus_ranked(store):
    hits = store.vector_search("deadline for filing an appeal", limit=100)
    assert len(hits) == 10
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_vector_search_respects_the_limit(store):
    assert len(store.vector_search("appeal", limit=3)) == 3


def test_searching_an_empty_store_returns_nothing():
    empty = InMemoryStore()
    assert empty.lexical_search("anything", 5) == []
    assert empty.vector_search("anything", 5) == []


def test_indexing_returns_the_number_of_documents_written(store):
    assert store.index([Document("doc-new", "A fresh passage about hearings.")]) == 1
    assert store.lexical_search("fresh passage hearings", 1)[0].doc_id == "doc-new"


def test_indexing_the_same_id_twice_overwrites_rather_than_duplicates(store):
    store.index([Document("doc-114", "Completely different content about parking permits.")])
    hits = store.lexical_search("parking permits", 5)
    assert [h.doc_id for h in hits] == ["doc-114"]


def test_indexing_nothing_is_a_no_op(store):
    assert store.index([]) == 0


def test_hits_carry_their_text_back(store):
    assert "sixty days" in store.lexical_search("sixty days notification", 1)[0].text


class TestHashingEmbedder:
    def test_is_deterministic(self):
        embedder = HashingEmbedder()
        assert embedder.embed(["the same text"]) == embedder.embed(["the same text"])

    def test_produces_unit_vectors(self):
        vector = HashingEmbedder().embed(["some words here"])[0]
        assert sum(v * v for v in vector) == pytest.approx(1.0)

    def test_similar_text_scores_higher_than_unrelated_text(self):
        embedder = HashingEmbedder()
        base, near, far = embedder.embed(
            [
                "the appeal must be filed within sixty days",
                "an appeal is filed within sixty days of notice",
                "espresso machines require descaling every month",
            ]
        )
        assert cosine_similarity(base, near) > cosine_similarity(base, far)

    def test_empty_text_gives_a_zero_vector_not_a_crash(self):
        assert HashingEmbedder().embed([""])[0] == [0.0] * 256

    def test_dimension_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
