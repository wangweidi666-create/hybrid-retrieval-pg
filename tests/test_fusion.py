import pytest

from hybridsearch import reciprocal_rank_fusion, weighted_score_fusion
from hybridsearch.fusion import normalise_scores
from hybridsearch.types import Hit


def run(*doc_ids):
    return [Hit(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(doc_ids)]


def test_a_document_both_retrievers_rank_highly_wins():
    fused = reciprocal_rank_fusion({"lexical": run("a", "b", "c"), "vector": run("b", "a", "d")})
    # 'b' is 2nd and 1st; 'a' is 1st and 2nd -- equal, so the tie-break by
    # doc_id decides. The point is that both beat anything found once.
    assert {fused[0].doc_id, fused[1].doc_id} == {"a", "b"}
    assert fused[0].score == pytest.approx(fused[1].score)
    assert fused[-1].score < fused[0].score


def test_a_document_found_by_one_retriever_still_survives():
    fused = reciprocal_rank_fusion({"lexical": run("a"), "vector": run("z")})
    assert {h.doc_id for h in fused} == {"a", "z"}


def test_ranks_record_where_each_retriever_placed_the_document():
    fused = reciprocal_rank_fusion({"lexical": run("a", "b"), "vector": run("b")})
    by_id = {h.doc_id: h for h in fused}
    assert by_id["b"].ranks == {"lexical": 2, "vector": 1}
    assert by_id["b"].found_by == ["lexical", "vector"]
    assert by_id["a"].found_by == ["lexical"]


def test_weights_can_favour_one_retriever():
    runs = {"lexical": run("a"), "vector": run("z")}
    lexical_heavy = reciprocal_rank_fusion(runs, weights={"lexical": 3.0, "vector": 1.0})
    assert lexical_heavy[0].doc_id == "a"
    vector_heavy = reciprocal_rank_fusion(runs, weights={"lexical": 1.0, "vector": 3.0})
    assert vector_heavy[0].doc_id == "z"


def test_a_zero_weight_retriever_contributes_nothing_to_the_score():
    fused = reciprocal_rank_fusion(
        {"lexical": run("a"), "vector": run("z")}, weights={"vector": 0.0}
    )
    by_id = {h.doc_id: h for h in fused}
    assert by_id["z"].score == 0.0
    assert by_id["a"].score > 0.0


def test_smaller_k_sharpens_the_advantage_of_rank_one():
    sharp = reciprocal_rank_fusion({"r": run("a", "b")}, k=1)
    flat = reciprocal_rank_fusion({"r": run("a", "b")}, k=1000)
    assert sharp[0].score / sharp[1].score > flat[0].score / flat[1].score


def test_ordering_is_stable_for_tied_scores():
    runs = {"lexical": run("b", "a"), "vector": run("a", "b")}
    assert [h.doc_id for h in reciprocal_rank_fusion(runs)] == ["a", "b"]


def test_text_is_carried_through_from_whichever_retriever_had_it():
    runs = {
        "lexical": [Hit("a", 1.0)],
        "vector": [Hit("a", 0.9, text="the passage")],
    }
    assert reciprocal_rank_fusion(runs)[0].text == "the passage"


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion({"lexical": [], "vector": []}) == []


@pytest.mark.parametrize("k", [0, -5])
def test_non_positive_k_is_rejected(k):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"r": run("a")}, k=k)


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"r": run("a")}, weights={"r": -1.0})


def test_normalise_scores_maps_onto_zero_to_one():
    scaled = normalise_scores([Hit("a", 10.0), Hit("b", 5.0), Hit("c", 0.0)])
    assert [h.score for h in scaled] == pytest.approx([1.0, 0.5, 0.0])


def test_normalise_scores_handles_a_flat_run_without_dividing_by_zero():
    scaled = normalise_scores([Hit("a", 3.0), Hit("b", 3.0)])
    assert [h.score for h in scaled] == [1.0, 1.0]


def test_weighted_score_fusion_reaches_the_same_verdict_on_a_clear_case():
    runs = {"lexical": run("a", "b", "c"), "vector": run("a", "c", "b")}
    assert weighted_score_fusion(runs)[0].doc_id == "a"
