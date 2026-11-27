"""Activity D metrics - hand-worked examples (evaluation.retrieval_metrics).

Toy index (chunk -> canonical facts it fully contains):
    A: {}        B: {e1}      C: {e1, x}  (C repeats e1: an overlap duplicate)
    D: {e2}      E: {y}       F: {e3}
Relevant chunks per required fact: e1 -> B, C   e2 -> D   e3 -> F
"""

from __future__ import annotations

import pytest

from evaluation.metrics import query_metrics
from evaluation.retrieval_metrics import aggregate, compare_k, evaluate_ranking

FACTS = {"A": set(), "B": {"e1"}, "C": {"e1", "x"}, "D": {"e2"}, "E": {"y"}, "F": {"e3"}}


def ev(qid, groups, ranked, answerable=True, ks=(3, 5)):
    return evaluate_ranking(qid, answerable, groups, ranked, FACTS, ks)


# ------------------------------------------------------------------ single queries
def test_one_relevant_chunk_at_rank_1():
    m = ev("Q1", [["e3"]], ["F", "A", "B"]).metrics[3]
    # relevant in top-3: F (1 of 3); relevant in index: F (1)
    assert (m.hit, m.reciprocal_rank, m.precision, m.recall, m.coverage) == (
        1,
        1.0,
        pytest.approx(1 / 3),
        1.0,
        1.0,
    )


def test_relevant_chunk_at_rank_3_and_effect_of_k():
    r = ev("Q2", [["e2"]], ["A", "E", "D", "B", "C"])
    m3, m5 = r.metrics[3], r.metrics[5]
    assert m3.reciprocal_rank == pytest.approx(1 / 3) and m3.precision == pytest.approx(1 / 3)
    assert m5.reciprocal_rank == pytest.approx(1 / 3) and m5.precision == pytest.approx(
        1 / 5
    )  # larger K dilutes P
    assert m3.recall == m5.recall == 1.0 and m3.coverage == 1.0


def test_relevant_chunk_only_beyond_k():
    m3, m5 = (ev("Q", [["e2"]], ["A", "E", "B", "C", "D"]).metrics[k] for k in (3, 5))
    assert (m3.hit, m3.reciprocal_rank, m3.coverage) == (0, 0.0, 0.0)  # D at rank 5: missed at K=3
    assert (m5.hit, m5.reciprocal_rank, m5.coverage) == (1, pytest.approx(1 / 5), 1.0)


def test_no_relevant_chunk_retrieved():
    m = ev("Q3", [["e3"]], ["A", "E"]).metrics[3]
    assert (m.hit, m.reciprocal_rank, m.precision, m.recall, m.coverage) == (0, 0.0, 0.0, 0.0, 0.0)
    assert m.relevant_in_index == 1 and m.first_relevant_rank is None


def test_multiple_relevant_chunks_and_duplicate_fact():
    # B and C both contain e1 (overlap duplicate): two relevant chunks, but ONE fact.
    m = ev("Q4", [["e1"]], ["B", "C", "A"]).metrics[3]
    assert m.precision == pytest.approx(2 / 3)  # chunk level: both count
    assert m.recall == 1.0 and m.relevant_in_index == 2  # 2 relevant chunks in the index, both retrieved
    assert m.coverage == 1.0 and m.groups_satisfied == 1  # fact level: e1 counted once
    only_b = ev("Q4b", [["e1"]], ["B", "A", "E"]).metrics[3]
    assert only_b.recall == 0.5 and only_b.coverage == 1.0  # duplicate moves chunk recall, not coverage


def test_multiple_required_facts_partial_then_full():
    r = ev("Q5", [["e1"], ["e2"]], ["B", "A", "E", "D"])
    m3, m5 = r.metrics[3], r.metrics[5]
    # relevant in index: B, C (e1) and D (e2) = 3
    assert (m3.precision, m3.recall, m3.coverage) == (pytest.approx(1 / 3), pytest.approx(1 / 3), 0.5)
    assert (m5.precision, m5.recall, m5.coverage) == (pytest.approx(2 / 5), pytest.approx(2 / 3), 1.0)


def test_alternative_evidence_within_a_group():
    assert ev("Q8", [["zz", "e2"]], ["D"]).metrics[3].coverage == 1.0  # e2 satisfies the group


def test_empty_retrieval_result():
    m = ev("Q6", [["e1"]], []).metrics[3]
    assert (m.hit, m.reciprocal_rank, m.precision, m.recall, m.coverage) == (0, 0.0, 0.0, 0.0, 0.0)


def test_unanswerable_query_has_no_retrieval_metrics():
    r = ev("Q7", [], ["A", "B"], answerable=False)
    assert r.metrics is None and r.returned == 2
    with pytest.raises(ValueError):
        ev("Qbad", [["e1"]], ["A"], answerable=False)  # unanswerable with evidence is a label error


def test_invalid_inputs():
    with pytest.raises(ValueError):
        ev("Qx", [], ["A"])  # answerable without evidence
    with pytest.raises(ValueError):
        ev("Qy", [["e1"]], ["NOT-IN-INDEX"])


def test_fact_missing_from_index_gives_zero_recall_and_is_counted():
    r = ev("Qs", [["gone"]], ["A"])
    assert r.metrics[3].relevant_in_index == 0 and r.metrics[3].recall == 0.0
    assert aggregate([r], 3)["recall_zero_denominator"] == 1


def test_same_numbers_as_the_phase3_function():
    ranked, groups = ["B", "A", "E", "D"], [["e1"], ["e2"]]
    assert ev("Q", groups, ranked).metrics[3] == query_metrics(ranked, FACTS, groups, 3)


# ------------------------------------------------------------------ aggregation
def suite():
    return [
        ev("Q1", [["e3"]], ["F", "A", "B"]),  # hit 1, RR 1,   P 1/3, R 1,   cov 1
        ev("Q2", [["e2"]], ["A", "E", "D", "B", "C"]),  # hit 1, RR 1/3, P 1/3, R 1,   cov 1
        ev("Q3", [["e3"]], ["A", "E"]),  # hit 0, RR 0,   P 0,   R 0,   cov 0
        ev("Q4", [["e1"]], ["B", "C", "A"]),  # hit 1, RR 1,   P 2/3, R 1,   cov 1
        ev(
            "Q5", [["e1"], ["e2"]], ["B", "A", "E", "D"]
        ),  # hit 1, RR 1,   P 1/3, R 1/3, cov 1/2 (1 of 2 groups)
        ev("Q6", [["e1"]], []),  # hit 0, RR 0,   P 0,   R 0,   cov 0
        ev("Q7", [], ["A", "B"], answerable=False),  # excluded
    ]


def test_macro_aggregation_at_k3_by_hand():
    a = aggregate(suite(), 3)
    assert (a["n_queries"], a["n_answerable"], a["n_unanswerable_excluded"], a["empty_results"]) == (
        7,
        6,
        1,
        1,
    )
    assert a["hit_rate"] == pytest.approx(4 / 6)
    assert a["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 + 1 + 0) / 6)  # = 10/18
    assert a["precision"] == pytest.approx((1 / 3 + 1 / 3 + 0 + 2 / 3 + 1 / 3 + 0) / 6)  # = 5/18
    assert a["recall"] == pytest.approx((1 + 1 + 0 + 1 + 1 / 3 + 0) / 6)  # = 10/18
    assert a["coverage_macro"] == pytest.approx((1 + 1 + 0 + 1 + 0.5 + 0) / 6)  # = 7/12 = 0.5833
    assert a["coverage_micro"] == pytest.approx(4 / 7)  # pooled groups: 4 of 7 = 0.5714
    assert a["coverage_macro"] != a["coverage_micro"]  # macro is NOT the pooled value


def test_k3_vs_k5_comparison():
    c = compare_k(suite(), (3, 5))
    # at K=5, Q5 also finds D (cov 1/2 -> 1); Q2's precision falls from 1/3 to 1/5
    assert c[5]["coverage_macro"] == pytest.approx((1 + 1 + 0 + 1 + 1 + 0) / 6)
    assert c[5]["coverage_macro"] > c[3]["coverage_macro"]
    assert c[5]["hit_rate"] == c[3]["hit_rate"]


def test_aggregate_with_only_unanswerable_queries():
    a = aggregate([ev("U", [], [], answerable=False)], 3)
    assert a["n_answerable"] == 0 and a["hit_rate"] is None and a["coverage_micro"] is None
