"""Retrieval metrics - hand-calculated examples (evaluation.metrics)."""

from __future__ import annotations

import pytest

from evaluation.metrics import mean_metrics, query_metrics, relevant_chunks

# Index with 6 chunks. Required evidence: group 1 = e1, group 2 = e2 OR e2b.
CHUNK_FACTS = {
    "A": set(),
    "B": {"e1"},
    "C": {"e1", "x"},  # duplicate of e1 (overlap)
    "D": {"e2"},
    "E": {"y"},
    "F": {"e2b"},
}
GROUPS = [["e1"], ["e2", "e2b"]]
RANKED = ["A", "B", "C", "D", "E"]  # F (relevant) is not retrieved


def test_relevance_definition():
    assert relevant_chunks(CHUNK_FACTS, {"e1", "e2", "e2b"}) == {"B", "C", "D", "F"}


def test_metrics_at_3_by_hand():
    # top-3 = A B C; relevant: B, C -> 2 of 3; relevant in index = 4
    m = query_metrics(RANKED, CHUNK_FACTS, GROUPS, k=3)
    assert m.hit == 1
    assert m.first_relevant_rank == 2 and m.reciprocal_rank == pytest.approx(1 / 2)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 4)  # chunk-level: B and C both count
    assert m.coverage == pytest.approx(1 / 2)  # fact-level: e1 counted ONCE, group 2 missing
    assert (m.relevant_retrieved, m.relevant_in_index, m.groups_satisfied) == (2, 4, 1)


def test_metrics_at_5_by_hand():
    # top-5 = A B C D E; relevant: B C D -> 3; groups: e1 yes, e2 yes -> 2/2
    m = query_metrics(RANKED, CHUNK_FACTS, GROUPS, k=5)
    assert m.precision == pytest.approx(3 / 5)
    assert m.recall == pytest.approx(3 / 4)
    assert m.coverage == 1.0


def test_duplicate_fact_inflates_chunk_recall_but_not_coverage():
    # Retrieving B and its overlap-duplicate C: chunk recall doubles, fact coverage does not move.
    only_b = query_metrics(["B"], CHUNK_FACTS, GROUPS, k=2)
    b_and_c = query_metrics(["B", "C"], CHUNK_FACTS, GROUPS, k=2)
    assert b_and_c.recall == 2 * only_b.recall
    assert b_and_c.coverage == only_b.coverage == 0.5


def test_no_relevant_chunk_retrieved_and_repeated_ids():
    m = query_metrics(["A", "E"], CHUNK_FACTS, GROUPS, k=3)
    assert (m.hit, m.reciprocal_rank, m.precision, m.coverage, m.first_relevant_rank) == (
        0,
        0.0,
        0.0,
        0.0,
        None,
    )
    assert query_metrics(["B", "B", "B"], CHUNK_FACTS, GROUPS, k=3).relevant_retrieved == 1  # counted once


def test_fact_absent_from_index():
    # e.g. a fact split by fixed-size chunking: no relevant chunk exists, recall reported as 0.
    m = query_metrics(["A"], {"A": set()}, [["gone"]], k=3)
    assert m.relevant_in_index == 0 and m.recall == 0.0 and m.hit == 0


def test_mean_metrics_macro_average():
    rows = [query_metrics(RANKED, CHUNK_FACTS, GROUPS, k=3), query_metrics(["A"], CHUNK_FACTS, GROUPS, k=3)]
    avg = mean_metrics(rows)
    assert avg["hit_rate"] == 0.5 and avg["mrr"] == pytest.approx(0.25)
    assert avg["groups_satisfied"] == 1 and avg["groups_total"] == 4
    with pytest.raises(ValueError):
        query_metrics(RANKED, CHUNK_FACTS, GROUPS, k=0)
