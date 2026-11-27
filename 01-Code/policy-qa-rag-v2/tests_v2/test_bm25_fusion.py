"""BM25 ranking and Reciprocal Rank Fusion (hand-calculated examples)."""

from __future__ import annotations

import math

import pytest

from rag_v2.bm25 import BM25_PARAMS, BM25Index
from rag_v2.fusion import rrf_fuse
from src.ingestion import Chunk


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(cid, text, {"chunk_id": cid, "policy_id": "XX-001", "title": "Toy"})


TOY = [
    chunk("XX-001-CH01", "KeyVault is the approved password manager."),
    chunk("XX-001-CH02", "Passwords must be at least 14 characters."),
    chunk("XX-001-CH03", "Annual leave is 20 days."),
]


# ------------------------------------------------------------------ BM25
def test_bm25_returns_only_chunks_sharing_a_term():
    hits = BM25Index(TOY).search("What is KeyVault?", top_k=10)
    assert [h.chunk_id for h in hits] == ["XX-001-CH01"]  # CH02/CH03 score 0 -> not retrieved
    assert hits[0].matched_terms == ("keyvault",)


def test_bm25_score_equals_okapi_formula():
    index = BM25Index(TOY)
    hit = index.search("keyvault", top_k=1)[0]
    n_docs, df, tf = 3, 1, 1
    dl = len(index.doc_tokens[0])
    avgdl = sum(len(t) for t in index.doc_tokens) / n_docs
    k1, b = BM25_PARAMS["k1"], BM25_PARAMS["b"]
    idf = math.log((n_docs - df + 0.5) / (df + 0.5))  # = ln(2.5 / 1.5) = 0.5108
    expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert hit.score == pytest.approx(expected, rel=1e-12)


def test_bm25_plural_folding_lets_query_match_both_forms():
    hits = BM25Index(TOY).search("password", top_k=10)
    assert {h.chunk_id for h in hits} == {"XX-001-CH01", "XX-001-CH02"}  # "passwords" folds to "password"


def test_bm25_ties_broken_by_index_position_and_deterministic():
    # "laptop" in 2 of 5 chunks (a rare term, positive IDF); CH02 and CH04 are identical -> exact tie.
    docs = [
        chunk("XX-001-CH01", "annual leave"),
        chunk("XX-001-CH02", "laptop return"),
        chunk("XX-001-CH03", "sick leave"),
        chunk("XX-001-CH04", "laptop return"),
        chunk("XX-001-CH05", "hotel limit"),
    ]
    index = BM25Index(docs)
    first = index.search("laptop", top_k=10)
    assert [h.chunk_id for h in first] == ["XX-001-CH02", "XX-001-CH04"]
    assert first[0].score == first[1].score > 0
    assert index.search("laptop", top_k=10) == first


def test_bm25_term_in_more_than_half_of_chunks_gets_no_positive_weight():
    # rank_bm25 quirk worth knowing: IDF = ln((N - n + 0.5) / (n + 0.5)) is negative when n > N/2
    # and is replaced by epsilon * mean IDF. In this 3-chunk toy the mean IDF is itself negative,
    # so "laptop" (2 of 3 chunks) scores <= 0 and is treated as "no keyword match".
    docs = [
        chunk("XX-001-CH01", "laptop return"),
        chunk("XX-001-CH02", "laptop return"),
        chunk("XX-001-CH03", "x"),
    ]
    index = BM25Index(docs)
    assert index.bm25.idf["laptop"] <= 0
    assert index.search("laptop", top_k=10) == []


def test_bm25_empty_query_and_top_k():
    index = BM25Index(TOY)
    assert index.search("what is the", top_k=5) == []  # only stopwords
    assert len(index.search("password leave keyvault", top_k=2)) == 2


# ------------------------------------------------------------------ RRF
def test_rrf_hand_calculated_example():
    # dense: A B C   bm25: C A D   k=60
    # A = 1/61 + 1/62 = 0.032522   C = 1/63 + 1/61 = 0.032266   B = 1/62 = 0.016129   D = 1/63 = 0.015873
    fused = rrf_fuse({"dense": ["A", "B", "C"], "bm25": ["C", "A", "D"]}, k=60)
    assert [h.chunk_id for h in fused] == ["A", "C", "B", "D"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1].score == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[3].score == pytest.approx(1 / 63)
    assert fused[0].source_ranks == {"dense": 1, "bm25": 2}
    assert fused[3].source_ranks == {"bm25": 3}
    assert [h.rank for h in fused] == [1, 2, 3, 4]


def test_rrf_rewards_agreement_over_a_single_top_rank():
    # X is 1st in dense only; Y is 3rd in both. 1/61 = 0.0164 < 2/63 = 0.0317
    fused = rrf_fuse({"dense": ["X", "Z", "Y"], "bm25": ["W", "V", "Y"]}, k=60)
    assert fused[0].chunk_id == "Y"


def test_rrf_ties_are_deterministic():
    # A and B both get exactly 1/61; same best rank; dense list was passed first -> A first.
    assert [h.chunk_id for h in rrf_fuse({"dense": ["A"], "bm25": ["B"]})] == ["A", "B"]
    assert [h.chunk_id for h in rrf_fuse({"bm25": ["B"], "dense": ["A"]})] == ["B", "A"]


def test_rrf_ignores_duplicates_within_one_list():
    fused = rrf_fuse({"dense": ["A", "A", "B"]}, k=60)
    assert [h.chunk_id for h in fused] == ["A", "B"]
    assert fused[0].score == pytest.approx(1 / 61) and fused[1].score == pytest.approx(1 / 62)


def test_rrf_k_and_top_n():
    assert rrf_fuse({"d": ["A"]}, k=0)[0].score == 1.0
    assert len(rrf_fuse({"d": ["A", "B", "C"], "b": ["D"]}, top_n=2)) == 2
    assert rrf_fuse({"d": [], "b": []}) == []
    with pytest.raises(ValueError):
        rrf_fuse({"d": ["A"]}, k=-1)
