"""Cross-encoder reranking (rag_v2.reranker, rag_v2.rerank_search)."""

from __future__ import annotations

import pytest

from evaluation.metrics import query_metrics
from rag_v2.chunking import load_corpus
from rag_v2.hybrid import HybridSearcher
from rag_v2.indexing import build_index
from rag_v2.rerank_search import RerankingSearcher
from rag_v2.reranker import Candidate, CrossEncoderReranker, format_pairs
from src.config import load_settings
from src.embeddings import LocalEmbedder


def cands(*ids):
    return [Candidate(cid, f"text of {cid}", i, 1.0 / (60 + i)) for i, cid in enumerate(ids, start=1)]


class Stub:
    """Deterministic test scorer: score looked up from a table by passage text; records calls."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, pairs):
        self.calls.append(list(pairs))
        return [self.table[p] for _, p in pairs]


# ------------------------------------------------------------------ input formatting
def test_format_pairs_query_first_passage_second_in_candidate_order():
    pairs = format_pairs("  What is KeyVault?  ", cands("A", "B"))
    assert pairs == [("What is KeyVault?", "text of A"), ("What is KeyVault?", "text of B")]
    with pytest.raises(ValueError):
        format_pairs("   ", cands("A"))


def test_scorer_receives_exactly_the_formatted_pairs():
    stub = Stub({"text of A": 0.1, "text of B": 0.2})
    CrossEncoderReranker(stub).rerank("q", cands("A", "B"))
    assert stub.calls == [[("q", "text of A"), ("q", "text of B")]]  # one batch, candidate order


# ------------------------------------------------------------------ ordering and preservation
def test_rerank_orders_by_score_and_keeps_prior_information():
    stub = Stub({"text of A": -3.0, "text of B": 5.0, "text of C": 1.0})
    out = CrossEncoderReranker(stub).rerank("q", cands("A", "B", "C"))
    assert [h.chunk_id for h in out] == ["B", "C", "A"]
    assert [h.rank for h in out] == [1, 2, 3]
    assert [(h.chunk_id, h.prior_rank) for h in out] == [("B", 2), ("C", 3), ("A", 1)]
    assert out[0].score == 5.0 and out[0].prior_score == pytest.approx(1 / 62)


def test_rerank_is_a_permutation_no_new_or_lost_chunks():
    pool = cands(*"ABCDEFGHIJ")
    stub = Stub({c.text: float((i * 7) % 10) for i, c in enumerate(pool)})
    out = CrossEncoderReranker(stub).rerank("q", pool)
    assert sorted(h.chunk_id for h in out) == sorted(c.chunk_id for c in pool)
    assert len(out) == len(pool) == len({h.chunk_id for h in out})


def test_ties_keep_incoming_order_and_ranking_is_deterministic():
    stub = Stub({"text of A": 1.0, "text of B": 2.0, "text of C": 1.0})
    rr = CrossEncoderReranker(stub)
    first = rr.rerank("q", cands("A", "B", "C"))
    assert [h.chunk_id for h in first] == ["B", "A", "C"]  # A before C: same score, A came first
    assert rr.rerank("q", cands("A", "B", "C")) == first


def test_empty_candidates_return_empty_without_calling_the_model():
    stub = Stub({})
    assert CrossEncoderReranker(stub).rerank("q", []) == []
    assert stub.calls == []


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        CrossEncoderReranker(Stub({"text of A": 1.0})).rerank("q", cands("A") + cands("A"))
    with pytest.raises(RuntimeError):
        CrossEncoderReranker(lambda pairs: [1.0]).rerank("q", cands("A", "B"))


# ------------------------------------------------------------------ metrics before / after
def test_metrics_before_and_after_reranking_by_hand():
    # Pool (hybrid order) A B C D; only D contains the required fact e1.
    facts = {"A": set(), "B": set(), "C": set(), "D": {"e1"}, "Z": {"e1"}}  # Z: relevant but NOT in pool
    pool = cands("A", "B", "C", "D")
    stub = Stub({"text of A": 0.0, "text of B": -1.0, "text of C": -2.0, "text of D": 9.0})
    after = [h.chunk_id for h in CrossEncoderReranker(stub).rerank("q", pool)]
    before = [c.chunk_id for c in pool]
    b3, a3 = (query_metrics(x, facts, [["e1"]], k=3) for x in (before, after))
    assert (b3.hit, b3.reciprocal_rank, b3.coverage, b3.precision) == (0, 0.0, 0.0, 0.0)
    assert (a3.hit, a3.reciprocal_rank, a3.coverage) == (1, 1.0, 1.0) and a3.precision == pytest.approx(1 / 3)
    assert b3.relevant_in_index == a3.relevant_in_index == 2  # D and Z
    assert a3.recall == pytest.approx(1 / 2)  # Z cannot be recovered: it is not in the pool
    # Pool-level coverage is identical before and after (reranking only reorders).
    assert (
        query_metrics(before, facts, [["e1"]], k=10).coverage
        == query_metrics(after, facts, [["e1"]], k=10).coverage
    )


# ------------------------------------------------------------------ with the real hybrid searcher
@pytest.fixture(scope="module")
def hybrid(tmp_path_factory):
    docs = load_corpus(load_settings().policy_dir)
    index = build_index(
        "v1_paragraph_800_150", docs, LocalEmbedder(), chroma_dir=tmp_path_factory.mktemp("ch")
    )
    return HybridSearcher(index)


def test_reranking_searcher_reorders_exactly_the_hybrid_pool(hybrid):
    stub_scores = {c.embedding_text: -float(len(c.text)) for c in hybrid.index.chunks}  # shorter = better
    searcher = RerankingSearcher(
        hybrid, CrossEncoderReranker(lambda pairs: [stub_scores[p] for _, p in pairs])
    )
    r = searcher.search("What is the hotel limit per night?")
    assert len(r.pool_ids) == 10 and sorted(r.reranked_ids) == sorted(r.pool_ids)
    assert [c.text for c in searcher.candidates(r.hybrid)] == [
        hybrid.index.chunks_by_id[cid].embedding_text for cid in r.pool_ids
    ]
    assert {"dense", "bm25", "fusion", "hybrid_total", "rerank", "total"} <= set(r.timings_ms)


def _real_reranker():
    try:
        return CrossEncoderReranker()  # pinned revision, local cache only
    except Exception as exc:  # model not downloaded
        pytest.skip(
            f"cross-encoder not in local cache ({exc.__class__.__name__}); run scripts/download_cross_encoder.py"
        )


def test_real_cross_encoder_is_deterministic_and_preserves_pool(hybrid):
    rr = _real_reranker()
    searcher = RerankingSearcher(hybrid, rr)
    q = "Can I use public Wi-Fi when working from home?"
    a, b = searcher.search(q), searcher.search(q)
    assert a.reranked == b.reranked
    assert sorted(a.reranked_ids) == sorted(a.pool_ids)
    assert rr.model.activation_fn.__class__.__name__ == "Identity"  # raw logits, ordering only
