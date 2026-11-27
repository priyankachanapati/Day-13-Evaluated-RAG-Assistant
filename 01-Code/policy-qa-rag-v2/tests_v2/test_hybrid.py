"""Hybrid searcher on the real corpus (local embeddings; temporary Chroma directory)."""

from __future__ import annotations

import pytest

from rag_v2.chunking import STRATEGIES, load_corpus
from rag_v2.fusion import rrf_fuse
from rag_v2.hybrid import HybridSearcher
from rag_v2.indexing import build_index
from src.config import load_settings
from src.embeddings import LocalEmbedder


@pytest.fixture(scope="module")
def searchers(tmp_path_factory):
    docs = load_corpus(load_settings().policy_dir)
    embedder = LocalEmbedder()
    chroma = tmp_path_factory.mktemp("chroma-v2-test")
    return {n: HybridSearcher(build_index(n, docs, embedder, chroma_dir=chroma)) for n in STRATEGIES}


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_hybrid_is_rrf_of_the_two_candidate_lists(searchers, name):
    r = searchers[name].search("What is the hotel limit per night in USD?")
    assert len(r.dense) == 10 and 0 < len(r.bm25) <= 10
    expected = rrf_fuse({"dense": r.ranked_ids("dense"), "bm25": r.ranked_ids("bm25")}, k=60, top_n=10)
    assert r.hybrid == expected
    assert set(r.timings_ms) == {"dense", "bm25", "fusion"}


def test_keyword_query_found_by_bm25(searchers):
    r = searchers["v1_paragraph_800_150"].search("KeyVault")
    assert r.bm25[0].chunk_id.startswith("IT-001") and "keyvault" in r.bm25[0].matched_terms


def test_dense_gate_is_the_v1_rule_on_dense_scores(searchers):
    s = searchers["v1_paragraph_800_150"]
    r = s.search("How many days of annual leave do I get?")
    v1 = s.index.retriever.retrieve("How many days of annual leave do I get?", top_k=10)
    assert r.dense_gate_passes == v1.has_relevant_context == (max(h.score for h in r.dense) >= 0.25)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_search_is_deterministic(searchers, name):
    q = "Can I use public Wi-Fi when working from home?"
    a, b = searchers[name].search(q), searchers[name].search(q)
    assert (a.dense, a.bm25, a.hybrid) == (b.dense, b.bm25, b.hybrid)
