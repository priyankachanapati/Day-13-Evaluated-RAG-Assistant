"""Dense, BM25 and hybrid (RRF) retrieval over one chunking strategy (Activity B).

All three methods use the SAME candidate depth: dense top-N (v1 ``Retriever``: MiniLM query
embedding + Chroma cosine search) and BM25 top-N over the same chunks; the hybrid list is the
RRF fusion of exactly those two top-N lists. Each method is timed separately.

The v1 refusal rule is evaluated on the dense result only (best cosine similarity >= the v1
threshold 0.25) and is reported independently; BM25 and RRF scores are never compared with it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .bm25 import BM25Hit, BM25Index
from .fusion import RRF_K, FusedHit, rrf_fuse
from .indexing import StrategyIndex

CANDIDATES = 10  # same depth for dense, BM25 and the lists fused by RRF (Phase 4 reranks top 10)


@dataclass(frozen=True)
class DenseHit:
    rank: int
    chunk_id: str
    score: float  # cosine similarity = 1 - cosine distance (v1 convention)


@dataclass
class SearchResult:
    query: str
    dense: list[DenseHit]
    bm25: list[BM25Hit]
    hybrid: list[FusedHit]
    dense_gate_passes: bool  # v1 rule on dense similarity only
    timings_ms: dict[str, float] = field(default_factory=dict)

    def ranked_ids(self, method: str) -> list[str]:
        return [h.chunk_id for h in getattr(self, method)]


class HybridSearcher:
    def __init__(self, index: StrategyIndex, candidates: int = CANDIDATES, rrf_k: int = RRF_K) -> None:
        self.index = index
        self.candidates = candidates
        self.rrf_k = rrf_k
        t0 = time.perf_counter()
        self.bm25 = BM25Index(index.chunks)
        self.bm25_build_ms = (time.perf_counter() - t0) * 1000

    def search(self, query: str) -> SearchResult:
        t0 = time.perf_counter()
        dense_result = self.index.retriever.retrieve(query, top_k=self.candidates)
        t1 = time.perf_counter()
        bm25_hits = self.bm25.search(query, self.candidates)
        t2 = time.perf_counter()
        dense_hits = [DenseHit(c.rank, c.chunk_id, c.similarity) for c in dense_result.chunks]
        fused = rrf_fuse(
            {"dense": [h.chunk_id for h in dense_hits], "bm25": [h.chunk_id for h in bm25_hits]},
            k=self.rrf_k,
            top_n=self.candidates,
        )
        t3 = time.perf_counter()
        return SearchResult(
            query=query,
            dense=dense_hits,
            bm25=bm25_hits,
            hybrid=fused,
            dense_gate_passes=dense_result.has_relevant_context,
            timings_ms={"dense": (t1 - t0) * 1000, "bm25": (t2 - t1) * 1000, "fusion": (t3 - t2) * 1000},
        )
