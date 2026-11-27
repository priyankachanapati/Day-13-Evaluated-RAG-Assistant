"""Hybrid retrieval followed by cross-encoder reranking (Activity C).

Wraps the Phase 3 ``HybridSearcher`` unchanged: the candidate pool is exactly its hybrid
(RRF, k=60) top-10. The cross-encoder only re-orders that pool. Timings are kept per stage:
dense, BM25 and fusion (from the hybrid searcher), the hybrid end-to-end time, the reranking
time (pair formatting + cross-encoder scoring + sort), and the total.

The v1 refusal gate stays on the dense similarity (``hybrid.dense_gate_passes``); the
cross-encoder score is never compared with it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .hybrid import HybridSearcher, SearchResult
from .reranker import Candidate, CrossEncoderReranker, RerankedHit


@dataclass
class RerankResult:
    hybrid: SearchResult
    reranked: list[RerankedHit]
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def pool_ids(self) -> list[str]:
        return self.hybrid.ranked_ids("hybrid")

    @property
    def reranked_ids(self) -> list[str]:
        return [h.chunk_id for h in self.reranked]


class RerankingSearcher:
    def __init__(self, hybrid: HybridSearcher, reranker: CrossEncoderReranker) -> None:
        self.hybrid = hybrid
        self.reranker = reranker

    def candidates(self, result: SearchResult) -> list[Candidate]:
        chunks = self.hybrid.index.chunks_by_id
        return [
            Candidate(h.chunk_id, chunks[h.chunk_id].embedding_text, h.rank, h.score) for h in result.hybrid
        ]

    def search(self, query: str) -> RerankResult:
        t0 = time.perf_counter()
        hybrid = self.hybrid.search(query)
        t1 = time.perf_counter()
        reranked = self.reranker.rerank(query, self.candidates(hybrid))
        t2 = time.perf_counter()
        timings = dict(hybrid.timings_ms)
        timings.update(
            {"hybrid_total": (t1 - t0) * 1000, "rerank": (t2 - t1) * 1000, "total": (t2 - t0) * 1000}
        )
        return RerankResult(hybrid, reranked, timings)
