"""Reciprocal Rank Fusion (RRF) - Activity B.

RRF combines several *ranked lists* using only the ranks, never the raw scores:

    RRF(d) = sum over lists L containing d of  1 / (k + rank_L(d))        (rank starts at 1)

Why ranks only: cosine similarity (about -1..1) and BM25 (unbounded, corpus-dependent) are
on different scales, so adding or averaging them would be meaningless. RRF sidesteps that.
The constant k (60, from Cormack et al. 2009) flattens the curve: with k=60, rank 1 is worth
1/61 = 0.01639 and rank 10 is worth 1/70 = 0.01429, so a document found by BOTH lists
usually beats one found at the top of only one list. With two lists the maximum is 2/61.

RRF scores are only meaningful for ordering the fused list; they are not similarities.

Deterministic tie-breaking: higher RRF score, then better (smaller) best rank in any list,
then the order in which the lists were passed (dense before BM25 in this project), then
chunk ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    rank: int
    chunk_id: str
    score: float  # RRF score (sum of 1/(k + rank)); ordering only
    source_ranks: dict[str, int] = field(default_factory=dict)  # list name -> rank in that list


def _dedupe(ids: list[str]) -> list[str]:
    """Keep the first occurrence of each ID (a list must not credit a chunk twice)."""
    return list(dict.fromkeys(ids))


def rrf_fuse(rankings: dict[str, list[str]], k: int = RRF_K, top_n: int | None = None) -> list[FusedHit]:
    """Fuse ranked lists of chunk IDs (best first) with Reciprocal Rank Fusion."""
    if k < 0:
        raise ValueError("k must be >= 0")
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(_dedupe(ids), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, {})[name] = rank
    order = list(rankings)

    def key(chunk_id: str):
        src = ranks[chunk_id]
        best = min(src.values())
        first_list = min(order.index(n) for n, r in src.items() if r == best)
        return (-scores[chunk_id], best, first_list, chunk_id)

    fused = sorted(scores, key=key)
    if top_n is not None:
        fused = fused[:top_n]
    return [FusedHit(i, cid, scores[cid], dict(ranks[cid])) for i, cid in enumerate(fused, start=1)]
