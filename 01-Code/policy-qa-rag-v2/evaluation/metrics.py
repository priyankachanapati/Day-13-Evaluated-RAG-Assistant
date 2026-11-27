"""Retrieval metrics with one explicit relevance definition (used from Activity B onward).

Labels
    A query has required-evidence *groups* G = [[e1, e1'], [e2], ...]: each group is one fact
    the answer needs; alternatives inside a group are interchangeable sentences.
    E = all evidence IDs in G.

Relevance (the same rule for every metric)
    A retrieved chunk is **relevant** iff it FULLY contains at least one evidence unit in E
    (offset containment, see ``evaluation.evidence``). Relevance is therefore defined by
    canonical evidence IDs, not chunk IDs, and works for every chunking strategy.

Conventional, chunk-level metrics (each retrieved chunk is one item)
    Hit@K        1 if any of the top-K chunks is relevant, else 0.
    RR@K         1 / rank of the first relevant chunk within the top K, else 0  (MRR = mean RR).
    Precision@K  relevant chunks in top K / K   (K in the denominator even if fewer were returned).
    Recall@K     relevant chunks in top K / relevant chunks in the whole index.
                 Depends on the chunking: with overlap the same fact sits in several chunks,
                 which enlarges the denominator, and retrieving the duplicate counts again.
                 Undefined when the index has no relevant chunk (e.g. the fact was split);
                 reported as 0 with ``relevant_in_index == 0`` so it is visible.

Evidence-fact metric (chunk-independent)
    Coverage@K   required groups satisfied by the top-K chunks / number of groups.
                 Each fact counts once, however many retrieved chunks contain it, so overlap
                 duplicates cannot inflate it. This is the fair cross-strategy measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .evidence import satisfied_groups


@dataclass(frozen=True)
class QueryMetrics:
    k: int
    hit: int
    reciprocal_rank: float
    precision: float
    recall: float
    coverage: float
    relevant_retrieved: int
    relevant_in_index: int
    groups_satisfied: int
    groups_total: int
    first_relevant_rank: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def relevant_chunks(chunk_facts: dict[str, set[str]], required: set[str]) -> set[str]:
    """IDs of chunks (in this index) that fully contain at least one required evidence unit."""
    return {cid for cid, facts in chunk_facts.items() if facts & required}


def query_metrics(
    ranked_ids: list[str], chunk_facts: dict[str, set[str]], groups: list[list[str]], k: int
) -> QueryMetrics:
    """All metrics for one ranked list. ``chunk_facts`` maps every chunk ID in the index to the
    evidence IDs it fully contains."""
    if k <= 0:
        raise ValueError("k must be positive")
    required = {e for g in groups for e in g}
    relevant = relevant_chunks(chunk_facts, required)
    top = list(dict.fromkeys(ranked_ids))[:k]  # a chunk can only be retrieved once
    flags = [cid in relevant for cid in top]
    first = next((i for i, f in enumerate(flags, start=1) if f), None)
    found = set().union(*(chunk_facts.get(cid, set()) for cid in top)) if top else set()
    satisfied = satisfied_groups(groups, found)
    n_rel = sum(flags)
    return QueryMetrics(
        k=k,
        hit=int(n_rel > 0),
        reciprocal_rank=1.0 / first if first else 0.0,
        precision=n_rel / k,
        recall=n_rel / len(relevant) if relevant else 0.0,
        coverage=sum(satisfied) / len(groups) if groups else 0.0,
        relevant_retrieved=n_rel,
        relevant_in_index=len(relevant),
        groups_satisfied=sum(satisfied),
        groups_total=len(groups),
        first_relevant_rank=first,
    )


def mean_metrics(rows: list[QueryMetrics]) -> dict[str, float]:
    """Macro-average over queries (each query weighs the same)."""
    n = len(rows)
    if n == 0:
        return {}
    return {
        "hit_rate": sum(r.hit for r in rows) / n,
        "mrr": sum(r.reciprocal_rank for r in rows) / n,
        "precision": sum(r.precision for r in rows) / n,
        "recall": sum(r.recall for r in rows) / n,
        "coverage": sum(r.coverage for r in rows) / n,
        "groups_satisfied": sum(r.groups_satisfied for r in rows),
        "groups_total": sum(r.groups_total for r in rows),
    }
