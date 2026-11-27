"""Activity D - retrieval metrics for ranked results, with explicit aggregation rules.

Per-query metrics come from ``evaluation.metrics.query_metrics`` (Phase 3, frozen), so every
retrieval system is scored with exactly the same relevance definition. This module adds what a
dataset-level evaluation needs: unanswerable handling, macro aggregation with counts, a
separately reported micro fact coverage, and comparison across K.

RELEVANCE (one rule for every metric and system)
    A retrieved chunk is relevant iff it fully contains >= 1 canonical evidence ID that the
    query requires (any group, any alternative). Labels are evidence IDs, never chunk IDs.

CHUNK-LEVEL metrics (items = retrieved chunks)
    Precision@K = relevant chunks in the top K / K
                  (K in the denominator even if fewer than K chunks were returned)
    Recall@K    = relevant chunks in the top K / relevant chunks in the whole index
                  Depends on the chunking: overlap duplicates a fact into several chunks, each of
                  which is a separate relevant item. If no chunk in the index contains a required
                  fact intact (e.g. it was split), the denominator is 0: Recall@K is reported as 0
                  and the query is counted in ``recall_zero_denominator``.
    Hit@K       = 1 if any relevant chunk is in the top K, else 0
    RR@K        = 1 / rank of the first relevant chunk in the top K, else 0;  MRR@K = mean RR@K

FACT-LEVEL metric (items = required facts; never mixed into the chunk-level metrics)
    Coverage@K  = required fact groups satisfied by the top K / number of groups.
                  A group is satisfied if ANY of its alternative evidence IDs is fully contained in
                  a top-K chunk. Each fact counts once, however many retrieved chunks contain it,
                  so overlapping chunks cannot inflate it. Multiple required facts give partial
                  credit (e.g. 1 of 2 groups = 0.5).

UNANSWERABLE queries have no relevant items, so every retrieval metric is undefined for them.
    They are EXCLUDED from all retrieval-metric averages (n = answerable queries only) and
    reported as a count. Whether a system correctly refuses them is an answer-level property
    evaluated in Phase 7 (refusal accuracy), not a retrieval metric.

AGGREGATION
    Macro average over answerable queries: each query weighs the same (primary numbers).
    Coverage is additionally reported as a micro value (satisfied groups / all groups, pooled),
    clearly labelled, because queries have different numbers of required facts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .metrics import QueryMetrics, query_metrics


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    answerable: bool
    returned: int  # chunks the system returned
    metrics: dict[int, QueryMetrics] | None  # K -> metrics; None for unanswerable queries


def evaluate_ranking(
    query_id: str,
    answerable: bool,
    required_evidence: list[list[str]],
    ranked_ids: list[str],
    chunk_facts: dict[str, set[str]],
    ks: Iterable[int] = (3, 5),
) -> QueryEvaluation:
    """Metrics of one ranked list at each K. ``chunk_facts`` maps EVERY chunk ID of the index to
    the evidence IDs it fully contains (needed for the Recall@K denominator)."""
    unknown = [cid for cid in ranked_ids if cid not in chunk_facts]
    if unknown:
        raise ValueError(f"{query_id}: ranked IDs not in the index: {unknown}")
    if not answerable:
        if required_evidence:
            raise ValueError(f"{query_id}: an unanswerable query must not have required evidence")
        return QueryEvaluation(query_id, False, len(ranked_ids), None)
    if not required_evidence or not all(required_evidence):
        raise ValueError(f"{query_id}: an answerable query needs non-empty evidence groups")
    return QueryEvaluation(
        query_id,
        True,
        len(ranked_ids),
        {k: query_metrics(ranked_ids, chunk_facts, required_evidence, k) for k in ks},
    )


def aggregate(results: list[QueryEvaluation], k: int) -> dict:
    """Macro averages over answerable queries at K (plus micro coverage and bookkeeping counts)."""
    rows = [r.metrics[k] for r in results if r.answerable and r.metrics is not None]
    n = len(rows)
    out = {
        "k": k,
        "n_queries": len(results),
        "n_answerable": n,
        "n_unanswerable_excluded": sum(not r.answerable for r in results),
        "recall_zero_denominator": sum(m.relevant_in_index == 0 for m in rows),
        "empty_results": sum(r.returned == 0 for r in results if r.answerable),
    }
    if n == 0:
        return out | {
            "hit_rate": None,
            "mrr": None,
            "precision": None,
            "recall": None,
            "coverage_macro": None,
            "coverage_micro": None,
            "groups_satisfied": 0,
            "groups_total": 0,
        }
    satisfied, total = sum(m.groups_satisfied for m in rows), sum(m.groups_total for m in rows)
    return out | {
        "hit_rate": sum(m.hit for m in rows) / n,
        "mrr": sum(m.reciprocal_rank for m in rows) / n,
        "precision": sum(m.precision for m in rows) / n,
        "recall": sum(m.recall for m in rows) / n,
        "coverage_macro": sum(m.coverage for m in rows) / n,
        "coverage_micro": satisfied / total,
        "groups_satisfied": satisfied,
        "groups_total": total,
    }


def compare_k(results: list[QueryEvaluation], ks: Iterable[int] = (3, 5)) -> dict[int, dict]:
    """Aggregates for several K side by side (e.g. K=3 vs K=5)."""
    return {k: aggregate(results, k) for k in ks}
