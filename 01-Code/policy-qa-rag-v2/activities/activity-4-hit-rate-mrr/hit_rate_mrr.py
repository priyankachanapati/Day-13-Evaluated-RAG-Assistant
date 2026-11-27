"""Activity 4 - Hit Rate@K and MRR@K for the Day 12 retriever, K = 3 and K = 5.

Dataset (10 labelled questions): the first 10 ANSWERABLE questions, in ID order, of the frozen
held-out set evaluation/datasets/eval_queries_v1_1.json (hash-checked). Phase 7 is complete and
its configuration selection is already registered, so reusing them here influences no decision.
Known relevant chunk IDs are derived from the frozen fact labels (see activity_common.py) and
written to labelled_questions.json before any retrieval runs.

The metric function ``hit_rate_and_mrr`` below is the compact implementation (14 lines). It is checked
against the project's evaluation.metrics.query_metrics and against the saved Phase 7 v1 records.

Local only: no LLM, no API key. Usage (project root):
    .venv/bin/python activities/activity-4-hit-rate-mrr/hit_rate_mrr.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import activity_common as ac  # noqa: E402

from evaluation.metrics import mean_metrics, query_metrics  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

STRATEGY = "v1_paragraph_800_150"
N_QUESTIONS = 10
KS = (3, 5)


def hit_rate_and_mrr(rankings: list[list[str]], relevant: list[set[str]], k: int) -> tuple[float, float]:
    """Hit Rate@k and MRR@k over queries.

    rankings[i] = retrieved chunk IDs for query i, best first; relevant[i] = its known relevant IDs.
    Hit@k = 1 if any relevant ID is in the top k. RR@k = 1/rank of the first relevant ID in the top k, else 0.
    """
    if not rankings or len(rankings) != len(relevant):
        raise ValueError("need one non-empty relevance set per ranking")
    hits, rr_sum = 0, 0.0
    for ranked, rel in zip(rankings, relevant, strict=True):
        rank = next((i for i, cid in enumerate(ranked[:k], start=1) if cid in rel), None)
        if rank is not None:
            hits, rr_sum = hits + 1, rr_sum + 1.0 / rank
    return hits / len(rankings), rr_sum / len(rankings)


def main() -> int:
    assert ac.sha256(ac.EVAL_V11) == ac.manifest_hash("eval-dataset-v1_1.sha256", "eval_queries_v1_1.json")
    questions = [q for q in ac.load_queries(ac.EVAL_V11) if q["answerable"]][:N_QUESTIONS]
    facts = ac.chunk_facts(STRATEGY, ac.corpus())
    # Question texts stay only in the frozen dataset (tests_v2/test_held_out_guard.py): store IDs and labels.
    text = {q["id"]: q["question"] for q in questions}
    labelled = [
        {
            "id": q["id"],
            "category": q["category"],
            "required_evidence": q["required_evidence"],
            "known_relevant_chunk_ids": ac.relevant_ids(facts, q["required_evidence"]),
        }
        for q in questions
    ]
    ac.write_json(
        HERE / "labelled_questions.json",
        {
            "source": "evaluation/datasets/eval_queries_v1_1.json (frozen) - first 10 answerable questions",
            "question_text": "read from the frozen dataset by ID (held-out texts are not copied, see test_held_out_guard)",
            "relevance_rule": "chunk fully contains at least one required evidence sentence",
            "chunking": STRATEGY,
            "questions": labelled,
        },
    )

    pipeline = RAGPipeline(load_settings())  # Day 12 retriever; the LLM client is lazy and never created
    assert pipeline.store.count() == len(facts) == 22
    rankings = [
        [c.chunk_id for c in pipeline.retriever.retrieve(text[q["id"]], top_k=max(KS)).chunks]
        for q in labelled
    ]
    relevant = [set(q["known_relevant_chunk_ids"]) for q in labelled]

    print("=" * 92)
    print(
        f"ACTIVITY 4 - HIT RATE@K AND MRR@K | Day 12 retriever ({STRATEGY}) | {len(labelled)} labelled questions"
    )
    print("=" * 92)
    print(f"{'ID':<6} {'relevant chunk IDs':<28} {'top-5 retrieved (rank order)':<62}")
    per_q = []
    for q, ranked in zip(labelled, rankings, strict=True):
        first = next((i for i, c in enumerate(ranked, 1) if c in set(q["known_relevant_chunk_ids"])), None)
        marked = ", ".join(f"*{c}" if c in q["known_relevant_chunk_ids"] else c for c in ranked)
        print(f"{q['id']:<6} {', '.join(q['known_relevant_chunk_ids']):<28} {marked}")
        per_q.append(
            {
                "id": q["id"],
                "relevant": q["known_relevant_chunk_ids"],
                "top5": ranked,
                "first_relevant_rank": first,
                **{f"hit@{k}": int(first is not None and first <= k) for k in KS},
                **{f"rr@{k}": (1 / first if first and first <= k else 0.0) for k in KS},
            }
        )
    print("(* = known relevant chunk)\n")

    results, checks = {}, []
    phase7 = {
        r["id"]: r
        for r in json.loads(ac.PHASE7_RETRIEVAL.read_text(encoding="utf-8"))["configs"][f"{STRATEGY}+dense"][
            "per_query"
        ]
    }
    for k in KS:
        hit, mrr = hit_rate_and_mrr(rankings, relevant, k)
        ref = mean_metrics(
            [
                query_metrics(r, facts, q["required_evidence"], k)
                for r, q in zip(rankings, labelled, strict=True)
            ]
        )
        saved = [phase7[q["id"]]["metrics"][str(k)] for q in labelled]
        checks += [
            abs(hit - ref["hit_rate"]) < 1e-12 and abs(mrr - ref["mrr"]) < 1e-12,
            abs(hit - sum(s["hit"] for s in saved) / len(saved)) < 1e-12
            and abs(mrr - sum(s["reciprocal_rank"] for s in saved) / len(saved)) < 1e-12,
        ]
        results[k] = {"hit_rate": round(hit, 4), "mrr": round(mrr, 4)}
    checks.append(
        all(phase7[q["id"]]["final_ranking"][:5] == r for q, r in zip(labelled, rankings, strict=True))
    )

    print(f"{'Metric':<10} {'K=3':>8} {'K=5':>8} {'Change':>8}")
    for m in ("hit_rate", "mrr"):
        print(f"{m:<10} {results[3][m]:>8.3f} {results[5][m]:>8.3f} {results[5][m] - results[3][m]:>+8.3f}")
    ranks = [p["first_relevant_rank"] for p in per_q]
    print(f"\nFirst-relevant rank per question: {ranks}")
    ok = all(checks)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {len(checks)} checks: compact function == evaluation.metrics (K=3, K=5); "
        "== saved Phase 7 v1 records (K=3, K=5); live top-5 == saved Phase 7 ranking (10/10)"
    )
    ac.write_json(
        HERE / "results" / "hit-rate-mrr.json",
        {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "retriever": f"Day 12 dense, {STRATEGY}",
            "questions": len(labelled),
            "per_question": per_q,
            "results": {f"k={k}": v for k, v in results.items()},
            "all_checks_pass": ok,
        },
    )
    print("Wrote labelled_questions.json and results/hit-rate-mrr.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
