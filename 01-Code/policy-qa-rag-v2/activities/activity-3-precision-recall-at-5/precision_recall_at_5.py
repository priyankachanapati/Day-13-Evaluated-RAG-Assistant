"""Activity 3 - Precision@5 and Recall@5 for five labelled questions (Day 12 retriever).

    Precision@k = relevant retrieved in top k / total retrieved in top k
    Recall@k    = relevant retrieved in top k / total known relevant

Steps
  1. Five questions = the 5 frozen development questions (evaluation/datasets/dev_queries.json,
     hash-checked). Their labels are canonical evidence sentences.
  2. Known relevant chunk IDs = Day 12 chunks that fully contain a required evidence sentence
     (derived from the frozen labels, before retrieval; written to labelled_questions.json).
  3. Top-5 = the unchanged Day 12 retriever (src/ RAGPipeline.retriever, persisted .chroma index).
  4. For each question print relevant IDs, retrieved IDs, intersection, numerator, denominator.
  5. Verify every number programmatically: exact fractions, an independent re-implementation
     (evaluation.metrics.query_metrics), and a cross-check against the saved Activity A/B rankings.

Local only: no LLM, no API key. Usage (project root):
    .venv/bin/python activities/activity-3-precision-recall-at-5/precision_recall_at_5.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import activity_common as ac  # noqa: E402

from evaluation.metrics import query_metrics  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

K = 5
STRATEGY = "v1_paragraph_800_150"  # the Day 12 chunking (800/150, paragraph-aware)
SAVED_B = ac.PROJECT_ROOT / "activities/b-hybrid-search-rrf/results/hybrid-retrieval.json"


def precision_recall(retrieved: list[str], relevant: set[str], k: int) -> dict:
    top = retrieved[:k]
    hits = [cid for cid in top if cid in relevant]
    p, r = Fraction(len(hits), len(top)), Fraction(len(hits), len(relevant))
    return {
        "top_k": top,
        "intersection": hits,
        "num": len(hits),
        "p_den": len(top),
        "r_den": len(relevant),
        "precision": p,
        "recall": r,
    }


def main() -> int:
    assert ac.sha256(ac.DEV_QUERIES) == ac.manifest_hash("dev-queries.sha256", "dev_queries.json"), (
        "dev set changed"
    )
    questions = ac.load_queries(ac.DEV_QUERIES)
    assert len(questions) == 5
    facts = ac.chunk_facts(STRATEGY, ac.corpus())
    pipeline = RAGPipeline(load_settings())  # LLM client is lazy: never created here
    retriever = pipeline.retriever
    assert pipeline.store.count() == len(facts) == 22, "Day 12 index must hold the 22 v1 chunks"

    saved_b = {q["id"]: q for q in json.loads(SAVED_B.read_text(encoding="utf-8"))["per_query"]}
    print("=" * 96)
    print(f"ACTIVITY 3 - PRECISION@{K} AND RECALL@{K}  |  Day 12 retriever (dense, {STRATEGY}, 22 chunks)")
    print("=" * 96)
    print(
        f"Precision@{K} = |relevant ∩ top-{K}| / |top-{K}|      Recall@{K} = |relevant ∩ top-{K}| / |relevant|\n"
    )

    rows, labelled, checks = [], [], []
    for q in questions:
        relevant = ac.relevant_ids(facts, q["required_evidence"])
        result = retriever.retrieve(q["query"], top_k=K)
        retrieved = [c.chunk_id for c in result.chunks]
        m = precision_recall(retrieved, set(relevant), K)
        ref = query_metrics(retrieved, facts, q["required_evidence"], K)  # independent implementation
        checks.append(
            abs(float(m["precision"]) - ref.precision) < 1e-12
            and abs(float(m["recall"]) - ref.recall) < 1e-12
        )
        saved = [t["chunk_id"] for t in saved_b[q["id"]]["strategies"][STRATEGY]["dense"]["top"]][:K]
        checks.append(saved == retrieved)
        labelled.append(
            {
                "id": q["id"],
                "question": q["query"],
                "required_evidence": q["required_evidence"],
                "known_relevant_chunk_ids": relevant,
            }
        )
        print(f"{q['id']}  {q['query']}")
        print(f"  relevant chunk IDs  : {relevant}")
        print(
            "  top-5 retrieved     : "
            + ", ".join(f"{i}.{c.chunk_id}({c.similarity:.3f})" for i, c in enumerate(result.chunks, 1))
        )
        print(f"  intersection        : {m['intersection'] or '(none)'}")
        print(
            f"  Precision@{K} = {m['num']}/{m['p_den']} = {float(m['precision']):.3f}    "
            f"Recall@{K} = {m['num']}/{m['r_den']} = {float(m['recall']):.3f}\n"
        )
        rows.append(
            {
                "id": q["id"],
                "relevant": relevant,
                "retrieved_top5": retrieved,
                "similarities": [round(c.similarity, 4) for c in result.chunks],
                "intersection": m["intersection"],
                "numerator": m["num"],
                "precision_denominator": m["p_den"],
                "recall_denominator": m["r_den"],
                "precision_at_5": str(m["precision"]),
                "recall_at_5": str(m["recall"]),
                "precision_at_5_value": round(float(m["precision"]), 4),
                "recall_at_5_value": round(float(m["recall"]), 4),
            }
        )

    mean_p = sum(Fraction(r["precision_at_5"]) for r in rows) / len(rows)
    mean_r = sum(Fraction(r["recall_at_5"]) for r in rows) / len(rows)
    print(f"{'Question':<8} {'|rel|':>5} {'hits':>4}  {'Precision@5':>12}  {'Recall@5':>10}")
    for r in rows:
        print(
            f"{r['id']:<8} {r['recall_denominator']:>5} {r['numerator']:>4}  "
            f"{r['numerator']}/5 = {r['precision_at_5_value']:.3f}  {r['numerator']}/{r['recall_denominator']} = {r['recall_at_5_value']:.3f}"
        )
    print(
        f"{'Mean':<8} {'':>5} {'':>4}  {float(mean_p):>12.3f}  {float(mean_r):>10.3f}   (macro average over 5 questions)\n"
    )

    ok = all(checks)
    print(
        f"[{'PASS' if ok else 'FAIL'}] {len(checks)} checks: Fraction results == evaluation.metrics.query_metrics "
        f"(5/5) and live top-5 == saved Activity B dense ranking (5/5)"
    )
    out = {
        "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "k": K,
        "retriever": f"Day 12 dense retriever, {STRATEGY}",
        "definition": {
            "precision_at_k": "relevant retrieved in top k / total retrieved in top k",
            "recall_at_k": "relevant retrieved in top k / total known relevant",
        },
        "per_question": rows,
        "mean_precision_at_5": round(float(mean_p), 4),
        "mean_recall_at_5": round(float(mean_r), 4),
        "all_checks_pass": ok,
    }
    ac.write_json(
        HERE / "labelled_questions.json",
        {
            "source": "evaluation/datasets/dev_queries.json (frozen)",
            "relevance_rule": "chunk fully contains at least one required evidence sentence",
            "chunking": STRATEGY,
            "questions": labelled,
        },
    )
    ac.write_json(HERE / "results" / "precision-recall-at-5.json", out)
    print("Wrote labelled_questions.json and results/precision-recall-at-5.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
