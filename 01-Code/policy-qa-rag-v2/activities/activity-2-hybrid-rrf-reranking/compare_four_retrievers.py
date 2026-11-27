"""Activity 2 - BM25 hybrid search with Reciprocal Rank Fusion, then cross-encoder reranking.

Pipeline (Day 12 corpus and chunking, v1_paragraph_800_150, 22 chunks):
  1. Vector retriever: the unchanged v1 dense retriever (MiniLM + Chroma), top 10.
  2. BM25 retriever: rank_bm25.BM25Okapi (rag_v2/bm25.py), top 10.
  3. Reciprocal Rank Fusion of both lists, k = 60 (rag_v2/fusion.py): score = sum 1/(60 + rank).
  4. Top 10 fused results are reranked by the pinned sentence-transformers cross-encoder
     cross-encoder/ms-marco-MiniLM-L6-v2 @ 233902d2 (rag_v2/reranker.py, loaded from the local cache).
Compared on the same 5 frozen development questions: vector-only, BM25-only, hybrid (RRF) and
hybrid + rerank, with Hit Rate@K and MRR@K (K = 3, 5) using the project's fact-level relevance rule.
Live rankings are cross-checked against the saved Activity B (hybrid) and Activity C (reranked)
results; the 30-question Phase 7 metrics of the same four configurations are shown, labelled saved.

Local only: no LLM, no API key, no model download. Usage (project root):
    .venv/bin/python activities/activity-2-hybrid-rrf-reranking/compare_four_retrievers.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import activity_common as ac  # noqa: E402

from evaluation.metrics import mean_metrics, query_metrics  # noqa: E402
from rag_v2.chunking import STRATEGIES  # noqa: E402
from rag_v2.fusion import RRF_K  # noqa: E402
from rag_v2.hybrid import CANDIDATES, HybridSearcher  # noqa: E402
from rag_v2.indexing import V2_CHROMA_DIR, StrategyIndex, collection_name  # noqa: E402
from rag_v2.rerank_search import RerankingSearcher  # noqa: E402
from rag_v2.reranker import MODEL_ID, MODEL_REVISION, CrossEncoderReranker  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.vector_store import PolicyVectorStore  # noqa: E402

STRATEGY = "v1_paragraph_800_150"
SYSTEMS = {
    "dense": "Vector only",
    "bm25": "BM25 only",
    "hybrid": "Hybrid (RRF)",
    "hybrid+rerank": "Hybrid + rerank",
}
KS = (3, 5)
SAVED_B = ac.PROJECT_ROOT / "activities/b-hybrid-search-rrf/results/hybrid-retrieval.json"
SAVED_C = ac.PROJECT_ROOT / "activities/c-cross-encoder-reranking/results/reranking.json"


def main() -> int:
    assert ac.sha256(ac.DEV_QUERIES) == ac.manifest_hash("dev-queries.sha256", "dev_queries.json")
    docs, questions = ac.corpus(), ac.load_queries(ac.DEV_QUERIES)
    chunks = STRATEGIES[STRATEGY].chunk_documents(docs)
    store = PolicyVectorStore(V2_CHROMA_DIR, collection_name(STRATEGY))  # existing index, opened read-only
    index = StrategyIndex(STRATEGIES[STRATEGY], chunks, store, Retriever(LocalEmbedder(), store, top_k=3))
    hybrid = HybridSearcher(index, candidates=CANDIDATES, rrf_k=RRF_K)
    searcher = RerankingSearcher(hybrid, CrossEncoderReranker(local_files_only=True))
    facts = ac.chunk_facts(STRATEGY, docs)
    saved_b = {
        q["id"]: q["strategies"][STRATEGY]
        for q in json.loads(SAVED_B.read_text(encoding="utf-8"))["per_query"]
    }
    saved_c = {
        q["id"]: q["strategies"][STRATEGY]
        for q in json.loads(SAVED_C.read_text(encoding="utf-8"))["per_query"]
    }

    print("=" * 100)
    print(
        f"ACTIVITY 2 - VECTOR vs BM25 vs HYBRID (RRF k={RRF_K}) vs HYBRID + CROSS-ENCODER | {STRATEGY}, {store.count()} chunks"
    )
    print(
        f"Reranker: {MODEL_ID} @ {MODEL_REVISION[:8]} (local cache) | fused candidates reranked: top {CANDIDATES}"
    )
    print("=" * 100)
    checks = [store.count() == len(chunks)]
    per_q, metrics = [], {s: {k: [] for k in KS} for s in SYSTEMS}
    for q in questions:
        t0 = time.perf_counter()
        r = searcher.search(q["query"])
        ms = (time.perf_counter() - t0) * 1000
        ranked = {
            "dense": r.hybrid.ranked_ids("dense"),
            "bm25": r.hybrid.ranked_ids("bm25"),
            "hybrid": r.pool_ids,
            "hybrid+rerank": r.reranked_ids,
        }
        checks.append(ranked["hybrid"] == [t["chunk_id"] for t in saved_b[q["id"]]["hybrid"]["top"]])
        checks.append(ranked["hybrid+rerank"] == [t["chunk_id"] for t in saved_c[q["id"]]["rows"]["after"]])
        rel = set(ac.relevant_ids(facts, q["required_evidence"]))
        print(f"\n{q['id']}  {q['query']}   (relevant: {', '.join(sorted(rel))})")
        entry = {
            "id": q["id"],
            "query": q["query"],
            "relevant": sorted(rel),
            "rankings_top5": {},
            "metrics": {},
            "rrf_source_ranks_top5": [
                {"chunk_id": h.chunk_id, "rrf": round(h.score, 5), **h.source_ranks}
                for h in r.hybrid.hybrid[:5]
            ],
            "search_plus_rerank_ms": round(ms, 1),
        }
        for s, label in SYSTEMS.items():
            m = {k: query_metrics(ranked[s], facts, q["required_evidence"], k) for k in KS}
            for k in KS:
                metrics[s][k].append(m[k])
            top = ", ".join(f"{'*' if c in rel else ''}{c}" for c in ranked[s][:5])
            print(f"  {label:<16} {top:<66} RR@5={m[5].reciprocal_rank:.2f}")
            entry["rankings_top5"][s] = ranked[s][:5]
            entry["metrics"][s] = {f"hit@{k}": m[k].hit for k in KS} | {
                f"rr@{k}": m[k].reciprocal_rank for k in KS
            }
        per_q.append(entry)
    print("\n(* = chunk fully contains a required fact)")

    phase7 = json.loads(ac.PHASE7_RETRIEVAL.read_text(encoding="utf-8"))["configs"]
    summary = {}
    print(
        f"\n{'System':<16} {'Hit@3':>6} {'Hit@5':>6} {'MRR@3':>6} {'MRR@5':>6}  | saved Phase 7 (25 answerable): "
        f"{'Hit@3':>6} {'MRR@3':>6} {'MRR@5':>6}"
    )
    for s, label in SYSTEMS.items():
        dev = {k: mean_metrics(metrics[s][k]) for k in KS}
        p7 = phase7[f"{STRATEGY}+{s}"]["aggregate"]
        summary[s] = {
            "dev5": {f"hit@{k}": round(dev[k]["hit_rate"], 4) for k in KS}
            | {f"mrr@{k}": round(dev[k]["mrr"], 4) for k in KS},
            "phase7_saved_25": {
                "hit@3": p7["3"]["hit_rate"],
                "mrr@3": p7["3"]["mrr"],
                "mrr@5": p7["5"]["mrr"],
            },
        }
        print(
            f"{label:<16} {dev[3]['hit_rate']:>6.2f} {dev[5]['hit_rate']:>6.2f} {dev[3]['mrr']:>6.2f} {dev[5]['mrr']:>6.2f}  | "
            f"{'':<31}{p7['3']['hit_rate']:>6.2f} {p7['3']['mrr']:>6.2f} {p7['5']['mrr']:>6.2f}"
        )
    ok = all(checks)
    print(
        f"\n[{'PASS' if ok else 'FAIL'}] {len(checks)} checks: index complete (1/1); live hybrid top-10 == saved Activity B "
        "(5/5); live reranked top-10 == saved Activity C (5/5)"
    )
    ac.write_json(
        HERE / "results" / "four-retrievers-comparison.json",
        {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "chunking": STRATEGY,
            "rrf_k": RRF_K,
            "candidates": CANDIDATES,
            "reranker": {"model": MODEL_ID, "revision": MODEL_REVISION},
            "per_query": per_q,
            "summary": summary,
            "all_checks_pass": ok,
            "note": "latency is a single warm local run on CPU (indicative only)",
        },
    )
    print("Wrote results/four-retrievers-comparison.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
