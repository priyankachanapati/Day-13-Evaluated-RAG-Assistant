"""Activity 1 - Compare chunking strategies: fixed 500 characters vs recursive 800/100.

The Day 12 corpus (6 policies, hash-checked) was re-indexed with both strategies by the project's
chunkers (rag_v2/chunking.py) into separate Chroma collections (.chroma-v2/d13_fixed_500,
.chroma-v2/d13_recursive_800_100). This script opens those collections read-only (no rebuild),
runs the same 5 frozen development questions against both with the unchanged v1 dense retriever,
takes the top 3, and measures relevance with the project's fact-level labels:

  relevant chunk = fully contains a required evidence sentence
  Hit@3, MRR@3, Precision@3 (chunk level) and fact Coverage@3 (share of required facts present)

It cross-checks the live top-3 against the saved Activity A results and reports, separately and
labelled as saved, the 30-question Phase 7 metrics of the same two dense configurations.

Local only: no LLM, no API key. Usage (project root):
    .venv/bin/python activities/activity-1-chunking-comparison/compare_chunking_top3.py
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
from rag_v2.chunking import STRATEGIES  # noqa: E402
from rag_v2.indexing import V2_CHROMA_DIR, collection_name  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.vector_store import PolicyVectorStore  # noqa: E402

COMPARED = {
    "fixed_500": "Fixed character, 500 chars, no overlap",
    "recursive_800_100": "Recursive character, 800 chars, 100 overlap",
}
K = 3
SAVED_A = ac.PROJECT_ROOT / "activities/a-chunking-strategies/results/chunking-comparison.json"


def main() -> int:
    assert ac.sha256(ac.DEV_QUERIES) == ac.manifest_hash("dev-queries.sha256", "dev_queries.json")
    assert ac.sha256(ac.PROJECT_ROOT / "evaluation/datasets/evidence_catalog.json") == ac.manifest_hash(
        "evidence-catalog.sha256", "evidence_catalog.json"
    )
    docs, questions, embedder = ac.corpus(), ac.load_queries(ac.DEV_QUERIES), LocalEmbedder()
    saved_a = {q["id"]: q for q in json.loads(SAVED_A.read_text(encoding="utf-8"))["dev_queries"]}
    phase7 = json.loads(ac.PHASE7_RETRIEVAL.read_text(encoding="utf-8"))["configs"]

    print("=" * 100)
    print("ACTIVITY 1 - CHUNKING COMPARISON: same 5 questions, dense top-3, Day 12 corpus (6 policies)")
    print("=" * 100)
    stores, facts, checks = {}, {}, []
    for name, desc in COMPARED.items():
        chunks = STRATEGIES[name].chunk_documents(docs)
        stores[name] = PolicyVectorStore(V2_CHROMA_DIR, collection_name(name))
        facts[name] = ac.chunk_facts(name, docs)
        checks.append(stores[name].count() == len(chunks))
        lengths = [len(c.text) for c in chunks]
        print(
            f"{name:<18} {desc:<46} chunks={len(chunks):>2}  mean={sum(lengths) / len(lengths):6.1f}  "
            f"max={max(lengths)}  index={collection_name(name)} ({stores[name].count()})"
        )

    per_q, rows = [], {n: [] for n in COMPARED}
    for q in questions:
        print(f"\n{q['id']}  {q['query']}")
        entry = {
            "id": q["id"],
            "query": q["query"],
            "required_evidence": q["required_evidence"],
            "results": {},
        }
        for name in COMPARED:
            result = Retriever(embedder, stores[name], top_k=K, min_similarity=0.25).retrieve(q["query"])
            ranked = [c.chunk_id for c in result.chunks]
            m = query_metrics(ranked, facts[name], q["required_evidence"], K)
            rows[name].append(m)
            rel = set(ac.relevant_ids(facts[name], q["required_evidence"]))
            checks.append(ranked == [t["chunk_id"] for t in saved_a[q["id"]]["results"][name]["top"]])
            shown = ", ".join(
                f"{'*' if c.chunk_id in rel else ''}{c.chunk_id}({c.similarity:.3f})" for c in result.chunks
            )
            print(
                f"  {name:<18} {shown:<62} hit={m.hit} RR={m.reciprocal_rank:.2f} P@3={m.precision:.2f} "
                f"facts={m.groups_satisfied}/{m.groups_total}"
            )
            entry["results"][name] = {
                "top3": [
                    {
                        "rank": c.rank,
                        "chunk_id": c.chunk_id,
                        "similarity": round(c.similarity, 4),
                        "section": c.section,
                        "relevant": c.chunk_id in rel,
                    }
                    for c in result.chunks
                ],
                "metrics": m.as_dict(),
            }
        per_q.append(entry)
    print("\n(* = chunk fully contains a required fact)")

    summary = {}
    print(
        f"\n{'Strategy':<18} {'Hit@3':>6} {'MRR@3':>6} {'P@3':>6} {'Cov@3':>6}   | saved Phase 7, 25 answerable: "
        f"{'Hit@3':>6} {'MRR@3':>6} {'Cov@3':>6}"
    )
    for name in COMPARED:
        mm = mean_metrics(rows[name])
        p7 = phase7[f"{name}+dense"]["aggregate"]["3"]
        summary[name] = {
            "dev5": {k: round(mm[k], 4) for k in ("hit_rate", "mrr", "precision", "coverage")},
            "phase7_saved_25": {
                "hit_rate": p7["hit_rate"],
                "mrr": p7["mrr"],
                "coverage": p7["coverage_macro"],
            },
        }
        print(
            f"{name:<18} {mm['hit_rate']:>6.2f} {mm['mrr']:>6.2f} {mm['precision']:>6.2f} {mm['coverage']:>6.2f}   | "
            f"{'':<30}{p7['hit_rate']:>6.2f} {p7['mrr']:>6.2f} {p7['coverage_macro']:>6.2f}"
        )
    ok = all(checks)
    print(
        f"\n[{'PASS' if ok else 'FAIL'}] {len(checks)} checks: both indexes hold every chunk (2/2); live top-3 == saved "
        "Activity A top-3 (10/10)"
    )
    ac.write_json(
        HERE / "results" / "chunking-top3-comparison.json",
        {
            "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "k": K,
            "strategies": COMPARED,
            "per_query": per_q,
            "summary": summary,
            "all_checks_pass": ok,
        },
    )
    print("Wrote results/chunking-top3-comparison.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
