"""Activity D - retrieval metrics: worked example and K=3 vs K=5 on the DEVELOPMENT queries.

Part 1  step-by-step Precision@K, Recall@K, Hit@K, RR/MRR and fact Coverage@K on a hand-worked
        toy index, plus macro vs micro aggregation (the same numbers the unit tests assert).
Part 2  K=3 vs K=5 on the 5 development queries, recomputed with ``evaluation.retrieval_metrics``
        from the ranked lists ALREADY SAVED by Phase 3 (dense, BM25, hybrid) and Phase 4
        (hybrid + cross-encoder) for all three chunking strategies. No retrieval system is run,
        and the frozen 30-query evaluation set is never read (Phase 7).
        Cross-check: every per-query metric must equal the value saved in Phase 3 / 4.

Usage (project root):  .venv/bin/python activities/d-retrieval-metrics/metrics_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evidence import chunk_evidence, load_catalog  # noqa: E402
from evaluation.retrieval_metrics import aggregate, compare_k, evaluate_ranking  # noqa: E402
from rag_v2.chunking import STRATEGIES, load_corpus  # noqa: E402
from src.config import load_settings  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PHASE3 = PROJECT_ROOT / "activities" / "b-hybrid-search-rrf" / "results" / "hybrid-retrieval.json"
PHASE4 = PROJECT_ROOT / "activities" / "c-cross-encoder-reranking" / "results" / "reranking.json"
DEV = PROJECT_ROOT / "evaluation" / "datasets" / "dev_queries.json"
SYSTEMS = {
    "dense": "Dense only",
    "bm25": "BM25 only",
    "hybrid": "Hybrid (RRF)",
    "rerank": "Hybrid + cross-encoder",
}

TOY = {"A": set(), "B": {"e1"}, "C": {"e1", "x"}, "D": {"e2"}, "E": {"y"}, "F": {"e3"}}


def f(x):
    return "—" if x is None else f"{x:.3f}"


def part1(L: list[str]) -> None:
    L += [
        "## Part 1 – Worked example (hand-checkable)",
        "",
        "Toy index: `A:{}  B:{e1}  C:{e1,x}  D:{e2}  E:{y}  F:{e3}` (C repeats e1, like an overlap duplicate).",
        "Query requires two facts: group 1 = e1, group 2 = e2. Relevant chunks in the index: B, C (e1), D (e2) → 3.",
        "Ranked result: `B, A, E, D`.",
        "",
        "| K | Top-K | Relevant in top-K | Precision@K | Recall@K (chunks) | Hit@K | RR@K | Fact groups satisfied | Coverage@K |",
        "|---:|---|---|---|---|---:|---:|---|---:|",
    ]
    r = evaluate_ranking("toy", True, [["e1"], ["e2"]], ["B", "A", "E", "D"], TOY, (1, 3, 5))
    for k, m in r.metrics.items():
        top = ["B", "A", "E", "D"][:k]
        rel = [c for c in top if TOY[c] & {"e1", "e2"}]
        L.append(
            f"| {k} | {', '.join(top)} | {', '.join(rel) or '—'} | {m.relevant_retrieved}/{k} = {m.precision:.3f} | "
            f"{m.relevant_retrieved}/{m.relevant_in_index} = {m.recall:.3f} | {m.hit} | {m.reciprocal_rank:.3f} | "
            f"{m.groups_satisfied}/{m.groups_total} | {m.coverage:.3f} |"
        )
    L += [
        "",
        "Reading the table: at K=3 the list has found e1 (in B) but not e2 → coverage 1/2. At K=5 D is included → "
        "coverage 2/2, but precision falls (more non-relevant chunks per slot). Retrieving C as well would raise "
        "chunk recall but **not** coverage: e1 is one fact.",
        "",
        "### Aggregation over several queries (K=3)",
        "",
        "| Query | Required | Ranked | Hit | RR | P@3 | R@3 | Coverage@3 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    suite = [
        ("Q1", [["e3"]], ["F", "A", "B"], True),
        ("Q2", [["e2"]], ["A", "E", "D", "B", "C"], True),
        ("Q3", [["e3"]], ["A", "E"], True),
        ("Q4", [["e1"]], ["B", "C", "A"], True),
        ("Q5", [["e1"], ["e2"]], ["B", "A", "E", "D"], True),
        ("Q6", [["e1"]], [], True),
        ("Q7 (unanswerable)", [], ["A", "B"], False),
    ]
    results = [evaluate_ranking(q, a, g, r, TOY, (3, 5)) for q, g, r, a in suite]
    for (q, g, rk, _), res in zip(suite, results, strict=True):
        if res.metrics is None:
            L.append(f"| {q} | — | {', '.join(rk)} | excluded | | | | |")
            continue
        m = res.metrics[3]
        L.append(
            f"| {q} | {g} | {', '.join(rk) or '(empty)'} | {m.hit} | {m.reciprocal_rank:.3f} | {m.precision:.3f} | "
            f"{m.recall:.3f} | {m.groups_satisfied}/{m.groups_total} = {m.coverage:.2f} |"
        )
    a = aggregate(results, 3)
    L += [
        "",
        f"Macro averages over the **{a['n_answerable']} answerable** queries ({a['n_unanswerable_excluded']} "
        f"unanswerable excluded): Hit@3 = {a['hit_rate']:.3f}, MRR@3 = {a['mrr']:.3f}, P@3 = {a['precision']:.3f}, "
        f"R@3 = {a['recall']:.3f}, Coverage@3 (macro) = {a['coverage_macro']:.4f}; Coverage@3 (micro, pooled "
        f"{a['groups_satisfied']}/{a['groups_total']}) = {a['coverage_micro']:.4f}. Macro weighs every query equally; "
        "micro weighs every fact equally — they differ because Q5 has two facts.",
        "",
    ]


def part2(L: list[str]) -> bool:
    for m in (
        "phase3-artifacts.sha256",
        "phase4-artifacts.sha256",
        "dev-queries.sha256",
        "evidence-catalog.sha256",
    ):
        if subprocess.run(
            ["shasum", "-a", "256", "-c", "--quiet", f"manifests/{m}"], cwd=PROJECT_ROOT
        ).returncode:
            raise SystemExit(f"manifest {m} does not match - refusing to use saved results")
    docs = load_corpus(load_settings().policy_dir)
    by_policy: dict = {}
    for u in load_catalog().values():
        by_policy.setdefault(u.policy_id, []).append(u)
    facts = {
        n: {c.chunk_id: set(chunk_evidence(c.metadata, by_policy)) for c in s.chunk_documents(docs)}
        for n, s in STRATEGIES.items()
    }
    dev = {q["id"]: q for q in json.loads(DEV.read_text())["queries"]}
    p3 = {e["id"]: e for e in json.loads(PHASE3.read_text())["per_query"]}
    p4 = {e["id"]: e for e in json.loads(PHASE4.read_text())["per_query"]}

    checked = mismatches = 0
    table = {}
    for n in STRATEGIES:
        for system in SYSTEMS:
            results = []
            for qid, q in dev.items():
                if system == "rerank":
                    ranked = [r["chunk_id"] for r in p4[qid]["strategies"][n]["rows"]["after"]]
                    saved = p4[qid]["strategies"][n]["metrics"]["after"]
                else:
                    ranked = [h["chunk_id"] for h in p3[qid]["strategies"][n][system]["top"]]
                    saved = p3[qid]["strategies"][n][system]["metrics"]
                res = evaluate_ranking(qid, True, q["required_evidence"], ranked, facts[n], (3, 5))
                for k in (3, 5):
                    checked += 1
                    mismatches += res.metrics[k].as_dict() != saved[f"@{k}"]
                results.append(res)
            table[(n, system)] = compare_k(results, (3, 5))
    L += [
        "## Part 2 – K=3 vs K=5 on the 5 development queries (saved Phase 3/4 rankings)",
        "",
        "Recomputed with `evaluation.retrieval_metrics` from the ranked lists saved by Phase 3 and Phase 4; no "
        "retrieval was run and the 30-query evaluation set was not used. `v1_paragraph_800_150` + Dense only is the "
        "v1 retriever (verified identical in Phases 2–3).",
        "",
        f"Cross-check against the saved per-query metrics: **{checked - mismatches}/{checked} identical**.",
        "",
        "| Strategy | System | Hit@3 | Hit@5 | MRR@3 | MRR@5 | P@3 | P@5 | R@3* | R@5* | Coverage@3 macro (micro) | "
        "Coverage@5 macro (micro) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for (n, system), c in table.items():
        a3, a5 = c[3], c[5]
        L.append(
            f"| `{n}` | {SYSTEMS[system]} | {f(a3['hit_rate'])} | {f(a5['hit_rate'])} | {f(a3['mrr'])} | "
            f"{f(a5['mrr'])} | {f(a3['precision'])} | {f(a5['precision'])} | {f(a3['recall'])} | {f(a5['recall'])} | "
            f"{f(a3['coverage_macro'])} ({a3['groups_satisfied']}/{a3['groups_total']}) | "
            f"{f(a5['coverage_macro'])} ({a5['groups_satisfied']}/{a5['groups_total']}) |"
        )
    L += [
        "",
        "\\* chunk-level recall; depends on the chunking, not comparable across strategies.",
        "",
        "**K=3 vs K=5.** Going from 3 to 5 retrieved chunks can only keep or raise Hit, MRR, recall and coverage "
        "(more chances to include a relevant chunk) and usually lowers precision (the extra slots are mostly "
        "non-relevant). On these 5 dev queries every system reaches full coverage at K=5. The trade-off for the "
        "assistant is context size: v1 sends 4 chunks to the LLM, and more chunks mean more prompt tokens (Phase 8). "
        "These are development numbers only; the held-out comparison is Phase 7.",
        "",
    ]
    return mismatches == 0


def main() -> int:
    L = [
        "# Activity D – Retrieval Metrics: Worked Example and K=3 vs K=5 (generated report)",
        "",
        "> Generated by `activities/d-retrieval-metrics/metrics_demo.py`. Definitions: `evaluation/retrieval_metrics.py` "
        "and `02-Documentation/Day-13-Activity-D-Retrieval-Metrics.md`.",
        "",
    ]
    part1(L)
    ok = part2(L)
    report = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics-demo-report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'} (recomputed metrics identical to saved Phase 3/4 values)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
