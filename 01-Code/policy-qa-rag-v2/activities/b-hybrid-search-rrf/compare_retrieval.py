"""Activity B - dense vs BM25 vs hybrid (RRF, k=60) on all three chunking strategies.

Same corpus, same chunks per strategy, same embedding model, same 5 frozen development
queries and the same candidate depth (top-10) for every method. Reports top-3 per query,
strategy and method with chunk IDs, sections, ranks, method-specific scores and canonical
fact IDs; retrieval metrics at K=3 and K=5 under one relevance definition; an ablation
(dense only / BM25 only / hybrid); where hybrid helps or not; the v1 dense refusal gate; and
latency per component (measured separately, excluded from the reproducibility hash).

No reranking (Phase 4). No strategy selection (decided on the 30-query set in Phase 7).

Usage (project root):  .venv/bin/python activities/b-hybrid-search-rrf/compare_retrieval.py
Outputs: activities/b-hybrid-search-rrf/results/hybrid-retrieval-report.md (generated)
         activities/b-hybrid-search-rrf/results/hybrid-retrieval.json
No API key, no network.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evidence import CATALOG_PATH, chunk_evidence, load_catalog, sha256_file  # noqa: E402
from evaluation.metrics import mean_metrics, query_metrics  # noqa: E402
from rag_v2.bm25 import BM25_PARAMS  # noqa: E402
from rag_v2.chunking import STRATEGIES, load_corpus  # noqa: E402
from rag_v2.fusion import RRF_K  # noqa: E402
from rag_v2.hybrid import CANDIDATES, HybridSearcher  # noqa: E402
from rag_v2.indexing import build_index  # noqa: E402
from rag_v2.tokenizer import tokenize  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEV_QUERIES = PROJECT_ROOT / "evaluation" / "datasets" / "dev_queries.json"
PHASE2_JSON = PROJECT_ROOT / "activities" / "a-chunking-strategies" / "results" / "chunking-comparison.json"
MANIFESTS = PROJECT_ROOT / "manifests"
METHODS = ("dense", "bm25", "hybrid")
METHOD_LABEL = {"dense": "Dense only", "bm25": "BM25 only", "hybrid": "Hybrid (RRF k=60)"}
KS = (3, 5)
WARMUP, REPEATS = 3, 20
TOKENIZER_EXAMPLES = [
    "HR-003",
    "USD 1,000",
    "USD 0.40 per kilometre",
    "core hours 10:00 and 16:00",
    "work-from-home days",
    "public Wi-Fi",
    "hr-ops@brightlake.example",
    "IT Service Desk's extension",
    "It applies to all policies",
]


def sh_check(manifest: str) -> bool:
    return (
        subprocess.run(
            ["shasum", "-a", "256", "-c", "--quiet", f"manifests/{manifest}"], cwd=PROJECT_ROOT
        ).returncode
        == 0
    )


def manifest_hash(name: str) -> str:
    return (MANIFESTS / name).read_text().split()[0]


def pctl(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def score_label(method: str, hit) -> str:
    if method == "dense":
        return f"cos {hit.score:.4f}"
    if method == "bm25":
        return f"bm25 {hit.score:.3f}"
    src = "+".join(f"{'d' if n == 'dense' else 'b'}{r}" for n, r in hit.source_ranks.items())
    return f"rrf {hit.score:.5f} ({src})"


def compare(a, b) -> str:
    """Compare two QueryMetrics on (coverage, reciprocal rank)."""
    ka, kb = (a.coverage, a.reciprocal_rank), (b.coverage, b.reciprocal_rank)
    return "better" if ka > kb else "worse" if ka < kb else "same"


def main() -> int:
    started = datetime.now(UTC)
    settings = load_settings()
    docs = load_corpus(settings.policy_dir)
    catalog = load_catalog()
    units_by_policy: dict = {}
    for u in catalog.values():
        units_by_policy.setdefault(u.policy_id, []).append(u)
    queries = json.loads(DEV_QUERIES.read_text(encoding="utf-8"))["queries"]

    checks = [
        ("corpus matches manifests/corpus.sha256", sh_check("corpus.sha256"), "6 policy files"),
        ("frozen v1 code/tests match manifest", sh_check("v1-baseline-code.sha256"), "20 files"),
        (
            "Phase 2 artifacts unchanged (manifests/phase2-artifacts.sha256)",
            sh_check("phase2-artifacts.sha256"),
            "chunkers, analysis, catalog, dev queries, Phase 2 results",
        ),
        (
            "evidence catalog matches frozen hash",
            sha256_file(CATALOG_PATH) == manifest_hash("evidence-catalog.sha256"),
            "102 units",
        ),
        (
            "dev queries match frozen hash",
            sha256_file(DEV_QUERIES) == manifest_hash("dev-queries.sha256"),
            "5 queries",
        ),
    ]
    if not all(ok for _, ok, _ in checks):
        for n, ok, d in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")
        print("Integrity check failed - not running.", file=sys.stderr)
        return 1

    # ---- one-time costs (measured once; warm OS file cache)
    one_time = {}
    t0 = time.perf_counter()
    embedder = LocalEmbedder()
    one_time["embedding model load (ONNX MiniLM, from local cache)"] = (time.perf_counter() - t0) * 1000
    searchers, chunk_facts = {}, {}
    for name in STRATEGIES:
        t0 = time.perf_counter()
        index = build_index(name, docs, embedder, min_similarity=settings.min_similarity)
        one_time[f"{name}: dense index build (chunk + embed + Chroma insert)"] = (
            time.perf_counter() - t0
        ) * 1000
        searchers[name] = HybridSearcher(index, candidates=CANDIDATES, rrf_k=RRF_K)
        one_time[f"{name}: BM25 index build (tokenize + BM25Okapi)"] = searchers[name].bm25_build_ms
        chunk_facts[name] = {
            c.chunk_id: set(chunk_evidence(c.metadata, units_by_policy)) for c in index.chunks
        }

    # ---- consistency with Phase 2: dense top-3 must be identical
    phase2 = {q["id"]: q for q in json.loads(PHASE2_JSON.read_text(encoding="utf-8"))["dev_queries"]}

    # ---- retrieval
    per_query = []
    for q in queries:
        entry = {
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "query_tokens": tokenize(q["query"]),
            "required_evidence": q["required_evidence"],
            "strategies": {},
        }
        for name, s in searchers.items():
            r = s.search(q["query"])
            p2 = phase2[q["id"]]["results"][name]["top"]
            same = [h.chunk_id for h in r.dense[:3]] == [t["chunk_id"] for t in p2] and all(
                h.score == t["similarity"] for h, t in zip(r.dense[:3], p2, strict=True)
            )
            entry.setdefault("phase2_dense_identical", []).append(same)
            methods = {}
            for m in METHODS:
                ids = r.ranked_ids(m)
                full = query_metrics(ids, chunk_facts[name], q["required_evidence"], k=CANDIDATES)
                hits = getattr(r, m)
                methods[m] = {
                    "top": [
                        {
                            "rank": h.rank,
                            "chunk_id": h.chunk_id,
                            "score": h.score,
                            "score_type": {"dense": "cosine_similarity", "bm25": "bm25", "hybrid": "rrf"}[m],
                            "section": searchers[name].index.chunks_by_id[h.chunk_id].metadata["section"],
                            "facts": sorted(chunk_facts[name][h.chunk_id]),
                            "matched_terms": list(getattr(h, "matched_terms", ())),
                            "source_ranks": dict(getattr(h, "source_ranks", {})),
                        }
                        for h in hits
                    ],
                    "returned": len(hits),
                    "first_relevant_rank_in_candidates": full.first_relevant_rank,
                    "metrics": {
                        f"@{k}": query_metrics(ids, chunk_facts[name], q["required_evidence"], k).as_dict()
                        for k in KS
                    },
                }
            methods["dense_gate_passes"] = r.dense_gate_passes
            methods["best_dense_similarity"] = r.dense[0].score if r.dense else None
            entry["strategies"][name] = methods
        per_query.append(entry)
    checks.append(
        (
            "dense top-3 identical to Phase 2 (IDs and exact scores)",
            all(all(e["phase2_dense_identical"]) for e in per_query),
            f"{sum(sum(e['phase2_dense_identical']) for e in per_query)}/{len(per_query) * len(searchers)} "
            "query x strategy",
        )
    )

    # ---- aggregate ablation
    from evaluation.metrics import QueryMetrics  # local import keeps the top tidy

    def qm(d):
        return QueryMetrics(**d)

    ablation = {
        name: {
            m: {
                f"@{k}": mean_metrics([qm(e["strategies"][name][m]["metrics"][f"@{k}"]) for e in per_query])
                for k in KS
            }
            for m in METHODS
        }
        for name in searchers
    }
    verdicts = {
        name: {
            e["id"]: {
                "hybrid_vs_dense": compare(
                    qm(e["strategies"][name]["hybrid"]["metrics"]["@3"]),
                    qm(e["strategies"][name]["dense"]["metrics"]["@3"]),
                ),
                "hybrid_vs_bm25": compare(
                    qm(e["strategies"][name]["hybrid"]["metrics"]["@3"]),
                    qm(e["strategies"][name]["bm25"]["metrics"]["@3"]),
                ),
                "bm25_vs_dense": compare(
                    qm(e["strategies"][name]["bm25"]["metrics"]["@3"]),
                    qm(e["strategies"][name]["dense"]["metrics"]["@3"]),
                ),
            }
            for e in per_query
        }
        for name in searchers
    }

    # ---- latency benchmark (separate components)
    lat: dict = {
        name: {c: [] for c in ("dense_total", "query_embedding", "vector_search", "bm25", "fusion")}
        for name in searchers
    }
    for name, s in searchers.items():
        for q in queries:
            for i in range(WARMUP + REPEATS):
                r = s.search(q["query"])
                t0 = time.perf_counter()
                vec = embedder.embed_one(q["query"])
                t1 = time.perf_counter()
                s.index.store.query(vec, CANDIDATES)
                t2 = time.perf_counter()
                if i >= WARMUP:
                    lat[name]["dense_total"].append(r.timings_ms["dense"])
                    lat[name]["bm25"].append(r.timings_ms["bm25"])
                    lat[name]["fusion"].append(r.timings_ms["fusion"])
                    lat[name]["query_embedding"].append((t1 - t0) * 1000)
                    lat[name]["vector_search"].append((t2 - t1) * 1000)
    latency = {
        name: {
            c: {
                "median_ms": statistics.median(v),
                "p95_ms": pctl(v, 0.95),
                "mean_ms": statistics.mean(v),
                "n": len(v),
            }
            for c, v in comps.items()
        }
        for name, comps in lat.items()
    }

    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        cpu = platform.processor()
    run = {
        "started_utc": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "rank_bm25": metadata.version("rank-bm25"),
        "chromadb": metadata.version("chromadb"),
        "machine": f"{cpu} ({platform.machine()}), macOS {platform.mac_ver()[0]}",
    }
    results = {
        "inputs": {
            "corpus_manifest_sha256": sha256_file(MANIFESTS / "corpus.sha256"),
            "evidence_catalog_sha256": sha256_file(CATALOG_PATH),
            "dev_queries_sha256": sha256_file(DEV_QUERIES),
        },
        "settings": {
            "candidates": CANDIDATES,
            "rrf_k": RRF_K,
            "bm25": BM25_PARAMS,
            "ks": list(KS),
            "bm25_text": "title + chunk (v1 embedding_text)",
            "dense_gate": settings.min_similarity,
        },
        "integrity_checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
        "tokenizer_examples": {t: tokenize(t) for t in TOKENIZER_EXAMPLES},
        "per_query": per_query,
        "ablation": ablation,
        "verdicts": verdicts,
    }
    results_sha = hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = render(run, results, latency, one_time, catalog)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "hybrid-retrieval-report.md").write_text(report + "\n", encoding="utf-8")
    (RESULTS_DIR / "hybrid-retrieval.json").write_text(
        json.dumps(
            {
                "run": run,
                "results_sha256": results_sha,
                **results,
                "latency_not_hashed": {
                    "per_component": latency,
                    "one_time_ms": one_time,
                    "warmup": WARMUP,
                    "repeats_per_query": REPEATS,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Results SHA-256 (everything except run metadata and latency): {results_sha}")
    ok = all(c[1] for c in checks)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'} (integrity and Phase 2 consistency checks)")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- report
def f2(x: float) -> str:
    return f"{x:.2f}"


def render(run, res, latency, one_time, catalog) -> str:
    names = list(STRATEGIES)
    L = [
        "# Activity B – Dense vs BM25 vs Hybrid (RRF) Retrieval (generated report)",
        "",
        "> Generated by `activities/b-hybrid-search-rrf/compare_retrieval.py` from a real run. "
        "Do not edit by hand; re-run the script instead.",
        "",
        f"- Run (UTC): {run['started_utc']} · Python {run['python']} · rank-bm25 {run['rank_bm25']} · "
        f"chromadb {run['chromadb']} · {run['machine']}",
        "",
        "## 0. Integrity checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    L += [
        f"| {c['name']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |"
        for c in res["integrity_checks"]
    ]

    s = res["settings"]
    L += [
        "",
        "## 1. Methods and settings (identical for every strategy)",
        "",
        "| Method | What it does | Candidates | Score |",
        "|---|---|---:|---|",
        f"| Dense only | v1 `Retriever`: MiniLM query embedding + Chroma cosine search | {s['candidates']} | "
        "cosine similarity = 1 − distance (≈ −1…1) |",
        f"| BM25 only | `rank_bm25.BM25Okapi` (k1={s['bm25']['k1']}, b={s['bm25']['b']}, "
        f"ε={s['bm25']['epsilon']}, defaults) over {s['bm25_text']}; zero-score chunks not returned | "
        f"≤ {s['candidates']} | BM25 (unbounded, ≥ 0, corpus-dependent) |",
        f"| Hybrid | RRF of the dense top-{s['candidates']} and BM25 top-{s['candidates']} lists, k={s['rrf_k']} | "
        f"{s['candidates']} | RRF = Σ 1/(k + rank) (max 2/61 ≈ 0.0328) |",
        "",
        "**The three scores are on different scales and are never compared numerically.** Cosine similarity "
        "measures embedding closeness; BM25 measures weighted term overlap and has no upper bound; RRF only "
        "encodes the ranks a chunk received. Within one table, compare *ranks*. In hybrid rows, "
        "`(d2+b1)` means rank 2 in the dense list and rank 1 in the BM25 list.",
        "",
        "### Tokenizer (BM25) examples",
        "",
        "| Input | Tokens |",
        "|---|---|",
    ]
    L += [f"| `{t}` | `{' '.join(toks)}` |" for t, toks in res["tokenizer_examples"].items()]

    L += [
        "",
        "## 2. Relevance and metric definitions",
        "",
        "- **Relevant chunk**: fully contains at least one required canonical fact (evidence ID) of the query. "
        "Same rule for every metric, strategy and method.",
        "- **Hit@K**: a relevant chunk is in the top K. **RR@K**: 1/rank of the first relevant chunk in the top K "
        "(0 if none); **MRR** = mean RR over the 5 queries.",
        "- **P@K** (conventional): relevant chunks in top K ÷ K. **R@K** (conventional, chunk-level): relevant "
        "chunks in top K ÷ relevant chunks in the index — depends on the chunking (overlap duplicates enlarge "
        "it), so it is **not** comparable across strategies.",
        "- **Coverage@K** (evidence facts): required fact groups satisfied by the top-K chunks ÷ groups. Each "
        "fact counts once even if several retrieved chunks contain it (no double counting from overlap). "
        "This is the cross-strategy measure.",
        "",
    ]

    L += [
        "## 3. Per-query results (top-3)",
        "",
        "*Facts* lists canonical evidence IDs fully contained in the chunk (required ones ✅). "
        "*Notes*: matched query terms for BM25; source ranks for hybrid.",
        "",
    ]
    for e in res["per_query"]:
        groups = " AND ".join("(" + " OR ".join(f"`{x}`" for x in g) + ")" for g in e["required_evidence"])
        required = {x for g in e["required_evidence"] for x in g}
        L += [
            f"### {e['id']} – {e['query']}",
            "",
            f"Category: {e['category']} · Query tokens (BM25): `{' '.join(e['query_tokens'])}` · Required: {groups}",
            "",
        ]
        L += [f"- `{x}`: {catalog[x].text}" for g in e["required_evidence"] for x in g]
        L += [
            "",
            "| Strategy | Method | Rank | Chunk | Score | Section(s) | Facts | Notes |",
            "|---|---|---:|---|---|---|---|---|",
        ]
        for n in names:
            for m in METHODS:
                for h in e["strategies"][n][m]["top"][:3]:
                    facts = ", ".join(f"{x} ✅" if x in required else x for x in h["facts"]) or "—"
                    note = ("terms: " + ", ".join(h["matched_terms"])) if m == "bm25" else ""
                    L.append(
                        f"| `{n}` | {METHOD_LABEL[m]} | {h['rank']} | `{h['chunk_id']}` | "
                        f"{score_label(m, _Hit(h))} | {h['section']} | {facts} | {note} |"
                    )
                if not e["strategies"][n][m]["top"]:
                    L.append(
                        f"| `{n}` | {METHOD_LABEL[m]} | — | — | — | — | — | no chunk shares a query term |"
                    )
        L += [
            "",
            "| Strategy | Method | First relevant rank (top-10) | Hit@3 | RR@3 | P@3 | R@3 (rel./in index) | "
            "Coverage@3 | Coverage@5 |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
        for n in names:
            for m in METHODS:
                d = e["strategies"][n][m]
                m3, m5 = d["metrics"]["@3"], d["metrics"]["@5"]
                L.append(
                    f"| `{n}` | {METHOD_LABEL[m]} | {d['first_relevant_rank_in_candidates'] or '—'} | {m3['hit']} | "
                    f"{f2(m3['reciprocal_rank'])} | {f2(m3['precision'])} | {f2(m3['recall'])} "
                    f"({m3['relevant_retrieved']}/{m3['relevant_in_index']}) | "
                    f"{m3['groups_satisfied']}/{m3['groups_total']} | {m5['groups_satisfied']}/{m5['groups_total']} |"
                )
        L.append("")

    L += [
        "## 4. Ablation summary (mean over 5 development queries)",
        "",
        "| Strategy | Method | Hit@3 | MRR@3 | P@3 | R@3* | Coverage@3 (groups) | Hit@5 | MRR@5 | P@5 | R@5* | "
        "Coverage@5 (groups) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for n in names:
        for m in METHODS:
            a3, a5 = res["ablation"][n][m]["@3"], res["ablation"][n][m]["@5"]
            L.append(
                f"| `{n}` | {METHOD_LABEL[m]} | {f2(a3['hit_rate'])} | {f2(a3['mrr'])} | {f2(a3['precision'])} | "
                f"{f2(a3['recall'])} | {f2(a3['coverage'])} ({a3['groups_satisfied']}/{a3['groups_total']}) | "
                f"{f2(a5['hit_rate'])} | {f2(a5['mrr'])} | {f2(a5['precision'])} | {f2(a5['recall'])} | "
                f"{f2(a5['coverage'])} ({a5['groups_satisfied']}/{a5['groups_total']}) |"
            )
    L += [
        "",
        "\\* chunk-level recall; not comparable across strategies (see §2).",
        "",
        "## 5. Where hybrid helps and where it does not (top-3; compared on coverage, then reciprocal rank)",
        "",
        "| Strategy | Query | Hybrid vs dense | Hybrid vs BM25 | BM25 vs dense | First relevant rank: dense / BM25 / hybrid |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        for e in res["per_query"]:
            v = res["verdicts"][n][e["id"]]
            fr = " / ".join(
                str(e["strategies"][n][m]["first_relevant_rank_in_candidates"] or "—") for m in METHODS
            )
            L.append(
                f"| `{n}` | {e['id']} | {v['hybrid_vs_dense']} | {v['hybrid_vs_bm25']} | {v['bm25_vs_dense']} | {fr} |"
            )
    L += [
        "",
        "| Strategy | Hybrid vs dense (better / same / worse) | Hybrid vs BM25 (better / same / worse) |",
        "|---|---|---|",
    ]
    for n in names:
        vs = res["verdicts"][n].values()
        cnt = {
            k: [sum(v[k] == x for v in vs) for x in ("better", "same", "worse")]
            for k in ("hybrid_vs_dense", "hybrid_vs_bm25")
        }
        L.append(
            f"| `{n}` | {' / '.join(map(str, cnt['hybrid_vs_dense']))} | {' / '.join(map(str, cnt['hybrid_vs_bm25']))} |"
        )

    L += [
        "",
        "## 6. v1 refusal gate (dense similarity only, unchanged)",
        "",
        "Evaluated on the dense list alone: best cosine similarity ≥ 0.25. BM25 and RRF scores are not "
        "compared with this threshold and no new threshold is introduced.",
        "",
        "| Query | " + " | ".join(f"`{n}` best cos (gate)" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for e in res["per_query"]:
        L.append(
            f"| {e['id']} | "
            + " | ".join(
                f"{e['strategies'][n]['best_dense_similarity']:.4f} ({'pass' if e['strategies'][n]['dense_gate_passes'] else 'refuse'})"
                for n in names
            )
            + " |"
        )

    L += [
        "",
        "## 7. Latency (measured, not part of the reproducibility hash)",
        "",
        f"Per strategy: 5 queries × {REPEATS} timed repetitions after {WARMUP} warm-up runs (n = {5 * REPEATS} per "
        "cell), single process, CPU. *Dense total* = query embedding + vector search (+ result conversion); "
        "the two sub-components were timed in separate calls.",
        "",
        "| Strategy | Component | Median ms | p95 ms | Mean ms |",
        "|---|---|---:|---:|---:|",
    ]
    comp_label = {
        "dense_total": "Dense retrieval total",
        "query_embedding": "  └ query embedding (MiniLM)",
        "vector_search": "  └ vector search (Chroma)",
        "bm25": "BM25 (tokenize + score + sort)",
        "fusion": "RRF fusion",
    }
    for n in names:
        for c, lab in comp_label.items():
            v = latency[n][c]
            L.append(f"| `{n}` | {lab} | {v['median_ms']:.3f} | {v['p95_ms']:.3f} | {v['mean_ms']:.3f} |")
    L += ["", "One-time costs (single measurement each):", "", "| Item | ms |", "|---|---:|"]
    L += [f"| {k} | {v:.1f} |" for k, v in one_time.items()]
    L += [
        "",
        "## 8. Interpretation limits",
        "",
        "Five development queries are a tuning sample: a difference of one query is not evidence that a method "
        "or strategy is better. **No chunking strategy and no retrieval method is selected here.** Phase 4 "
        "adds reranking on the same candidates; the final configuration is chosen on the frozen 30-query "
        "evaluation (Phase 7).",
        "",
    ]
    return "\n".join(L)


class _Hit:
    """Adapter so ``score_label`` can format a saved row."""

    def __init__(self, row: dict) -> None:
        self.score = row["score"]
        self.source_ranks = row["source_ranks"]


if __name__ == "__main__":
    sys.exit(main())
