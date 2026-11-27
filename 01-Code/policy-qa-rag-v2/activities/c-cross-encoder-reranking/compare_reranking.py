"""Activity C - cross-encoder reranking of the hybrid top-10, for all three chunking strategies.

Before = Phase 3 hybrid list (RRF k=60 of dense top-10 + BM25 top-10), top-10 candidate pool.
After  = the same 10 chunks re-ordered by cross-encoder/ms-marco-MiniLM-L6-v2 (pinned revision).
Reports top-3 before/after for the 5 frozen dev queries, Hit/P/R/MRR/Coverage at K=3 and 5 and
pool coverage (K=10) under the Phase 3 relevance definition, how each required fact moved,
which facts were absent from the pool (unrecoverable by reranking), the unchanged v1 dense
gate, and latency per stage (one-time costs reported separately).

No generator, prompt, refusal threshold or context-size change; no LLM calls.
No strategy or method is selected (decided on the 30-query evaluation in Phase 7).

Usage (project root):  HF_HUB_OFFLINE=1 .venv/bin/python activities/c-cross-encoder-reranking/compare_reranking.py
Requires the model in the local cache (scripts/download_cross_encoder.py, once).
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
from evaluation.metrics import QueryMetrics, mean_metrics, query_metrics  # noqa: E402
from rag_v2.chunking import STRATEGIES, load_corpus  # noqa: E402
from rag_v2.hybrid import CANDIDATES, HybridSearcher  # noqa: E402
from rag_v2.indexing import build_index  # noqa: E402
from rag_v2.rerank_search import RerankingSearcher  # noqa: E402
from rag_v2.reranker import MAX_LENGTH, MODEL_ID, MODEL_REVISION, REQUESTED_MODEL_ID, CrossEncoderReranker  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEV_QUERIES = PROJECT_ROOT / "evaluation" / "datasets" / "dev_queries.json"
PHASE3_JSON = PROJECT_ROOT / "activities" / "b-hybrid-search-rrf" / "results" / "hybrid-retrieval.json"
MODEL_INFO = PROJECT_ROOT / "results" / "cross-encoder-model-info.json"
MANIFESTS = PROJECT_ROOT / "manifests"
STAGES = ("before", "after")
STAGE_LABEL = {"before": "Hybrid (RRF) – before", "after": "Hybrid + cross-encoder – after"}
KS = (3, 5)
WARMUP, REPEATS = 3, 20


def sh_check(manifest: str) -> bool:
    return (
        subprocess.run(
            ["shasum", "-a", "256", "-c", "--quiet", f"manifests/{manifest}"], cwd=PROJECT_ROOT
        ).returncode
        == 0
    )


def manifest_hash(name: str) -> str:
    return (MANIFESTS / name).read_text().split()[0]


def pctl(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def rounded(obj, nd=4):
    """Copy with floats rounded (used only for the reproducibility hash: cross-encoder float noise ~1e-6)."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: rounded(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, nd) for v in obj]
    return obj


def first_rank(ids, chunk_facts, eid):
    return next((i for i, cid in enumerate(ids, start=1) if eid in chunk_facts[cid]), None)


def verdict(a: QueryMetrics, b: QueryMetrics) -> str:
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
    model_info = json.loads(MODEL_INFO.read_text(encoding="utf-8"))

    checks = [
        ("corpus matches manifest", sh_check("corpus.sha256"), "6 policy files"),
        ("frozen v1 code/tests match manifest", sh_check("v1-baseline-code.sha256"), "20 files"),
        ("Phase 2 artifacts unchanged", sh_check("phase2-artifacts.sha256"), "14 files"),
        ("Phase 3 artifacts unchanged", sh_check("phase3-artifacts.sha256"), "13 files"),
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
        (
            "cross-encoder revision = pinned revision",
            model_info["revision"] == MODEL_REVISION,
            MODEL_REVISION[:12],
        ),
    ]
    if not all(ok for _, ok, _ in checks):
        for n, ok, d in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")
        print("Integrity check failed - not running.", file=sys.stderr)
        return 1

    # ---- one-time costs
    one_time = {
        "cross-encoder download (separate run, scripts/download_cross_encoder.py)": model_info[
            "download_seconds_this_run"
        ]
        * 1000
    }
    t0 = time.perf_counter()
    import sentence_transformers  # noqa: F401  (first import in this process: torch + transformers)

    one_time["sentence-transformers import (first in process; loads torch/transformers)"] = (
        time.perf_counter() - t0
    ) * 1000
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    one_time["cross-encoder load from local cache (CPU)"] = (time.perf_counter() - t0) * 1000
    import torch

    model = reranker.model
    embedder = LocalEmbedder()
    searchers, chunk_facts = {}, {}
    for name in STRATEGIES:
        index = build_index(name, docs, embedder, min_similarity=settings.min_similarity)
        searchers[name] = RerankingSearcher(HybridSearcher(index), reranker)
        chunk_facts[name] = {
            c.chunk_id: set(chunk_evidence(c.metadata, units_by_policy)) for c in index.chunks
        }
    in_index = {name: set().union(*cf.values()) for name, cf in chunk_facts.items()}

    # ---- warm-up (model + caches) before anything is timed or compared
    for s in searchers.values():
        for q in queries:
            s.search(q["query"])

    phase3 = {e["id"]: e for e in json.loads(PHASE3_JSON.read_text(encoding="utf-8"))["per_query"]}
    longest_pair = 0
    per_query, movements = [], []
    same_pool = 0
    for q in queries:
        groups = q["required_evidence"]
        required = [e for g in groups for e in g]
        entry = {
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "required_evidence": groups,
            "strategies": {},
        }
        for name, s in searchers.items():
            r = s.search(q["query"])
            p3 = phase3[q["id"]]["strategies"][name]["hybrid"]["top"]
            same_pool += [h.chunk_id for h in r.hybrid.hybrid] == [t["chunk_id"] for t in p3] and all(
                h.score == t["score"] for h, t in zip(r.hybrid.hybrid, p3, strict=True)
            )
            for c in s.candidates(r.hybrid):
                longest_pair = max(longest_pair, len(model.tokenizer(q["query"], c.text)["input_ids"]))
            lists = {"before": r.pool_ids, "after": r.reranked_ids}
            if sorted(lists["before"]) != sorted(lists["after"]):
                raise RuntimeError("reranking changed the candidate pool")  # contract violation
            chunks = s.hybrid.index.chunks_by_id
            rows = {
                "before": [
                    {
                        "rank": h.rank,
                        "chunk_id": h.chunk_id,
                        "score": h.score,
                        "score_type": "rrf",
                        "source_ranks": h.source_ranks,
                    }
                    for h in r.hybrid.hybrid
                ],
                "after": [
                    {
                        "rank": h.rank,
                        "chunk_id": h.chunk_id,
                        "score": h.score,
                        "score_type": "cross_encoder_logit",
                        "prior_rank": h.prior_rank,
                    }
                    for h in r.reranked
                ],
            }
            for st in STAGES:
                for row in rows[st]:
                    row["section"] = chunks[row["chunk_id"]].metadata["section"]
                    row["facts"] = sorted(chunk_facts[name][row["chunk_id"]])
            metrics = {
                st: {
                    f"@{k}": query_metrics(lists[st], chunk_facts[name], groups, k).as_dict()
                    for k in (*KS, CANDIDATES)
                }
                for st in STAGES
            }
            for e in required:
                pre, post = (
                    first_rank(lists["before"], chunk_facts[name], e),
                    first_rank(lists["after"], chunk_facts[name], e),
                )
                if e not in in_index[name]:
                    status = "split – in no chunk of this index"
                elif pre is None:
                    status = "absent from pool – not recoverable by reranking"
                else:
                    status = "moved up" if post < pre else "moved down" if post > pre else "unchanged"
                top3 = (
                    "entered top-3"
                    if pre and pre > 3 and post <= 3
                    else "left top-3"
                    if pre and pre <= 3 and post > 3
                    else ""
                )
                movements.append(
                    {
                        "strategy": name,
                        "query": q["id"],
                        "evidence_id": e,
                        "status": status,
                        "pool_rank_before": pre,
                        "rank_after": post,
                        "top3_change": top3,
                        "dense_rank": first_rank(r.hybrid.ranked_ids("dense"), chunk_facts[name], e),
                        "bm25_rank": first_rank(r.hybrid.ranked_ids("bm25"), chunk_facts[name], e),
                    }
                )
            entry["strategies"][name] = {
                "rows": rows,
                "metrics": metrics,
                "dense_gate_passes": r.hybrid.dense_gate_passes,
                "best_dense_similarity": r.hybrid.dense[0].score,
            }
        per_query.append(entry)
    n_pairs = len(queries) * len(searchers)
    checks.append(
        (
            "candidate pool = Phase 3 hybrid top-10 (IDs and exact RRF scores)",
            same_pool == n_pairs,
            f"{same_pool}/{n_pairs} query x strategy",
        )
    )
    checks.append(
        (
            "reranked list is a permutation of the pool (no chunk added or removed)",
            True,
            f"{n_pairs}/{n_pairs} (enforced at run time)",
        )
    )
    checks.append(
        (
            f"no (query, chunk) pair exceeds max_length {MAX_LENGTH} (no truncation)",
            longest_pair <= MAX_LENGTH,
            f"longest pair {longest_pair} tokens",
        )
    )

    def qm(d):
        return QueryMetrics(**d)

    summary = {
        name: {
            st: {
                f"@{k}": mean_metrics([qm(e["strategies"][name]["metrics"][st][f"@{k}"]) for e in per_query])
                for k in (*KS, CANDIDATES)
            }
            for st in STAGES
        }
        for name in searchers
    }
    verdicts = {
        name: {
            e["id"]: verdict(
                qm(e["strategies"][name]["metrics"]["after"]["@3"]),
                qm(e["strategies"][name]["metrics"]["before"]["@3"]),
            )
            for e in per_query
        }
        for name in searchers
    }

    # ---- latency (model already warm)
    lat = {
        name: {c: [] for c in ("dense", "bm25", "fusion", "hybrid_total", "rerank", "total")}
        for name in searchers
    }
    for name, s in searchers.items():
        for q in queries:
            for i in range(WARMUP + REPEATS):
                r = s.search(q["query"])
                if i >= WARMUP:
                    for c in lat[name]:
                        lat[name][c].append(r.timings_ms[c])
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

    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    run = {
        "started_utc": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sentence_transformers": metadata.version("sentence-transformers"),
        "torch": metadata.version("torch"),
        "transformers": metadata.version("transformers"),
        "torch_threads": torch.get_num_threads(),
        "machine": f"{cpu} ({platform.machine()}), macOS {platform.mac_ver()[0]}",
    }
    results = {
        "inputs": {
            "corpus_manifest_sha256": sha256_file(MANIFESTS / "corpus.sha256"),
            "evidence_catalog_sha256": sha256_file(CATALOG_PATH),
            "dev_queries_sha256": sha256_file(DEV_QUERIES),
        },
        "model": {
            "requested_model_id": REQUESTED_MODEL_ID,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "architecture": model_info["config"]["architectures"],
            "layers": model_info["config"]["num_hidden_layers"],
            "hidden_size": model_info["config"]["hidden_size"],
            "max_length": MAX_LENGTH,
            "num_labels": model.num_labels,
            "activation_fn": type(model.activation_fn).__name__,
            "device": str(model.device),
            "batch": "whole candidate pool in one batch",
            "pair_format": "(query, policy title + chunk text)",
            "longest_pair_tokens": longest_pair,
            "download_bytes": model_info["download_bytes"],
        },
        "settings": {
            "candidate_pool": CANDIDATES,
            "rrf_k": 60,
            "ks": list(KS),
            "dense_gate": settings.min_similarity,
        },
        "integrity_checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
        "per_query": per_query,
        "summary": summary,
        "verdicts": verdicts,
        "fact_movements": movements,
    }
    results_sha = hashlib.sha256(
        json.dumps(rounded(results), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    report = render(run, results, latency, one_time, catalog)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "reranking-report.md").write_text(report + "\n", encoding="utf-8")
    (RESULTS_DIR / "reranking.json").write_text(
        json.dumps(
            {
                "run": run,
                "results_sha256_floats_rounded_4dp": results_sha,
                **results,
                "latency_not_hashed": {
                    "per_stage": latency,
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
    print(f"Results SHA-256 (floats rounded to 4 dp; excludes run metadata and latency): {results_sha}")
    ok = all(c[1] for c in checks)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'} (integrity, same-pool and no-truncation checks)")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- report
def f2(x):
    return f"{x:.2f}"


def render(run, res, latency, one_time, catalog) -> str:
    names = list(STRATEGIES)
    m = res["model"]
    L = [
        "# Activity C – Cross-Encoder Reranking of the Hybrid Top-10 (generated report)",
        "",
        "> Generated by `activities/c-cross-encoder-reranking/compare_reranking.py` from a real run. "
        "Do not edit by hand; re-run the script instead.",
        "",
        f"- Run (UTC): {run['started_utc']} · Python {run['python']} · sentence-transformers "
        f"{run['sentence_transformers']} · torch {run['torch']} · transformers {run['transformers']} · "
        f"{run['torch_threads']} CPU threads · {run['machine']}",
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
    L += [
        "",
        "## 1. Model and settings",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Requested model ID (plan) | `{m['requested_model_id']}` (Hub alias) |",
        f"| Resolved / loaded model ID | `{m['model_id']}` |",
        f"| Pinned revision (Hub commit) | `{m['revision']}` |",
        f"| Architecture | {', '.join(m['architecture'])}, {m['layers']} layers, hidden size {m['hidden_size']} |",
        f"| Output | {m['num_labels']} label, activation `{m['activation_fn']}` → raw relevance logit |",
        f"| Input | {m['pair_format']}; max_length {m['max_length']}; longest pair {m['longest_pair_tokens']} tokens "
        "(no truncation) |",
        f"| Inference | {m['device']}, {m['batch']} (fixed batch → deterministic) |",
        f"| Candidate pool | hybrid top-{res['settings']['candidate_pool']} (RRF k={res['settings']['rrf_k']} of dense "
        "top-10 + BM25 top-10), identical to Phase 3 |",
        "",
        "**Score types are kept separate.** *RRF* (before) only encodes the ranks a chunk got from dense and BM25 "
        "(≈ 0.014–0.033). The *cross-encoder logit* (after) is the model's relevance estimate for one "
        "(query, chunk) pair; it is unbounded, can be negative, and is used **only to order the 10 candidates**. "
        "Neither score is a cosine similarity: the v1 refusal gate stays on the dense similarity (≥ "
        f"{res['settings']['dense_gate']}), and no cross-encoder threshold is introduced.",
        "",
        "## 2. Definitions (same relevance rule as Phase 3)",
        "",
        "- **Relevant chunk**: fully contains at least one required canonical fact. Hit@K, RR@K/MRR, P@K, R@K "
        "(chunk-level; not comparable across strategies) and **Coverage@K** (fact groups satisfied; each fact "
        "counted once) as in Phase 3.",
        "- **Pool coverage** = Coverage@10 of the candidate pool. Reranking only re-orders the pool, so pool "
        "coverage is identical before and after; it is the upper bound for what reranking can deliver.",
        "",
        "## 3. Per-query top-3 before and after reranking",
        "",
        "*Prior* = the chunk's rank in the hybrid list before reranking. *Facts* = canonical evidence IDs fully "
        "contained (required ✅).",
        "",
    ]
    for e in res["per_query"]:
        required = {x for g in e["required_evidence"] for x in g}
        groups = " AND ".join("(" + " OR ".join(f"`{x}`" for x in g) + ")" for g in e["required_evidence"])
        L += [f"### {e['id']} – {e['query']}", "", f"Category: {e['category']} · Required: {groups}", ""]
        L += [f"- `{x}`: {catalog[x].text}" for g in e["required_evidence"] for x in g]
        L += [
            "",
            "| Strategy | Stage | Rank | Chunk | Score | Prior | Section(s) | Facts |",
            "|---|---|---:|---|---|---:|---|---|",
        ]
        for n in names:
            for st in STAGES:
                for row in e["strategies"][n]["rows"][st][:3]:
                    facts = ", ".join(f"{x} ✅" if x in required else x for x in row["facts"]) or "—"
                    if st == "before":
                        src = "+".join(
                            f"{'d' if k == 'dense' else 'b'}{v}" for k, v in row["source_ranks"].items()
                        )
                        score, prior = f"rrf {row['score']:.5f} ({src})", "—"
                    else:
                        score, prior = f"ce {row['score']:.3f}", str(row["prior_rank"])
                    L.append(
                        f"| `{n}` | {STAGE_LABEL[st]} | {row['rank']} | `{row['chunk_id']}` | {score} | {prior} | "
                        f"{row['section']} | {facts} |"
                    )
        L += [
            "",
            "| Strategy | Stage | Hit@3 | RR@3 | P@3 | R@3 (rel./in index) | Coverage@3 | Coverage@5 | Pool coverage (@10) |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for n in names:
            for st in STAGES:
                mm = e["strategies"][n]["metrics"][st]
                a, b, p = mm["@3"], mm["@5"], mm[f"@{CANDIDATES}"]
                L.append(
                    f"| `{n}` | {STAGE_LABEL[st]} | {a['hit']} | {f2(a['reciprocal_rank'])} | {f2(a['precision'])} | "
                    f"{f2(a['recall'])} ({a['relevant_retrieved']}/{a['relevant_in_index']}) | "
                    f"{a['groups_satisfied']}/{a['groups_total']} | {b['groups_satisfied']}/{b['groups_total']} | "
                    f"{p['groups_satisfied']}/{p['groups_total']} |"
                )
        L.append("")

    L += [
        "## 4. Summary (mean over 5 development queries)",
        "",
        "| Strategy | Stage | Hit@3 | MRR@3 | P@3 | R@3* | Coverage@3 | Hit@5 | MRR@5 | P@5 | R@5* | Coverage@5 | Pool coverage |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for n in names:
        for st in STAGES:
            a, b, p = (res["summary"][n][st][f"@{k}"] for k in (3, 5, CANDIDATES))
            L.append(
                f"| `{n}` | {STAGE_LABEL[st]} | {f2(a['hit_rate'])} | {f2(a['mrr'])} | {f2(a['precision'])} | "
                f"{f2(a['recall'])} | {f2(a['coverage'])} ({a['groups_satisfied']}/{a['groups_total']}) | "
                f"{f2(b['hit_rate'])} | {f2(b['mrr'])} | {f2(b['precision'])} | {f2(b['recall'])} | "
                f"{f2(b['coverage'])} ({b['groups_satisfied']}/{b['groups_total']}) | "
                f"{p['groups_satisfied']}/{p['groups_total']} |"
            )
    L += [
        "",
        "\\* chunk-level recall; not comparable across strategies.",
        "",
        "## 5. How each required fact moved",
        "",
        "Rank = first rank of a chunk containing the fact. *Dense / BM25 rank* = its rank in the two lists that "
        "were fused (— = not in that top-10).",
        "",
        "| Strategy | Query | Fact | Status | Pool rank before → after | Top-3 change | Dense rank | BM25 rank |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for mv in res["fact_movements"]:
        L.append(
            f"| `{mv['strategy']}` | {mv['query']} | `{mv['evidence_id']}` | {mv['status']} | "
            f"{mv['pool_rank_before'] or '—'} → {mv['rank_after'] or '—'} | {mv['top3_change'] or '—'} | "
            f"{mv['dense_rank'] or '—'} | {mv['bm25_rank'] or '—'} |"
        )
    cnt: dict = {}
    for mv in res["fact_movements"]:
        cnt[mv["status"]] = cnt.get(mv["status"], 0) + 1
    L += [
        "",
        "Totals over all strategy × query × required-fact rows: "
        + "; ".join(f"{k}: {v}" for k, v in sorted(cnt.items())),
        "",
        "## 6. Effect of reranking per query (top-3; coverage, then reciprocal rank)",
        "",
        "| Strategy | " + " | ".join(e["id"] for e in res["per_query"]) + " | better / same / worse |",
        "|---|" + "---|" * (len(res["per_query"]) + 1),
    ]
    for n in names:
        v = res["verdicts"][n]
        L.append(
            f"| `{n}` | "
            + " | ".join(v[e["id"]] for e in res["per_query"])
            + " | "
            + " / ".join(str(sum(x == k for x in v.values())) for k in ("better", "same", "worse"))
            + " |"
        )

    L += [
        "",
        "## 7. v1 refusal gate (dense similarity only, unchanged)",
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
        "## 8. Latency (model warm; not part of the reproducibility hash)",
        "",
        f"Per strategy: 5 queries × {REPEATS} timed searches after {WARMUP} warm-up searches per query and a full "
        "warm-up pass (n = 100 per cell). CPU only. *Hybrid* = dense + BM25 + RRF; *Rerank* = pair formatting + "
        "cross-encoder scoring of 10 pairs + sort; *Total* = hybrid + rerank in the same call.",
        "",
        "| Strategy | Stage | Median ms | p95 ms | Mean ms |",
        "|---|---|---:|---:|---:|",
    ]
    label = {
        "dense": "  dense retrieval",
        "bm25": "  BM25",
        "fusion": "  RRF fusion",
        "hybrid_total": "Hybrid retrieval (no reranking)",
        "rerank": "Cross-encoder reranking (10 pairs)",
        "total": "Hybrid + reranking (total)",
    }
    for n in names:
        for c in ("hybrid_total", "dense", "bm25", "fusion", "rerank", "total"):
            v = latency[n][c]
            L.append(
                f"| `{n}` | {label[c]} | {v['median_ms']:.3f} | {v['p95_ms']:.3f} | {v['mean_ms']:.3f} |"
            )
    L += ["", "| Strategy | Added by reranking (median) | Total ÷ hybrid (median) |", "|---|---:|---:|"]
    for n in names:
        h, t = latency[n]["hybrid_total"]["median_ms"], latency[n]["total"]["median_ms"]
        L.append(f"| `{n}` | +{latency[n]['rerank']['median_ms']:.1f} ms | {t / h:.2f}× |")
    L += ["", "One-time costs (single measurements, not included above):", "", "| Item | ms |", "|---|---:|"]
    L += [f"| {k} | {v:.1f} |" for k, v in one_time.items()]
    L += [
        "",
        "## 9. Interpretation limits",
        "",
        "Five development queries, four of which are already at rank 1 before reranking: differences of one query "
        "are not evidence. **No chunking strategy or retrieval configuration is selected here**; the decision is "
        "made on the frozen 30-query held-out evaluation (Phase 7).",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
