"""Phase 7 - RAG v1 vs v2 on the frozen evaluation set v1.1, following evaluation/phase7/preregistration.json.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_phase7_eval.py retrieval            # Part A: 18 configs, no LLM
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_phase7_eval.py answers [--resume]   # Part B: 3 configs, real LLM
    .venv/bin/python scripts/run_phase7_eval.py report                                 # report from saved results

Integrity: refuses to run unless the pre-registration, dataset v1.1, v1.0, Phase 2-6 artifacts and the frozen v1
code match their manifests. EV-06 (exposed) is never part of the totals; Part A reports it separately.
Part B saves every answer immediately (results/phase7/answers.jsonl), resumes, reuses the response cache, stops
on authentication failure or quota exhaustion (status incomplete), and never exceeds the registered call budget.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import groq  # noqa: E402

from evaluation.answer_eval import evaluate_answer  # noqa: E402
from evaluation.evidence import chunk_evidence, load_catalog  # noqa: E402
from evaluation.metrics import query_metrics  # noqa: E402
from evaluation.retrieval_metrics import compare_k, evaluate_ranking  # noqa: E402
from rag_v2.chunking import STRATEGIES, load_corpus  # noqa: E402
from rag_v2.config_v2 import RAGv2Config, all_configs  # noqa: E402
from rag_v2.indexing import build_index  # noqa: E402
from rag_v2.llm_cache import CachingLLM  # noqa: E402
from rag_v2.pipeline_v2 import RAGPipelineV2  # noqa: E402
from rag_v2.reranker import CrossEncoderReranker  # noqa: E402
from src.config import ConfigurationError, load_settings  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402
from src.generation import GenerationError, GroqGenerator, QuotaExhaustedError  # noqa: E402

D = PROJECT_ROOT / "evaluation" / "datasets"
PREREG = PROJECT_ROOT / "evaluation" / "phase7" / "preregistration.json"
OUT = PROJECT_ROOT / "results" / "phase7"
MANIFESTS = [
    "phase7-preregistration.sha256",
    "eval-dataset-v1_1.sha256",
    "eval-dataset.sha256",
    "corpus.sha256",
    "v1-baseline-code.sha256",
    "phase2-artifacts.sha256",
    "phase3-artifacts.sha256",
    "phase4-artifacts.sha256",
    "phase5-artifacts.sha256",
    "phase6-artifacts.sha256",
    "evidence-catalog.sha256",
]


def integrity() -> None:
    bad = [
        m
        for m in MANIFESTS
        if subprocess.run(
            ["shasum", "-a", "256", "-c", "--quiet", f"manifests/{m}"], cwd=PROJECT_ROOT
        ).returncode
    ]
    if bad:
        raise SystemExit(f"integrity check failed: {bad}")


def load_sets():
    clean = json.loads((D / "eval_queries_v1_1.json").read_text(encoding="utf-8"))["queries"]
    assert len(clean) == 30 and all(q["id"] != "EV-06" for q in clean)
    exposed = [
        e["query"]
        for e in json.loads((D / "eval_queries_exposed.json").read_text(encoding="utf-8"))["queries"]
    ]
    return clean, exposed


def facts_by_chunking(docs):
    by_policy: dict = {}
    for u in load_catalog().values():
        by_policy.setdefault(u.policy_id, []).append(u)
    return {
        n: {c.chunk_id: set(chunk_evidence(c.metadata, by_policy)) for c in s.chunk_documents(docs)}
        for n, s in STRATEGIES.items()
    }


def smean(v):
    return statistics.mean(v) if v else None


def median(v):
    return statistics.median(v) if v else None


# ----------------------------------------------------------------------------- Part A
def part_retrieval() -> int:
    settings = load_settings()
    clean, exposed = load_sets()
    docs = load_corpus(settings.policy_dir)
    facts = facts_by_chunking(docs)
    embedder, reranker = LocalEmbedder(), CrossEncoderReranker()
    indexes = {n: build_index(n, docs, embedder) for n in STRATEGIES}
    out = {"run_utc": datetime.now(UTC).isoformat(timespec="seconds"), "configs": {}}
    for cfg in all_configs():
        pipe = RAGPipelineV2(
            settings,
            cfg,
            llm=None,
            embedder=embedder,
            index=indexes[cfg.chunking],
            reranker=reranker if cfg.rerank else None,
        )
        rows, evals, sens = [], [], []
        for q in clean + exposed:
            r = pipe.retrieve(q["question"])
            row = {
                "id": q["id"],
                "answerable": q["answerable"],
                "category": q["category"],
                "final_ranking": r.final_ranking,
                "context": [c.chunk_id for c in r.context],
                "dropped_by_dense_filter": r.dropped_by_dense_filter,
                "best_dense": r.best_dense_similarity,
                "gate_passes": r.gate_passes,
                "timings_ms": r.timings_ms,
            }
            if q["answerable"]:
                ctx = query_metrics(
                    row["context"], facts[cfg.chunking], q["required_evidence"], k=cfg.context_k
                )
                row["context_coverage"] = ctx.coverage
                row["dropped_required"] = sorted(
                    {e for cid in r.dropped_by_dense_filter for e in facts[cfg.chunking][cid]}
                    & {e for g in q["required_evidence"] for e in g}
                )
            ev = evaluate_ranking(
                q["id"], q["answerable"], q["required_evidence"], r.final_ranking, facts[cfg.chunking], (3, 5)
            )
            row["metrics"] = {str(k): m.as_dict() for k, m in (ev.metrics or {}).items()}
            (sens if q["id"] == "EV-06" else rows).append(row)
            if q["id"] != "EV-06":
                evals.append(ev)
        ans = [r for r in rows if r["answerable"]]
        out["configs"][cfg.name] = {
            "aggregate": {str(k): v for k, v in compare_k(evals, (3, 5)).items()},
            "context_coverage_mean": statistics.mean(r["context_coverage"] for r in ans),
            "context_full_coverage_queries": sum(r["context_coverage"] == 1.0 for r in ans),
            "required_fact_dropped_by_dense_filter": sum(bool(r["dropped_required"]) for r in ans),
            "unanswerable_refused_by_dense_gate": sum(
                not r["gate_passes"] for r in rows if not r["answerable"]
            ),
            "answerable_refused_by_dense_gate": sum(not r["gate_passes"] for r in ans),
            "median_ms": {
                s: median([r["timings_ms"][s] for r in rows if s in r["timings_ms"]])
                for s in ("query_embedding", "vector_search", "bm25", "fusion", "rerank", "retrieval_total")
            },
            "per_query": rows,
            "ev06_sensitivity": sens,
        }
        a3 = out["configs"][cfg.name]["aggregate"]["3"]
        print(
            f"{cfg.name:<40} Hit@3 {a3['hit_rate']:.2f} MRR@3 {a3['mrr']:.2f} Cov@3 {a3['coverage_macro']:.2f} "
            f"ctx-cov {out['configs'][cfg.name]['context_coverage_mean']:.2f}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT / 'retrieval.json'}")
    return 0


# ----------------------------------------------------------------------------- Part B
def part_answers(resume: bool, delay: float) -> int:
    prereg = json.loads(PREREG.read_text())
    configs = [RAGv2Config.from_name(n) for n in prereg["part_b_answers"]["configs"]]
    budget = prereg["part_b_answers"]["max_llm_calls"]
    settings = load_settings()
    try:
        key = settings.require_groq_key()
    except ConfigurationError as exc:
        print(f"Missing credentials: {exc}", file=sys.stderr)
        return 3
    llm = CachingLLM(
        GroqGenerator(api_key=key, model=settings.llm_model, temperature=settings.llm_temperature),
        PROJECT_ROOT / "results" / "llm-cache",
    )
    del key
    clean, _ = load_sets()
    docs = load_corpus(settings.policy_dir)
    facts = facts_by_chunking(docs)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "answers.jsonl"
    if path.exists() and not resume:
        print("answers.jsonl exists; use --resume", file=sys.stderr)
        return 1
    done = set()
    if path.exists():
        latest = {(r["config"], r["id"]): r for r in map(json.loads, path.read_text().splitlines())}
        done = {k for k, r in latest.items() if "error" not in r}
    embedder, reranker = LocalEmbedder(), None
    status = "complete"
    for cfg in configs:
        if status != "complete":
            break
        pipe = RAGPipelineV2(settings, cfg, llm=llm, embedder=embedder, reranker=reranker)
        reranker = pipe.reranker or reranker
        for q in clean:
            if (cfg.name, q["id"]) in done:
                continue
            if llm.real_calls >= budget:
                status = f"incomplete: registered call budget {budget} reached"
                break
            before = llm.real_calls
            try:
                a = pipe.ask(q["question"])
            except QuotaExhaustedError as exc:
                status = f"incomplete: provider quota exhausted ({exc})"
                break
            except GenerationError as exc:
                rec = {"config": cfg.name, "id": q["id"], "error": str(exc)}
                if isinstance(exc.__cause__, groq.AuthenticationError):
                    status = "incomplete: authentication failed"
            else:
                ev = evaluate_answer(
                    answerable=q["answerable"],
                    status=a.status,
                    answer_text=a.raw_llm_answer or a.answer,
                    citations=a.citations,
                    context_chunks=a.retrieval.context,
                    answer_terms=q["answer_terms"],
                    required_evidence=q["required_evidence"],
                    expected_policy_ids=q["expected_policy_ids"],
                    chunk_facts=facts[cfg.chunking],
                )
                rec = {
                    "config": cfg.name,
                    "id": q["id"],
                    "answerable": q["answerable"],
                    "category": q["category"],
                    "status": a.status,
                    "answer": a.answer,
                    "raw_llm_answer": a.raw_llm_answer,
                    "context": [c.chunk_id for c in a.retrieval.context],
                    "valid_citations": [str(c) for c in a.citations.valid] if a.citations else [],
                    "llm_usage": a.llm_usage,
                    "estimated_cost_usd": a.estimated_cost_usd,
                    "timings_ms": a.timings_ms,
                    "eval": ev,
                    "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"  {cfg.name:<40} {q['id']:<6} {rec.get('status', 'ERROR'):<27} "
                f"{'ERROR' if 'error' in rec else ('correct' if rec['eval']['correct'] else 'not correct')}"
            )
            if status != "complete":
                break
            if llm.real_calls > before:
                time.sleep(delay)
    latest = {(r["config"], r["id"]): r for r in map(json.loads, path.read_text().splitlines())}
    missing = [
        (c.name, q["id"])
        for c in configs
        for q in clean
        if (c.name, q["id"]) not in latest or "error" in latest[(c.name, q["id"])]
    ]
    if status == "complete" and missing:
        status = f"incomplete: {len(missing)} items missing or failed"
    (OUT / "answers-status.json").write_text(
        json.dumps(
            {
                "status": status,
                "real_calls_this_invocation": llm.real_calls,
                "cache_hits_this_invocation": llm.cache_hits,
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    print(f"Status: {status} | real calls {llm.real_calls} | cache hits {llm.cache_hits}")
    return 0 if status == "complete" else (3 if "authentication" in status else 2)


# ----------------------------------------------------------------------------- report
def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def part_report() -> int:
    prereg = json.loads(PREREG.read_text())
    ret = json.loads((OUT / "retrieval.json").read_text())
    status = json.loads((OUT / "answers-status.json").read_text())["status"]
    latest = {}
    for r in map(json.loads, (OUT / "answers.jsonl").read_text().splitlines()):
        latest[(r["config"], r["id"])] = r
    clean, _ = load_sets()
    names = prereg["part_b_answers"]["configs"]
    base = prereg["part_b_answers"]["baseline"].split()[0]
    L = [
        "# Phase 7 – RAG v1 vs v2 on the frozen evaluation set v1.1 (generated report)",
        "",
        f"- Pre-registration: `evaluation/phase7/preregistration.json` ({prereg['registered_utc']}), hash in "
        f"`manifests/phase7-preregistration.sha256`. Dataset v1.1 SHA-256 `{prereg['dataset']['sha256'][:12]}…` "
        f"(30 queries: 25 answerable, 5 unanswerable). EV-06 excluded (exposed).",
        f"- Retrieval run: {ret['run_utc']} (no LLM). Answer run status: **{status}**.",
        "",
        "## A. Retrieval – all 18 configurations (macro over 25 answerable; no LLM)",
        "",
        "| Config | Hit@3 | Hit@5 | MRR@3 | MRR@5 | P@3 | P@5 | R@3* | R@5* | Cov@3 | Cov@5 | Context cov. (≤4 after dense filter) | "
        "Full-context queries | Required fact dropped by dense filter | Unanswerable refused by dense gate | Median retrieval ms |",
        "|---|" + "---:|" * 15,
    ]
    for n, c in ret["configs"].items():
        a3, a5 = c["aggregate"]["3"], c["aggregate"]["5"]
        mark = " **(v1)**" if n == base else ""
        L.append(
            f"| `{n}`{mark} | {f(a3['hit_rate'])} | {f(a5['hit_rate'])} | {f(a3['mrr'])} | {f(a5['mrr'])} | "
            f"{f(a3['precision'])} | {f(a5['precision'])} | {f(a3['recall'])} | {f(a5['recall'])} | "
            f"{f(a3['coverage_macro'])} | {f(a5['coverage_macro'])} | {f(c['context_coverage_mean'])} | "
            f"{c['context_full_coverage_queries']}/25 | {c['required_fact_dropped_by_dense_filter']} | "
            f"{c['unanswerable_refused_by_dense_gate']}/5 | {f(c['median_ms']['retrieval_total'], 1)} |"
        )
    L += [
        "",
        "\\* chunk-level recall depends on the chunking (not comparable across chunkings). Coverage = distinct "
        "required facts (fact level).",
        "",
        "### Coverage@3 by category (answerable)",
        "",
    ]
    cats = list(dict.fromkeys(q["category"] for q in clean if q["answerable"]))
    L += [
        "| Config | "
        + " | ".join(f"{c} (n={sum(q['category'] == c and q['answerable'] for q in clean)})" for c in cats)
        + " |",
        "|---|" + "---:|" * len(cats),
    ]
    for n, c in ret["configs"].items():
        cells = []
        for cat in cats:
            rows = [r for r in c["per_query"] if r["answerable"] and r["category"] == cat]
            cells.append(f(statistics.mean(r["metrics"]["3"]["coverage"] for r in rows)))
        L.append(f"| `{n}` | " + " | ".join(cells) + " |")

    L += [
        "",
        "## B. Answers – pre-registered configurations (real LLM, deterministic checks)",
        "",
        "| Config | Correct (25) | Refusals correct (5) | **Score /30** | Answered | Answer terms (mean) | Citations valid | "
        "Cites expected policy | Citation support | Cited-fact cov. | Context-fact cov. | Unsupported numbers | "
        "Withheld (invalid citations) | Prompt / completion tokens | Est. cost USD | Median generation ms | Median total ms |",
        "|---|" + "---:|" * 16,
    ]
    scores = {}
    for n in names:
        rs = [latest.get((n, q["id"])) for q in clean]
        rs = [r for r in rs if r and "error" not in r]
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"]]
        ev = [r["eval"] for r in ans]
        correct = sum(e["correct"] for e in ev)
        refused = sum(r["eval"]["refusal_correct"] for r in una)
        real = [r for r in rs if r["llm_usage"].get("llm_called") and not r["llm_usage"].get("cached")]
        pt = sum(r["llm_usage"].get("prompt_tokens", 0) for r in rs if r["llm_usage"].get("llm_called"))
        ct = sum(r["llm_usage"].get("completion_tokens", 0) for r in rs if r["llm_usage"].get("llm_called"))
        scores[n] = {
            "score": correct + refused,
            "correct": correct,
            "refused": refused,
            "items": len(rs),
            "ctx": ret["configs"][n]["context_coverage_mean"],
            "latency": median([r["timings_ms"]["total"] for r in real]),
            "by_id": {r["id"]: (r["eval"]["correct"]) for r in rs},
        }
        done = (
            "COMPLETE (30/30)"
            if len(rs) == len(clean)
            else f"NOT RUN (0/{len(clean)})"
            if not rs
            else f"PARTIAL ({len(rs)}/{len(clean)})"
        )
        L.append(
            f"| `{n}` — {done} | {correct}/{len(ans)} | {refused}/{len(una)} | **{correct + refused}/{len(rs)}** | "
            f"{sum(e['answered'] for e in ev)}/{len(ans)} | {f(smean([e['answer_term_coverage'] for e in ev]))} | "
            f"{sum(e['citations_valid'] for e in ev)}/{len(ans)} | {sum(e['cites_expected_policy'] for e in ev)}/{len(ans)} | "
            f"{sum(all(e['citation_support']) for e in ev)}/{len(ans)} | {f(smean([e['cited_fact_coverage'] for e in ev]))} | "
            f"{f(smean([e['context_fact_coverage'] for e in ev]))} | {sum(len(e['unsupported_numbers']) for e in ev)} | "
            f"{sum(r['status'] == 'rejected_invalid_citations' for r in ans)} | {pt} / {ct} | "
            f"{sum(r['estimated_cost_usd'] or 0 for r in rs):.6f} | {f(median([r['timings_ms']['generation'] for r in real]), 0)} | "
            f"{f(scores[n]['latency'], 0)} |"
        )
    L += [
        "",
        "### Paired comparison with v1 (per query, 'correct' incl. refusals)",
        "",
        "| v2 config | v2 better | v1 better | ties | exact sign test p (two-sided) |",
        "|---|---:|---:|---:|---:|",
    ]
    for n in names:
        if n == base:
            continue
        common = set(scores[n]["by_id"]) & set(scores[base]["by_id"])
        w = sum(scores[n]["by_id"][i] and not scores[base]["by_id"][i] for i in common)
        lo = sum(scores[base]["by_id"][i] and not scores[n]["by_id"][i] for i in common)
        L.append(f"| `{n}` | {w} | {lo} | {len(common) - w - lo} | {sign_test(w, lo):.3f} |")
    L += [
        "",
        "### Same-subset comparison (only the queries both configurations answered)",
        "",
        "| Config pair | Queries in common | v1 correct | v2 correct | Completeness |",
        "|---|---:|---:|---:|---|",
    ]
    for n in names:
        if n == base:
            continue
        common = sorted(set(scores[n]["by_id"]) & set(scores[base]["by_id"]))
        if not common:
            L.append(f"| v1 vs `{n}` | 0 | — | — | no answers recorded for `{n}` |")
            continue
        L.append(
            f"| v1 vs `{n}` | {len(common)} ({common[0]}–{common[-1]}) | "
            f"{sum(scores[base]['by_id'][i] for i in common)}/{len(common)} | "
            f"{sum(scores[n]['by_id'][i] for i in common)}/{len(common)} | "
            f"{'complete' if len(common) == len(clean) else 'PARTIAL – not a full-set result'} |"
        )
    L += [
        "",
        "### Per-query 'correct' (✓ correct, ✗ not correct)",
        "",
        "| Query | Category | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for q in clean:
        L.append(
            f"| {q['id']} | {q['category']} | "
            + " | ".join(
                ("✓" if scores[n]["by_id"].get(q["id"]) else "✗") if q["id"] in scores[n]["by_id"] else "—"
                for n in names
            )
            + " |"
        )
    complete = all(scores[n]["items"] == len(clean) for n in names)
    ranked = (
        sorted(names, key=lambda n: (-scores[n]["score"], -scores[n]["ctx"], scores[n]["latency"] or 1e9))
        if complete
        else None
    )
    L += [
        "",
        "## C. Pre-registered selection",
        "",
        "Rule: highest score /30; tie-break context coverage, then lower median end-to-end latency.",
        "",
    ]
    if complete:
        L += [
            f"{i}. `{n}` – score {scores[n]['score']}/30, context coverage {f(scores[n]['ctx'])}, "
            f"median total {f(scores[n]['latency'], 0)} ms"
            for i, n in enumerate(ranked, start=1)
        ]
        L += ["", f"Selected by the rule: **`{ranked[0]}`**", ""]
    else:
        L += [
            "**The selection rule cannot be applied.** It needs all three pre-registered configurations answered on "
            "all 30 queries; the answer run is incomplete ("
            + ", ".join(f"`{n}` {scores[n]['items']}/{len(clean)}" for n in names)
            + "). No winning configuration is declared.",
            "",
        ]
    L += [
        "## D. EV-06 sensitivity (exposed query; retrieval only; not in any total)",
        "",
        "| Config | Hit@3 | Coverage@3 | Context coverage |",
        "|---|---:|---:|---:|",
    ]
    for n in names:
        s = ret["configs"][n]["ev06_sensitivity"][0]
        L.append(
            f"| `{n}` | {s['metrics']['3']['hit']} | {f(s['metrics']['3']['coverage'])} | {f(s['context_coverage'])} |"
        )
    (OUT / "phase7-report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "selection.json").write_text(
        json.dumps(
            {
                "selection_applicable": ranked is not None,
                "ranking": ranked,
                "scores": {n: {k: v for k, v in s.items() if k != "by_id"} for n, s in scores.items()},
                "answer_status": status,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["retrieval", "answers", "report"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--delay", type=float, default=3.0)
    args = ap.parse_args()
    integrity()
    if args.part == "retrieval":
        return part_retrieval()
    if args.part == "answers":
        return part_answers(args.resume, args.delay)
    return part_report()


if __name__ == "__main__":
    sys.exit(main())
