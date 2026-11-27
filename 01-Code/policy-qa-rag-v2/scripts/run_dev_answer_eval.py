"""Phase 6 - answer evaluation of RAG v2 configurations on the DEVELOPMENT queries only.

    .venv/bin/python scripts/run_dev_answer_eval.py --run-name smoke-1                 # real Groq LLM
    .venv/bin/python scripts/run_dev_answer_eval.py --run-name smoke-1 --resume        # continue after an interruption
    .venv/bin/python scripts/run_dev_answer_eval.py --run-name mock-1 --mock           # no API: scripted mock LLM

Queries: the 5 frozen development queries (dev_queries.json + dev_answer_keys.json) and, as
refusal probes, the 3 unanswerable Day 12 smoke questions (SM-13..15). The frozen 30-query
held-out set is never loaded; any other query ID is refused.

Safety and quota:
* the key is read from .env by src.config; it is never printed or written;
* every query result is appended to ``records.jsonl`` immediately (nothing is lost on interruption);
* ``--resume`` skips (config, query) pairs already recorded;
* identical prompts are answered from ``results/llm-cache/`` (no repeated calls);
* ``--max-llm-calls`` caps real LLM calls; ``--delay`` spaces them (free-tier tokens/minute);
* a provider quota stop (QuotaExhaustedError) ends the run as INCOMPLETE (exit 2), not as wrong answers.

Exit codes: 0 complete (every item answered without error), 1 integrity failure, 2 incomplete
(quota, call budget or items that failed with errors), 3 missing or rejected credentials.
An authentication failure stops the run after the first attempt. ``--resume`` retries items whose
latest record is an error; summaries use the latest record per (config, query).
``--summarize-only`` rebuilds summary.json / summary.md from records.jsonl without any API call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import groq  # noqa: E402  (only to recognise authentication failures)

from evaluation.answer_eval import evaluate_answer  # noqa: E402
from evaluation.evidence import chunk_evidence, load_catalog  # noqa: E402
from rag_v2.config_v2 import RAGv2Config  # noqa: E402
from rag_v2.llm_cache import CachingLLM  # noqa: E402
from rag_v2.pipeline_v2 import RAGPipelineV2  # noqa: E402
from src.config import ConfigurationError, load_settings  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402
from src.generation import GenerationError, GroqGenerator, QuotaExhaustedError  # noqa: E402

DATASETS = PROJECT_ROOT / "evaluation" / "datasets"
DEFAULT_CONFIGS = ["v1_paragraph_800_150+dense", "v1_paragraph_800_150+hybrid+rerank"]
INTEGRITY = [
    "corpus.sha256",
    "v1-baseline-code.sha256",
    "phase2-artifacts.sha256",
    "phase3-artifacts.sha256",
    "phase4-artifacts.sha256",
    "phase5-artifacts.sha256",
    "eval-dataset.sha256",
    "dev-queries.sha256",
    "dev-answer-keys.sha256",
    "evidence-catalog.sha256",
]


class MockLLM:
    """MOCK for dry runs and tests: cites the first context chunk with its first sentence. Not a real model."""

    model = "mock-llm"
    temperature = 0.0
    max_tokens = 0

    def __init__(self) -> None:
        self.prompt_tokens = self.completion_tokens = self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import re

        self.calls += 1
        self.prompt_tokens += len(user_prompt.split())
        m = re.search(r"policy_id: (\S+) \| chunk_id: (\S+) \|.*?\n(.*?)(?:\n|$)", user_prompt)
        self.completion_tokens += 20
        if not m:
            return "The policy documents do not contain enough information to answer this question."
        sentence = re.split(r"(?<=[.!?])\s", m.group(3).strip())[0]
        return f"{sentence} [Source: {m.group(1)}, Chunk: {m.group(2)}]"


def load_queries() -> list[dict]:
    dev = json.loads((DATASETS / "dev_queries.json").read_text(encoding="utf-8"))["queries"]
    keys = json.loads((DATASETS / "dev_answer_keys.json").read_text(encoding="utf-8"))
    catalog = load_catalog()
    out = [
        {
            "id": q["id"],
            "question": q["query"],
            "answerable": True,
            "kind": "development query",
            "required_evidence": q["required_evidence"],
            "answer_terms": keys["keys"][q["id"]]["answer_terms"],
            "reference_answer": keys["keys"][q["id"]]["reference_answer"],
            "expected_policy_ids": sorted({catalog[e].policy_id for g in q["required_evidence"] for e in g}),
        }
        for q in dev
    ]
    smoke = {
        q["id"]: q
        for q in json.loads((PROJECT_ROOT / "tests" / "smoke_questions.json").read_text())["questions"]
    }
    out += [
        {
            "id": sid,
            "question": smoke[sid]["question"],
            "answerable": False,
            "kind": "refusal probe (Day 12 smoke)",
            "required_evidence": [],
            "answer_terms": [],
            "reference_answer": smoke[sid]["expected_answer"],
            "expected_policy_ids": [],
        }
        for sid in keys["refusal_probes"]["ids"]
    ]
    bad = [q["id"] for q in out if not q["id"].startswith(("DEV-", "SM-"))]
    if bad:  # the held-out EV- queries must never reach this script
        raise SystemExit(f"refusing non-development query IDs: {bad}")
    return out


def median(values):
    return statistics.median(values) if values else None


def latest(records: list[dict]) -> list[dict]:
    """Last record per (config, query): records.jsonl is append-only, so a retried item supersedes its error."""
    by_key: dict = {}
    for r in records:
        by_key[(r["config"], r["query_id"])] = r
    return list(by_key.values())


def run_status(records: list[dict], configs: list[str], query_ids: list[str], loop_status: str) -> str:
    """'complete' only if every requested (config, query) has a non-error latest record."""
    if loop_status != "complete":
        return loop_status
    have = {(r["config"], r["query_id"]): r for r in latest(records)}
    expected = [(c, q) for c in configs for q in query_ids]
    errors = sum(1 for k in expected if k in have and "error" in have[k])
    missing = sum(1 for k in expected if k not in have)
    if errors or missing:
        return f"incomplete: {errors} of {len(expected)} items failed with errors, {missing} not run"
    return "complete"


def summarise(records: list[dict]) -> dict:
    records = latest(records)
    out = {}
    for cfg in dict.fromkeys(r["config"] for r in records):
        rs = [r for r in records if r["config"] == cfg and "error" not in r]
        ans = [r for r in rs if r["answerable"]]
        una = [r for r in rs if not r["answerable"]]
        ev = [r["eval"] for r in ans]
        real = [r for r in rs if r["llm_usage"].get("llm_called") and not r["llm_usage"].get("cached")]
        stages = ("query_embedding", "vector_search", "bm25", "fusion", "rerank", "retrieval_total")
        out[cfg] = {
            "answerable_n": len(ans),
            "unanswerable_n": len(una),
            "errors": sum(1 for r in records if r["config"] == cfg and "error" in r),
            "correct": sum(e["correct"] for e in ev),
            "status_answered": sum(e["answered"] for e in ev),
            "answer_term_coverage_mean": statistics.mean(e["answer_term_coverage"] for e in ev)
            if ev
            else None,
            "citations_valid": sum(e["citations_valid"] for e in ev),
            "cites_expected_policy": sum(e["cites_expected_policy"] for e in ev),
            "citation_support_all": sum(all(e["citation_support"]) for e in ev),
            "cited_fact_coverage_mean": statistics.mean(e["cited_fact_coverage"] for e in ev) if ev else None,
            "context_fact_coverage_mean": statistics.mean(e["context_fact_coverage"] for e in ev)
            if ev
            else None,
            "unsupported_numbers_total": sum(len(e["unsupported_numbers"]) for e in ev),
            "invalid_citations_total": sum(e["n_invalid_citations"] for e in ev),
            "refusal_correct": sum(r["eval"]["refusal_correct"] for r in una),
            "dropped_by_dense_filter_total": sum(len(r["dropped_by_dense_filter"]) for r in rs),
            "llm_calls_real": len(real),
            "llm_cache_hits": sum(1 for r in rs if r["llm_usage"].get("cached")),
            "llm_not_called": sum(1 for r in rs if not r["llm_usage"].get("llm_called")),
            "prompt_tokens_total": sum(
                r["llm_usage"].get("prompt_tokens", 0) for r in rs if r["llm_usage"].get("llm_called")
            ),
            "completion_tokens_total": sum(
                r["llm_usage"].get("completion_tokens", 0) for r in rs if r["llm_usage"].get("llm_called")
            ),
            "estimated_cost_usd_total": sum(r["estimated_cost_usd"] or 0 for r in rs),
            "median_ms": {s: median([r["timings_ms"][s] for r in rs if s in r["timings_ms"]]) for s in stages}
            | {
                "generation (real calls only)": median([r["timings_ms"]["generation"] for r in real]),
                "total (real calls only)": median([r["timings_ms"]["total"] for r in real]),
            },
        }
    return out


def f(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def render(meta: dict, summary: dict, records: list[dict]) -> str:
    L = [
        f"# Phase 6 – Development-query answer evaluation: run `{meta['run_name']}`",
        "",
        f"- Started (UTC): {meta['started_utc']} · LLM: `{meta['llm_model']}` ({meta['llm_kind']}) · temperature "
        f"{meta['temperature']} · dense refusal threshold {meta['min_similarity']} · context_k 4 · status: "
        f"**{meta['status']}**",
        f"- Integrity: {meta['integrity']}",
        "- Queries: 5 development queries + 3 refusal probes (Day 12 SM-13..15). The 30-query held-out set was not used.",
        "- All checks are deterministic (no LLM-as-judge); see `02-Documentation/Day-13-Answer-Evaluation-Methodology.md`.",
        "",
        "## Summary per configuration",
        "",
        "| Config | Correct (dev) | Answered | Answer terms (mean) | Citations valid | Cites expected policy | "
        "Citation support | Cited-fact cov. | Context-fact cov. | Unsupported numbers | Refusals correct | Errors |",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---|---:|",
    ]
    for cfg, s in summary.items():
        n = s["answerable_n"]
        L.append(
            f"| `{cfg}` | {s['correct']}/{n} | {s['status_answered']}/{n} | {f(s['answer_term_coverage_mean'])} | "
            f"{s['citations_valid']}/{n} | {s['cites_expected_policy']}/{n} | {s['citation_support_all']}/{n} | "
            f"{f(s['cited_fact_coverage_mean'])} | {f(s['context_fact_coverage_mean'])} | "
            f"{s['unsupported_numbers_total']} | {s['refusal_correct']}/{s['unanswerable_n']} | {s['errors']} |"
        )
    L += [
        "",
        "## Tokens, cost and latency per configuration",
        "",
        "| Config | Real LLM calls | Cache hits | LLM not called | Prompt tokens | Completion tokens | "
        "Est. cost (USD, list price) | Median ms: embed / vector / BM25 / RRF / rerank / retrieval | Median generation ms | "
        "Median total ms |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for cfg, s in summary.items():
        m = s["median_ms"]
        stages = " / ".join(
            "—" if m[k] is None else f"{m[k]:.1f}"
            for k in ("query_embedding", "vector_search", "bm25", "fusion", "rerank", "retrieval_total")
        )
        L.append(
            f"| `{cfg}` | {s['llm_calls_real']} | {s['llm_cache_hits']} | {s['llm_not_called']} | "
            f"{s['prompt_tokens_total']} | {s['completion_tokens_total']} | {s['estimated_cost_usd_total']:.6f} | "
            f"{stages} | {f(m['generation (real calls only)'])} | {f(m['total (real calls only)'])} |"
        )
    L += [
        "",
        "Estimated cost uses the published list price in `rag_v2/pricing.py` (source and date recorded there); it "
        "is not the amount charged on the account's plan. Generation latency includes any rate-limit waits.",
        "",
        "## Per query",
        "",
        "| Config | Query | Status | Correct | Answer terms | Citations | Support | Unsupported numbers | "
        "Context chunks (dropped by dense filter) | Answer (shown) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if "error" in r:
            L.append(f"| `{r['config']}` | {r['query_id']} | ERROR | — | — | — | — | — | — | {r['error']} |")
            continue
        e = r["eval"]
        ans = r["answer"].replace("|", "\\|").replace("\n", " ")
        ctx = ", ".join(r["context_ids"]) + (
            f" (dropped: {', '.join(r['dropped_by_dense_filter'])})" if r["dropped_by_dense_filter"] else ""
        )
        if r["answerable"]:
            L.append(
                f"| `{r['config']}` | {r['query_id']} | {r['status']} | {'yes' if e['correct'] else 'no'} | "
                f"{e['answer_terms_found']} | {'valid' if e['citations_valid'] else 'NOT valid'} | {e['citation_support']} | "
                f"{e['unsupported_numbers'] or '—'} | {ctx} | {ans} |"
            )
        else:
            L.append(
                f"| `{r['config']}` | {r['query_id']} | {r['status']} | {'yes' if e['correct'] else 'no'} (refusal) | — | "
                f"{'has citations' if e['has_citations'] else 'none'} | — | — | {ctx or '—'} | {ans} |"
            )
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--mock", action="store_true", help="scripted mock LLM (no API calls)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--delay", type=float, default=8.0, help="seconds after each real LLM call")
    ap.add_argument("--max-llm-calls", type=int, default=20)
    ap.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "dev-answer-eval")
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help="LLM response cache (default results/llm-cache[-mock])"
    )
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="rebuild summary.json/summary.md from records.jsonl; no LLM, no API call",
    )
    args = ap.parse_args()

    run_dir = args.results_root / args.run_name
    records_path = run_dir / "records.jsonl"
    if args.summarize_only:
        return summarize_only(run_dir, args.configs)
    if run_dir.exists() and not args.resume and records_path.exists():
        print(f"{run_dir} already has records; use --resume or a new --run-name.", file=sys.stderr)
        return 1
    failed = [
        m
        for m in INTEGRITY
        if subprocess.run(
            ["shasum", "-a", "256", "-c", "--quiet", f"manifests/{m}"], cwd=PROJECT_ROOT
        ).returncode
    ]
    if failed:
        print(f"Integrity check failed: {failed}", file=sys.stderr)
        return 1

    settings = load_settings()
    if args.mock:
        inner, kind = MockLLM(), "MOCK - not a real model"
    else:
        try:
            key = settings.require_groq_key()
        except ConfigurationError as exc:
            print(f"Missing credentials: {exc}", file=sys.stderr)
            return 3
        inner, kind = (
            GroqGenerator(api_key=key, model=settings.llm_model, temperature=settings.llm_temperature),
            "real",
        )
        del key
    cache_dir = args.cache_dir or PROJECT_ROOT / "results" / ("llm-cache-mock" if args.mock else "llm-cache")
    llm = CachingLLM(inner, cache_dir)
    configs = [RAGv2Config.from_name(n) for n in args.configs]
    queries = load_queries()
    run_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if records_path.exists():
        old = latest([json.loads(line) for line in records_path.read_text().splitlines()])
        done = {
            (r["config"], r["query_id"]) for r in old if "error" not in r
        }  # errors are retried on --resume

    meta = {
        "run_name": args.run_name,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm_model": llm.model,
        "llm_kind": kind,
        "temperature": settings.llm_temperature,
        "min_similarity": settings.min_similarity,
        "configs": [c.as_dict() for c in configs],
        "integrity": f"{len(INTEGRITY)} manifests PASS",
        "status": "running",
        "one_time_ms": {},
    }
    print(
        f"Run {args.run_name}: {len(configs)} configs x {len(queries)} queries, LLM {llm.model} ({kind}); "
        f"{len(done)} already done"
    )

    embedder, reranker = LocalEmbedder(), None
    catalog = load_catalog()
    by_policy: dict = {}
    for u in catalog.values():
        by_policy.setdefault(u.policy_id, []).append(u)
    status, stop = "complete", False
    for cfg in configs:
        if stop:
            break
        pipe = RAGPipelineV2(settings, cfg, llm=llm, embedder=embedder, reranker=reranker)
        reranker = pipe.reranker or reranker
        meta["one_time_ms"][cfg.name] = pipe.one_time_ms
        facts = {c.chunk_id: set(chunk_evidence(c.metadata, by_policy)) for c in pipe.index.chunks}
        for q in queries:
            if (cfg.name, q["id"]) in done:
                continue
            if llm.real_calls >= args.max_llm_calls:
                status, stop = f"incomplete: --max-llm-calls {args.max_llm_calls} reached", True
                break
            calls_before = llm.real_calls
            try:
                a = pipe.ask(q["question"])
            except QuotaExhaustedError as exc:
                status, stop = f"incomplete: provider quota exhausted ({exc})", True
                break
            except GenerationError as exc:
                if isinstance(exc.__cause__, groq.AuthenticationError):  # not per-query: stop after the first
                    status, stop = "incomplete: authentication failed (Groq rejected the API key)", True
                rec = {
                    "config": cfg.name,
                    "query_id": q["id"],
                    "answerable": q["answerable"],
                    "error": str(exc),
                }
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
                    chunk_facts=facts,
                )
                rec = {
                    "config": cfg.name,
                    "query_id": q["id"],
                    "kind": q["kind"],
                    "question": q["question"],
                    "answerable": q["answerable"],
                    "status": a.status,
                    "answer": a.answer,
                    "raw_llm_answer": a.raw_llm_answer,
                    "context_ids": [c.chunk_id for c in a.retrieval.context],
                    "context_dense_similarity": [round(c.similarity, 4) for c in a.retrieval.context],
                    "dropped_by_dense_filter": a.retrieval.dropped_by_dense_filter,
                    "best_dense_similarity": a.retrieval.best_dense_similarity,
                    "final_ranking": a.retrieval.final_ranking,
                    "valid_citations": [str(c) for c in a.citations.valid] if a.citations else [],
                    "invalid_citations": [str(c) for c, _ in a.citations.invalid] if a.citations else [],
                    "llm_usage": a.llm_usage,
                    "estimated_cost_usd": a.estimated_cost_usd,
                    "timings_ms": a.timings_ms,
                    "eval": ev,
                    "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            with records_path.open("a", encoding="utf-8") as fh:  # incremental: saved before the next call
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            verdict = "ERROR" if "error" in rec else ("correct" if rec["eval"]["correct"] else "not correct")
            print(f"  {cfg.name:<42} {q['id']:<7} {rec.get('status', 'error'):<27} {verdict}")
            if stop:
                break
            if llm.real_calls > calls_before and not args.mock:
                time.sleep(args.delay)

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    status = run_status(records, [c.name for c in configs], [q["id"] for q in queries], status)
    meta["status"] = status
    meta["llm_real_calls_this_invocation"] = llm.real_calls
    meta["llm_cache_hits_this_invocation"] = llm.cache_hits
    summary = summarise(records)
    (run_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render(meta, summary, latest(records)), encoding="utf-8")
    print(
        f"Status: {status} | real LLM calls this invocation: {llm.real_calls} | cache hits: {llm.cache_hits}"
    )
    print(f"Saved: {run_dir / 'records.jsonl'}, summary.json, summary.md")
    if status.startswith("incomplete: authentication"):
        return 3
    return 0 if status == "complete" else 2


def summarize_only(run_dir: Path, config_names: list[str]) -> int:
    """Rebuild the summary of an existing run from its records (no LLM client is created)."""
    records_path = run_dir / "records.jsonl"
    if not records_path.exists():
        print(f"no records in {run_dir}", file=sys.stderr)
        return 1
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    sources = sorted(run_dir.glob("summary-original*.json")) + [run_dir / "summary.json"]
    previous = next((json.loads(s.read_text())["meta"] for s in sources if s.exists()), {})
    settings = load_settings()
    defaults = {
        "run_name": run_dir.name,
        "started_utc": next((r["recorded_utc"] for r in records if r.get("recorded_utc")), "unknown"),
        "llm_model": settings.llm_model,
        "llm_kind": "unknown (rebuilt from records)",
        "temperature": settings.llm_temperature,
        "min_similarity": settings.min_similarity,
        "integrity": "not re-checked by --summarize-only",
    }
    previous = defaults | {k: v for k, v in previous.items() if k != "status"}
    names = list(dict.fromkeys(r["config"] for r in records)) or config_names
    query_ids = [q["id"] for q in load_queries()]
    errors = {r["error"] for r in latest(records) if "error" in r}
    loop = (
        "incomplete: authentication failed (Groq rejected the API key)"
        if any("authentication error" in e for e in errors)
        else "complete"
    )
    status = run_status(records, names, query_ids, loop)
    meta = previous | {
        "status": status,
        "summary_rebuilt_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary_rebuilt_note": "rebuilt from records.jsonl with --summarize-only (no API call)",
    }
    summary = summarise(records)
    (run_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.md").write_text(render(meta, summary, latest(records)), encoding="utf-8")
    print(f"Status: {status} | rebuilt from {len(records)} records (no API call)")
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
