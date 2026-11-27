"""Build the Day 13 final project report; every table is read from saved results (no retrieval, no LLM call).

.venv/bin/python scripts/build_final_report.py   ->  ../../04-Submission/Day-13-Final-Project-Report.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAY = ROOT.parent.parent
OUT = DAY / "04-Submission" / "Day-13-Final-Project-Report.md"
V1, V2A, V2B = (
    "v1_paragraph_800_150+dense",
    "v1_paragraph_800_150+hybrid+rerank",
    "recursive_800_100+hybrid+rerank",
)


def j(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def latest(path):
    d = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d[(r["config"], r.get("id") or r.get("query_id"))] = r
    return [r for r in d.values() if "error" not in r]


def main() -> int:
    ret = j("results/phase7/retrieval.json")["configs"]
    sel = j("results/phase7/selection.json")
    cl = j("results/phase8/cost-latency.json")
    prereg = j("evaluation/phase7/preregistration.json")
    v11 = j("evaluation/datasets/eval_dataset_v1_1_freeze.json")
    smoke = j("results/dev-answer-eval/real-smoke-1/summary.json")["summary"]
    man = j("evaluation/phase7/manual_review.json")["summary"]
    s = sel["scores"]

    rt = [
        "| Config | Hit@3 | MRR@3 | P@3 | Cov@3 | Hit@5 | Cov@5 | Context coverage (4 chunks) | Median retrieval ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, c in ret.items():
        a3, a5 = c["aggregate"]["3"], c["aggregate"]["5"]
        rt.append(
            f"| `{n}`{' **(v1)**' if n == V1 else ''} | {a3['hit_rate']:.2f} | {a3['mrr']:.2f} | {a3['precision']:.2f} | "
            f"{a3['coverage_macro']:.2f} | {a5['hit_rate']:.2f} | {a5['coverage_macro']:.2f} | "
            f"{c['context_coverage_mean']:.2f} | {c['median_ms']['retrieval_total']:.1f} |"
        )

    def arow(n):
        x = s[n]
        state = (
            "COMPLETE"
            if x["items"] == 30
            else ("NOT RUN (0/30)" if x["items"] == 0 else f"PARTIAL ({x['items']}/30)")
        )
        return f"| `{n}` | {state} | {x['correct']} | {x['refused']} | {x['score']}/{x['items']} |"

    b = cl["retrieval_benchmark_median_ms"]["v1_paragraph_800_150"]
    gate = {n: c["unanswerable_refused_by_dense_gate"] for n, c in ret.items()}
    by_value: dict = {}
    for n, v in gate.items():
        by_value.setdefault(v, []).append(n)
    gate_text = (
        "; ".join(
            f"{v}/5 refused without an LLM call for {len(ns)} configuration(s)"
            + (f" ({', '.join(sorted({x.split('+')[0] for x in ns}))} chunking)" if len(ns) < 18 else "")
            for v, ns in sorted(by_value.items())
        )
        + "; the LLM refused the rest."
    )
    ans_gate = f"{sum(c['answerable_refused_by_dense_gate'] for c in ret.values())} cases"
    ans = latest("results/phase7/answers.jsonl")
    ok = {(r["config"], r["id"]): r["eval"]["correct"] for r in ans}
    ids = sorted({i for _, i in ok})

    def paired(c):
        w = sum(ok[(c, i)] and not ok[(V1, i)] for i in ids)
        lo = sum(ok[(V1, i)] and not ok[(c, i)] for i in ids)
        return w, lo, len(ids) - w - lo

    pa, pb = paired(V2A), paired(V2B)
    winner = sel["ranking"][0] if sel["selection_applicable"] else None
    tie = s[sel["ranking"][0]]["score"] == s[sel["ranking"][1]]["score"] if winner else False
    clean_rr = cl["retrieval_benchmark_median_ms"]["v1_paragraph_800_150"]
    selection_text = (
        f"**Pre-registered selection (rule applied exactly as written): `{winner}`.** "
        + (
            f"It ties `{sel['ranking'][1]}` on the primary score ({s[winner]['score']}/30) and on tie-break 1 (context "
            f"coverage 1.00); tie-break 2 (median end-to-end latency in the answer run, {s[winner]['latency']:.0f} vs "
            f"{s[sel['ranking'][1]]['latency']:.0f} ms) decides. **Caveat:** that latency input is confounded — the runs "
            "were made at different times under different free-tier rate-limit pressure and cached answers are excluded — "
            f"while the controlled benchmark shows the selected configuration is ≈ {clean_rr['total'] - clean_rr['hybrid']:.0f} ms "
            "*slower* per query (reranking). The selection is the formal outcome of the registered rule, not evidence that "
            "the configuration is better: answer quality is identical (paired ties on all 30 queries)."
            if tie
            else ""
        )
        if winner
        else "**The pre-registered selection rule cannot be applied** (incomplete answer run)."
    )
    L = f"""# Day 13 – AI05 Mini-Project 1: Evaluated RAG Assistant (Policy Q&A RAG v2) — Final Report

**Status:** Completed - Pending Review · **Date:** 2026-11-27 (answer evaluation completed 2026-11-27 02:58 UTC)
**Project:** `Day-13/01-Code/policy-qa-rag-v2/` · Tables in this report are generated from saved results by
`scripts/build_final_report.py`.

## 1. Executive summary

- **What was built:** RAG v2 — a configurable extension of the Day 12 Policy Q&A RAG (v1) with three chunking
  strategies, BM25 + dense hybrid retrieval with Reciprocal Rank Fusion, and cross-encoder reranking (18
  configurations), on the unchanged v1 generator, prompt, citation validation and dense refusal threshold (0.25). The
  v1-equivalent configuration sends byte-identical prompts to RAG v1 (15/15).
- **How it was evaluated:** a frozen, hash-verified 30-query labelled set (v1.1; 25 answerable, 5 unanswerable)
  with chunk-independent canonical fact labels, a pre-registered evaluation plan, retrieval metrics for all 18
  configurations, and deterministic answer checks with a real LLM (Groq `openai/gpt-oss-120b`).
- **Retrieval results (complete, 30 queries):** the v1 baseline is already near the ceiling on this 6-policy corpus
  (Hit@3 {ret[V1]["aggregate"]["3"]["hit_rate"]:.2f}, MRR@3 {ret[V1]["aggregate"]["3"]["mrr"]:.2f}, fact coverage@3
  {ret[V1]["aggregate"]["3"]["coverage_macro"]:.2f}). Reranking raises MRR@3 to {ret[V2A]["aggregate"]["3"]["mrr"]:.2f};
  recursive 800/100 chunking raises coverage@3 to {ret["recursive_800_100+dense"]["aggregate"]["3"]["coverage_macro"]:.2f};
  hybrid without reranking is slightly *worse* ({ret["v1_paragraph_800_150+hybrid"]["aggregate"]["3"]["coverage_macro"]:.2f});
  fixed 500-character chunking is clearly worse ({ret["fixed_500+dense"]["aggregate"]["3"]["coverage_macro"]:.2f}) because it
  splits 19 of 102 facts.
- **Answer results (complete, 3 × 30 queries, deterministic checks):** `{V1}` **{s[V1]["score"]}/30**, `{V2A}`
  **{s[V2A]["score"]}/30**, `{V2B}` **{s[V2B]["score"]}/30**; every configuration refused 5/5 unanswerable questions with
  valid citations on all answers. Paired vs v1: `{V2A}` {pa[0]} better / {pa[1]} worse / {pa[2]} ties; `{V2B}` {pb[0]} / {pb[1]}
  / {pb[2]} (sign test p = {2 * sum(__import__("math").comb(pb[0] + pb[1], k) for k in range(min(pb) + 1)) / 2 ** (pb[0] + pb[1]) if pb[0] + pb[1] else 1:.2f}).
  Manual review (separate): all three reach 29/30; every difference is an answer-checker false negative.
- {selection_text}
- **Cost/latency (measured):** hybrid adds ≈ {b["bm25"] + b["rrf"]:.2f} ms per query; the cross-encoder adds ≈ {b["rerank"]:.0f} ms
  on CPU; neither adds API cost and the LLM context stays at 4 chunks. LLM generation (≈ 1 s uncontended) dominates.
  Estimated LLM cost ≈ ${cl["v1_cost_per_query_usd_estimate"]:.6f} per query at list price; Day 13 spent
  {cl["day13_tokens_spent"]["prompt"] + cl["day13_tokens_spent"]["completion"]:,} tokens (≈ ${cl["day13_cost_estimate_usd"]:.4f} estimated; free tier).

## 2. Architecture and workflow

```
question ─► MiniLM query embedding ─► cosine similarity for every chunk (Chroma)          [v1 components]
              │                         └► dense top-10
              ├► BM25 top-10 (rank_bm25, identifier-preserving tokenizer)                 [Activity B]
              └► RRF k=60 (hybrid)  ─► optional cross-encoder rerank of top-10            [Activities B, C]
                                          └► first 4 chunks; keep only dense similarity ≥ 0.25 (v1 rule)
                                                └► none left → refuse without LLM call
                                                └► v1 prompt → Groq gpt-oss-120b (T=0) → v1 citation validation
chunking (index time): v1 paragraph 800/150 | fixed 500 | recursive 800/100                [Activity A]
```

Details, configuration guide and setup: `02-Documentation/Day-13-RAG-v2-Architecture-and-Configuration.md`.

## 3. Methodology

- **Corpus:** the unchanged Day 12 6-policy corpus (SHA-256 manifest).
- **Labels:** 102 canonical sentence-level facts (`HR-001-S02-E04`); queries require fact groups (with alternatives),
  so relevance does not depend on chunk IDs.
- **Datasets:** 5 development queries (only tuning set); 30-query held-out set v1.0 frozen before any run; EV-06 was
  exposed in Phase 6 and replaced by EV-31 in **v1.1** (SHA-256 `{v11["sha256"][:12]}…`, frozen {v11["frozen_utc"]}); v1.0
  kept unchanged.
- **Pre-registration:** `evaluation/phase7/preregistration.json` ({prereg["registered_utc"]}) fixed the configurations,
  metrics, selection rule and sign test before Phase 7 ran.
- **Retrieval metrics:** Hit@K, MRR@K, Precision@K, Recall@K (chunk level) and distinct fact Coverage@K (fact level),
  K = 3 and 5, macro over 25 answerable queries; unanswerable queries excluded from retrieval metrics.
- **Answer checks (deterministic, no LLM judge):** answer facts present, citations valid, expected policy cited,
  each fact supported by a cited chunk, no unsupported numbers; correct refusal without citations for unanswerable.
- **Integrity:** every phase froze its artifacts with SHA-256 manifests and re-verified earlier ones; Day 12 was never
  modified (git working tree clean throughout).

Full methodology: `02-Documentation/Day-13-Evaluation-Dataset-Methodology.md`,
`Day-13-Activity-D-Retrieval-Metrics.md`, `Day-13-Answer-Evaluation-Methodology.md`.

## 4. Activities A–D (development findings)

| Activity | Key measured result |
|---|---|
| A – Chunking | v1 22 chunks, fixed 500 → 28, recursive 800/100 → 20 (byte-identical to LangChain). fixed_500 splits 19/102 facts; recursive leaves 12 headings detached; v1 keeps all facts intact |
| B – BM25 + RRF | hybrid never below dense on dev queries, better once; adds ≈ 0.11 ms |
| C – Reranking | dev Coverage@3 for v1 chunking 6/7 → 7/7, MRR@3 0.80 → 1.00; +46–72 ms on CPU; no fact lost from the pool |
| D – Metrics | P@K, R@K, Hit@K, MRR, Coverage@K with hand-worked tests; K=3 vs K=5: coverage up, precision down |

## 5. Retrieval comparison – all 18 configurations (complete, 25 answerable queries)

{chr(10).join(rt)}

Charts: `01-Code/policy-qa-rag-v2/results/phase7/charts/coverage-at-3.svg`, `mrr-at-3.svg`, `retrieval-latency.svg`.
Context coverage = required facts inside the ≤ 4 chunks actually sent to the LLM. Dense refusal gate on the 5
unanswerable queries: {gate_text} No answerable query was refused by the gate in any configuration ({ans_gate}).

## 6. Answer evaluation (real LLM)

| Config | Completeness | Correct (deterministic) | Refusals correct | Score |
|---|---|---:|---:|---:|
{arow(V1)}
{arow(V2A)}
{arow(V2B)}

- Paired vs v1 ('correct' incl. refusals): `{V2A}` {pa[0]} better / {pa[1]} worse / {pa[2]} ties; `{V2B}` {pb[0]} / {pb[1]} / {pb[2]}.
- Manual review of failures (separate from the scores; `evaluation/phase7/manual_review.json`): `{V1}`
  {man[V1]["deterministic_score"]} → {man[V1]["manually_reviewed_score"]}; `{V2A}` {man[V2A]["deterministic_score"]} →
  {man[V2A]["manually_reviewed_score"]}; `{V2B}` {man[V2B]["deterministic_score"]} → {man[V2B]["manually_reviewed_score"]}. The
  only genuine miss (EV-09: "blocked by default" omitted) is shared by all three; the other failures are checker false
  negatives (phrasing, "two" vs "2", Unicode hyphens).
- {selection_text}
- Dev smoke test (complete, both configs): v1 {smoke[V1]["correct"]}/5 vs `{V2A}` {smoke[V2A]["correct"]}/5 correct,
  refusals 3/3 each, citations valid 5/5 each.

## 7. Cost and latency

See `01-Code/policy-qa-rag-v2/results/phase8/cost-latency-report.md` (measured vs estimate vs unavailable).

| Component | Median (Apple M3 CPU) | API cost |
|---|---:|---|
| Query embedding + Chroma search | {b["embed"] + b["search"]:.1f} ms | $0 |
| BM25 + RRF | {b["bm25"] + b["rrf"]:.2f} ms | $0 |
| Cross-encoder rerank (10 pairs) | {b["rerank"]:.1f} ms | $0 |
| LLM generation (uncontended, dev smoke) | ≈ 1.0 s | ≈ ${cl["v1_cost_per_query_usd_estimate"]:.6f}/query (estimate) |

Estimated LLM cost per query (30 queries, list price): `{V1}` ${s[V1]["cost"] if "cost" in s[V1] else cl["v1_cost_per_query_usd_estimate"]:.6f}; all three configurations are within ≈ 4 %
because the context stays at 4 chunks. Phase 7 LLM and end-to-end latencies were inflated by free-tier rate-limit waits
at different times and are not comparable between configurations; clean figures come from the benchmarks and the dev
smoke test.

## 8. Limitations

1. **Selection decided by a confounded tie-break:** the registered rule picks `{winner}` on end-to-end latency
   measured under different rate-limit conditions; answer quality is identical to v1 and the controlled benchmark shows
   reranking adds latency.
2. **Ceiling effect:** the v1 baseline retrieves well on a 6-policy corpus; improvements are small and confidence
   intervals with 25 answerable queries are wide.
3. **Deterministic answer checks** miss paraphrases (9 false negatives across the three configurations) and the numeric check can be satisfied by a
   heading number; no LLM judge was used.
4. **Single annotator** for the labels; sentence-level facts.
5. **CPU-only latency** on one machine; no load testing; the billed amount is not visible to the code.
6. **Process incidents (disclosed):** a held-out question (EV-06) was exposed in Phase 6 and replaced; a real-looking
   API key was found in `.env.example` and removed with approval (revocation recommended).

## 9. Future improvements

Re-measure end-to-end latency for the three configurations under identical, uncontended conditions; add a calibrated LLM-as-judge alongside the deterministic checks; normalise Unicode punctuation and allow inserted
words in answer matching; evaluate on a larger, multi-annotator corpus; try query rewriting; measure GPU / ONNX
cross-encoder latency.

## 10. Rubric mapping

| Criterion (weight) | Evidence | Where |
|---|---|---|
| Functional correctness (25%) | v2 pipeline with 18 configurations; v1 equivalence 15/15; 193 tests; real-LLM smoke test; 3 × 30 answer evaluations | §2, §6; `03-Evidence/24`, `27`, `33`; `04-Submission/test-log.txt` |
| Retrieval improvement evidence (25%) | 18-config retrieval comparison on 30 frozen queries; Activities A–C | §4, §5; charts; `results/phase7/phase7-report.md` |
| Evaluation methodology (20%) | canonical facts, frozen hashed datasets, pre-registration, exposure handling, metric tests | §3; dataset and metrics documents; `03-Evidence/18–21b`, `25` |
| Cost/latency analysis (15%) | per-stage measured latency, tokens, estimates vs measured vs unavailable | §7; `results/phase8/cost-latency-report.md`; `03-Evidence/32` |
| Code and documentation (15%) | modular `rag_v2/`, `evaluation/`, scripts, README, guides, evidence index | README; `02-Documentation/`; `03-Evidence/evidence-index.md` |
"""
    OUT.write_text(L, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
