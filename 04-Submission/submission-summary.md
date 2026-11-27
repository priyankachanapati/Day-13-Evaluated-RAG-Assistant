# Submission Summary – Day 13: AI05 Mini-Project 1 (Evaluated RAG Assistant)

**Status:** Completed - Pending Review (not Verified, not Submitted). **Module:** AI05 – Advanced RAG, Retrieval Evaluation & Mini-Project 1
**Repository:** [GITHUB LINK PENDING] (no repository created; GitHub only with explicit approval)

## Deliverables

| Deliverable | Location |
|---|---|
| Working code (RAG v2, 18 configs, CLI, evaluation harness) | `01-Code/policy-qa-rag-v2/` (`rag_v2/`, `evaluation/`, `scripts/`) |
| Evaluation report | `04-Submission/Day-13-Final-Project-Report.md`; `executive-summary.md` |
| Labelled 30-query dataset (frozen) | `evaluation/datasets/eval_queries_v1_1.json` (v1.0 kept; EV-06 exposure record) |
| Retrieval + answer evaluation | `results/phase7/phase7-report.md`, charts in `results/phase7/charts/` |
| Cost and latency | `results/phase8/cost-latency-report.md` |
| Demo | `02-Documentation/Day-13-Demo-Walkthrough.md` |
| Tests | `04-Submission/test-log.txt` |
| Evidence | `03-Evidence/` (`evidence-index.md`) |

## Results (actual)

| Item | Result |
|---|---|
| Retrieval, 18 configs × 25 answerable | v1 Hit@3 1.00 / MRR@3 0.98 / Cov@3 0.98; rerank MRR@3 1.00; recursive Cov@3 1.00; fixed 500 Cov@3 0.84 |
| Answers, 3 configs × 30 | v1 27/30; v1 + hybrid + rerank 27/30 (30 paired ties); recursive + hybrid + rerank 24/30 (p = 0.25); refusals 5/5 each |
| Manual review (separate) | all three 29/30; differences = answer-checker false negatives |
| Pre-registered selection | `v1_paragraph_800_150+hybrid+rerank`, by a confounded latency tie-break (documented) |
| Cost | ≈ $0.00026–0.00027 per query (list-price estimate); BM25/RRF/reranker $0 API cost |
| Latency (controlled) | hybrid +0.11 ms; reranking +47–72 ms; LLM ≈ 1 s uncontended |
| Tests / lint / hashes / secrets | 193 passed; ruff clean; all manifests PASS; 0 key-like strings |

## Rubric self-assessment (an estimate, not a reviewer score)

| Criterion | Weight | Self-score | Reason |
|---|---:|---:|---|
| Functional correctness | 25 | 22 | All 18 configs work; v1 equivalence proven; 193 tests; real-LLM evaluation complete for 3 configs. Answer checker has known false negatives |
| Retrieval improvement evidence | 25 | 19 | Full 18-config comparison on frozen held-out data; reranking and recursive chunking improve retrieval metrics; gains are small (ceiling) and do not translate into better answers |
| Evaluation methodology | 20 | 18 | Canonical fact labels, frozen hashed datasets, pre-registration, exposure handled, metrics hand-tested; single annotator, string-based answer checks |
| Cost/latency analysis | 15 | 12 | Per-stage measured latency, tokens and estimates clearly separated; end-to-end LLM latency confounded by rate limits |
| Code and documentation | 15 | 13 | Modular code, generated reports, guides, evidence index, 8 inspected demo screenshots; DOCX optional/pending |
| **Total** | **100** | **≈ 84** | Pass mark 75 |

## Before submitting (assignee actions)

1. ~~Capture the demo screenshots~~ — done 2026-11-27 (8 screenshots, `03-Evidence/34-…` to `41-…`, inspected).
2. Decide whether to publish a GitHub repository (Claude needs explicit approval) and add the link.
3. Revoke the Groq key that had been written into `.env.example`.
4. Optional: DOCX / Google Doc versions of the documentation.
5. Status becomes **Submitted** only after you confirm the LMS submission.
