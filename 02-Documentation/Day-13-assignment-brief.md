# Day 13 — AI05: Advanced RAG, Retrieval Evaluation & Mini-Project 1

**Date assigned:** 2026-11-26
**Due date:** [NOT YET PROVIDED]
**Status:** In Progress (Phases 1–5 completed; Phases 1–9 completed; Completed - Pending Review; screenshots, GitHub and LMS submission pending)
**Blocked reason (if Blocked):** —

> Status definitions: see `MASTER_INSTRUCTIONS.md` §7. Use **Submitted** only after the user confirms actual submission.

## 1. Assignment Brief

Extend the Day 12 Policy Q&A RAG (v1) into an evaluated and improved RAG v2.

- **Activity A – Chunking strategies:** understand character, recursive, document-specific and semantic splitting.
  Re-index the Day 12 corpus with (1) fixed 500-character chunks and (2) recursive 800-character chunks with
  100-character overlap. Compare the top-3 retrieved results for five queries.
- **Activity B – Hybrid search and RRF:** add BM25 keyword retrieval (`rank_bm25`), combine it with vector retrieval
  using Reciprocal Rank Fusion, compare retrieval quality, understand how RRF combines rankings.
- **Activity C – Reranking:** use a sentence-transformers cross-encoder to rerank the top 10 results; compare quality
  before and after reranking.
- **Activity D – Retrieval evaluation:** understand Precision@K and Recall@K and calculate them for ranked results;
  build a labelled question set; implement Hit Rate@K and MRR in Python; compare k=3 and k=5.
- **Activity E – Mini-Project 1: Evaluated RAG Assistant (main deliverable):** (1) labelled evaluation dataset of 30
  queries; (2) evaluate RAG v1 and establish baseline metrics; (3) implement at least two retrieval improvements
  (chunking, hybrid search, reranking, query rewriting); (4) report retrieval and answer-quality metrics before and
  after; (5) measure and report cost and latency impact; (6) deliver working code, evaluation report and a demo.

## 2. Learning Objectives

- Explain and compare chunking strategies and their effect on retrieval.
- Implement sparse (BM25) + dense hybrid retrieval and Reciprocal Rank Fusion.
- Apply cross-encoder reranking and explain the bi-encoder vs cross-encoder trade-off.
- Define, hand-calculate and implement Precision@K, Recall@K, Hit Rate@K and MRR.
- Run a fair, reproducible before/after evaluation and analyse quality vs cost vs latency.

## 3. Requirements & Constraints

- Language / tools: Python 3.12 (uv venv), ChromaDB 1.5.9, local all-MiniLM-L6-v2 (ONNX), Groq `openai/gpt-oss-120b`,
  `rank_bm25`, `sentence-transformers` cross-encoder.
- **Approved decisions (2026-11-26):**
  1. Corpus: only the existing 6-policy Day 12 corpus (not the 20-document activity corpus).
  2. Improvements: all three — chunking comparison (fixed 500, recursive 800/100, existing Day 12 chunking),
     BM25 + dense hybrid with RRF, cross-encoder reranking of the top 10. Query rewriting excluded (future extension).
  3. Layout: Day 12 v1 code, corpus and tests copied verbatim into `Day-13/01-Code/policy-qa-rag-v2/` as the frozen
     baseline (SHA-256 manifests).
  4. API key: Claude does not read or copy the Day 12 key; only `.env.example` is created; the user creates `.env`.
  5. GitHub: no repository creation, push or modification without explicit approval.
- **Additional technical requirements (2026-11-26):**
  1. *Metric definitions:* distinguish conventional Precision@K / Recall@K from evidence-fact coverage; use canonical
     evidence IDs so labels are independent of chunk IDs, strategies are compared fairly, overlapping chunks cannot
     inflate recall, and Hit Rate@K / MRR use the same relevance definition; hand-calculated examples and unit tests
     for every metric.
  2. *Refusal:* preserve v1 refusal behaviour; in v2 evaluate the dense-similarity gate independently of hybrid and
     cross-encoder scores; never compare RRF or cross-encoder scores with the 0.25 cosine threshold; no uncalibrated
     new threshold.
  3. *Integrity:* 5 development queries for tuning; freeze the 30-query dataset (version + SHA-256) before the first
     final v2 evaluation; identical questions, corpus and conditions for v1 and v2; report actual results including
     negative results and regressions; never fabricate or adjust metrics.
  4. *Cost/latency:* separate one-time install/model-load time, embedding + dense retrieval, BM25 + RRF,
     cross-encoder, LLM generation, token usage and estimated API cost; distinguish free-tier actual charges from
     estimated list pricing.

## 4. Evaluation Criteria

| # | Criterion (from assignment) | Weight |
|---|-----------------------------|-------:|
| E1 | Functional correctness | 25% |
| E2 | Retrieval improvement evidence | 25% |
| E3 | Evaluation methodology | 20% |
| E4 | Cost/latency analysis | 15% |
| E5 | Code and documentation | 15% |

Minimum passing score: 75%.

## 5. Task Checklist (phases)

- [x] Phase 1 – Workspace setup, verbatim v1 copy, manifests, uv environment, v1 tests, baseline reproduction
- [x] Phase 2 – Chunking comparison (Activity A)
- [x] Phase 3 – BM25 hybrid retrieval and RRF (Activity B)
- [x] Phase 4 – Cross-encoder reranking (Activity C)
- [x] Phase 5 – Retrieval metrics and labelled dataset (Activity D)
- [x] Phase 6 – RAG v2 integration (Mini-Project 1) — mocked-LLM integration complete; [x] real-LLM dev smoke test (completed 17:13 UTC via --resume: v1 baseline 4/5, hybrid+rerank 3/5 correct on dev; refusals 6/6)
- [x] Phase 7 – Baseline vs improved evaluation (complete: 18 configs retrieval, 3 × 30 answers)
- [x] Phase 8 – Cost and latency analysis
- [x] Phase 9 – Testing, documentation, evidence and demo (screenshots pending: user)

## 6. Dependencies & Missing Information

| Item | Type | Resolution |
|------|------|------------|
| Day 12 RAG v1 code, corpus, tests | Dependency | RESOLVED: copied verbatim, verified by SHA-256 against Day 12 files and git commit b366fc8 |
| `rank-bm25`, `sentence-transformers` | Dependency | RESOLVED: installed in Day 13 venv (rank-bm25 0.2.2, sentence-transformers 6.3.0, torch 2.15.0) |
| Groq API key in Day 13 `.env` | Dependency | RESOLVED: valid key configured by the user (smoke test completed); recommended: revoke the key that had been written into `.env.example` |
| Groq free-tier daily quota | Dependency / risk | PENDING: check with one tiny request before answer-quality runs |
| Cross-encoder model download (~90 MB, Hugging Face) | Dependency | RESOLVED: `cross-encoder/ms-marco-MiniLM-L6-v2` (canonical name of the plan's `…L-6-v2` alias) pinned at 233902d2…; 91.8 MB; 26.8 s one-time |
| Due date | Missing info | [NOT YET PROVIDED] |

## 7. Implementation Plan

See the approved 9-phase roadmap (Phase 1 report: `Day-13-Phase-1-Setup-and-Verification.md`).

## 8. Deliverables

| Deliverable | Location | Status |
|-------------|----------|--------|
| Frozen v1 baseline + manifests | `Day-13/01-Code/policy-qa-rag-v2/{src,tests,data,manifests}` | Completed - Pending Review |
| Day 13 environment (lock file) | `Day-13/01-Code/policy-qa-rag-v2/requirements.txt` | Completed - Pending Review |
| Activity A code, report and write-up | `01-Code/policy-qa-rag-v2/activities/a-chunking-strategies/`, `02-Documentation/Day-13-Activity-A-Chunking-Strategies.md` | Completed - Pending Review |
| Canonical evidence catalog (102 units) + 5 frozen dev queries | `01-Code/policy-qa-rag-v2/evaluation/datasets/` | Completed - Pending Review |
| Activity B code, report and write-up | `01-Code/policy-qa-rag-v2/activities/b-hybrid-search-rrf/`, `02-Documentation/Day-13-Activity-B-Hybrid-Search-RRF.md` | Completed - Pending Review |
| Retrieval metrics module (Hit@K, MRR, P@K, R@K, Coverage@K) | `01-Code/policy-qa-rag-v2/evaluation/metrics.py` | Completed - Pending Review (extended in Phase 5) |
| Activity C code, report and write-up | `01-Code/policy-qa-rag-v2/activities/c-cross-encoder-reranking/`, `02-Documentation/Day-13-Activity-C-Cross-Encoder-Reranking.md` | Completed - Pending Review |
| 30-query labelled evaluation dataset (frozen v1.0, SHA-256 c9677c0f…) | `01-Code/policy-qa-rag-v2/evaluation/datasets/eval_queries_30.json`, `02-Documentation/Day-13-Evaluation-Dataset-Methodology.md` | Completed - Pending Review |
| Activity D metrics, demo and write-up | `01-Code/policy-qa-rag-v2/evaluation/retrieval_metrics.py`, `activities/d-retrieval-metrics/`, `02-Documentation/Day-13-Activity-D-Retrieval-Metrics.md` | Completed - Pending Review |
| RAG v2 pipeline (configurable, 18 configs), CLI, dev answer-evaluation runner | `01-Code/policy-qa-rag-v2/rag_v2/`, `scripts/run_dev_answer_eval.py`, `02-Documentation/Day-13-RAG-v2-Architecture-and-Configuration.md` | Completed - Pending Review (real-LLM smoke pending) |
| Answer evaluation (deterministic) | `01-Code/policy-qa-rag-v2/evaluation/answer_eval.py`, `02-Documentation/Day-13-Answer-Evaluation-Methodology.md` | Completed - Pending Review |
| Evaluation report (incl. cost/latency) | `02-Documentation/Day-13-Mini-Project-1-Evaluation-Report.md` | Not Started |
| Demo | `03-Evidence/` | Not Started |
| Submission summary and checklist | `04-Submission/` | Not Started |

## 9. Test Results

| Test / Check | Command | Result | Notes |
|--------------|---------|--------|-------|
| Manifest verification (Day 13 vs Day 12 files vs git b366fc8) | `bash scripts/verify_manifests.sh` | Pass | 20 code/test files + 6 corpus files + 2 reference files |
| Copied v1 offline tests | `.venv/bin/python -m pytest -m "not integration" -v` | Pass | 42 passed, 2 deselected (same as Day 12) |
| v1 baseline reproduction | `.venv/bin/python scripts/verify_v1_baseline_reproduction.py` | Pass | 15/15 questions; 60/60 chunk scores exact at 4 dp (re-checked after Phase 2) |
| Activity A comparison (2 runs) | `.venv/bin/python activities/a-chunking-strategies/compare_chunking.py` | Pass | integrity gate PASS; identical results SHA-256 in both runs |
| v1 + v2 tests | `.venv/bin/python -m pytest tests tests_v2 -m "not integration"` | Pass | 67 passed, 2 deselected |
| Recursive splitter vs LangChain 1.2.0 | see `03-Evidence/06-…` | Pass | 6/6 documents byte-identical |
| Activity B comparison (3 runs) | `.venv/bin/python activities/b-hybrid-search-rrf/compare_retrieval.py` | Pass | integrity gate PASS; dense = Phase 2 15/15; identical results SHA-256 in all runs |
| v1 + v2 tests after Phase 3 | `.venv/bin/python -m pytest tests tests_v2 -m "not integration"` | Pass | 109 passed, 2 deselected |
| Activity C reranking (3 runs) | `HF_HUB_OFFLINE=1 .venv/bin/python activities/c-cross-encoder-reranking/compare_reranking.py` | Pass | pool = Phase 3 hybrid 15/15; identical hash; scores bit-identical across processes |
| v1 + v2 tests after Phase 4 | `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests tests_v2 -m "not integration"` | Pass | 119 passed, 2 deselected |
| Eval dataset validation + freeze | `.venv/bin/python scripts/validate_eval_dataset.py --freeze` | Pass | 30 queries (25/5); frozen 2026-11-26T16:26:53Z |
| Activity D demo | `.venv/bin/python activities/d-retrieval-metrics/metrics_demo.py` | Pass | 120/120 recomputed = saved Phase 3/4 metrics |
| v1 + v2 tests after Phase 5 | `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests tests_v2 -m "not integration"` | Pass | 140 passed, 2 deselected |
| v1 + v2 tests after Phase 6 | same | Pass | 185 passed, 2 deselected |
| v1 equivalence of RAG v2 | unit test + `03-Evidence/24-…` | Pass | identical prompts and statuses, 15/15 |
| Real-LLM dev smoke test | `scripts/run_dev_answer_eval.py --run-name real-smoke-1 [--resume]` | Pass (complete) | first attempt: 16/16 auth errors (runner bug fixed); resumed: 16/16 items, citations valid 10/10, refusals 6/6, 16,904 tokens spent, est. USD 0.0038 |
| Dataset v1.1 validation + freeze | `.venv/bin/python scripts/validate_eval_dataset_v1_1.py --freeze` | Pass | 18/18 checks; v1.0 unchanged |
| Tests before Phase 7 | `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests tests_v2 -m "not integration"` | Pass | 191 passed, 2 deselected |

## 10. Evidence Mapping

See `Day-13/03-Evidence/evidence-index.md`.

## 11. Outstanding Items

- [ ] User: create `Day-13/01-Code/policy-qa-rag-v2/.env` from `.env.example`.
- [ ] User (recommended): revoke the Groq key that had been written into `.env.example`.
- [ ] User: capture screenshots; decide on GitHub; LMS submission.

## 12. Questions / Clarifications

- Due date not yet provided.

## 13. Notes

- Day 12 was not modified (Day 12 git working tree clean before and after Phase 1).
