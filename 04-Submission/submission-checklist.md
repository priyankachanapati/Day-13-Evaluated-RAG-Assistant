# Submission Checklist – Day 13

| # | Requirement (assignment) | Status | Evidence |
|---|---|---|---|
| 1 | Labelled evaluation dataset of 30 queries | Done | `evaluation/datasets/eval_queries_v1_1.json` (frozen, SHA-256 e860b729…); `03-Evidence/18`, `25` |
| 2 | Evaluate RAG v1 and establish baseline metrics | Done | `results/phase7/phase7-report.md` (v1 retrieval + 27/30 answers) |
| 3 | At least two retrieval improvements | Done (3: chunking, hybrid BM25 + RRF, cross-encoder reranking) | Activities A–C write-ups; `03-Evidence/07–17` |
| 4 | Retrieval and answer-quality metrics before/after | Done | Final report §5–6; `03-Evidence/29`, `30`, `34` |
| 5 | Cost and latency impact | Done (measured / estimate / unavailable separated) | `results/phase8/cost-latency-report.md`; `03-Evidence/32`, `35` |
| 6 | Working code, evaluation report, demo | Done (code, report, demo walkthrough, 8 demo screenshots) | `01-Code/`; `04-Submission/`; `02-Documentation/Day-13-Demo-Walkthrough.md` |
| 7 | Activity A – chunking comparison, top-3 for 5 queries | Done | `Day-13-Activity-A-Chunking-Strategies.md` |
| 8 | Activity B – BM25 + RRF | Done | `Day-13-Activity-B-Hybrid-Search-RRF.md` |
| 9 | Activity C – cross-encoder rerank of top 10 | Done | `Day-13-Activity-C-Cross-Encoder-Reranking.md` |
| 10 | Activity D – P@K, R@K, Hit@K, MRR, K=3 vs K=5 | Done | `Day-13-Activity-D-Retrieval-Metrics.md` |
| 11 | Tests pass | Done (193 passed, 2 live tests deselected) | `04-Submission/test-log.txt` |
| 12 | Frozen work unchanged (Day 12, v1, datasets, pre-registration) | Done | `03-Evidence/36-final-checks.txt` |
| 13 | No secrets in files | Done (0 key-like strings; `.env` git-ignored) | `03-Evidence/36-final-checks.txt` |
| 14 | Screenshots | Done (8, captured by the user, inspected) | `03-Evidence/34-…` to `41-…`; evidence index |
| 15 | GitHub repository | PENDING (requires approval) | — |
| 16 | LMS submission | PENDING (user) | — |
