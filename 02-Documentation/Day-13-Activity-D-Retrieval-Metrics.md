# Day 13 – Activity D: Retrieval Evaluation Metrics

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Phase:** 5 of 9

## 1. Objective

Understand and implement Precision@K, Recall@K, Hit Rate@K and Mean Reciprocal Rank for ranked retrieval results,
plus distinct evidence-fact coverage, with one consistent relevance definition; build the labelled question set
(`Day-13-Evaluation-Dataset-Methodology.md`); and compare K = 3 with K = 5.

## 2. Relevance definition (identical for every system)

A retrieved chunk is **relevant** if it fully contains at least one canonical evidence ID required by the query (any
group, any alternative). Containment is by character offsets: half a sentence does not count. Because labels are
evidence IDs, the definition works unchanged for all three chunking strategies and for dense, BM25, hybrid and
reranked lists. Per-query numbers are computed by the Phase 3 function `evaluation.metrics.query_metrics` (frozen);
`evaluation/retrieval_metrics.py` adds dataset-level rules.

## 3. Metric definitions

| Metric | Counts | Definition | Notes |
|---|---|---|---|
| Precision@K | **chunks** | relevant chunks in top K ÷ K | K in the denominator even if fewer than K chunks are returned (worked example: 4 returned, P@5 = 2/5) |
| Recall@K | **chunks** | relevant chunks in top K ÷ relevant chunks in the whole index | depends on chunking: an overlap-duplicated fact is two relevant chunks; if no chunk contains a required fact intact, the denominator is 0 → reported as 0 and counted (`recall_zero_denominator`). Not compared across strategies |
| Hit Rate@K | chunks | 1 if any relevant chunk is in the top K | macro mean = share of queries with a hit |
| MRR@K | chunks | mean over queries of 1 / rank of the first relevant chunk within the top K (0 if none) | |
| Coverage@K | **facts** | required fact groups satisfied by the top K ÷ groups | each fact counted once however many chunks hold it; partial credit for multi-fact queries; alternatives in a group are interchangeable |

Chunk-level metrics and fact-level coverage are **reported side by side and never combined into one number**.

**Multiple relevant facts:** a chunk is relevant if it holds any required fact; Coverage@K gives 1/2 when one of two
required facts is retrieved. **Overlap and duplicates:** a chunk can only be retrieved once (repeated IDs in a list
count once); a duplicated fact raises chunk-level precision/recall (two relevant items) but not Coverage@K.
**Unanswerable queries:** they have no relevant items, so retrieval metrics are undefined; they are **excluded** from
all retrieval averages (n = 25 answerable queries in Phase 7) and their count is reported. Correct refusal is an
answer-level metric (Phase 7). **Empty results:** all metrics 0; counted (`empty_results`).

**Aggregation:** macro average over answerable queries (each query weighs the same) — the primary numbers. Coverage
is also reported as micro coverage (satisfied groups ÷ all groups, pooled), labelled separately; it weighs facts
equally. In the hand-worked suite: macro 7/12 = 0.5833, micro 4/7 = 0.5714.

## 4. Implementation

| File | Content |
|---|---|
| `evaluation/retrieval_metrics.py` | `evaluate_ranking` (per query, several K; unanswerable → no metrics; rejects unknown chunk IDs and inconsistent labels), `aggregate` (macro + micro coverage + counts), `compare_k` |
| `evaluation/metrics.py` (Phase 3, unchanged) | per-query Hit / RR / P / R / Coverage |
| `tests_v2/test_retrieval_metrics.py` | 15 hand-worked tests: one relevant chunk (rank 1, rank 3, beyond K), none retrieved, multiple relevant chunks, duplicate fact across overlapping chunks, multiple required facts (partial → full), alternatives, empty results, unanswerable, invalid labels, fact missing from the index, equality with the Phase 3 function, macro aggregation by hand, K=3 vs K=5, only-unanswerable aggregate |
| `tests_v2/test_eval_dataset.py` | 6 tests: freeze hash = record = manifest, validator passes, composition, canonical IDs only, dev separation, validator imports no retrieval code |
| `activities/d-retrieval-metrics/metrics_demo.py` | worked example + K=3 vs K=5 on the dev queries |

## 5. Results (actual; `activities/d-retrieval-metrics/results/metrics-demo-report.md`)

**Worked example** (query needs e1 and e2; ranked B, A, E, D; relevant chunks in index B, C, D):

| K | P@K | R@K | Hit | RR | Coverage |
|---:|---|---|---:|---:|---|
| 1 | 1/1 | 1/3 | 1 | 1.0 | 1/2 |
| 3 | 1/3 | 1/3 | 1 | 1.0 | 1/2 |
| 5 | 2/5 | 2/3 | 1 | 1.0 | 2/2 |

**K = 3 vs K = 5 on the 5 development queries** — recomputed from the rankings saved in Phases 3 and 4 (no retrieval
run, 30-query set not touched); **120/120** per-query values identical to the saved ones. Coverage@K (macro):

| Strategy | Dense | BM25 | Hybrid (RRF) | Hybrid + cross-encoder |
|---|---|---|---|---|
| v1_paragraph_800_150 | 0.80 → 1.00 | 1.00 → 1.00 | 0.80 → 1.00 | 1.00 → 1.00 |
| fixed_500 | 0.80 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |
| recursive_800_100 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 |

Precision falls from K=3 to K=5 for every system (e.g. v1 dense 0.467 → 0.320) while Hit/MRR/coverage rise or stay
equal — the expected trade-off: a larger K finds more but returns more irrelevant context, which costs prompt tokens
(Phase 8). Development numbers only; the held-out comparison is Phase 7.

## 6. Verification

`03-Evidence/18-…` (validation + freeze), `19-…` (label review sheet), `20-…` (metrics demo, 120/120 cross-check),
`21a-…` / `21b-…` (Phase 5 regression checks; 21a contains a flawed check of mine, corrected in 21b). DOCX version: [PENDING – Phase 9].
