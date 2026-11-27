# Day 13 – Activity C: Cross-Encoder Reranking

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Phase:** 4 of 9

## 1. Objective

Rerank the hybrid (BM25 + dense, RRF k = 60) top-10 candidate pool with a sentence-transformers cross-encoder for
all three chunking strategies, and compare before vs after on the same pool and the same five development queries —
retrieval metrics, fact movements, unrecoverable facts and latency — without changing generation, prompts, the
refusal threshold or the context size, and without LLM calls.

## 2. Concepts: bi-encoder vs cross-encoder

- The **dense retriever is a bi-encoder**: query and chunk are embedded *separately*, so chunk vectors are computed
  once at indexing time and search is a fast vector comparison over the whole index.
- A **cross-encoder** reads the query and one chunk *together* in one transformer pass, so every query token can
  attend to every chunk token; it outputs one relevance score per pair. More accurate, but it must run once per
  pair at query time — so it is applied only to a short candidate list (retrieve-then-rerank).
- **Reranking can only re-order what retrieval found.** A fact absent from the candidate pool cannot be recovered;
  pool coverage is the ceiling for reranking.

## 3. Model verification and configuration

| Item | Value / check |
|---|---|
| Requested ID (project plan) | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Hub resolution | the Hub resolves it to the renamed repository `cross-encoder/ms-marco-MiniLM-L6-v2`; both names return the same repository and commit (checked) |
| Pinned revision | `233902d25c440f23af6f7d6e94d2946bac0bee0a` (Hub last modified 2026-08-09); loaded with `revision=` and `local_files_only=True` |
| License / task | apache-2.0; `text-ranking`; library sentence-transformers |
| Files downloaded | 7 files, 91,819,431 bytes (`model.safetensors` + tokenizer/config/README); every size matches the Hub metadata. The ONNX/OpenVINO/Flax copies (repo total 890 MB) were not downloaded |
| Architecture | BertForSequenceClassification, 6 layers, hidden size 384, 12 heads, max 512 positions |
| Output | 1 label, `activation_fn = Identity` → raw relevance logit |
| Compatibility | sentence-transformers 6.3.0, transformers 5.24.1, torch 2.15.0 (already in the Day 13 lock; nothing new installed) |
| Input format | `(query, policy title + chunk text)` — the same text the dense and BM25 sides use; longest pair 206 tokens ≤ 512 (no truncation) |
| Determinism | CPU, the whole pool scored in one batch. In-process repeat identical; batch 1 vs batch 10 differs by ~1e-6 (padding), hence the fixed batch; scores **bit-identical across separate processes** (150/150, max diff 0) |

## 4. Implementation

| Component | File | Notes |
|---|---|---|
| Download | `scripts/download_cross_encoder.py` | pinned revision, allow-listed files, size verification, writes `results/cross-encoder-model-info.json` |
| Reranker | `rag_v2/reranker.py` | `format_pairs`; `CrossEncoderReranker.rerank`: same chunks re-ordered by logit; ties keep the incoming order; empty list → empty, model not called; duplicate IDs rejected; scorer injectable for tests |
| Pipeline | `rag_v2/rerank_search.py` | wraps the unchanged Phase 3 `HybridSearcher`; candidates = hybrid top-10; per-stage timings |
| Comparison | `activities/c-cross-encoder-reranking/compare_reranking.py` | integrity gate (v1, corpus, Phase 2 and 3 manifests, catalog, dev queries, model revision); pool = Phase 3 hybrid top-10 exactly; permutation enforced; no-truncation check; generated report + JSON |

**Score types.** RRF (before) only encodes ranks from dense and BM25 (≈ 0.014–0.033). The cross-encoder logit (after)
is an unbounded relevance estimate for one pair and is used **only to order the 10 candidates**. It is not a
probability, not comparable with cosine, BM25 or RRF, and **does not replace the v1 dense-similarity refusal gate**,
which is unchanged (best cosine ≥ 0.25; 15/15 pass). No cross-encoder threshold was introduced — and the data show why
an uncalibrated one would be wrong: for DEV-01 (v1) *every* candidate scores below 0, including the correct one
(−1.23), so a "score > 0" rule would refuse an answerable question.

## 5. Results (actual; `results/reranking-report.md`)

### 5.1 Before vs after (mean over 5 dev queries)

| Strategy | Stage | Hit@3 | MRR@3 | P@3 | Coverage@3 | Coverage@5 | Pool coverage (@10) |
|---|---|---:|---:|---:|---|---|---|
| v1_paragraph_800_150 | hybrid (before) | 0.80 | 0.80 | 0.47 | 0.80 (6/7) | 7/7 | 7/7 |
| v1_paragraph_800_150 | + rerank (after) | **1.00** | **1.00** | 0.53 | **1.00 (7/7)** | 7/7 | 7/7 |
| fixed_500 | hybrid (before) | 1.00 | 0.90 | 0.40 | 1.00 (7/7) | 7/7 | 7/7 |
| fixed_500 | + rerank (after) | 1.00 | **1.00** | 0.40 | 1.00 (7/7) | 7/7 | 7/7 |
| recursive_800_100 | hybrid (before) | 1.00 | 1.00 | 0.47 | 1.00 (7/7) | 7/7 | 7/7 |
| recursive_800_100 | + rerank (after) | 1.00 | 1.00 | 0.47 | 1.00 (7/7) | 7/7 | 7/7 |

Recall@K (chunk-level) is in the generated report; as in Phase 3 it is not compared across strategies. Per query:
reranking was **better on 2 of 15** query × strategy cases (DEV-01 for v1 and fixed_500), **same on 13, worse on 0**.

### 5.2 Fact movements within the pool (24 required-fact rows)

- **Moved up: 5** — notably DEV-01 v1: `HR-001-S02-E04` from pool rank **4 → 1** (entered the top-3). This is the
  case RRF could not fix in Phase 3: the cross-encoder scores the annual-leave chunk −1.23 and pushes RRF's #1 (the
  work-from-home "request … week" chunk) down to #4 (−6.30). fixed_500 DEV-01: 2 → 1.
- **Moved down: 2**, neither reducing Coverage@3:
  - DEV-03 v1: the two alternative sentences that both answer the question (extension 4357 is stated in IT-001 and
    IT-002) swapped ranks 1 and 2 — harmless.
  - DEV-04 v1: `HR-002-S06-E02` 1 → 2. The cross-encoder preferred `HR-002-CH04` (Wi-Fi rule only, via overlap) over
    `HR-002-CH03` (contains **both** required facts). A cross-encoder scores each chunk's relevance on its own; it
    does not reward a chunk for covering more parts of a multi-part question.
- **Unchanged: 17** (already at rank 1 or 2).
- **Absent from the pool (unrecoverable by reranking): 0.** Pool coverage was 7/7 for every strategy, so this case
  did not occur on these queries; the behaviour is covered by a unit test (a relevant chunk outside the pool keeps
  recall below 1 after reranking). No required fact was split in any strategy for these queries.

### 5.3 Latency (Apple M3 CPU, 4 threads; model warm; 100 timed searches per cell; ranges over 3 runs)

| Strategy | Hybrid (no rerank) median | Rerank 10 pairs median | Hybrid + rerank median | Total ÷ hybrid |
|---|---:|---:|---:|---:|
| v1_paragraph_800_150 | 19.4 – 20.5 ms | 60.9 – 63.3 ms | 80.7 – 83.8 ms | ≈ 4.2× |
| fixed_500 | 20.0 – 20.8 ms | 45.6 – 47.4 ms | 65.2 – 68.5 ms | ≈ 3.3× |
| recursive_800_100 | 20.4 – 21.4 ms | 66.6 – 71.7 ms | 87.1 – 93.2 ms | ≈ 4.3× |

- Reranking costs ~46–72 ms per query — **2–3.5× the whole hybrid retrieval**. It is faster for fixed_500, which is
  consistent with its shorter chunks (mean 442 vs ~638 characters, Phase 2).
- p95 was within ~13 % of the median in runs 1–2; in run 3 recursive_800_100 had tail outliers (hybrid p95 32.6 ms,
  rerank p95 92.9 ms, total p95 124.1 ms) while its medians stayed in range — reported as measured.
- **One-time costs (not in the per-query numbers):** download 26.8 s for 91.8 MB (first run only); sentence-transformers
  import (loads torch/transformers) 2.1–2.5 s; model load from local cache 0.15–0.77 s (the first load after
  download was slowest).
- No API cost: the cross-encoder runs locally. LLM cost/latency is analysed in Phase 8.

## 6. Conclusion — nothing selected

Reranking improved the top-3 where the pool contained the answer but the hybrid order did not (DEV-01 for v1 and
fixed_500) and never reduced Coverage@3; it adds roughly 45–70 ms per query on CPU. With four of five queries at the
ceiling and all differences coming from one query, this is development evidence only. **No chunking strategy or
retrieval configuration is selected**; that decision is made on the frozen 30-query held-out evaluation (Phase 7).
Generator, prompts, answer evaluation, refusal threshold and final context size are unchanged; no LLM was called.

## 7. Verification

| Check | Result | Evidence |
|---|---|---|
| Model ID / revision / file sizes | alias → canonical, same commit; 7/7 sizes match | `03-Evidence/14-…`, `results/cross-encoder-model-info.json` |
| Integrity gate incl. Phase 2 and Phase 3 manifests | PASS | report §0 |
| Pool identical to Phase 3 hybrid top-10 (IDs and exact RRF scores) | 15/15 | report §0 |
| Reranked list = permutation of the pool | 15/15 (enforced at run time) | report §0 |
| No truncation | longest pair 206 ≤ 512 tokens | report §0 |
| Reproducibility | 3 runs (run 3 after formatting): identical hash `e70b4ff0…`; full-precision scores bit-identical across processes | `15-…`, `16-…`, `17-…` |
| Tests | 119 passed, 2 deselected (42 frozen v1 + 77 v2; new: 10 reranker tests incl. one with the real model offline) | `17-…` |
| Frozen baseline, Phase 2/3 artifacts, catalog, Phase 1 reproduction, Day 12 | all PASS / unchanged; Day 12 git clean; 0 Phase 4 files reference an LLM client | `17-…` |

## 8. Issues encountered

- The plan's model ID is now a Hub alias (repository renamed to `…MiniLM-L6-v2`). Verified both resolve to the same
  repository and commit; the canonical ID plus pinned revision is used, and the requested ID is recorded.
- `ruff` fixes after the runs were cosmetic (import order, one unused test import); run 3 after the fixes gave the
  same results hash.

## 9. Files

`rag_v2/{reranker,rerank_search}.py`, `scripts/download_cross_encoder.py`, `results/cross-encoder-model-info.json`,
`activities/c-cross-encoder-reranking/{compare_reranking.py, README.md, results/}`, `tests_v2/test_reranker.py`,
`manifests/phase3-artifacts.sha256`. DOCX version: [PENDING – Phase 9].
