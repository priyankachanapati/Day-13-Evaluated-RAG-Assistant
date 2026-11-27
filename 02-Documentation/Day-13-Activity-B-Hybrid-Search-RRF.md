# Day 13 – Activity B: Hybrid Search with BM25 and Reciprocal Rank Fusion

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Phase:** 3 of 9

## 1. Objective

Add BM25 keyword retrieval (`rank_bm25`) for all three chunking strategies, combine it with dense vector retrieval
using Reciprocal Rank Fusion (k = 60), and compare dense-only, BM25-only and hybrid retrieval fairly on the five frozen
development queries — with evidence-fact relevance, explicit metric definitions and separately measured latency.

## 2. Concepts

- **Dense retrieval** embeds the query and chunks into vectors and ranks by cosine similarity. It handles paraphrases
  ("holiday" ≈ "leave") but can miss exact identifiers and can be pulled by topical words.
- **BM25** ranks by weighted term overlap: a term counts more if it is rare in the collection (IDF) and appears in
  the chunk, with saturation for repeated terms (k1) and length normalisation (b). It is precise for codes, numbers
  and rare names, but only matches tokens that are literally present — so tokenization matters.
- **Reciprocal Rank Fusion** merges ranked lists using ranks only: `RRF(d) = Σ 1/(k + rank(d))`. Raw scores are
  ignored because cosine (≈ −1…1) and BM25 (unbounded) are on incompatible scales. With k = 60 the weights are
  flat (rank 1 = 0.01639, rank 10 = 0.01429), so a chunk ranked reasonably by **both** lists usually beats a chunk
  ranked first by only one.

## 3. Implementation

| Component | File | Key decisions |
|---|---|---|
| Tokenizer | `rag_v2/tokenizer.py` | Rules fixed before any BM25 result: policy codes atomic (`HR-003` → `hr-003`); other hyphenated terms emit compound + parts (`work-from-home` → `work-from-home work home`); `1,000` → `1000`; decimals/times whole (`0.40`, `10:00`); e-mails whole + parts; ALL-CAPS acronyms protected from stopwords (`IT Service Desk` keeps `it`, "it applies" drops it); possessive `'s` removed; light plural folding; small explicit stopword list (negations kept) |
| BM25 | `rag_v2/bm25.py` | `BM25Okapi` with library defaults (k1 = 1.5, b = 0.75, ε = 0.25; not tuned); indexes **title + chunk**, the same text the dense side embeds; zero-score chunks are not returned; ties broken by index position |
| RRF | `rag_v2/fusion.py` | k = 60; duplicates inside one list ignored; deterministic ties (score → best rank → list order, dense first → chunk ID) |
| Hybrid searcher | `rag_v2/hybrid.py` | dense top-10 (v1 `Retriever`) + BM25 top-10 → RRF top-10; dense, BM25 and fusion timed separately; v1 dense gate reported separately |
| Metrics | `evaluation/metrics.py` | one relevance rule for all metrics (below) |
| Comparison | `activities/b-hybrid-search-rrf/compare_retrieval.py` | integrity gate; generated report + JSON; latency outside the reproducibility hash |

**Fairness controls:** same corpus, same chunks per strategy (Phase 2 chunkers, unchanged), same embedding model and
embedded text, same five frozen queries, same candidate depth (10) for every method; dense top-3 verified identical to
Phase 2 for 15/15 query × strategy pairs.

## 4. Relevance and metric definitions

- **Relevant chunk:** a chunk that fully contains at least one required canonical fact (evidence ID) of the query.
  The same rule is used for every metric, so Hit@K and MRR are consistent with precision and recall.
- **Hit@K**, **RR@K / MRR** (1 / rank of the first relevant chunk in the top K), **Precision@K** (relevant chunks in
  top K ÷ K) and **Recall@K** (relevant chunks in top K ÷ relevant chunks in the index) are the conventional,
  chunk-level metrics. Recall@K depends on the chunking — with overlap the same fact lives in several chunks and each
  counts — so it is **not** compared across strategies.
- **Coverage@K** (evidence-fact coverage) = required fact groups satisfied by the top-K chunks ÷ groups. Each fact
  counts once, however many retrieved chunks contain it, so overlap duplicates cannot inflate it.
- Hand-calculated unit tests cover every metric, including a case where retrieving an overlap duplicate doubles
  chunk recall while coverage stays unchanged.

## 5. Results (actual; `results/hybrid-retrieval-report.md`)

### 5.1 Ablation (mean over 5 dev queries, K = 3)

| Strategy | Method | Hit@3 | MRR@3 | P@3 | Coverage@3 (groups) |
|---|---|---:|---:|---:|---|
| v1_paragraph_800_150 | Dense only | 0.80 | 0.80 | 0.47 | 0.80 (6/7) |
| v1_paragraph_800_150 | BM25 only | **1.00** | **0.90** | 0.53 | **1.00 (7/7)** |
| v1_paragraph_800_150 | Hybrid (RRF) | 0.80 | 0.80 | 0.47 | 0.80 (6/7) |
| fixed_500 | Dense only | 0.80 | 0.80 | 0.33 | 0.80 (6/7) |
| fixed_500 | BM25 only | **1.00** | **1.00** | 0.40 | **1.00 (7/7)** |
| fixed_500 | Hybrid (RRF) | **1.00** | 0.90 | 0.40 | **1.00 (7/7)** |
| recursive_800_100 | Dense only | **1.00** | **1.00** | 0.47 | **1.00 (7/7)** |
| recursive_800_100 | BM25 only | **1.00** | 0.90 | 0.47 | **1.00 (7/7)** |
| recursive_800_100 | Hybrid (RRF) | **1.00** | **1.00** | 0.47 | **1.00 (7/7)** |

At K = 5 every strategy and method reaches Hit = 1.00 and Coverage = 7/7.

### 5.2 Where hybrid helps and where it does not

All differences come from **DEV-01** ("How far in advance do I have to request a two-week holiday?"). DEV-02 to
DEV-05 put a relevant chunk at rank 1 for every strategy and method, so they cannot separate the methods (ceiling).

| Strategy | First relevant rank: dense / BM25 / hybrid | Hybrid vs dense | Hybrid vs BM25 |
|---|---|---|---|
| v1_paragraph_800_150 | 4 / 2 / 4 | same (no gain) | worse |
| fixed_500 | 4 / 1 / 2 | **better** | worse |
| recursive_800_100 | 1 / 2 / 1 | same | **better** |

Why (from the RRF scores in the report):
- **Where BM25 helps:** BM25 matches `advance` and `request` in the annual-leave chunk, which the dense model ranks 4th
  for v1 and fixed_500 because "holiday" pulls the *Public Holidays* chunk up.
- **v1 – hybrid does not help.** The relevant chunk `HR-001-CH02` gets 1/64 + 1/62 = 0.03175 (dense 4, BM25 2). Three
  chunks score higher because both lists rank them high: `HR-002-CH02` (work-from-home requests; BM25 matched
  `request`, `week`) d3+b1 = 0.03227, `HR-001-CH05` d1+b4 = 0.03202 and `HR-001-CH04` (Public Holidays) d2+b3 =
  0.03200. RRF rewards agreement, and here the two lists agree on wrong chunks.
- **fixed_500 – hybrid helps (rank 4 → 2)**, but by a margin of 0.00002 (0.03202 vs 0.03200): RRF scores near the
  top are tightly packed, so small rank shifts reorder them.
- **recursive_800_100 – hybrid keeps dense's rank 1** and fixes BM25's rank 2.
- Hybrid never fell below dense on any query or strategy (0 worse of 15); it fell below BM25-only twice.

### 5.3 BM25 behaviour worth knowing

- In these small indexes (20–28 chunks), terms in more than half of the chunks get a *negative* IDF that rank_bm25
  replaces with ε × mean IDF ≈ 0.53–0.62 — e.g. `policy`, `employee`, `day`, `brightlake` (present via the title in
  every chunk). That floor gives very common terms slightly *more* weight than terms in just under half the chunks.
  Kept at library defaults (no tuning on 5 queries); documented and unit-tested.
- BM25 can also mislead: `request` + `week` made the work-from-home chunk BM25's #1 for DEV-01 (v1).

### 5.4 Refusal gate (unchanged v1 rule)

Evaluated on dense similarity only (best cosine ≥ 0.25); all 15 query × strategy cases pass (best cosine 0.2950 –
0.6565). BM25 and RRF scores were never compared with 0.25, and no new threshold was introduced.

### 5.5 Latency (CPU, Apple M3, 100 timed searches per cell after 3 warm-ups; ranges over 3 runs and 3 strategies)

| Component | Median | p95 (max over runs) |
|---|---:|---:|
| Dense retrieval total (query embedding + Chroma search) | 19.1 – 21.2 ms | ≤ 22.1 ms |
|   └ query embedding (MiniLM ONNX) | 18.0 – 20.0 ms | |
|   └ Chroma vector search (top-10) | 1.11 – 1.17 ms | |
| BM25 (tokenize + score + sort) | 0.083 – 0.092 ms | ≤ 0.15 ms |
| RRF fusion | 0.020 – 0.022 ms | ≤ 0.03 ms |

**Hybrid adds about 0.11 ms (BM25 + fusion) to a ~20 ms dense search (~0.5 %)**; the query embedding dominates.
One-time: BM25 index build 1.6–1.9 ms per strategy; dense index build (embedding all chunks) 382–638 ms; model load
22–79 ms (depends on OS file cache). Full cost/latency analysis including the LLM is Phase 8.

## 6. Conclusion — nothing selected

On these five queries, hybrid **never hurt** dense retrieval and **helped once** (fixed_500, DEV-01); BM25 alone did
best on DEV-01 for v1 and fixed_500. These are single-query differences on a development sample where four of five
queries are already at the ceiling, so they are **not** evidence for choosing a method or strategy. As agreed, no
chunking strategy or retrieval method is selected; all three strategies and all three methods remain available for
reranking (Phase 4) and the frozen 30-query evaluation (Phase 7).

## 7. Verification

| Check | Result | Evidence |
|---|---|---|
| Integrity gate (corpus, v1 code, Phase 2 artifacts, catalog, dev queries) | PASS | report §0 |
| Dense top-3 identical to Phase 2 | 15/15 query × strategy | report §0 |
| Reproducibility (3 runs, indexes rebuilt each time; run 3 after formatting) | identical results SHA-256 `db9a570e…` | `03-Evidence/11-…`, `12-…`, `13-…` |
| Tests | 109 passed, 2 deselected: 42 frozen v1 + 67 v2 (new in Phase 3: 16 tokenizer, 11 BM25/RRF, 7 metrics, 8 hybrid) | `03-Evidence/13-…` |
| Frozen baseline, Phase 2 artifacts, catalog, Phase 1 reproduction, Day 12 | all PASS / unchanged; Day 12 git clean | `03-Evidence/13-…` |

## 8. Issues encountered

1. **Tokenizer bug (fixed before the comparison ran):** plural folding applied only to words longer than 4 letters,
   so `days` stayed `days` although my own documented rule says `days → day`. Changed the limit to longer than 3
   letters to match the specification; covered by a test.
2. **Wrong test assumption:** a BM25 tie test used a 3-chunk toy where the tied term occurs in 2 of 3 chunks; its IDF
   is negative (and the floor is negative too because the toy's mean IDF is negative), so BM25 correctly returned
   nothing. Rewrote the tie test with a rare term and added a test documenting this property.
3. One `# noqa: SIM905` (stopword list kept as a readable word block).

## 9. Files

`rag_v2/{tokenizer,bm25,fusion,hybrid}.py`, `evaluation/metrics.py`,
`activities/b-hybrid-search-rrf/{compare_retrieval.py, README.md, results/}`,
`tests_v2/{test_tokenizer,test_bm25_fusion,test_metrics,test_hybrid}.py`, `manifests/phase2-artifacts.sha256`.
DOCX version: [PENDING – Phase 9].
