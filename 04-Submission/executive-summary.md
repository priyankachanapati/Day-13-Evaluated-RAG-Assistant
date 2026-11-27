# Day 13 – Mini-Project 1: Evaluated RAG Assistant — Executive Summary

**Status:** Completed - Pending Review · Full report: `Day-13-Final-Project-Report.md` (tables generated from saved results)

**Built.** RAG v2 extends the Day 12 Policy Q&A RAG with three chunking strategies (v1 paragraph 800/150, fixed 500,
recursive 800/100), BM25 + dense hybrid retrieval with Reciprocal Rank Fusion (k = 60) and cross-encoder reranking
(`ms-marco-MiniLM-L6-v2`, pinned) — 18 selectable configurations on the unchanged v1 prompt, Groq `gpt-oss-120b`
generator, citation validation and 0.25 dense refusal threshold. The v1-equivalent configuration reproduces v1's
prompts byte for byte (15/15).

**Evaluated.** A frozen 30-query labelled set (v1.1; 25 answerable, 5 unanswerable) with chunk-independent fact labels,
a pre-registered plan, retrieval metrics (Hit@K, MRR, Precision@K, Recall@K, fact coverage@K, K = 3/5) for all 18
configurations, and deterministic answer checks with the real LLM for three pre-registered configurations (30/30 each).

**Results (measured).**
- Retrieval: v1 is near the ceiling (Hit@3 1.00, MRR@3 0.98, coverage@3 0.98). Reranking → MRR@3 1.00; recursive chunking →
  coverage@3 1.00; hybrid without reranking 0.94; fixed 500 chunks 0.84 (splits 19 of 102 facts).
- Answers (deterministic): v1 baseline **27/30**, v1 chunking + hybrid + rerank **27/30** (identical outcome on all 30
  queries), recursive + hybrid + rerank **24/30** (3 fewer, sign test p = 0.25). All three refused 5/5 unanswerable
  questions and had valid citations on every answer. Manual review (reported separately, scores unchanged): all three
  reach 29/30; every difference between them is an answer-checker false negative; the one genuine miss (EV-09) is shared.
- **Selection (pre-registered rule, applied as written): `v1_paragraph_800_150+hybrid+rerank`**, decided by the
  latency tie-break after exact ties on score and context coverage. That latency input is confounded (runs at different
  times under different free-tier rate limits); controlled benchmarks show reranking adds ≈ 63 ms per query. The
  selection is the rule's formal outcome, not evidence that v2 answers better than v1.
- Cost/latency: hybrid +0.11 ms, reranking +47–72 ms on CPU, $0 API cost for both; LLM ≈ 1 s uncontended and
  ≈ $0.00026–0.00027 per query at list price for all three configurations (context stays at 4 chunks).

**Main limitations.** Ceiling effect on a small corpus; confounded latency tie-break; string-based answer checks
(9 false negatives found); single annotator; CPU-only latency measured on one machine.

**Recommendation.** On this corpus RAG v1 is already sufficient for answer quality; reranking gives a measurable
retrieval-ranking gain at ≈ 63 ms and no API cost. Before adopting v2 in production, re-measure end-to-end latency under
identical conditions and evaluate on a larger, harder corpus where retrieval is not at the ceiling.
