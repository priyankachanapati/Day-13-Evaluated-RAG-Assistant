# Activity 2 – BM25 Hybrid Search, Reciprocal Rank Fusion and Cross-Encoder Reranking

**Requirement.** Add BM25 (`rank_bm25`) next to the vector retriever, fuse both rankings with Reciprocal Rank Fusion,
rerank the top 10 fused results with a `sentence-transformers` cross-encoder, and compare vector-only, BM25,
hybrid (RRF) and hybrid + rerank with hit rate and MRR.

| Component | Implementation |
|---|---|
| Vector retriever | unchanged Day 12 dense retriever – [`src/retrieval.py`](../../src/retrieval.py) (MiniLM ONNX + Chroma, cosine) |
| BM25 retriever | [`rag_v2/bm25.py`](../../rag_v2/bm25.py) – `rank_bm25.BM25Okapi` (k1 1.5, b 0.75), tokenizer [`rag_v2/tokenizer.py`](../../rag_v2/tokenizer.py) keeps codes like `HR-003` and numbers |
| RRF | [`rag_v2/fusion.py`](../../rag_v2/fusion.py) – score(d) = Σ 1 / (60 + rank of d in each list); both lists top 10 |
| Hybrid searcher | [`rag_v2/hybrid.py`](../../rag_v2/hybrid.py) – dense top 10 + BM25 top 10 → RRF → top 10 |
| Cross-encoder reranking | [`rag_v2/reranker.py`](../../rag_v2/reranker.py), [`rag_v2/rerank_search.py`](../../rag_v2/rerank_search.py) – `cross-encoder/ms-marco-MiniLM-L6-v2` pinned at `233902d2`, local cache, re-orders exactly the 10 fused candidates |
| This activity's runner | [`compare_four_retrievers.py`](compare_four_retrievers.py) → [`results/four-retrievers-comparison.json`](results/four-retrievers-comparison.json) |
| Earlier detailed reports (unchanged) | [`../b-hybrid-search-rrf/`](../b-hybrid-search-rrf/), [`../c-cross-encoder-reranking/`](../c-cross-encoder-reranking/) |
| Unit tests | [`tests_v2/test_bm25_fusion.py`](../../tests_v2/test_bm25_fusion.py), [`tests_v2/test_reranker.py`](../../tests_v2/test_reranker.py) (hand-calculated RRF scores, pinned model) |

```bash
HF_HUB_OFFLINE=1 .venv/bin/python activities/activity-2-hybrid-rrf-reranking/compare_four_retrievers.py   # local only
```

Same corpus and chunking for every system (Day 12 corpus, `v1_paragraph_800_150`, 22 chunks), same 5 frozen
development questions, same relevance rule (chunk fully contains a required fact). The runner checks that the live
hybrid and reranked top-10 lists equal the saved Activity B and C results (11/11 checks PASS).

## Measured results

| System | Hit@3 | Hit@5 | MRR@3 | MRR@5 | Saved Phase 7 (25 answerable): Hit@3 / MRR@3 / MRR@5 |
|---|---:|---:|---:|---:|---|
| Vector only | 0.80 | 1.00 | 0.80 | 0.85 | 1.00 / 0.98 / 0.98 |
| BM25 only | 1.00 | 1.00 | 0.90 | 0.90 | 0.96 / 0.94 / 0.94 |
| Hybrid (RRF) | 0.80 | 1.00 | 0.80 | 0.85 | 0.96 / 0.96 / 0.97 |
| Hybrid + cross-encoder | **1.00** | **1.00** | **1.00** | **1.00** | **1.00 / 1.00 / 1.00** |

Before/after example (DEV-01, relevant chunk `HR-001-CH02`): vector rank 4, BM25 rank 2, hybrid (RRF) rank 4,
after reranking rank 1 (RR@5 0.25 → 1.00).

**What each part contributes.**
- **BM25** scores exact shared words, codes and numbers, weighted by rarity, which embeddings can blur. It ranks
  DEV-01's relevant chunk 2nd where the vector retriever ranks it 4th, and it is the best single retriever on the
  5 dev questions (MRR@3 0.90), though not on the 30-question set (0.94 vs 0.98).
- **RRF** merges two rankings with incompatible score scales using ranks only; a chunk ranked well by both lists
  rises. Here it helps no more than the vector retriever alone (DEV-01 stays 4th, and on the 30-question set hybrid
  is slightly below vector-only on MRR@3, 0.96 vs 0.98): fusion is only as good as the agreement between the lists.
- **The cross-encoder** reads the question and each candidate together and re-orders the 10 fused candidates. It
  moved the relevant chunk to rank 1 in every case, giving the best MRR on both the 5 dev questions and the saved
  30-question evaluation. Cost: ≈ 47–72 ms per query on CPU (Phase 8 controlled benchmark), no API cost.

The gains are small because the Day 12 retriever is already near the ceiling on this six-policy corpus.
