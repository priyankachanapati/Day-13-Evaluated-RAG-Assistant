# Activity B – Hybrid Search (BM25 + Dense) with Reciprocal Rank Fusion

Compares three retrieval methods on each of the three Activity A chunking strategies, with everything else held
constant (frozen corpus, same chunks, all-MiniLM-L6-v2 ONNX embeddings, same 5 frozen dev queries, top-10 candidates
for every method):

| Method | Implementation |
|---|---|
| Dense only | frozen v1 `Retriever` (query embedding + Chroma cosine search), top-10 |
| BM25 only | `rag_v2.bm25.BM25Index` – `rank_bm25.BM25Okapi` (defaults k1=1.5, b=0.75, ε=0.25) over title + chunk text, tokenized by `rag_v2.tokenizer.tokenize`; zero-score chunks not returned |
| Hybrid | `rag_v2.fusion.rrf_fuse` – RRF (k=60) of the dense top-10 and BM25 top-10 lists |

`rag_v2.hybrid.HybridSearcher` runs all three and times dense, BM25 and fusion separately. The v1 refusal gate is
evaluated on dense similarity only. No reranking here (Activity C).

## Run

```bash
.venv/bin/python activities/b-hybrid-search-rrf/compare_retrieval.py
```

No API key, no network. Refuses to run unless the corpus, frozen v1 code, Phase 2 artifacts, evidence catalog and dev
queries match their manifests; checks that dense top-3 equals the Phase 2 results exactly.

## Outputs (`results/`)

- `hybrid-retrieval-report.md` – generated: settings and score scales, tokenizer examples, metric definitions, per-query
  top-3 for 3 strategies × 3 methods (chunk IDs, sections, ranks, method-specific scores, fact IDs, BM25 matched terms,
  RRF source ranks), per-query metrics at K=3/5, ablation summary, hybrid-vs-dense/BM25 verdicts, dense gate, latency.
- `hybrid-retrieval.json` – all raw data. `results_sha256` covers everything except run metadata and latency
  (latency is under `latency_not_hashed`).
