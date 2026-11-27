# Activity D – Retrieval Evaluation Metrics

- Metrics: `evaluation/retrieval_metrics.py` (dataset-level rules) on top of `evaluation/metrics.py` (per query).
  Chunk-level: Precision@K, Recall@K, Hit Rate@K, MRR. Fact-level (separate): distinct evidence Coverage@K.
  Unanswerable queries are excluded from retrieval averages; macro averages are primary.
- Labelled dataset: `evaluation/datasets/eval_queries_30.json` (v1.0, frozen; SHA-256 in
  `manifests/eval-dataset.sha256`), validated by `scripts/validate_eval_dataset.py`. Held out for Phase 7.

```bash
.venv/bin/python scripts/validate_eval_dataset.py             # label validation (no retrieval code)
.venv/bin/python activities/d-retrieval-metrics/metrics_demo.py # worked example + K=3 vs K=5 on DEV queries
```

`metrics_demo.py` uses only the 5 development queries and the rankings already saved by Phases 3–4; it never reads
the 30-query set. Output: `results/metrics-demo-report.md`.
