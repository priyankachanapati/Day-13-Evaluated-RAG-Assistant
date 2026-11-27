# Activity C – Cross-Encoder Reranking

Reranks the Phase 3 hybrid top-10 candidate pool (RRF k=60 of dense top-10 + BM25 top-10) with
`cross-encoder/ms-marco-MiniLM-L6-v2` for all three chunking strategies, on the 5 frozen dev queries.

| Item | Value |
|---|---|
| Requested ID (plan) | `cross-encoder/ms-marco-MiniLM-L-6-v2` – a Hub alias of the renamed repository below |
| Loaded model | `cross-encoder/ms-marco-MiniLM-L6-v2` at pinned commit `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| Files | 7 files, 91.8 MB (PyTorch weights + tokenizer/config only; ONNX/OpenVINO/Flax copies not downloaded) |
| Inference | CPU, whole pool in one batch, `(query, title + chunk)` pairs, raw logits (activation Identity) |

## Run

```bash
.venv/bin/python scripts/download_cross_encoder.py                                    # once (network)
HF_HUB_OFFLINE=1 .venv/bin/python activities/c-cross-encoder-reranking/compare_reranking.py   # offline
```

The comparison refuses to run unless all frozen artifacts match their manifests and the model revision matches the
pin; it verifies that the candidate pool equals the Phase 3 hybrid top-10 exactly and that reranking only re-orders it.

## Outputs (`results/`)

- `reranking-report.md` – generated: model/settings, score types, per-query top-3 before/after, metrics at K=3/5 and
  pool coverage, fact movements, per-query effect, dense gate, latency (per stage + one-time).
- `reranking.json` – raw data; `results_sha256_floats_rounded_4dp` excludes run metadata and latency.

`results/cross-encoder-model-info.json` (project root `results/`) records revision, file sizes and config.
