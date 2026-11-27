# Activity A – Chunking Strategies

Compares three chunking strategies on the frozen 6-policy Day 12 corpus, with everything else held constant
(v1 loader, all-MiniLM-L6-v2 ONNX embeddings, title-prefixed embedded text, cosine distance, v1 `Retriever`).

| Strategy | Family | Implementation |
|---|---|---|
| `v1_paragraph_800_150` | document-specific (Markdown/paragraph-aware) | frozen Day 12 `src.ingestion.chunk_document(800, 150)`, unchanged |
| `fixed_500` | character | `rag_v2.chunking.fixed_spans` – 500-character windows, no overlap |
| `recursive_800_100` | recursive character | `rag_v2.chunking.recursive_spans` – LangChain `RecursiveCharacterTextSplitter` algorithm, separators `\n\n`, `\n`, space, `""`; byte-identical to LangChain 1.2.0 on all 6 documents (`03-Evidence/06-…`) |

## Run

```bash
.venv/bin/python activities/a-chunking-strategies/compare_chunking.py
```

No API key, no network. The script refuses to run if the corpus, evidence catalog or dev queries do not match their
frozen SHA-256 hashes, and checks that the v1-strategy index reproduces Day 12's run-1 top-3 for all 15 smoke questions.

## Outputs (`results/`)

- `chunking-comparison-report.md` – generated report: chunk statistics, boundary quality, key-fact integrity,
  representative samples, top-3 per dev query with evidence, summary.
- `chunking-comparison.json` – all raw data incl. every chunk's text and offsets; `results_sha256` covers everything
  except the run timestamp (identical across runs = reproducible).

## Indexes (kept for Activities B and C)

`.chroma-v2/` collections `d13_v1_paragraph_800_150`, `d13_fixed_500`, `d13_recursive_800_100`
(rebuilt from scratch on every run; the Phase 1 v1 index in `.chroma/` is not touched).
