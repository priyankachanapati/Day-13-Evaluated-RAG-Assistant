# Activity 1 – Compare Chunking Strategies (fixed 500 vs recursive 800/100)

**Requirement.** Re-index the Day 12 corpus with fixed 500-character chunks and recursive 800/100 chunks, run the same
five queries against both, take the top 3, compare the chunks and their relevance, and record a structured comparison.

| Item | Where |
|---|---|
| Chunkers (both strategies) | [`rag_v2/chunking.py`](../../rag_v2/chunking.py) – `chunk_fixed_500`, `chunk_recursive_800_100` |
| Re-indexing (separate Chroma collections) | [`rag_v2/indexing.py`](../../rag_v2/indexing.py) → `.chroma-v2/d13_fixed_500` (28 chunks), `.chroma-v2/d13_recursive_800_100` (20 chunks) |
| Five test queries (frozen, fact-labelled) | [`evaluation/datasets/dev_queries.json`](../../evaluation/datasets/dev_queries.json) |
| This activity's runner | [`compare_chunking_top3.py`](compare_chunking_top3.py) |
| Structured output | [`results/chunking-top3-comparison.json`](results/chunking-top3-comparison.json) |
| Full chunk analysis (lengths, split facts, samples) | [`../a-chunking-strategies/`](../a-chunking-strategies/) (earlier Activity A, unchanged) |

```bash
.venv/bin/python activities/activity-1-chunking-comparison/compare_chunking_top3.py   # local only, no API calls
```

The runner opens the two existing indexes read-only, retrieves the dense top 3 for each query with the unchanged v1
retriever, and checks that every top-3 list equals the saved Activity A result (12/12 checks PASS).

**Relevance rule.** A retrieved chunk is relevant if it fully contains a required evidence sentence of the question
(project-wide rule, `evaluation/metrics.py`). Coverage@3 = share of the question's required facts present in the top 3.

## Measured results

| Strategy | Chunks | Mean chars | Hit@3 | MRR@3 | P@3 | Coverage@3 | Saved Phase 7 (25 answerable): Hit@3 / MRR@3 / Cov@3 |
|---|---:|---:|---:|---:|---:|---:|---|
| Fixed 500 | 28 | 442.0 | 0.80 | 0.80 | 0.33 | 0.80 | 0.88 / 0.80 / 0.84 |
| Recursive 800/100 | 20 | 637.8 | **1.00** | **1.00** | **0.47** | **1.00** | 1.00 / 0.95 / 1.00 |

Measured on 5 development questions (live run); the Phase 7 column is the saved held-out evaluation of the same two
dense configurations (30 questions, 25 answerable), shown for scale.

**How chunking affects retrieval.** Fixed 500-character chunks ignore structure: 19 chunks start and 19 end in the
middle of a sentence, there are 22 mid-word cuts and 6 broken headings, and 19 of the 102 policy facts exist in no
fixed chunk intact, so those facts can never be retrieved whole. Short, cut chunks also carry less context for the
embedding: for DEV-01 the fact (21-day notice for 5+ consecutive days of leave) is intact in fixed chunk
`HR-001-CH02`, but three other fixed chunks score higher and it falls out of the top 3. Recursive splitting breaks at
paragraph, line and sentence boundaries first and keeps up to 100 characters of overlap: no mid-sentence or mid-word
cuts, every fact intact (102/102), and the relevant chunk ranked first for all five questions. Its costs are longer
chunks (more prompt tokens per chunk) and 12 chunks that end with a heading detached from the text it introduces.
The 5-question sample is small; the saved 30-question evaluation shows the same direction.
