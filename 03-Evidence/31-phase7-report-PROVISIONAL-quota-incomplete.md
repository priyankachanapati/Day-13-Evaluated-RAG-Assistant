# Phase 7 – RAG v1 vs v2 on the frozen evaluation set v1.1 (generated report)

- Pre-registration: `evaluation/phase7/preregistration.json` (2026-11-26T17:20:54Z), hash in `manifests/phase7-preregistration.sha256`. Dataset v1.1 SHA-256 `e860b7294c57…` (30 queries: 25 answerable, 5 unanswerable). EV-06 excluded (exposed).
- Retrieval run: 2026-11-26T17:22:11+00:00 (no LLM). Answer run status: **incomplete: provider quota exhausted (Groq quota exhausted: Rate limit reached for model `openai/gpt-oss-120b` in organization `org_[redacted]` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199071, Requested 1506. Please try again in 4m9.264s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing)**.

## A. Retrieval – all 18 configurations (macro over 25 answerable; no LLM)

| Config | Hit@3 | Hit@5 | MRR@3 | MRR@5 | P@3 | P@5 | R@3* | R@5* | Cov@3 | Cov@5 | Context cov. (≤4 after dense filter) | Full-context queries | Required fact dropped by dense filter | Unanswerable refused by dense gate | Median retrieval ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1_paragraph_800_150+dense` **(v1)** | 1.00 | 1.00 | 0.98 | 0.98 | 0.41 | 0.26 | 0.95 | 0.96 | 0.98 | 1.00 | 1.00 | 25/25 | 0 | 0/5 | 19.6 |
| `v1_paragraph_800_150+dense+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 0.43 | 0.27 | 0.97 | 0.99 | 0.98 | 1.00 | 1.00 | 25/25 | 0 | 0/5 | 82.4 |
| `v1_paragraph_800_150+bm25` | 0.96 | 0.96 | 0.94 | 0.94 | 0.41 | 0.25 | 0.93 | 0.93 | 0.94 | 0.94 | 0.94 | 23/25 | 0 | 0/5 | 19.7 |
| `v1_paragraph_800_150+bm25+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 0.43 | 0.27 | 0.97 | 0.99 | 0.98 | 1.00 | 1.00 | 25/25 | 0 | 0/5 | 78.6 |
| `v1_paragraph_800_150+hybrid` | 0.96 | 1.00 | 0.96 | 0.97 | 0.41 | 0.26 | 0.93 | 0.98 | 0.94 | 1.00 | 0.94 | 23/25 | 0 | 0/5 | 20.2 |
| `v1_paragraph_800_150+hybrid+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 0.43 | 0.27 | 0.97 | 0.99 | 0.98 | 1.00 | 1.00 | 25/25 | 0 | 0/5 | 81.7 |
| `fixed_500+dense` | 0.88 | 0.88 | 0.80 | 0.80 | 0.33 | 0.20 | 0.88 | 0.88 | 0.84 | 0.84 | 0.84 | 20/25 | 0 | 0/5 | 20.8 |
| `fixed_500+dense+rerank` | 0.88 | 0.92 | 0.86 | 0.87 | 0.32 | 0.21 | 0.86 | 0.90 | 0.82 | 0.86 | 0.84 | 20/25 | 0 | 0/5 | 64.3 |
| `fixed_500+bm25` | 0.88 | 0.92 | 0.79 | 0.80 | 0.33 | 0.21 | 0.88 | 0.90 | 0.84 | 0.86 | 0.84 | 20/25 | 0 | 0/5 | 21.2 |
| `fixed_500+bm25+rerank` | 0.88 | 0.92 | 0.86 | 0.87 | 0.32 | 0.22 | 0.86 | 0.92 | 0.82 | 0.86 | 0.86 | 20/25 | 0 | 0/5 | 59.2 |
| `fixed_500+hybrid` | 0.88 | 0.92 | 0.82 | 0.83 | 0.33 | 0.22 | 0.88 | 0.92 | 0.84 | 0.86 | 0.86 | 20/25 | 0 | 0/5 | 21.4 |
| `fixed_500+hybrid+rerank` | 0.88 | 0.92 | 0.86 | 0.87 | 0.32 | 0.21 | 0.86 | 0.90 | 0.82 | 0.86 | 0.84 | 20/25 | 0 | 0/5 | 65.6 |
| `recursive_800_100+dense` | 1.00 | 1.00 | 0.95 | 0.95 | 0.37 | 0.22 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 25/25 | 0 | 1/5 | 21.1 |
| `recursive_800_100+dense+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 | 0.22 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 25/25 | 0 | 1/5 | 85.5 |
| `recursive_800_100+bm25` | 0.96 | 0.96 | 0.94 | 0.94 | 0.35 | 0.21 | 0.93 | 0.93 | 0.94 | 0.94 | 0.94 | 23/25 | 0 | 1/5 | 21.2 |
| `recursive_800_100+bm25+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 | 0.22 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 25/25 | 0 | 1/5 | 82.7 |
| `recursive_800_100+hybrid` | 0.96 | 1.00 | 0.96 | 0.97 | 0.35 | 0.22 | 0.93 | 0.98 | 0.94 | 1.00 | 1.00 | 25/25 | 0 | 1/5 | 21.5 |
| `recursive_800_100+hybrid+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 | 0.22 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 25/25 | 0 | 1/5 | 86.0 |

\* chunk-level recall depends on the chunking (not comparable across chunkings). Coverage = distinct required facts (fact level).

### Coverage@3 by category (answerable)

| Config | paraphrase / synonym (n=5) | exact keyword / identifier (n=4) | numerical / date (n=6) | multi-fact (n=4) | multi-fact, cross-section (n=3) | multi-fact, cross-policy (n=1) | near-topic distractor (n=2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_paragraph_800_150+dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `v1_paragraph_800_150+dense+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `v1_paragraph_800_150+bm25` | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `v1_paragraph_800_150+bm25+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `v1_paragraph_800_150+hybrid` | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `v1_paragraph_800_150+hybrid+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `fixed_500+dense` | 1.00 | 1.00 | 0.83 | 0.75 | 1.00 | 0.00 | 0.50 |
| `fixed_500+dense+rerank` | 1.00 | 1.00 | 0.83 | 0.75 | 0.83 | 0.00 | 0.50 |
| `fixed_500+bm25` | 1.00 | 1.00 | 0.83 | 0.75 | 1.00 | 0.00 | 0.50 |
| `fixed_500+bm25+rerank` | 1.00 | 1.00 | 0.83 | 0.75 | 0.83 | 0.00 | 0.50 |
| `fixed_500+hybrid` | 1.00 | 1.00 | 0.83 | 0.75 | 1.00 | 0.00 | 0.50 |
| `fixed_500+hybrid+rerank` | 1.00 | 1.00 | 0.83 | 0.75 | 0.83 | 0.00 | 0.50 |
| `recursive_800_100+dense` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `recursive_800_100+dense+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `recursive_800_100+bm25` | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `recursive_800_100+bm25+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `recursive_800_100+hybrid` | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 |
| `recursive_800_100+hybrid+rerank` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## B. Answers – pre-registered configurations (real LLM, deterministic checks)

| Config | Correct (25) | Refusals correct (5) | **Score /30** | Answered | Answer terms (mean) | Citations valid | Cites expected policy | Citation support | Cited-fact cov. | Context-fact cov. | Unsupported numbers | Withheld (invalid citations) | Prompt / completion tokens | Est. cost USD | Median generation ms | Median total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1_paragraph_800_150+dense` | 22/25 | 5/5 | **27/30** | 25/25 | 0.96 | 25/25 | 25/25 | 22/25 | 1.00 | 1.00 | 0 | 0 | 30620 / 5964 | 0.008171 | 1877 | 1913 |
| `v1_paragraph_800_150+hybrid+rerank` | 8/9 | 0/0 | **8/9** | 9/9 | 0.94 | 9/9 | 9/9 | 8/9 | 1.00 | 1.00 | 0 | 0 | 9110 / 1394 | 0.002203 | 5822 | 5918 |
| `recursive_800_100+hybrid+rerank` | 0/0 | 0/0 | **0/0** | 0/0 | — | 0/0 | 0/0 | 0/0 | — | — | 0 | 0 | 0 / 0 | 0.000000 | — | — |

### Paired comparison with v1 (per query, 'correct' incl. refusals)

| v2 config | v2 better | v1 better | ties | exact sign test p (two-sided) |
|---|---:|---:|---:|---:|
| `v1_paragraph_800_150+hybrid+rerank` | 0 | 0 | 9 | 1.000 |
| `recursive_800_100+hybrid+rerank` | 0 | 0 | 0 | 1.000 |

### Per-query 'correct' (✓ correct, ✗ not correct)

| Query | Category | `v1_paragraph_800_150+dense` | `v1_paragraph_800_150+hybrid+rerank` | `recursive_800_100+hybrid+rerank` |
|---|---|---|---|---|
| EV-01 | paraphrase / synonym | ✓ | ✓ | — |
| EV-02 | paraphrase / synonym | ✓ | ✓ | — |
| EV-03 | paraphrase / synonym | ✓ | ✓ | — |
| EV-04 | paraphrase / synonym | ✓ | ✓ | — |
| EV-05 | paraphrase / synonym | ✓ | ✓ | — |
| EV-07 | exact keyword / identifier | ✓ | ✓ | — |
| EV-08 | exact keyword / identifier | ✓ | ✓ | — |
| EV-09 | exact keyword / identifier | ✗ | ✗ | — |
| EV-10 | numerical / date | ✓ | ✓ | — |
| EV-11 | numerical / date | ✓ | — | — |
| EV-12 | numerical / date | ✓ | — | — |
| EV-13 | numerical / date | ✓ | — | — |
| EV-14 | numerical / date | ✓ | — | — |
| EV-15 | numerical / date | ✓ | — | — |
| EV-16 | multi-fact | ✓ | — | — |
| EV-17 | multi-fact | ✓ | — | — |
| EV-18 | multi-fact | ✓ | — | — |
| EV-19 | multi-fact | ✓ | — | — |
| EV-20 | multi-fact, cross-section | ✓ | — | — |
| EV-21 | multi-fact, cross-section | ✓ | — | — |
| EV-22 | multi-fact, cross-section | ✓ | — | — |
| EV-23 | multi-fact, cross-policy | ✓ | — | — |
| EV-24 | near-topic distractor | ✗ | — | — |
| EV-25 | near-topic distractor | ✓ | — | — |
| EV-26 | unanswerable (near-topic) | ✓ | — | — |
| EV-27 | unanswerable (near-topic) | ✓ | — | — |
| EV-28 | unanswerable (near-topic) | ✓ | — | — |
| EV-29 | unanswerable (off-topic) | ✓ | — | — |
| EV-30 | unanswerable (off-topic) | ✓ | — | — |
| EV-31 | exact keyword / identifier | ✗ | — | — |

## C. Pre-registered selection

Rule: highest score /30; tie-break context coverage, then lower median end-to-end latency.

1. `v1_paragraph_800_150+dense` – score 27/30, context coverage 1.00, median total 1913 ms
2. `v1_paragraph_800_150+hybrid+rerank` – score 8/9, context coverage 1.00, median total 5918 ms
3. `recursive_800_100+hybrid+rerank` – score 0/0, context coverage 1.00, median total — ms

Selected by the rule: **`v1_paragraph_800_150+dense`** — PROVISIONAL: the answer run is incomplete

## D. EV-06 sensitivity (exposed query; retrieval only; not in any total)

| Config | Hit@3 | Coverage@3 | Context coverage |
|---|---:|---:|---:|
| `v1_paragraph_800_150+dense` | 1 | 1.00 | 0.00 |
| `v1_paragraph_800_150+hybrid+rerank` | 1 | 1.00 | 0.00 |
| `recursive_800_100+hybrid+rerank` | 1 | 1.00 | 0.00 |
