# Phase 6 – Development-query answer evaluation: run `real-smoke-1`

- Started (UTC): 2026-11-26T17:01:30+00:00 · LLM: `openai/gpt-oss-120b` (real) · temperature 0.0 · dense refusal threshold 0.25 · context_k 4 · status: **complete**
- Integrity: 10 manifests PASS
- Queries: 5 development queries + 3 refusal probes (Day 12 SM-13..15). The 30-query held-out set was not used.
- All checks are deterministic (no LLM-as-judge); see `02-Documentation/Day-13-Answer-Evaluation-Methodology.md`.

## Summary per configuration

| Config | Correct (dev) | Answered | Answer terms (mean) | Citations valid | Cites expected policy | Citation support | Cited-fact cov. | Context-fact cov. | Unsupported numbers | Refusals correct | Errors |
|---|---|---|---:|---|---|---|---:|---:|---:|---|---:|
| `v1_paragraph_800_150+dense` | 0/0 | 0/0 | — | 0/0 | 0/0 | 0/0 | — | — | 0 | 0/0 | 8 |
| `v1_paragraph_800_150+hybrid+rerank` | 0/0 | 0/0 | — | 0/0 | 0/0 | 0/0 | — | — | 0 | 0/0 | 8 |

## Tokens, cost and latency per configuration

| Config | Real LLM calls | Cache hits | LLM not called | Prompt tokens | Completion tokens | Est. cost (USD, list price) | Median ms: embed / vector / BM25 / RRF / rerank / retrieval | Median generation ms | Median total ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `v1_paragraph_800_150+dense` | 0 | 0 | 0 | 0 | 0 | 0.000000 | — / — / — / — / — / — | — | — |
| `v1_paragraph_800_150+hybrid+rerank` | 0 | 0 | 0 | 0 | 0 | 0.000000 | — / — / — / — / — / — | — | — |

Estimated cost uses the published list price in `rag_v2/pricing.py` (source and date recorded there); it is not the amount charged on the account's plan. Generation latency includes any rate-limit waits.

## Per query

| Config | Query | Status | Correct | Answer terms | Citations | Support | Unsupported numbers | Context chunks (dropped by dense filter) | Answer (shown) |
|---|---|---|---|---|---|---|---|---|---|
| `v1_paragraph_800_150+dense` | DEV-01 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | DEV-02 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | DEV-03 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | DEV-04 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | DEV-05 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | SM-13 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | SM-14 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+dense` | SM-15 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-01 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-02 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-03 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-04 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-05 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | SM-13 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | SM-14 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
| `v1_paragraph_800_150+hybrid+rerank` | SM-15 | ERROR | — | — | — | — | — | — | Groq rejected the API key (authentication error). |
