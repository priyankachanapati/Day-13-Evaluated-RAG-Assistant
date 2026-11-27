# Phase 6 – Development-query answer evaluation: run `real-smoke-1`

- Started (UTC): 2026-11-26T17:13:15+00:00 · LLM: `openai/gpt-oss-120b` (real) · temperature 0.0 · dense refusal threshold 0.25 · context_k 4 · status: **complete**
- Integrity: 10 manifests PASS
- Queries: 5 development queries + 3 refusal probes (Day 12 SM-13..15). The 30-query held-out set was not used.
- All checks are deterministic (no LLM-as-judge); see `02-Documentation/Day-13-Answer-Evaluation-Methodology.md`.

## Summary per configuration

| Config | Correct (dev) | Answered | Answer terms (mean) | Citations valid | Cites expected policy | Citation support | Cited-fact cov. | Context-fact cov. | Unsupported numbers | Refusals correct | Errors |
|---|---|---|---:|---|---|---|---:|---:|---:|---|---:|
| `v1_paragraph_800_150+dense` | 4/5 | 5/5 | 0.80 | 5/5 | 5/5 | 4/5 | 1.00 | 1.00 | 0 | 3/3 | 0 |
| `v1_paragraph_800_150+hybrid+rerank` | 3/5 | 5/5 | 0.70 | 5/5 | 5/5 | 3/5 | 1.00 | 1.00 | 0 | 3/3 | 0 |

## Tokens, cost and latency per configuration

| Config | Real LLM calls | Cache hits | LLM not called | Prompt tokens | Completion tokens | Est. cost (USD, list price) | Median ms: embed / vector / BM25 / RRF / rerank / retrieval | Median generation ms | Median total ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `v1_paragraph_800_150+dense` | 8 | 0 | 0 | 7653 | 1667 | 0.002148 | 38.4 / 2.6 / — / — / — / 41.2 | 974.82 | 1016.16 |
| `v1_paragraph_800_150+hybrid+rerank` | 6 | 2 | 0 | 7559 | 1442 | 0.001999 | 35.4 / 2.2 / 0.1 / 0.0 / 66.8 / 109.1 | 1058.68 | 1186.32 |

Estimated cost uses the published list price in `rag_v2/pricing.py` (source and date recorded there); it is not the amount charged on the account's plan. Generation latency includes any rate-limit waits.

## Per query

| Config | Query | Status | Correct | Answer terms | Citations | Support | Unsupported numbers | Context chunks (dropped by dense filter) | Answer (shown) |
|---|---|---|---|---|---|---|---|---|---|
| `v1_paragraph_800_150+dense` | DEV-01 | answered | yes | [True] | valid | [True] | — | HR-001-CH05, HR-001-CH04, HR-002-CH02, HR-001-CH02 | You must submit a request at least 21 calendar days before the first day of a two‐week holiday. [Source: HR-001, Chunk: HR-001-CH02] |
| `v1_paragraph_800_150+dense` | DEV-02 | answered | yes | [True] | valid | [True] | — | IT-001-CH03 (dropped: IT-001-CH02, IT-001-CH01, IT-002-CH01) | KeyVault is the company‐approved password manager that is installed on all Brightlake laptops. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+dense` | DEV-03 | answered | yes | [True] | valid | [True] | — | IT-002-CH03, IT-001-CH03, IT-002-CH01, IT-002-CH02 | The IT Service Desk can be reached on internal extension 4357. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+dense` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH04, HR-002-CH02, IT-002-CH02 | Yes, you may use public Wi‐Fi for work only when you are connected through the company VPN. [Source: HR-002, Chunk: HR-002-CH03] |
| `v1_paragraph_800_150+dense` | DEV-05 | answered | yes | [True, True] | valid | [True, True] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-003-CH04, HR-002-CH04) | Yes, premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH02] Meals during business travel are reimbursed up to USD 60 per day. [Source: HR-003, Chunk: HR-003-CH03] |
| `v1_paragraph_800_150+dense` | SM-13 | refused | yes (refusal) | — | none | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, HR-003-CH04 | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+dense` | SM-14 | refused | yes (refusal) | — | none | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-001-CH01) | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+dense` | SM-15 | refused | yes (refusal) | — | none | — | — | HR-003-CH02, HR-003-CH03, HR-003-CH04, HR-003-CH01 | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-01 | answered | yes | [True] | valid | [True] | — | HR-001-CH02, HR-001-CH05, HR-001-CH04, HR-002-CH02 | You need to submit the request at least 21 calendar days before the first day of your two‐week holiday. [Source: HR-001, Chunk: HR-001-CH02] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-02 | answered | yes | [True] | valid | [True] | — | IT-001-CH03 (dropped: IT-001-CH01, HR-004-CH01, IT-001-CH02) | KeyVault is the company‐approved password manager that is installed on all Brightlake laptops. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-03 | answered | yes | [True] | valid | [True] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02 (dropped: IT-001-CH02) | The IT Service Desk’s internal extension is 4357 [Source: IT-001, Chunk: IT-001-CH03]. |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | You may use public Wi‐Fi for work only when you are connected through the company VPN. [Source: HR-002, Chunk: HR-002-CH03] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-05 | answered | no | [False, True] | valid | [False, True] | — | HR-003-CH03, HR-003-CH02 (dropped: HR-001-CH02, HR-003-CH04) | Yes, you may book premium economy for a 7‐hour business flight [Source: HR-003, Chunk: HR-003-CH03]. Meals during business travel are reimbursed up to USD 60 per day [Source: HR-003, Chunk: HR-003-CH03]. |
| `v1_paragraph_800_150+hybrid+rerank` | SM-13 | refused | yes (refusal) | — | none | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+hybrid+rerank` | SM-14 | refused | yes (refusal) | — | none | — | — | HR-004-CH02 (dropped: HR-004-CH03, IT-002-CH02, HR-004-CH01) | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+hybrid+rerank` | SM-15 | refused | yes (refusal) | — | none | — | — | HR-003-CH03, HR-003-CH02, HR-003-CH04, HR-002-CH04 | The policy documents do not contain enough information to answer this question. |
