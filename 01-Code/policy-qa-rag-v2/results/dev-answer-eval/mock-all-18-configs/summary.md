# Phase 6 – Development-query answer evaluation: run `mock-all-18-configs`

- Started (UTC): 2026-11-26T16:42:23+00:00 · LLM: `mock-llm` (MOCK - not a real model) · temperature 0.0 · dense refusal threshold 0.25 · context_k 4 · status: **complete**
- Integrity: 10 manifests PASS
- Queries: 5 development queries + 3 refusal probes (Day 12 SM-13..15). The 30-query held-out set was not used.
- All checks are deterministic (no LLM-as-judge); see `02-Documentation/Day-13-Answer-Evaluation-Methodology.md`.

## Summary per configuration

| Config | Correct (dev) | Answered | Answer terms (mean) | Citations valid | Cites expected policy | Citation support | Cited-fact cov. | Context-fact cov. | Unsupported numbers | Refusals correct | Errors |
|---|---|---|---:|---|---|---|---:|---:|---:|---|---:|
| `v1_paragraph_800_150+dense` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.70 | 1.00 | 0 | 0/3 | 0 |
| `v1_paragraph_800_150+dense+rerank` | 0/5 | 5/5 | 0.20 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 0/3 | 0 |
| `v1_paragraph_800_150+bm25` | 0/5 | 5/5 | 0.10 | 5/5 | 4/5 | 0/5 | 0.80 | 1.00 | 0 | 1/3 | 0 |
| `v1_paragraph_800_150+bm25+rerank` | 0/5 | 5/5 | 0.20 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 1/3 | 0 |
| `v1_paragraph_800_150+hybrid` | 0/5 | 5/5 | 0.00 | 5/5 | 4/5 | 0/5 | 0.70 | 1.00 | 0 | 0/3 | 0 |
| `v1_paragraph_800_150+hybrid+rerank` | 0/5 | 5/5 | 0.20 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 0/3 | 0 |
| `fixed_500+dense` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.80 | 1.00 | 0 | 0/3 | 0 |
| `fixed_500+dense+rerank` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 1.00 | 1.00 | 0 | 0/3 | 0 |
| `fixed_500+bm25` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 1.00 | 1.00 | 0 | 0/3 | 0 |
| `fixed_500+bm25+rerank` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 1.00 | 1.00 | 0 | 0/3 | 0 |
| `fixed_500+hybrid` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.80 | 1.00 | 0 | 0/3 | 0 |
| `fixed_500+hybrid+rerank` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 1.00 | 1.00 | 0 | 0/3 | 0 |
| `recursive_800_100+dense` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 0/3 | 0 |
| `recursive_800_100+dense+rerank` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 0/3 | 0 |
| `recursive_800_100+bm25` | 0/5 | 5/5 | 0.00 | 5/5 | 4/5 | 0/5 | 0.70 | 1.00 | 0 | 1/3 | 0 |
| `recursive_800_100+bm25+rerank` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 1/3 | 0 |
| `recursive_800_100+hybrid` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 0/3 | 0 |
| `recursive_800_100+hybrid+rerank` | 0/5 | 5/5 | 0.00 | 5/5 | 5/5 | 0/5 | 0.90 | 1.00 | 0 | 0/3 | 0 |

## Tokens, cost and latency per configuration

| Config | Real LLM calls | Cache hits | LLM not called | Prompt tokens | Completion tokens | Est. cost (USD, list price) | Median ms: embed / vector / BM25 / RRF / rerank / retrieval | Median generation ms | Median total ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `v1_paragraph_800_150+dense` | 8 | 0 | 0 | 3225 | 160 | 0.000000 | 17.9 / 1.8 / — / — / — / 19.8 | 0.25 | 20.10 |
| `v1_paragraph_800_150+dense+rerank` | 6 | 2 | 0 | 3296 | 160 | 0.000000 | 18.0 / 1.9 / — / — / 64.6 / 84.7 | 0.35 | 85.89 |
| `v1_paragraph_800_150+bm25` | 5 | 2 | 1 | 2527 | 140 | 0.000000 | 17.9 / 1.7 / 0.1 / — / — / 19.7 | 0.25 | 19.97 |
| `v1_paragraph_800_150+bm25+rerank` | 2 | 5 | 1 | 2898 | 140 | 0.000000 | 18.0 / 1.8 / 0.1 / — / 58.1 / 78.4 | 0.42 | 81.79 |
| `v1_paragraph_800_150+hybrid` | 3 | 5 | 0 | 3323 | 160 | 0.000000 | 17.9 / 1.8 / 0.1 / 0.0 / — / 19.9 | 0.28 | 20.27 |
| `v1_paragraph_800_150+hybrid+rerank` | 0 | 8 | 0 | 3163 | 160 | 0.000000 | 18.0 / 1.9 / 0.1 / 0.0 / 61.3 / 81.3 | — | — |
| `fixed_500+dense` | 8 | 0 | 0 | 2671 | 160 | 0.000000 | 18.6 / 2.1 / — / — / — / 20.7 | 0.25 | 21.00 |
| `fixed_500+dense+rerank` | 5 | 3 | 0 | 2733 | 160 | 0.000000 | 18.0 / 2.1 / — / — / 45.1 / 66.5 | 0.34 | 67.30 |
| `fixed_500+bm25` | 6 | 2 | 0 | 2166 | 160 | 0.000000 | 19.0 / 2.0 / 0.1 / — / — / 21.2 | 0.27 | 21.51 |
| `fixed_500+bm25+rerank` | 0 | 8 | 0 | 2269 | 160 | 0.000000 | 18.4 / 2.1 / 0.1 / — / 41.0 / 62.0 | — | — |
| `fixed_500+hybrid` | 3 | 5 | 0 | 2482 | 160 | 0.000000 | 19.4 / 2.1 / 0.1 / 0.0 / — / 21.6 | 0.31 | 21.65 |
| `fixed_500+hybrid+rerank` | 0 | 8 | 0 | 2631 | 160 | 0.000000 | 18.6 / 2.1 / 0.1 / 0.0 / 44.6 / 65.3 | — | — |
| `recursive_800_100+dense` | 8 | 0 | 0 | 3471 | 160 | 0.000000 | 19.5 / 1.8 / — / — / — / 21.3 | 0.31 | 21.66 |
| `recursive_800_100+dense+rerank` | 5 | 3 | 0 | 3357 | 160 | 0.000000 | 18.7 / 1.8 / — / — / 65.7 / 86.9 | 0.35 | 87.79 |
| `recursive_800_100+bm25` | 4 | 3 | 1 | 2607 | 140 | 0.000000 | 19.6 / 1.7 / 0.1 / — / — / 21.4 | 0.35 | 21.78 |
| `recursive_800_100+bm25+rerank` | 2 | 5 | 1 | 2896 | 140 | 0.000000 | 19.1 / 1.8 / 0.1 / — / 60.3 / 81.2 | 0.50 | 82.56 |
| `recursive_800_100+hybrid` | 1 | 7 | 0 | 3447 | 160 | 0.000000 | 19.6 / 1.8 / 0.1 / 0.0 / — / 21.6 | 0.26 | 21.57 |
| `recursive_800_100+hybrid+rerank` | 0 | 8 | 0 | 3217 | 160 | 0.000000 | 18.6 / 1.8 / 0.1 / 0.0 / 66.8 / 87.3 | — | — |

Estimated cost uses the published list price in `rag_v2/pricing.py` (source and date recorded there); it is not the amount charged on the account's plan. Generation latency includes any rate-limit waits.

## Per query

| Config | Query | Status | Correct | Answer terms | Citations | Support | Unsupported numbers | Context chunks (dropped by dense filter) | Answer (shown) |
|---|---|---|---|---|---|---|---|---|---|
| `v1_paragraph_800_150+dense` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH05, HR-001-CH04, HR-002-CH02, HR-001-CH02 | Employees required to work on a public holiday receive one compensatory day off, which must be taken within 60 days. [Source: HR-001, Chunk: HR-001-CH05] |
| `v1_paragraph_800_150+dense` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH02, IT-001-CH01, IT-002-CH01) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+dense` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-002-CH03, IT-001-CH03, IT-002-CH01, IT-002-CH02 | USB storage devices are blocked on company laptops by default. [Source: IT-002, Chunk: IT-002-CH03] |
| `v1_paragraph_800_150+dense` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH04, HR-002-CH02, IT-002-CH02 | Managers may decline requests during critical project phases, client visits, or mandatory in-office events. [Source: HR-002, Chunk: HR-002-CH03] |
| `v1_paragraph_800_150+dense` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-003-CH04, HR-002-CH04) | Claims submitted after 30 days will be rejected unless the Finance Controller approves a late submission. [Source: HR-003, Chunk: HR-003-CH02] |
| `v1_paragraph_800_150+dense` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, HR-003-CH04 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+dense` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-001-CH01) | HR sends a welcome email with first-day instructions 5 working days before the joining date. [Source: HR-004, Chunk: HR-004-CH02] |
| `v1_paragraph_800_150+dense` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH02, HR-003-CH03, HR-003-CH04, HR-003-CH01 | Claims submitted after 30 days will be rejected unless the Finance Controller approves a late submission. [Source: HR-003, Chunk: HR-003-CH02] |
| `v1_paragraph_800_150+dense+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-001-CH05, HR-001-CH04, HR-002-CH02 | Annual leave accrues at 1.67 days per completed month of service. [Source: HR-001, Chunk: HR-001-CH02] |
| `v1_paragraph_800_150+dense+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH01, HR-004-CH01, IT-001-CH02) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+dense+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02, IT-002-CH01 | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+dense+rerank` | DEV-04 | answered | no | [True, False] | valid | [True, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | Use of public Wi-Fi networks for work is permitted only when connected through the company VPN. [Source: HR-002, Chunk: HR-002-CH04] |
| `v1_paragraph_800_150+dense+rerank` | DEV-05 | answered | no | [True, False] | valid | [True, False] | — | HR-003-CH03, HR-003-CH02 (dropped: HR-001-CH02, HR-004-CH02) | Premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH03] |
| `v1_paragraph_800_150+dense+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+dense+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: HR-004-CH03, IT-002-CH02, HR-004-CH01) | HR sends a welcome email with first-day instructions 5 working days before the joining date. [Source: HR-004, Chunk: HR-004-CH02] |
| `v1_paragraph_800_150+dense+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH03, HR-003-CH02, HR-003-CH04, HR-002-CH04 | Premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH03] |
| `v1_paragraph_800_150+bm25` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-002-CH02, HR-001-CH02, HR-001-CH04, HR-001-CH05 | Employees still on probation must work from the office unless their department head grants a written exception. [Source: HR-002, Chunk: HR-002-CH02] |
| `v1_paragraph_800_150+bm25` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+bm25` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03 (dropped: IT-001-CH02, HR-002-CH02) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+bm25` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH04, HR-002-CH02 (dropped: HR-001-CH04) | Managers may decline requests during critical project phases, client visits, or mandatory in-office events. [Source: HR-002, Chunk: HR-002-CH03] |
| `v1_paragraph_800_150+bm25` | DEV-05 | answered | no | [True, False] | valid | [True, False] | — | HR-003-CH03, HR-003-CH02 (dropped: HR-003-CH04, HR-001-CH02) | Premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH03] |
| `v1_paragraph_800_150+bm25` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-002-CH01, HR-003-CH01, HR-001-CH01, HR-004-CH01 | # Work-From-Home Policy (HR-002) [Source: HR-002, Chunk: HR-002-CH01] |
| `v1_paragraph_800_150+bm25` | SM-14 | refused_no_context | yes (refusal) | — | none | — | — |  (dropped: IT-002-CH02) | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+bm25` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH04 (dropped: HR-001-CH02, IT-002-CH01) | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+bm25+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-001-CH05, HR-001-CH04, HR-002-CH02 | Annual leave accrues at 1.67 days per completed month of service. [Source: HR-001, Chunk: HR-001-CH02] |
| `v1_paragraph_800_150+bm25+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+bm25+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02 (dropped: IT-001-CH02) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+bm25+rerank` | DEV-04 | answered | no | [True, False] | valid | [True, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | Use of public Wi-Fi networks for work is permitted only when connected through the company VPN. [Source: HR-002, Chunk: HR-002-CH04] |
| `v1_paragraph_800_150+bm25+rerank` | DEV-05 | answered | no | [True, False] | valid | [True, False] | — | HR-003-CH03, HR-003-CH02 (dropped: HR-001-CH02, IT-001-CH03) | Premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH03] |
| `v1_paragraph_800_150+bm25+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+bm25+rerank` | SM-14 | refused_no_context | yes (refusal) | — | none | — | — |  (dropped: IT-002-CH02) | The policy documents do not contain enough information to answer this question. |
| `v1_paragraph_800_150+bm25+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-002-CH04, HR-003-CH01, HR-001-CH01 (dropped: HR-001-CH02) | Use of public Wi-Fi networks for work is permitted only when connected through the company VPN. [Source: HR-002, Chunk: HR-002-CH04] |
| `v1_paragraph_800_150+hybrid` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-002-CH02, HR-001-CH05, HR-001-CH04, HR-001-CH02 | Employees still on probation must work from the office unless their department head grants a written exception. [Source: HR-002, Chunk: HR-002-CH02] |
| `v1_paragraph_800_150+hybrid` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH02, IT-001-CH01, IT-002-CH01) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+hybrid` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-002-CH03, IT-001-CH03, IT-002-CH01, IT-002-CH02 | USB storage devices are blocked on company laptops by default. [Source: IT-002, Chunk: IT-002-CH03] |
| `v1_paragraph_800_150+hybrid` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH04, HR-002-CH02, HR-002-CH01 | Managers may decline requests during critical project phases, client visits, or mandatory in-office events. [Source: HR-002, Chunk: HR-002-CH03] |
| `v1_paragraph_800_150+hybrid` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-003-CH04, HR-002-CH04) | Claims submitted after 30 days will be rejected unless the Finance Controller approves a late submission. [Source: HR-003, Chunk: HR-003-CH02] |
| `v1_paragraph_800_150+hybrid` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+hybrid` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-001-CH01) | HR sends a welcome email with first-day instructions 5 working days before the joining date. [Source: HR-004, Chunk: HR-004-CH02] |
| `v1_paragraph_800_150+hybrid` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH04, HR-001-CH01, IT-001-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-001-CH05, HR-001-CH04, HR-002-CH02 | Annual leave accrues at 1.67 days per completed month of service. [Source: HR-001, Chunk: HR-001-CH02] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH01, HR-004-CH01, IT-001-CH02) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02 (dropped: IT-001-CH02) | If the account is locked again within 24 hours, it remains locked until it is unlocked by the IT Service Desk. [Source: IT-001, Chunk: IT-001-CH03] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-04 | answered | no | [True, False] | valid | [True, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | Use of public Wi-Fi networks for work is permitted only when connected through the company VPN. [Source: HR-002, Chunk: HR-002-CH04] |
| `v1_paragraph_800_150+hybrid+rerank` | DEV-05 | answered | no | [True, False] | valid | [True, False] | — | HR-003-CH03, HR-003-CH02 (dropped: HR-001-CH02, HR-003-CH04) | Premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH03] |
| `v1_paragraph_800_150+hybrid+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `v1_paragraph_800_150+hybrid+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: HR-004-CH03, IT-002-CH02, HR-004-CH01) | HR sends a welcome email with first-day instructions 5 working days before the joining date. [Source: HR-004, Chunk: HR-004-CH02] |
| `v1_paragraph_800_150+hybrid+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH03, HR-003-CH02, HR-003-CH04, HR-002-CH04 | Premium economy may be booked for journeys of 6 hours or longer. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+dense` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH05, HR-002-CH02, HR-001-CH03, HR-001-CH02 | ement Leave [Source: HR-001, Chunk: HR-001-CH05] |
| `fixed_500+dense` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH02, IT-001-CH01, IT-001-CH04) | nt is automatically locked for 15 minutes after 5 consecutive failed sign-in attempts. [Source: IT-001, Chunk: IT-001-CH03] |
| `fixed_500+dense` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH04, IT-002-CH03, IT-002-CH04, IT-002-CH02 | Compromised Account [Source: IT-001, Chunk: IT-001-CH04] |
| `fixed_500+dense` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, IT-002-CH02 | t be seen or overheard. [Source: HR-002, Chunk: HR-002-CH04] |
| `fixed_500+dense` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH03, HR-003-CH04 (dropped: HR-003-CH02, HR-004-CH02) | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+dense` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, HR-003-CH05 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+dense` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02, IT-002-CH02 (dropped: HR-004-CH03, HR-004-CH01) | before their joining date: a government-issued photo ID, bank account details for payroll, and the signed offer letter. [Source: HR-004, Chunk: HR-004-CH02] |
| `fixed_500+dense` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH03, HR-003-CH04, HR-003-CH02, HR-003-CH05 | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+dense+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH04, HR-001-CH05 | employees receive 20 days of paid annual leave per calendar year. [Source: HR-001, Chunk: HR-001-CH02] |
| `fixed_500+dense+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH01, HR-004-CH01, IT-001-CH02) | nt is automatically locked for 15 minutes after 5 consecutive failed sign-in attempts. [Source: IT-001, Chunk: IT-001-CH03] |
| `fixed_500+dense+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH04, IT-002-CH04, IT-002-CH03, IT-002-CH01 | Compromised Account [Source: IT-001, Chunk: IT-001-CH04] |
| `fixed_500+dense+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | t be seen or overheard. [Source: HR-002, Chunk: HR-002-CH04] |
| `fixed_500+dense+rerank` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH03, HR-003-CH04 (dropped: HR-003-CH05, HR-003-CH02) | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+dense+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+dense+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02, IT-002-CH02 (dropped: HR-004-CH03, HR-002-CH05) | before their joining date: a government-issued photo ID, bank account details for payroll, and the signed offer letter. [Source: HR-004, Chunk: HR-004-CH02] |
| `fixed_500+dense+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH04, HR-003-CH03, HR-003-CH02, HR-003-CH05 | onal Vehicle [Source: HR-003, Chunk: HR-003-CH04] |
| `fixed_500+bm25` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-001-CH05, HR-002-CH02, HR-002-CH03 | employees receive 20 days of paid annual leave per calendar year. [Source: HR-001, Chunk: HR-001-CH02] |
| `fixed_500+bm25` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 | nt is automatically locked for 15 minutes after 5 consecutive failed sign-in attempts. [Source: IT-001, Chunk: IT-001-CH03] |
| `fixed_500+bm25` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH04, IT-002-CH04, IT-002-CH03 (dropped: IT-001-CH03) | Compromised Account [Source: IT-001, Chunk: IT-001-CH04] |
| `fixed_500+bm25` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02 (dropped: HR-001-CH05) | t be seen or overheard. [Source: HR-002, Chunk: HR-002-CH04] |
| `fixed_500+bm25` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH03, HR-003-CH04 (dropped: IT-001-CH04, HR-001-CH02) | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+bm25` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-004-CH01, HR-001-CH01, HR-002-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+bm25` | SM-14 | answered | no (refusal) | — | has citations | — | — | IT-002-CH02 (dropped: IT-002-CH03) | ed on a standard refresh cycle of every 4 years, or earlier if the device is faulty and cannot be repaired. [Source: IT-002, Chunk: IT-002-CH02] |
| `fixed_500+bm25` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01 (dropped: HR-002-CH05, HR-001-CH02, IT-002-CH01) | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+bm25+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH04, HR-001-CH05 | employees receive 20 days of paid annual leave per calendar year. [Source: HR-001, Chunk: HR-001-CH02] |
| `fixed_500+bm25+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 | nt is automatically locked for 15 minutes after 5 consecutive failed sign-in attempts. [Source: IT-001, Chunk: IT-001-CH03] |
| `fixed_500+bm25+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH04, IT-002-CH04, IT-002-CH03 (dropped: IT-001-CH03) | Compromised Account [Source: IT-001, Chunk: IT-001-CH04] |
| `fixed_500+bm25+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | t be seen or overheard. [Source: HR-002, Chunk: HR-002-CH04] |
| `fixed_500+bm25+rerank` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH03, HR-003-CH04 (dropped: HR-001-CH02, HR-002-CH05) | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+bm25+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+bm25+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | IT-002-CH02 (dropped: IT-002-CH03) | ed on a standard refresh cycle of every 4 years, or earlier if the device is faulty and cannot be repaired. [Source: IT-002, Chunk: IT-002-CH02] |
| `fixed_500+bm25+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01 (dropped: HR-002-CH04, HR-001-CH02, HR-002-CH05) | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+hybrid` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH05, HR-001-CH02, HR-002-CH02, HR-002-CH03 | ement Leave [Source: HR-001, Chunk: HR-001-CH05] |
| `fixed_500+hybrid` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH02, IT-001-CH01, IT-001-CH04) | nt is automatically locked for 15 minutes after 5 consecutive failed sign-in attempts. [Source: IT-001, Chunk: IT-001-CH03] |
| `fixed_500+hybrid` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH04, IT-002-CH03, IT-002-CH04, IT-002-CH02 | Compromised Account [Source: IT-001, Chunk: IT-001-CH04] |
| `fixed_500+hybrid` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, IT-002-CH02 | t be seen or overheard. [Source: HR-002, Chunk: HR-002-CH04] |
| `fixed_500+hybrid` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH03, HR-003-CH04 (dropped: HR-002-CH03, IT-002-CH02) | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+hybrid` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+hybrid` | SM-14 | answered | no (refusal) | — | has citations | — | — | IT-002-CH02, HR-004-CH02 (dropped: IT-002-CH03, HR-004-CH03) | ed on a standard refresh cycle of every 4 years, or earlier if the device is faulty and cannot be repaired. [Source: IT-002, Chunk: IT-002-CH02] |
| `fixed_500+hybrid` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01 (dropped: HR-001-CH02, IT-001-CH01, IT-002-CH04) | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+hybrid+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH04, HR-001-CH05 | employees receive 20 days of paid annual leave per calendar year. [Source: HR-001, Chunk: HR-001-CH02] |
| `fixed_500+hybrid+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH03 (dropped: IT-001-CH01, HR-004-CH01, IT-001-CH02) | nt is automatically locked for 15 minutes after 5 consecutive failed sign-in attempts. [Source: IT-001, Chunk: IT-001-CH03] |
| `fixed_500+hybrid+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH04, IT-002-CH04, IT-002-CH03 (dropped: IT-001-CH03) | Compromised Account [Source: IT-001, Chunk: IT-001-CH04] |
| `fixed_500+hybrid+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH04, HR-002-CH03, HR-002-CH02, HR-002-CH01 | t be seen or overheard. [Source: HR-002, Chunk: HR-002-CH04] |
| `fixed_500+hybrid+rerank` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH03, HR-003-CH04 (dropped: HR-001-CH02, HR-003-CH02) | ent head. [Source: HR-003, Chunk: HR-003-CH03] |
| `fixed_500+hybrid+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `fixed_500+hybrid+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02, IT-002-CH02 (dropped: HR-004-CH03, HR-002-CH05) | before their joining date: a government-issued photo ID, bank account details for payroll, and the signed offer letter. [Source: HR-004, Chunk: HR-004-CH02] |
| `fixed_500+hybrid+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH04, HR-003-CH03, HR-003-CH02, HR-003-CH05 | onal Vehicle [Source: HR-003, Chunk: HR-003-CH04] |
| `recursive_800_100+dense` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH04, HR-001-CH03 | Employees must submit annual leave requests through the HR Portal at least 7 calendar days before the first day of leave. [Source: HR-001, Chunk: HR-001-CH02] |
| `recursive_800_100+dense` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH02 (dropped: IT-001-CH01, IT-002-CH01, IT-001-CH03) | ## 3. [Source: IT-001, Chunk: IT-001-CH02] |
| `recursive_800_100+dense` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH01, IT-002-CH02 | ## 6. [Source: IT-001, Chunk: IT-001-CH03] |
| `recursive_800_100+dense` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH02, HR-002-CH01, IT-002-CH02 | ## 6. [Source: HR-002, Chunk: HR-002-CH03] |
| `recursive_800_100+dense` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-002-CH02, HR-002-CH03) | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+dense` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, IT-002-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+dense` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-002-CH03) | ## 3. [Source: HR-004, Chunk: HR-004-CH02] |
| `recursive_800_100+dense` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH02, HR-003-CH03, HR-003-CH01, HR-001-CH01 | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+dense+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH03, HR-001-CH04 | Employees must submit annual leave requests through the HR Portal at least 7 calendar days before the first day of leave. [Source: HR-001, Chunk: HR-001-CH02] |
| `recursive_800_100+dense+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH02 (dropped: IT-001-CH01, HR-004-CH01, HR-001-CH01) | ## 3. [Source: IT-001, Chunk: IT-001-CH02] |
| `recursive_800_100+dense+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02, IT-002-CH01 | ## 6. [Source: IT-001, Chunk: IT-001-CH03] |
| `recursive_800_100+dense+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH02, HR-002-CH01 (dropped: HR-002-CH04) | ## 6. [Source: HR-002, Chunk: HR-002-CH03] |
| `recursive_800_100+dense+rerank` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-001-CH02, HR-002-CH02) | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+dense+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+dense+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-001-CH01) | ## 3. [Source: HR-004, Chunk: HR-004-CH02] |
| `recursive_800_100+dense+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH03, HR-003-CH02, HR-003-CH01, HR-002-CH03 | ## 5. [Source: HR-003, Chunk: HR-003-CH03] |
| `recursive_800_100+bm25` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-002-CH02, HR-001-CH02, HR-001-CH04, HR-004-CH02 | ## 3. [Source: HR-002, Chunk: HR-002-CH02] |
| `recursive_800_100+bm25` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH02 | ## 3. [Source: IT-001, Chunk: IT-001-CH02] |
| `recursive_800_100+bm25` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03 (dropped: IT-001-CH02, HR-002-CH01) | ## 6. [Source: IT-001, Chunk: IT-001-CH03] |
| `recursive_800_100+bm25` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH02, HR-002-CH01 (dropped: HR-001-CH04) | ## 6. [Source: HR-002, Chunk: HR-002-CH03] |
| `recursive_800_100+bm25` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-002-CH02, IT-002-CH03) | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+bm25` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+bm25` | SM-14 | refused_no_context | yes (refusal) | — | none | — | — |  (dropped: IT-002-CH02) | The policy documents do not contain enough information to answer this question. |
| `recursive_800_100+bm25` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH03 (dropped: HR-001-CH02, IT-002-CH01) | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+bm25+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH03, HR-001-CH04 | Employees must submit annual leave requests through the HR Portal at least 7 calendar days before the first day of leave. [Source: HR-001, Chunk: HR-001-CH02] |
| `recursive_800_100+bm25+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH02 | ## 3. [Source: IT-001, Chunk: IT-001-CH02] |
| `recursive_800_100+bm25+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02 (dropped: IT-001-CH02) | ## 6. [Source: IT-001, Chunk: IT-001-CH03] |
| `recursive_800_100+bm25+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH02, HR-002-CH01 (dropped: HR-002-CH04) | ## 6. [Source: HR-002, Chunk: HR-002-CH03] |
| `recursive_800_100+bm25+rerank` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-001-CH02, HR-002-CH02) | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+bm25+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+bm25+rerank` | SM-14 | refused_no_context | yes (refusal) | — | none | — | — |  (dropped: IT-002-CH02) | The policy documents do not contain enough information to answer this question. |
| `recursive_800_100+bm25+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH03, HR-001-CH01 (dropped: HR-001-CH02) | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+hybrid` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH04, HR-001-CH03 | Employees must submit annual leave requests through the HR Portal at least 7 calendar days before the first day of leave. [Source: HR-001, Chunk: HR-001-CH02] |
| `recursive_800_100+hybrid` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH02 (dropped: IT-001-CH01, IT-002-CH01, IT-001-CH03) | ## 3. [Source: IT-001, Chunk: IT-001-CH02] |
| `recursive_800_100+hybrid` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02, IT-002-CH01 | ## 6. [Source: IT-001, Chunk: IT-001-CH03] |
| `recursive_800_100+hybrid` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH02, HR-002-CH01, IT-002-CH02 | ## 6. [Source: HR-002, Chunk: HR-002-CH03] |
| `recursive_800_100+hybrid` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-002-CH02, HR-002-CH03) | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+hybrid` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-001-CH01, HR-002-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+hybrid` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-002-CH03) | ## 3. [Source: HR-004, Chunk: HR-004-CH02] |
| `recursive_800_100+hybrid` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH03, HR-004-CH02, HR-001-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+hybrid+rerank` | DEV-01 | answered | no | [False] | valid | [False] | — | HR-001-CH02, HR-002-CH02, HR-001-CH03, HR-001-CH04 | Employees must submit annual leave requests through the HR Portal at least 7 calendar days before the first day of leave. [Source: HR-001, Chunk: HR-001-CH02] |
| `recursive_800_100+hybrid+rerank` | DEV-02 | answered | no | [False] | valid | [False] | — | IT-001-CH02 (dropped: IT-001-CH01, HR-004-CH01, HR-001-CH01) | ## 3. [Source: IT-001, Chunk: IT-001-CH02] |
| `recursive_800_100+hybrid+rerank` | DEV-03 | answered | no | [False] | valid | [False] | — | IT-001-CH03, IT-002-CH03, IT-002-CH02 (dropped: IT-001-CH02) | ## 6. [Source: IT-001, Chunk: IT-001-CH03] |
| `recursive_800_100+hybrid+rerank` | DEV-04 | answered | no | [False, False] | valid | [False, False] | — | HR-002-CH03, HR-002-CH02, HR-002-CH01 (dropped: HR-002-CH04) | ## 6. [Source: HR-002, Chunk: HR-002-CH03] |
| `recursive_800_100+hybrid+rerank` | DEV-05 | answered | no | [False, False] | valid | [False, False] | — | HR-003-CH02, HR-003-CH03 (dropped: HR-001-CH02, HR-002-CH02) | An itemised receipt is required for every individual expense above USD 25. [Source: HR-003, Chunk: HR-003-CH02] |
| `recursive_800_100+hybrid+rerank` | SM-13 | answered | no (refusal) | — | has citations | — | — | HR-003-CH01, HR-002-CH01, HR-001-CH01, HR-004-CH01 | # Employee Reimbursement Policy (HR-003) [Source: HR-003, Chunk: HR-003-CH01] |
| `recursive_800_100+hybrid+rerank` | SM-14 | answered | no (refusal) | — | has citations | — | — | HR-004-CH02 (dropped: IT-002-CH02, HR-004-CH01, HR-001-CH01) | ## 3. [Source: HR-004, Chunk: HR-004-CH02] |
| `recursive_800_100+hybrid+rerank` | SM-15 | answered | no (refusal) | — | has citations | — | — | HR-003-CH03, HR-003-CH02, HR-003-CH01, HR-002-CH03 | ## 5. [Source: HR-003, Chunk: HR-003-CH03] |
