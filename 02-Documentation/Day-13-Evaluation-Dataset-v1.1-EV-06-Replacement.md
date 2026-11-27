# Day 13 – Evaluation Dataset v1.1: Replacement of Exposed EV-06

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Between Phase 6 and Phase 7**

## 1. Why

During Phase 6 I used held-out question **EV-06** ("What is TechDesk?") in two CLI runs and two mocked unit tests,
so retrieval had been run on it before the evaluation (see `Day-13-Phase-6-Integration-Results.md` §4). An exposed
question can no longer count as held-out, so it is excluded from the clean Phase 7 set and replaced.

## 2. What changed (and what did not)

| Item | Value |
|---|---|
| v1.0 (`evaluation/datasets/eval_queries_30.json`) | **unchanged**, immutable record; SHA-256 `c9677c0fe123a97ff65bf462bd3655fe0aed2717b2e58c1e49ae3322750046bb` (re-verified) |
| v1.1 (`evaluation/datasets/eval_queries_v1_1.json`) | the clean Phase 7 set: the 29 unexposed v1.0 questions (identical, field by field) + **EV-31** |
| v1.1 SHA-256 | `e860b7294c57c2275c8d9a886b39fb55470158be103ca3288d478cdad73e39ca` |
| v1.1 frozen (UTC) | 2026-11-26T17:00:48+00:00 (`evaluation/datasets/eval_dataset_v1_1_freeze.json`, `manifests/eval-dataset-v1_1.sha256`) |
| Exposure record (`evaluation/datasets/eval_queries_exposed.json`) | EV-06 exactly as in v1.0 + what exposed it, where the output is kept, "no LLM answer generated", "labels/configuration not changed"; SHA-256 `2c078822bb05da74fe93b31c880711ce00b1db420a4cb1e703ff9d2017d79ae6` |
| Composition | still 30 questions, 25 answerable / 5 unanswerable; category and difficulty mixture identical to v1.0 |
| EV-06 in Phase 7 | not in the clean set; may be reported separately as a clearly labelled sensitivity analysis |

## 3. EV-31

| Field | Value |
|---|---|
| Question | What can the Finance Controller approve? |
| Category / difficulty | exact keyword / identifier / medium — same as EV-06 |
| Expected policy | HR-003 (Employee Reimbursement Policy) |
| Required evidence | `HR-003-S02-E02`: "Claims submitted after 30 days will be rejected unless the Finance Controller approves a late submission." |
| Answer terms | (late submission \| late submissions \| late claim \| late claims) AND (30 days \| 30-day \| thirty days) |
| Reference answer | A late submission: expense claims submitted after 30 days are rejected unless the Finance Controller approves a late submission. |
| Distractors | same-policy approvals in HR-003 §3 (line manager for all claims; department head above USD 1,000 = EV-13) |

**Selection procedure (no retrieval, no embeddings ranking, no generation):** candidate sentences naming a system, role,
portal or address were listed from the frozen catalog and filtered by four criteria set beforehand — not used by any dev
or v1.0 query, self-contained, same category as EV-06, and intact in all three chunking strategies (Phase 2 static
analysis; picking a sentence that one strategy splits would bias against that strategy). Three passed: the Learning
Portal sentence was rejected (not self-contained: "Both courses…"), the HR-Portal leave-notice sentence was rejected
(adjacent to the DEV-01 fact), and `HR-003-S02-E02` was chosen.

**Verification** (`scripts/validate_eval_dataset_v1_1.py`, no retrieval imports; all 18 checks PASS,
`03-Evidence/25-…`): v1.0 hash unchanged; v1.1 supersedes v1.0 by hash; IDs = v1.0 − EV-06 + EV-31; 29 carried-over
queries identical; exposure record = v1.0 EV-06; 25/5 split and category mixture unchanged; EV-31 evidence exists and
occurs verbatim in the corpus; expected policy = evidence policy; every answer-term group occurs in the evidence text;
fact unused by dev and by all v1.0 queries (incl. EV-06); max word Jaccard vs any dev / smoke / v1.0 question 0.17.
The held-out guard test now covers v1.0 and v1.1; EV-31's text appears nowhere outside the dataset files.
