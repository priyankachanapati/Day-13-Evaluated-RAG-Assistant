# Day 13 – Answer Evaluation Methodology

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Phase:** 6 of 9

## 1. Deterministic checks only

Every check in `evaluation/answer_eval.py` is deterministic (string, ID and offset comparisons, plus the v1 citation
validator). **No LLM-as-judge is used** in Phase 6: it would spend the same limited free-tier quota as the system
under test and add its own variance. Consequence: paraphrased-but-correct answers can be missed by string matching,
and non-numeric unsupported claims are not detected automatically; both limitations are stated in every report.

## 2. Checks for an answerable query

| Check | Definition | Why |
|---|---|---|
| Status | must be `answered` (not refused, not withheld) | |
| Answer terms | every answer-term group appears in the answer (case-insensitive; alternatives per group) | correctness against reference facts |
| Citation validity | ≥ 1 citation and all valid (v1 validator: cited chunk was in the context and the policy ID matches), none malformed | |
| Cites expected policy | at least one valid citation to an expected policy | |
| Citation support | each answer-term group also appears in the text of a **validly cited** chunk | a syntactically valid citation is not enough — it must contain the stated fact |
| Cited-fact coverage | required canonical fact groups fully contained in validly cited chunks | fact-level grounding |
| Context-fact coverage | required fact groups present in the context sent to the LLM | separates retrieval failures from generation failures |
| Unsupported numbers | numbers / times / amounts in the answer prose (citations and policy IDs removed) that occur in no cited chunk | deterministic groundedness proxy for invented figures |
| Uncited sentences | informational (v1) | |
| **Correct** | status answered AND all answer terms AND citations valid AND cites expected policy AND all citation support AND no unsupported numbers | |

## 3. Unanswerable queries

`refusal_correct` = status `refused` or `refused_no_context` AND no citations. A refusal without an LLM call (dense
gate) and an LLM refusal are both correct refusals; the record shows which one happened.

## 4. Data used in Phase 6

- The 5 frozen development queries (`dev_queries.json`, unchanged) with answer keys in a new file
  `evaluation/datasets/dev_answer_keys.json` (answer terms and reference answers; every answer term occurs in the
  required evidence text; SHA-256 `4de6c03b…` in `manifests/dev-answer-keys.sha256`, frozen before any answer was
  generated).
- The development set contains no unanswerable query, so the three unanswerable Day 12 smoke questions (SM-13 pet
  insurance, SM-14 stock options, SM-15 gym membership) are used as **refusal probes**. They are v1's own test
  questions, not part of the 30-query held-out set.
- The 30-query held-out set is not used (Phase 7).

## 5. Tested behaviour

`tests_v2/test_answer_eval.py`: correct answer; valid citation to a chunk that does not contain the fact (not
correct); invented number flagged; missing fact; withheld answer with invalid citation; context coverage independent of
citations; refusal correct / refusal with citations; number extraction ignores citations and policy IDs.
