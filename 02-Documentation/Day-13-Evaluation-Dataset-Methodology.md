# Day 13 – Evaluation Dataset: Annotation and Ground-Truth Methodology

**Status:** Completed - Pending Review (dataset frozen) · **Date:** 2026-11-26 · **Phase:** 5 of 9

> **Update (before Phase 7):** EV-06 was exposed in Phase 6. The clean Phase 7 set is **v1.1**
> (`eval_queries_v1_1.json`, SHA-256 `e860b729…`): the 29 unexposed questions below + EV-31. This v1.0 document and
> file are unchanged. See `Day-13-Evaluation-Dataset-v1.1-EV-06-Replacement.md`.

## 1. Dataset at a glance

| Item | Value |
|---|---|
| File | `01-Code/policy-qa-rag-v2/evaluation/datasets/eval_queries_30.json` |
| Version | 1.0 |
| SHA-256 | `c9677c0fe123a97ff65bf462bd3655fe0aed2717b2e58c1e49ae3322750046bb` (`manifests/eval-dataset.sha256`) |
| Frozen (UTC) | 2026-11-26T16:26:53+00:00 (`evaluation/datasets/eval_dataset_freeze.json`) |
| Retrieval systems run on it before the freeze | **none** |
| Queries | 30: 25 answerable, 5 unanswerable |
| Labels | 34 required fact groups over 36 distinct canonical evidence IDs |
| Purpose | Held-out evaluation only (Phase 7). Not used to tune chunking, BM25, RRF, reranking, thresholds or prompts. |

## 2. Composition

| Category | n | Queries | What it tests |
|---|---:|---|---|
| Paraphrase / synonym | 5 | EV-01 – EV-05 | vocabulary mismatch ("time off for being ill" → sick leave, "vacation days" → annual leave) |
| Exact keyword / identifier | 4 | EV-06 – EV-09 | product names (TechDesk, Brightlake Authenticator), policy codes (HR-004, IT-002), acronyms (MFA, USB) |
| Numerical / date | 6 | EV-10 – EV-15 | dates, times, amounts, counts (31 March, 17:00 Friday, USD 0.40, USD 1,000, 10 passwords, 12 holidays) |
| Multi-fact (same section) | 4 | EV-16 – EV-19 | two required facts in adjacent sentences |
| Multi-fact, cross-section | 3 | EV-20 – EV-22 | facts in two different sections of one policy |
| Multi-fact, cross-policy | 1 | EV-23 | HR contact (stated in 3 policies) + Expense Portal (HR-003) |
| Near-topic distractor (answerable) | 2 | EV-24, EV-25 | a misleading nearby fact (HR-003 receipt rules; "90 days" probation check-ins) |
| Unanswerable, near-topic | 3 | EV-26 – EV-28 | jury-duty leave, taxi fares, phone-bill allowance: topic sounds covered, fact is not |
| Unanswerable, off-topic | 2 | EV-29, EV-30 | dental insurance, dogs at the office |

Difficulty (assigned before any run, with the reason in `notes`): 12 easy, 11 medium, 7 hard. Answerable queries
cover every policy (HR-001: 8, HR-002: 5, IT-001: 4, IT-002: 4, HR-004: 4, HR-003: 3 — counting a cross-policy query
once per policy); HR-003 is lighter because two of its facts are development-query facts (excluded, §4).

## 3. Label schema (per query)

| Field | Meaning |
|---|---|
| `id`, `question`, `answerable`, `category`, `difficulty` | identity and design intent |
| `expected_policy_ids` | the policies that contain the required evidence (validator enforces equality) |
| `required_evidence` | list of **groups**; each group = one fact the answer needs; IDs in a group are interchangeable sentences (any one satisfies it) — e.g. EV-23 group 1 = the HR contact sentence in HR-001, HR-002 **or** HR-004 |
| `answer_terms` | groups of acceptable phrasings for the Phase 7 answer check; each group must be matched; at least one term per group occurs verbatim in the required evidence text (validator) |
| `reference_answer` | short answer written only from the required evidence |
| `distractor_policy_ids`, `notes` | known near-topic sources and ambiguity notes |
| `absent_terms` (unanswerable) | topic words verified **not** to occur anywhere in the corpus |

**Canonical fact IDs, not chunk IDs.** Every label is a sentence-level evidence ID from the frozen catalog
(`HR-001-S04-E01` = HR-001, section 4, sentence 1). The same labels therefore apply to every chunking strategy; the
test suite asserts that no chunk ID (`-CH`) appears anywhere in the dataset.

## 4. How the labels were produced and verified

1. **Written from the policy text only.** For each question the required sentences were picked from the evidence
   catalog (verbatim policy sentences). No retrieval system — v1, dense, BM25, hybrid or reranked — was run on these
   30 questions before the freeze; the validator imports no retrieval code (asserted by a test).
2. **Self-contained alternatives only.** An alternative in a group must answer that fact on its own. Example: for
   EV-22 ("first day") `HR-004-S03-E02` ("During orientation *they* receive their company laptop…") is *not* an
   alternative, because the sentence alone does not say "first day".
3. **Automated validation** (`scripts/validate_eval_dataset.py`, all PASS): schema and IDs; 25/5 split; category
   minimums; unique questions; every evidence ID exists; expected policies = evidence policies; answer terms occur in
   the evidence text; unanswerable queries have no evidence/terms and their topic terms are absent from the corpus
   (jury, taxi/cab, phone allowance/mobile allowance/phone bill, dental/insurance, dog/pet: 0 occurrences each).
4. **Manual review** of every question beside its evidence text (`03-Evidence/19-eval-dataset-label-review-sheet.txt`).
   The review, done **before** the freeze, led to these changes:
   - EV-07 and EV-24 reference answers contained facts from sentences that were not labelled (MFA mandatory; paid via
     payroll) → trimmed to the labelled evidence.
   - EV-22: removed the non-self-contained alternative (see 2).
   - EV-23 (originally a laptop/encryption question) relied on the same non-self-contained sentence for its
     cross-policy part → replaced by the HR-contact + Expense-Portal question.
   - Answer terms that were too loose (`"10"`, `"12"`, `"40"` would match unrelated numbers) or too narrow
     (no "ten days", "5 pm", "March 31", "1000") → tightened / widened.
5. **Freeze:** SHA-256 and timestamp recorded; re-freezing with a different hash is refused.

## 5. Independence

| Risk | Control | Result |
|---|---|---|
| Tuning on test data | dev queries (5) are the only tuning set; no evidence fact shared with them | 0 of 8 dev facts reused |
| Near-copies of dev questions | word Jaccard < 0.6 | max 0.50 (EV-06 "What is TechDesk?" vs DEV-02 "What is KeyVault?": same template, different entity, policy and fact — kept) |
| Near-copies of Day 12 smoke questions | word Jaccard < 0.6 | max 0.45 (EV-20 vs SM-04) |
| Labels fitted to system output | freeze before any run; validator has no retrieval imports | "retrieval systems run before freeze: none" |

**Overlap with the Day 12 smoke test (hand-checked).** 8 questions are newly worded but share a fact with a smoke
question: EV-10/SM-02, EV-17/SM-09, EV-18/SM-11, EV-19/SM-03, EV-20/SM-04, EV-21/SM-08, EV-22/SM-12, EV-23/SM-06. This is
unavoidable in a 102-fact corpus, and the smoke questions were used only to validate v1 on Day 12 — never to tune v2.
(The validator's automatic overlap heuristic ignores terms under 3 characters to avoid false matches such as "5" and
therefore misses EV-21/SM-08 via "14"; the hand-checked list is authoritative.)

Notes for EV-17 and EV-19 mention that their sentences are split by `fixed_500` chunking. That comes from the static
Phase 2 chunk analysis of the corpus, not from running retrieval on these questions, and did not influence the labels.

## 6. Known limitations

- One annotator (with automated checks); no inter-annotator agreement measure.
- 30 queries over a 6-policy corpus: confidence intervals are wide; differences of one or two queries should be
  read with caution in Phase 7.
- Answer terms are string matches; Phase 7 combines them with citation validity and citation support (as Day 12).
- Sentence-level evidence: a fact spread over two sentences is labelled as two groups.
