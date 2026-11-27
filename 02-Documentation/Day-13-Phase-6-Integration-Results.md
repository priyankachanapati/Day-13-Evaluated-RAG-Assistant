# Day 13 – Phase 6: RAG v2 Integration Results (development queries)

**Status:** Completed - Pending Review — integration tested with a mocked LLM and a **completed real-LLM smoke test on the
development queries** (§3b); a first attempt had failed on authentication (§3). · **Date:** 2026-11-26

## 1. What was verified

| Check | Result | Evidence |
|---|---|---|
| v1 equivalence: `v1_paragraph_800_150+dense` vs frozen v1 `RAGPipeline` on the 15 Day 12 smoke questions | byte-identical system + user prompts, 15/15 identical statuses | unit test; `03-Evidence/24-…` |
| All 18 configurations end-to-end (mocked LLM) | context ≤ 4 chunks, all with dense similarity ≥ 0.25; every context chunk in the prompt; valid citations accepted; stage timings present exactly for the stages each mode uses; reranking is a permutation of the candidates | unit tests (18 parametrised) |
| Refusal semantics | dense gate refuses without an LLM call even when BM25 and the reranker found candidates; per-chunk dense filter; LLM refusal; fabricated citation → withheld | unit tests |
| Credentials and failures | missing key → clear `ConfigurationError` (retrieval still works); `GenerationError` / `QuotaExhaustedError` propagate; 429 with retry-after → waits retry-after + 1 s and succeeds | unit tests; `23a-…`, `23b-…` |
| Instrumentation | tokens (prompt/completion), cost estimate = tokens × list price, per-stage ms, total ≥ retrieval + generation, one-time costs kept separate, cache hits flagged | unit tests |
| Runner | mocked run of 2 configs × 8 queries; resume adds no duplicates; quota stop after 2 calls keeps the 2 records, status "incomplete", exit 2; missing key → exit 3, nothing written, no secret printed | unit tests |
| Mock dry-run, all 18 configs × (5 dev + 3 probes) | complete; 144 records, 0 duplicates (run interrupted by the call budget and resumed) | `22-…`, `22b-…`, `results/dev-answer-eval/mock-all-18-configs/` |

## 2. Development-query results (mock LLM — retrieval-side numbers only)

The mock "answer" just quotes the first context chunk, so answer-quality numbers of the mock run are **not
meaningful** and are not reported as results. Retrieval-side numbers are real:

- Context-fact coverage (required facts inside the 4 context chunks) = **1.00 for all 18 configurations** on the 5 dev
  queries.
- The dense per-chunk filter removed 5–10 low-similarity chunks per configuration over the 8 queries (mostly for the
  off-topic probes); on dev queries it never removed a chunk holding a required fact (0 cases).
- In BM25-only modes the dense gate refused one probe without an LLM call (BM25 found keyword chunks, none with dense
  similarity ≥ 0.25) — v1 semantics.
- Median retrieval time per query: ~20–22 ms without reranking, ~62–87 ms with reranking (mock run).

## 3. Real-LLM smoke test — run 2026-11-26 17:01 UTC, INCOMPLETE (authentication failure)

| Item | Result |
|---|---|
| Command | `HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_dev_answer_eval.py --run-name real-smoke-1` (as instructed) |
| Items | 2 configs × (5 dev + 3 refusal probes) = 16, all attempted; no held-out question |
| Outcome | **16/16 errors: "Groq rejected the API key (authentication error)."** — the key in `.env` is not accepted by Groq |
| Answers / correctness / citations / support / refusals | **none produced — cannot be reported** |
| Tokens / estimated cost | 0 prompt, 0 completion tokens; USD 0 (rejected requests return no usage) |
| Latency | not meaningful (no completion); retrieval ran normally |
| Secrets | no key-like string in records, logs or summaries |
| Evidence | `03-Evidence/26-…` (console), `26b-…` (relabelled summary), `results/dev-answer-eval/real-smoke-1/` |

**Runner bugs found by this run (fixed, with regression tests):** (1) the run was labelled `complete` (exit 0) although all 16 items failed — status is now `complete` only if every requested item has a non-error latest record, otherwise `incomplete: …` (exit 2); (2) an authentication failure is not per-query — the runner now stops after the first one (exit 3) instead of trying all items; (3) `--resume` treated error records as done — it now retries them, and summaries use the latest record per (config, query). A new `--summarize-only` mode rebuilt the `real-smoke-1` summary from its records without any API call: **status `incomplete: authentication failed (Groq rejected the API key)`**. The original mislabelled summary is kept as `summary-original-MISLABELLED-complete.{json,md}`. `manifests/phase6-artifacts.sha256` was regenerated after the fix (2 files changed: the runner and its test).

**Credential handling:** before the run, `.env.example` (a file meant to be committed) was found to contain a real-looking Groq key, different from the one in `.env`. With the user's approval the single value was replaced by the original placeholder (rest of the file verified identical; no value printed). No git repository exists, so nothing was committed or pushed. Recommendation: revoke that key and create a new one for `.env`.

**To complete the smoke test:** put a valid key in `.env`, then (completed items are kept; failed ones are retried):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_dev_answer_eval.py --run-name real-smoke-1 --resume
```

Configurations: `v1_paragraph_800_150+dense` (v1 baseline) and `v1_paragraph_800_150+hybrid+rerank`. Results will be
written to `results/dev-answer-eval/real-smoke-1/` and reported as measured, including failures.

## 3b. Real-LLM smoke test — completed 2026-11-26 17:13 UTC (`--resume` after a valid key was configured)

Status **complete**: 16/16 items with a non-error latest record (the 16 earlier authentication-error records remain in
`records.jsonl`; summaries use the latest record per item). Only DEV-01–05 and SM-13–15 were used. 14 real calls + 2
cache hits (identical prompts). No key-like string in any record, log or summary.

| Metric (deterministic) | `v1_paragraph_800_150+dense` (v1 baseline) | `v1_paragraph_800_150+hybrid+rerank` |
|---|---:|---:|
| Correct (all checks) | 4/5 | 3/5 |
| Citations valid / invalid citations | 5/5 / 0 | 5/5 / 0 |
| Citation support (all answer facts in a cited chunk) | 4/5 | 3/5 |
| Cited-fact coverage / context-fact coverage | 1.00 / 1.00 | 1.00 / 1.00 |
| Unsupported numbers | 0 | 0 |
| Refusal probes correct (all via LLM refusal) | 3/3 | 3/3 |
| Tokens prompt / completion | 7,653 / 1,667 | 7,559 / 1,442 (incl. 2 cached) |
| Estimated cost at list price | USD 0.002148 | USD 0.001999 (incl. cached-equivalent) |
| Median retrieval / generation / total ms | 41 / 975 / 1,016 | 109 / 1,059 / 1,186 |

Tokens actually spent: 13,993 prompt + 2,911 completion → USD 0.003846 at the list price (an estimate, not the plan's
charge). Every number above was recomputed from `records.jsonl` and equals `summary.json` (`03-Evidence/28-…`).

**Why DEV-04 and DEV-05 failed:**
- **DEV-04 (both configs)** — partly a real omission, partly a string-match false negative. Both answers say "only when
  *you are* connected through the company VPN": correct, but the answer term "only when connected" does not match the
  inserted words. The second required fact (VPN **at all times** when working from home — the question asked "do I
  *always* need the VPN?") is genuinely missing from both answers.
- **DEV-05 (hybrid + rerank only)** — evaluator strictness, not a wrong answer: the answer applies the rule ("you may book
  premium economy for a 7-hour flight") instead of stating it ("6 hours or longer"), so the answer term "6 hours" is
  absent; the cited chunk contains the 6-hour sentence (cited-fact coverage 1.00). The v1 answer stated the rule and passed.

**Known limitations confirmed (no code change made, per instruction):** answer-term string matching produces false
negatives for inserted words and applied-rule answers; the unsupported-number check can pass a number by coincidence
(the "7" above matched the section heading "## 7." in the cited chunk). Phase 7 reports therefore show the
component checks alongside "correct", and these two dev cases are documented examples. Dev answer keys stay frozen.

Five development queries cannot rank the two configurations; that comparison is Phase 7.

## 4. Disclosure: held-out query used by mistake

While preparing the CLI demo I used **"What is TechDesk?" — held-out query EV-06** — in two CLI runs (`ask`, which
retrieved and then stopped for the missing key, and `retrieve` with `v1_paragraph_800_150+hybrid+rerank`), and in two
mocked unit tests (no assertions on its ranking). The CLI output (top-4 chunk IDs, dense similarities, timings for one
configuration) is kept unedited in `03-Evidence/23a-…-ACCIDENTAL-HELD-OUT-QUERY-EV-06.txt`.

Actions: EV-06 is excluded from the clean Phase 7 set and replaced by EV-31 in dataset v1.1 (v1.0 kept unchanged; see `Day-13-Evaluation-Dataset-v1.1-EV-06-Replacement.md`); the tests now use dev query DEV-02; the CLI demo was redone with DEV-02 (`23b-…`); a guard test
(`tests_v2/test_held_out_guard.py`) fails if any held-out question text appears in code, tests, scripts, activities or
results. **Nothing was changed because of it:** the dataset is byte-identical to its frozen hash, no configuration or
threshold was changed, and the observation is not used in any analysis. Phase 7 reports EV-06 like every other query.
