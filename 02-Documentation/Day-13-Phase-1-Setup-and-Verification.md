# Day 13 – Phase 1: Workspace Setup and RAG v1 Baseline Verification

**Status:** Completed - Pending Review · **Date:** 2026-11-26 · **Module:** AI05 – Advanced RAG, Retrieval Evaluation & Mini-Project 1

## 1. Objective

Prepare an isolated Day 13 project where the Day 12 RAG v1 is preserved **unchanged** as the baseline for every later
comparison, and prove that it behaves exactly like the original before any v2 work starts.

## 2. What was done

| Step | Result |
|------|--------|
| Folder structure | `Day-13/{01-Code,02-Documentation,03-Evidence,04-Submission}`; project `Day-13/01-Code/policy-qa-rag-v2/` |
| Verbatim copy of v1 | `src/` (10 files), `tests/` (9 files), `pytest.ini`, `data/policies/` (6 policies) from `Day-12/01-Code/policy-qa-rag` (`rsync`, `__pycache__` excluded) |
| Reference results | Day 12 run-1 `smoke-results-run1.json` and `smoke-test-log-run1.txt` copied to `reference/` |
| SHA-256 manifests | `manifests/v1-baseline-code.sha256` (20), `manifests/corpus.sha256` (6), `manifests/day12-reference-results.sha256` (2) |
| Environment | `.venv` via uv 0.13.9, Python 3.12.16; 109 packages from lock file `requirements.txt` |
| Config files | `.env.example` (names only), `.gitignore` (`.env` ignored), `pyproject.toml` (ruff; `src/`, `tests/` excluded from formatting so the baseline is never rewritten) |
| Verification scripts | `scripts/verify_manifests.sh`, `scripts/verify_v1_baseline_reproduction.py` |

Not done (by instruction): no `.env` created and no Day 12 key read; no GitHub action; no Phase 2 work.

## 3. Key technical decisions

- **Verbatim copy instead of importing Day 12 code.** Keeps Day 13 self-contained (day isolation) while v1 stays
  byte-identical. The copy is never edited; v2 code will live in new packages (`rag_v2/`, `evaluation/`).
- **Three-way hash verification.** The manifest is checked against (1) the Day 13 copy, (2) the Day 12 working files
  and (3) Day 12 git commit `b366fc8` via `git show` (reads git objects only). (3) proves the copy equals the committed,
  pushed v1 — not just whatever is on disk. A negative control (one altered hash) was confirmed to fail.
- **Constraints file for reproducible embeddings.** Day 12's exact environment (`uv pip freeze`) is saved as
  `constraints/day12-v1-environment.txt` and used as resolver constraints, so every one of the 91 packages shared with
  v1 (chromadb 1.5.9, onnxruntime 1.32.0, numpy 2.5.5, tokenizers 0.23.3, …) has the identical version. Different
  onnxruntime/numpy builds could otherwise shift embedding values. `requirements.in` lists direct dependencies;
  `requirements.txt` is the compiled lock; `uv pip sync` installs exactly the lock.
- **Compatibility checked before installing.** `uv pip compile` (resolution only, no install) first failed because
  I passed `--python-platform aarch64-apple-darwin`, which targets macOS 13, while onnxruntime 1.32.0 ships only
  macOS ≥ 14 wheels. Resolving against the real interpreter (macOS 26.6.2, arm64 — the machine Day 12 was installed on)
  succeeded: 109 packages, 0 Day 12 versions changed, 18 new (rank-bm25 0.2.2, sentence-transformers 6.3.0,
  torch 2.15.0, transformers 5.24.1, scikit-learn 1.9.1, scipy 1.18.1 and their dependencies).

## 4. Results (actual)

| Check | Result | Evidence |
|-------|--------|----------|
| Manifests: Day 13 copy / Day 12 files / Day 12 git `b366fc8` / reference results | **PASS** (all 4 checks) | `03-Evidence/01-hash-verification-log.txt` |
| Same check repeated at the end of Phase 1 (after tests and ingestion) | **PASS** | `03-Evidence/05-final-hash-verification-after-phase-1.txt` |
| `uv pip check`; installed == lock; Day 12 versions unchanged | **PASS**: 109 compatible; 109 == 109; 91/91 identical | `03-Evidence/02-environment-setup-log.txt` |
| Import smoke test (9 packages) | **PASS** (cold first-import times; not latency measurements) | same |
| Copied v1 offline tests | **42 passed, 2 deselected** (same as Day 12); full suite: 42 passed, 2 skipped (no key) | `03-Evidence/03-v1-offline-tests-log.txt` |
| v1 ingestion in Day 13 | 6 documents, 22 chunks, per policy 5/4/4/3/3/3 — equal to Day 12 run 1 | `03-Evidence/04b-…-strict-check.txt` |
| v1 retrieval, 15 run-1 questions, top-4 | **15/15** identical chunk IDs and order; max \|Δsimilarity\| = 4.99e-05 (saved values are rounded to 4 dp); **60/60** chunk scores exactly equal after rounding to 4 dp | same |
| Refusal gate (similarity ≥ 0.25) | same decision as run 1 for 15/15 (gate passes; run 1 had 0 no-context refusals) | same |
| v1 chunk-index fingerprint | `b00e0446692ffdd8455c337b160963bec0b4064024be0cd3d252f97422cd794d` (identical in both runs) | same; `01-Code/policy-qa-rag-v2/results/phase1-v1-baseline-reproduction.json` |
| Day 12 untouched | Day 12 git working tree: 0 changes after Phase 1 | `05-final-hash-verification-after-phase-1.txt` and command output |

**Not reproduced and not claimed:** the LLM answers of run 1 (need the Groq key; not bit-for-bit deterministic).

## 5. Issues encountered

1. `uv pip compile` platform flag (see §3) — my error; fixed by resolving for the real interpreter.
2. The first reproduction run used only a tolerance check. A one-off strict comparison then reported 117/120 exact,
   which turned out to be **my check's double rounding** (it re-rounded values I had already saved at 6 dp). A
   full-precision recheck gave 120/120 score values (60 chunks × similarity + distance). The script now stores full-precision
   scores and includes the strict check; run 2 = 60/60 chunks exact. Run 1's log is kept unchanged as `04a-…`.
3. After run 2, `ruff --fix`/`ruff format` made cosmetic changes to the reproduction script (import order, line
   wrapping); it was re-run afterwards with the same result (OVERALL PASS, same fingerprint).

## 6. Outstanding

- User: create `.env` from `.env.example` (only needed for answer-quality runs in Phases 6–7).
- Approval required before Phase 2.
