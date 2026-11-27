# Day 13 – Evidence Index

All files are unedited output captured with `tee` from real command runs. Screenshots will be added in Phase 9.

| # | File | What it proves | Phase |
|---|------|----------------|-------|
| 01 | `01-hash-verification-log.txt` | Day 13 v1 code/tests/corpus/reference = Day 12 files = Day 12 git commit b366fc8 (SHA-256) | 1 |
| 02 | `02-environment-setup-log.txt` | uv environment: 109 compatible packages, install == lock, 91/91 Day 12 versions identical, imports OK | 1 |
| 03 | `03-v1-offline-tests-log.txt` | Copied v1 test suite: 42 passed, 2 deselected; full suite 42 passed, 2 skipped (no key) | 1 |
| 04a | `04a-v1-baseline-reproduction-log-run1-tolerance-check.txt` | Run 1 of reproduction (tolerance check only): 15/15 PASS | 1 |
| 04b | `04b-v1-baseline-reproduction-log-run2-strict-check.txt` | Run 2 with strict check: 15/15 questions, 60/60 chunk scores exact at 4 dp | 1 |
| 05 | `05-final-hash-verification-after-phase-1.txt` | Manifests still PASS at the end of Phase 1 (baseline unchanged by test/ingest runs) | 1 |
| 06 | `06-recursive-splitter-langchain-crosscheck.txt` | Our recursive 800/100 splitter is byte-identical to LangChain 1.2.0 `RecursiveCharacterTextSplitter` on 6/6 documents (LangChain in a throwaway venv only) | 2 |
| 07 | `07-chunking-comparison-run1-log.txt` | Activity A full comparison run 1 (integrity gate, stats, boundaries, facts, samples, dev top-3, summary) | 2 |
| 08 | `08-chunking-comparison-run2-reproducibility-log.txt` | Run 2 with indexes rebuilt: identical results SHA-256 `0c6e3c18…`; logs identical except timestamp | 2 |
| 09 | `09-phase-2-regression-checks.txt` | 67 tests passed (42 frozen v1 + 25 new); lint clean; manifests PASS; catalog rebuild PASS; Phase 1 reproduction PASS; Day 12 untouched | 2 |
| 10 | `10-dev01-full-ranking-check.txt` | DEV-01 diagnostic: required sentence at rank 4 (v1, fixed_500) and rank 1 (recursive) in full rankings | 2 |
| 11 | `11-hybrid-retrieval-run1-log.txt` | Activity B run 1: dense vs BM25 vs hybrid (RRF k=60) for 3 strategies × 5 dev queries; integrity gate incl. dense top-3 = Phase 2 (15/15) | 3 |
| 12 | `12-hybrid-retrieval-run2-reproducibility-log.txt` | Activity B run 2 (indexes rebuilt): identical results SHA-256 `db9a570e…`; sections 0–6 identical, latency varies by design | 3 |
| 13 | `13-phase-3-regression-checks.txt` | 109 tests passed (42 frozen v1 + 67 v2); lint clean; v1 + corpus manifests PASS; Phase 2 artifacts 14/14 OK; catalog PASS; Phase 1 reproduction PASS; Activity B run 3 after formatting = same hash; Day 12 untouched | 3 |
| 14 | `14-cross-encoder-download-log.txt` | Model ID alias → canonical `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d2…`; 7 files / 91.8 MB, all sizes match the Hub; one-time download 26.8 s | 4 |
| 15 | `15-reranking-run1-log.txt` | Activity C run 1: hybrid top-10 before vs after cross-encoder for 3 strategies × 5 dev queries; integrity, same-pool (15/15), permutation and no-truncation checks | 4 |
| 16 | `16-reranking-run2-reproducibility-log.txt` | Activity C run 2 (new process): identical results hash `e70b4ff0…`; sections 0–7 identical | 4 |
| 17 | `17-phase-4-regression-checks.txt` | 119 tests passed (incl. real model offline); lint clean; v1/corpus/Phase 2/Phase 3 manifests PASS; catalog PASS; Phase 1 reproduction PASS; run 3 same hash; CE scores bit-identical across processes; no LLM client in Phase 4 code; Day 12 untouched | 4 |
| 18 | `18-eval-dataset-validation-and-freeze.txt` | 30-query dataset validation (all checks PASS) and freeze: v1.0, SHA-256 `c9677c0f…`, 2026-11-26T16:26:53Z; re-freeze is a no-op | 5 |
| 19 | `19-eval-dataset-label-review-sheet.txt` | Every question with its required facts (verbatim policy text), answer terms, reference answer and notes — for reviewer inspection | 5 |
| 20 | `20-activity-d-metrics-demo-log.txt` | Activity D worked example (P@K, R@K, Hit, RR, coverage; macro vs micro) and K=3 vs K=5 on the 5 DEV queries from saved Phase 3/4 rankings; 120/120 identical to saved metrics | 5 |
| 21a | `21a-phase-5-regression-checks-run1-flawed-ev-id-check.txt` | Phase 5 regression run 1: all PASS except check 6, which was a false positive of my regex (`EV-NN` also matched `DEV-NN`) | 5 |
| 21b | `21b-phase-5-regression-checks-run2.txt` | Run 2 with the corrected check: 140 tests passed; lint clean; v1/corpus + Phase 2/3/4 (36 files) manifests PASS; dataset/catalog/dev hashes OK; 0 result files mention a held-out ID; Phase 1 reproduction PASS; Day 12 untouched. (The description line shows `EV-[0-9]{2}` without `\b` because zsh `echo` interprets `\b`; the grep itself used `'\bEV-[0-9]{2}\b'`.) | 5 |
| 22 | `22-dev-answer-eval-mock-all-configs-log.txt` | MOCK LLM dry-run of all 18 configs × (5 dev + 3 refusal probes); stopped by the `--max-llm-calls 20` budget guard (as designed) | 6 |
| 22b | `22b-dev-answer-eval-mock-all-configs-resume-log.txt` | Same run resumed (`--resume`): complete, 144 records, 0 duplicates. MOCK answers are not answer-quality results; context-fact coverage 1.00 for all configs | 6 |
| 23a | `23a-missing-credentials-and-cli-log-ACCIDENTAL-HELD-OUT-QUERY-EV-06.txt` | Runner without credentials → exit 3, nothing written; CLI demo that **by mistake used held-out query EV-06** (kept unedited for transparency; disclosed in `Day-13-Phase-6-Integration-Results.md` §4) | 6 |
| 23b | `23b-cli-demo-dev-query-and-missing-credentials.txt` | CLI demo redone with dev query DEV-02 for 4 configurations (dense gate and per-chunk dense filter visible); `ask` without credentials → clear error | 6 |
| 24 | `24-phase-6-regression-checks.txt` | 185 tests passed; lint clean; v1/corpus + Phase 2–5 + dataset/dev/answer-key/catalog manifests (49 files) PASS; Phase 1 reproduction PASS; v1 equivalence 15/15 identical prompts; held-out guard PASS; secret scan 0; 0 real Groq calls; Day 12 untouched | 6 |
| 25 | `25-eval-dataset-v1_1-validation-and-freeze.txt` | Dataset v1.1 (29 unexposed v1.0 questions + EV-31) validation, all 18 checks PASS, frozen 2026-11-26T17:00:48Z, SHA-256 `e860b729…`; v1.0 unchanged | pre-7 |
| 26 | `26-real-llm-dev-smoke-test-log.txt` | Real-LLM dev smoke test console log: 16/16 items failed with "Groq rejected the API key (authentication error)"; printed `Status: complete` = runner bug | 6 |
| 26b | `26b-real-smoke-1-relabelled-incomplete-auth-failure.txt` | Summary rebuilt from the records with the fixed runner (`--summarize-only`, no API call): status **incomplete: authentication failed**; 0 tokens, USD 0; no key-like strings | 6 |
| 27 | `27-pre-phase-7-regression-checks.txt` | 191 tests passed; lint clean; v1/corpus + 49 original frozen files PASS; v1.0 hash unchanged; v1.1 + exposure record OK; Phase 6 artifacts (regenerated after the runner fix) PASS; Phase 1 reproduction PASS; held-out guard PASS (v1.0 + v1.1); secret scan 0; `.env.example` placeholder restored; Day 12 untouched | pre-7 |
| 28 | `28-phase-6-completion-review-checks.txt` | Review of the completed `real-smoke-1` (resumed 17:13 UTC, 16/16 items, status complete): 191 tests passed; lint clean; 66 frozen files PASS; v1.0 + v1.1 validators PASS; exposure record hashed; only dev/probe queries; secret scan 0; Phase 1 reproduction PASS; Day 12 untouched; heuristic limitation ('7' matched a heading) shown | 6 |
| 29 | `29-phase7-part-a-retrieval-log.txt` | Phase 7 Part A (pre-registered 17:20:54Z, hash 9ef14c50…): retrieval metrics for all 18 configs on dataset v1.1, no LLM | 7 |
| 30 | `30-phase7-part-b-answers-log.txt` | Phase 7 Part B real-LLM answers: stopped by the Groq free-tier daily token limit (200k TPD) after 35 real calls + 4 cache hits; status incomplete, completed answers kept | 7 |
| 31 | `31-phase7-report-PROVISIONAL-quota-incomplete.md` | Generated report from the saved results — PROVISIONAL (answer run incomplete: v1 30/30 items, v1+hybrid+rerank 9/30, recursive+hybrid+rerank 0/30) | 7 |
| 32 | `32-phase8-cost-latency-log.txt` | Phase 8 cost/latency analysis while the answer run was partial (superseded by 35; kept) | 8 |
| 33 | `33-final-checks.txt` | Checks at the end of the 2026-11-26 session: 193 tests, lint, manifests, secret scan 0, Day 12 untouched | 9 |
| 34 | `34-phase7-final-report-log.txt` | Phase 7 report after the user resumed the answer run (complete: 3 × 30, 43 real calls + 7 cache hits in the resume); selection by the pre-registered rule | 7 |
| 35 | `35-phase8-cost-latency-final-log.txt` | Phase 8 with all three configurations complete; per-config rate-limit caveats computed from records | 8 |
| 36 | `36-final-checks.txt` | Final checks: tests, lint, all manifests (incl. append-only check of the partial Phase 7 records), secret scan, Day 12 | 9 |

Generated report for Activity D: `01-Code/policy-qa-rag-v2/activities/d-retrieval-metrics/results/metrics-demo-report.md`.

Generated report for Activity C: `01-Code/policy-qa-rag-v2/activities/c-cross-encoder-reranking/results/reranking-report.md` (from run 3).

Generated report for Activity B: `01-Code/policy-qa-rag-v2/activities/b-hybrid-search-rrf/results/hybrid-retrieval-report.md` (from run 3).

Generated report for Activity A: `01-Code/policy-qa-rag-v2/activities/a-chunking-strategies/results/chunking-comparison-report.md`.

Development-query answer evaluation (mock): `01-Code/policy-qa-rag-v2/results/dev-answer-eval/mock-all-18-configs/summary.md`.
Real-LLM development smoke test: completed — `01-Code/policy-qa-rag-v2/results/dev-answer-eval/real-smoke-1/summary.md` (first attempt's auth failure kept in 26/26b).

Final Phase 7 report: `01-Code/policy-qa-rag-v2/results/phase7/phase7-report.md`; charts: `results/phase7/charts/`; Phase 8: `results/phase8/cost-latency-report.md`.

## Demo screenshots (captured by the user 2026-11-27, inspected)

Captured from this session's display of each command's output (steps follow `02-Documentation/Day-13-Demo-Walkthrough.md`).
Steps 5, 5b and 7 show **saved** results (no re-run, no API call); the others are fresh runs made on 2026-11-27.
File names are as saved (four differ slightly from the walkthrough plan; contents verified).

| File | Shows | Fresh run / saved |
|---|---|---|
| `34-demo-manifest.png` | `verify_manifests.sh` 03:10:10Z — v1 code, corpus and Day 12 reference results identical to Day 12 / commit b366fc8; OVERALL PASS | fresh |
| `35-demo-configs.png` | `rag_v2.cli configs` — all 18 configurations | fresh |
| `36-demo-retreive-v1.png` | v1 retrieval for DEV-02: dense scores, 0.25 per-chunk rule, 22.8 ms | fresh |
| `37-demo-retreive-v2.png` | recursive + hybrid + rerank retrieval for DEV-02 with per-stage timings (cold rerank 156.5 ms; warm median ≈ 72 ms) | fresh |
| `38-demo-phase7-report..png` | Phase 7 answer scores (3 × 30), paired comparison, pre-registered selection | saved (`results/phase7/phase7-report.md`) |
| `39-demo-chart-coverage.png` | Fact coverage@3 chart, 18 configurations | saved (`results/phase7/charts/coverage-at-3.svg`) |
| `40-demo-pytest.png` | 193 passed, 2 deselected | fresh |
| `41-demo-ask.png` | Real-LLM cited answer (EV-16) and correct refusal (EV-27, cached) | saved (`results/phase7/answers.jsonl`) |


## Day 13 Activities 1–4 (screenshots 42–47, captured 2026-11-27)

Manual captures by the project owner of live local runs (no LLM or API calls); each run also saved its text log
(`.txt`, same number). File names are as saved (two contain a double dot).

| File | Activity | Shows | Fresh run / saved |
|---|---|---|---|
| `42-activity1-chunking-top3.png` / `.txt` | 1 – Chunking | fixed 500 vs recursive 800/100: top-3 for 5 questions, Hit@3/MRR@3/P@3/Cov@3, 12/12 checks PASS | fresh |
| `43-activity2-hybrid-rrf-rerank.png` / `.txt` | 2 – BM25 + RRF + rerank | vector vs BM25 vs hybrid (RRF) vs hybrid + cross-encoder rankings and Hit/MRR@3/5, 11/11 checks PASS | fresh |
| `44-activity3-precision-recall-at-5..png` / `44-…at-5.txt` | 3 – P@5 / R@5 | relevant IDs, top-5, intersection, fractions per question; mean P@5 0.320, R@5 1.000, 10/10 checks PASS | fresh |
| `45-activity4-function..png` | 4 – Hit Rate / MRR | the 14-line `hit_rate_and_mrr` function (lines 36–49) | fresh (file listing) |
| `46-activity4-hit-rate-mrr.png` / `.txt` | 4 – Hit Rate / MRR | 10 labelled questions, Hit Rate 1.000 and MRR 0.950 at K=3 and K=5, 5/5 checks PASS | fresh |
| `47-activities-tests.png` / `.txt` | 3 and 4 – tests | 5 new hand-worked tests PASS; full suite 198 passed; ruff clean; manifests OVERALL PASS | fresh |
