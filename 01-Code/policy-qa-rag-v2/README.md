# Policy Q&A RAG v2 — Evaluated and Improved RAG (Day 13 / AI05)

> Status: **Completed - Pending Review** — RAG v2 (18 configurations) built, evaluated on the frozen 30-query set (retrieval: 18 configs; answers: 3 pre-registered configs × 30), cost/latency analysed. Results: `../../04-Submission/Day-13-Final-Project-Report.md`.

This project extends the Day 12 **Policy Q&A RAG v1** (cited RAG over 6 synthetic Brightlake Systems HR/IT policies)
into an evaluated RAG v2: chunking comparison, BM25 + dense hybrid retrieval with Reciprocal Rank Fusion, and
cross-encoder reranking, measured against the v1 baseline on the same corpus and labelled questions.

## Layout

```
policy-qa-rag-v2/
├── src/                 # RAG v1 — FROZEN BASELINE, copied verbatim from Day 12 (commit b366fc8). Never edit.
├── tests/               # v1 test suite, copied verbatim (42 offline tests)
├── data/policies/       # the 6-policy Day 12 corpus, copied verbatim
├── reference/           # Day 12 run-1 smoke results/log (original baseline numbers)
├── manifests/           # SHA-256 manifests of src/, tests/, pytest.ini, corpus and reference files
├── constraints/         # Day 12's exact environment (uv pip freeze), used as resolver constraints
├── scripts/             # verification scripts (manifests, v1 baseline reproduction)
├── results/             # machine-readable results of runs
├── rag_v2/              # v2: chunking, indexes, BM25, RRF, hybrid, reranker, config_v2, pipeline_v2, llm_cache, pricing, cli
├── evaluation/          # evidence catalog, dev queries, FROZEN 30-query eval set, retrieval metrics (Phase 7: harness)
├── tests_v2/            # tests for the new code (tests/ stays the frozen v1 suite)
└── activities/          # Activities A–D (all done)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip sync requirements.txt --python .venv/bin/python   # installs exactly the locked versions
cp .env.example .env                                      # then add your own GROQ_API_KEY (never commit .env)
```

`requirements.in` holds the direct dependencies; `requirements.txt` is the lock compiled with
`uv pip compile requirements.in -c constraints/day12-v1-environment.txt -o requirements.txt`, which keeps every
package shared with v1 at its Day 12 version. The key is only needed for answer generation; retrieval, indexing and
offline tests run without it.

## Verify the frozen baseline

```bash
bash scripts/verify_manifests.sh                              # v1 copy == Day 12 files == Day 12 git b366fc8
.venv/bin/python -m pytest -m "not integration"               # 42 passed, 2 deselected
.venv/bin/python scripts/verify_v1_baseline_reproduction.py   # 15/15 run-1 retrievals reproduced exactly
```

## Run v1 (unchanged CLI)

```bash
.venv/bin/python -m src.rag_pipeline ingest
.venv/bin/python -m src.rag_pipeline retrieve "How many days of annual leave do I get?"
.venv/bin/python -m src.rag_pipeline ask "How many days of annual leave do I get?"   # needs GROQ_API_KEY
```

## Activity A – chunking comparison

```bash
.venv/bin/python activities/a-chunking-strategies/compare_chunking.py   # no key needed
.venv/bin/python -m pytest tests tests_v2 -m "not integration"           # frozen v1 + new tests
```

See `activities/a-chunking-strategies/README.md`.

## Activity B – dense vs BM25 vs hybrid (RRF)

```bash
.venv/bin/python activities/b-hybrid-search-rrf/compare_retrieval.py   # no key needed
```

See `activities/b-hybrid-search-rrf/README.md`.

## Activity C – cross-encoder reranking

```bash
.venv/bin/python scripts/download_cross_encoder.py                                          # once (network, ~92 MB)
HF_HUB_OFFLINE=1 .venv/bin/python activities/c-cross-encoder-reranking/compare_reranking.py   # offline
```

See `activities/c-cross-encoder-reranking/README.md`.

## Activity D – retrieval metrics and the frozen evaluation set

```bash
.venv/bin/python scripts/validate_eval_dataset.py               # validate the frozen 30-query set (no retrieval code)
.venv/bin/python activities/d-retrieval-metrics/metrics_demo.py   # worked example + K=3 vs K=5 on DEV queries only
```

## RAG v2 (Phase 6)

See `../../02-Documentation/Day-13-RAG-v2-Architecture-and-Configuration.md` for the architecture, configuration
guide and all commands (`python -m rag_v2.cli`, `scripts/run_dev_answer_eval.py`).

The 30-query set (`evaluation/datasets/eval_queries_30.json`, v1.0, SHA-256 in `manifests/eval-dataset.sha256`) is
held out for Phase 7; do not use it for tuning. Run the tests with `HF_HUB_OFFLINE=1` too; the real-model
test is skipped (with a reason) if the model has not been downloaded.

## Phase 1 results

See `../../02-Documentation/Day-13-Phase-1-Setup-and-Verification.md`.

## Phase 7–9: evaluation, cost/latency, reports (no API needed except `answers`)

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_phase7_eval.py retrieval          # 18 configs, no LLM
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_phase7_eval.py answers --resume   # real LLM; resumable; quota-safe
.venv/bin/python scripts/run_phase7_eval.py report                               # results/phase7/phase7-report.md
.venv/bin/python scripts/phase7_charts.py                                        # results/phase7/charts/*.svg
.venv/bin/python scripts/phase8_cost_latency.py                                  # results/phase8/cost-latency-report.md
.venv/bin/python scripts/build_final_report.py                                   # ../../04-Submission/Day-13-Final-Project-Report.md
```

Selected by the pre-registered rule: `v1_paragraph_800_150+hybrid+rerank` (tie with v1 on answer score and context
coverage; decided by a latency tie-break that is confounded by rate limits — see the final report). Demo:
`../../02-Documentation/Day-13-Demo-Walkthrough.md`.
