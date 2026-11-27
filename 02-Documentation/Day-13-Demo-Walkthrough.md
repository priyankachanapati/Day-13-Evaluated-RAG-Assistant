# Day 13 – Demo Walkthrough and Screenshot Plan

Run from `Day-13/01-Code/policy-qa-rag-v2/` with the virtual environment set up (README). Steps 1–6 need no API key
and make no LLM call. Use only development queries (DEV-01…05) in live demos; never the held-out EV-xx questions.

| # | Show | Command | What to point out | Screenshot file (to capture) |
|---|---|---|---|---|
| 1 | Frozen baseline is intact | `bash scripts/verify_manifests.sh` | v1 code, tests and corpus = Day 12 git commit b366fc8 | `34-demo-manifests.png` |
| 2 | All configurations | `.venv/bin/python -m rag_v2.cli configs` | 18 = 3 chunkings × dense/bm25/hybrid × rerank | `35-demo-configs.png` |
| 3 | v1 retrieval | `HF_HUB_OFFLINE=1 .venv/bin/python -m rag_v2.cli retrieve "What is KeyVault?"` | dense similarities; v1 dense filter (0.25) marks chunks kept / dropped | `36-demo-retrieve-v1.png` |
| 4 | v2 retrieval | `HF_HUB_OFFLINE=1 .venv/bin/python -m rag_v2.cli retrieve "What is KeyVault?" --chunking recursive_800_100 --retrieval hybrid --rerank` | BM25, RRF and rerank timings per stage; refusal still on dense similarity | `37-demo-retrieve-v2.png` |
| 5 | Evaluation results | open `results/phase7/phase7-report.md` and `results/phase7/charts/*.svg` | 18-config retrieval table; COMPLETE vs PARTIAL answer rows; "selection rule cannot be applied" | `38-demo-phase7-report.png`, `39-demo-chart-coverage.png` |
| 6 | Tests | `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests tests_v2 -m "not integration" -q` | 193 passed | `40-demo-pytest.png` |
| 7 | Cited answer (optional, uses 1–2 LLM calls) | `HF_HUB_OFFLINE=1 .venv/bin/python -m rag_v2.cli ask "How far in advance do I have to request a two-week holiday?" --retrieval hybrid --rerank` | answer with `[Source: …, Chunk: …]`, citation validation, tokens, cost estimate, timings | `41-demo-ask.png` |

Recorded (unedited) outputs of steps 2–4 already exist as evidence: `03-Evidence/23b-cli-demo-dev-query-and-missing-credentials.txt`.
Screenshots must be genuine captures of these commands; until they exist they are marked `[EVIDENCE PENDING]` in the
evidence index.
