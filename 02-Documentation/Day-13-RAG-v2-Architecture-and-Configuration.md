# Day 13 – RAG v2: Architecture, Configuration and Setup

**Status:** Completed - Pending Review (real-LLM smoke test blocked on credentials) · **Date:** 2026-11-26 · **Phase:** 6 of 9

## 1. Architecture

```
question
   │
   ▼
query embedding (all-MiniLM-L6-v2 ONNX, v1 LocalEmbedder) ──► cosine similarity for EVERY chunk (Chroma, v1 store)
   │                                                               │
   │           ┌───────────────────────────────────────────────────┤ dense top-10
   ▼           ▼                                                   │
 BM25 top-10 (rank_bm25, deliberate tokenizer)        ─►  RRF k=60 (hybrid only)
   │                                                               │
   └──────────────► candidate list (dense | bm25 | hybrid) top-10 ◄┘
                              │
                   optional cross-encoder rerank (ms-marco-MiniLM-L6-v2, pinned)
                              │
                     first 4 chunks of the final ranking
                              │
         v1 per-chunk rule: keep only chunks with DENSE similarity ≥ 0.25
                              │
        none left / best dense similarity < 0.25 ──► refused_no_context (LLM not called)
                              │
   v1 SYSTEM_PROMPT + build_user_prompt ─► Groq openai/gpt-oss-120b, temperature 0 (v1 GroqGenerator)
                              │
   v1 citation validation against the context chunks ─► answered | refused | rejected_invalid_citations
```

| Module | Role | Reuses |
|---|---|---|
| `rag_v2/config_v2.py` | `RAGv2Config` (chunking, retrieval, rerank, context_k, candidates, rrf_k); `v1_equivalent()`; `all_configs()` (18) | Phase 2 strategy registry |
| `rag_v2/pipeline_v2.py` | `RAGPipelineV2.retrieve()` / `.ask()`, per-stage timings, tokens, cost estimate | v1 embedder, store, prompt, generator, citation validator, statuses; Phase 2 indexes; Phase 3 BM25/RRF; Phase 4 reranker — all unchanged |
| `rag_v2/llm_cache.py` | response cache keyed by SHA-256 of (model, temperature, max_tokens, prompts) | — |
| `rag_v2/pricing.py` | published list price for cost **estimates** (source + date) | — |
| `rag_v2/cli.py` | `configs`, `retrieve`, `ask` | — |
| `evaluation/answer_eval.py` | deterministic answer checks | v1 citation regexes |
| `scripts/run_dev_answer_eval.py` | dev-query answer evaluation runner (incremental, resumable, cached, budgeted) | — |

## 2. Semantics preserved from v1

| Property | v1 | v2 |
|---|---|---|
| Model / temperature / max tokens | `openai/gpt-oss-120b`, 0.0, 2048 | same (v1 `GroqGenerator`, settings from `src.config`) |
| System prompt, user prompt format | `SYSTEM_PROMPT`, `build_user_prompt` | same functions, unchanged |
| Context size | 4 (TOP_K) | `context_k = 4` |
| Refusal gate | no chunk with dense similarity ≥ 0.25 → refuse without LLM | same rule, **dense similarity only**, every mode |
| Per-chunk relevance | only chunks with dense similarity ≥ 0.25 are sent | same, applied to the final top-4 of any mode; dropped chunks are recorded (`dropped_by_dense_filter`) |
| Citation validation, statuses | v1 validator; answered / refused / refused_no_context / rejected | same |
| BM25 / RRF / cross-encoder scores | — | ordering only; never compared with 0.25 |

**Equivalence check:** `v1_paragraph_800_150 + dense` (no rerank) sends **byte-identical system and user prompts**
to the frozen v1 `RAGPipeline` and gives the same status for all 15 Day 12 smoke questions (unit test).

**Documented difference (by design, not a change to v1):** in BM25/hybrid/rerank modes a chunk ranked in the top 4 by
keywords or by the cross-encoder is still excluded if its dense similarity is below 0.25 — that is v1's per-chunk
rule applied faithfully. On the development queries this never removed a chunk holding a required fact (0 cases);
the Phase 7 runs report the count so its effect on the held-out set is measured, not assumed.

## 3. Configuration guide

| Option | Values | Default (= v1) |
|---|---|---|
| `chunking` | `v1_paragraph_800_150`, `fixed_500`, `recursive_800_100` | `v1_paragraph_800_150` |
| `retrieval` | `dense`, `bm25`, `hybrid` | `dense` |
| `rerank` | off / on (cross-encoder over the top-10 candidates of the chosen retrieval) | off |
| `context_k` | 1…candidates | 4 |
| `candidates` | ≥ context_k | 10 |
| `rrf_k` | ≥ 0 | 60 |
| `.env` | `GROQ_API_KEY` (required for `ask` only); optional v1 overrides (`LLM_MODEL`, `MIN_SIMILARITY`, …) — keep the defaults for comparisons | — |

Configuration names: `<chunking>+<retrieval>[+rerank]`, e.g. `recursive_800_100+hybrid+rerank`. No configuration is
marked as the winner; the choice is made in Phase 7 on the frozen 30-query set.

## 4. Setup and execution

```bash
cd Day-13/01-Code/policy-qa-rag-v2
uv venv --python 3.12 .venv && uv pip sync requirements.txt --python .venv/bin/python
.venv/bin/python scripts/download_cross_encoder.py     # once, only needed for +rerank
cp .env.example .env                                   # then add your own GROQ_API_KEY (never commit .env)

# retrieval only (no key)
HF_HUB_OFFLINE=1 .venv/bin/python -m rag_v2.cli configs
HF_HUB_OFFLINE=1 .venv/bin/python -m rag_v2.cli retrieve "What is KeyVault?" --retrieval hybrid --rerank

# answer (needs GROQ_API_KEY)
HF_HUB_OFFLINE=1 .venv/bin/python -m rag_v2.cli ask "What is KeyVault?" --chunking recursive_800_100 --retrieval hybrid --rerank

# development-query answer evaluation
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_dev_answer_eval.py --run-name mock-1 --mock       # no API
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_dev_answer_eval.py --run-name real-smoke-1        # real Groq (dev only)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_dev_answer_eval.py --run-name real-smoke-1 --resume

# tests
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests tests_v2 -m "not integration"
```

Runner defaults: configurations `v1_paragraph_800_150+dense` (the v1 baseline) and
`v1_paragraph_800_150+hybrid+rerank`; 5 dev queries + 3 refusal probes; `--max-llm-calls 20`; `--delay 8` s after each
real call (free-tier tokens/minute). Results: `results/dev-answer-eval/<run>/{records.jsonl, summary.json,
summary.md}`; cache: `results/llm-cache/`. Exit codes: 0 complete, 1 integrity failure, 2 incomplete (quota or call
budget — completed records are kept), 3 missing credentials.

## 5. Secrets

The key lives only in `.env` (git-ignored, verified in Phase 1); `src.config` reads it; it is never printed, logged,
cached or written to results (the cache stores prompts, responses and token counts). Provider error messages are
sanitised by the v1 generator (`org_[redacted]`). A missing key gives `GROQ_API_KEY is not set…` and exit code 3.
