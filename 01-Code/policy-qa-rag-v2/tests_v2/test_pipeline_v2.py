"""RAG v2 pipeline integration tests (mocked LLM; local embeddings; temporary Chroma)."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import groq
import httpx
import pytest

from rag_v2.chunking import STRATEGIES, load_corpus
from rag_v2.config_v2 import RETRIEVAL_MODES, RAGv2Config, all_configs, v1_equivalent
from rag_v2.indexing import build_index
from rag_v2.llm_cache import CachingLLM
from rag_v2.pipeline_v2 import RAGPipelineV2
from rag_v2.pricing import estimate_cost_usd
from rag_v2.reranker import CrossEncoderReranker
from src import generation
from src.config import ConfigurationError, load_settings
from src.embeddings import LocalEmbedder
from src.generation import REFUSAL_MESSAGE, GenerationError, GroqGenerator, QuotaExhaustedError
from src.rag_pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
SMOKE = [
    q["question"] for q in json.loads((ROOT / "tests" / "smoke_questions.json").read_text())["questions"]
]


class FakeLLM:
    """TEST DOUBLE: cites the first context chunk; counts tokens like GroqGenerator."""

    def __init__(self, model="openai/gpt-oss-120b", reply=None):
        self.model, self.reply, self.calls = model, reply, []
        self.prompt_tokens = self.completion_tokens = 0

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        self.prompt_tokens += 1000
        self.completion_tokens += 50
        if self.reply is not None:
            return self.reply
        m = re.search(r"policy_id: (\S+) \| chunk_id: (\S+)", user_prompt)
        return f"Here is the answer. [Source: {m.group(1)}, Chunk: {m.group(2)}]"


def stub_reranker():
    # deterministic: shorter passages first (enough to test the contract without the model)
    return CrossEncoderReranker(lambda pairs: [-float(len(p)) for _, p in pairs])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    settings = dataclasses.replace(load_settings(), groq_api_key=None)
    embedder = LocalEmbedder()
    chroma = tmp_path_factory.mktemp("chroma")
    docs = load_corpus(settings.policy_dir)
    indexes = {n: build_index(n, docs, embedder, chroma_dir=chroma) for n in STRATEGIES}
    return settings, embedder, indexes, chroma


def make(env, cfg, llm=None, settings=None, reranker=None):
    s, embedder, indexes, _ = env
    return RAGPipelineV2(
        settings or s,
        cfg,
        llm=llm or FakeLLM(),
        embedder=embedder,
        index=indexes[cfg.chunking],
        reranker=reranker or (stub_reranker() if cfg.rerank else None),
    )


# ------------------------------------------------------------------ configuration
def test_config_names_validation_and_grid():
    assert v1_equivalent().name == "v1_paragraph_800_150+dense"
    assert RAGv2Config.from_name("fixed_500+hybrid+rerank") == RAGv2Config("fixed_500", "hybrid", True)
    assert len({c.name for c in all_configs()}) == 18
    for bad in ("x+dense", "fixed_500+magic", "fixed_500+dense+boost"):
        with pytest.raises(ValueError):
            RAGv2Config.from_name(bad)
    with pytest.raises(ValueError):
        RAGv2Config(context_k=11)


# ------------------------------------------------------------------ every mode x chunking x rerank
@pytest.mark.parametrize("cfg", all_configs(), ids=lambda c: c.name)
def test_every_configuration_end_to_end(env, cfg):
    llm = FakeLLM()
    pipe = make(env, cfg, llm)
    a = pipe.ask("What is the hotel limit per night on business travel?")
    r = a.retrieval
    ids = set(pipe.chunks)
    assert set(r.final_ranking) <= ids and len(r.final_ranking) == len(set(r.final_ranking)) <= cfg.candidates
    assert 1 <= len(r.context) <= 4 and all(c.similarity >= 0.25 for c in r.context)
    assert [c.chunk_id for c in r.context] == [
        x for x in r.final_ranking[:4] if x not in r.dropped_by_dense_filter
    ]
    expected_stages = {"query_embedding", "vector_search", "retrieval_total"}
    expected_stages |= {"bm25"} if cfg.retrieval in ("bm25", "hybrid") else set()
    expected_stages |= {"fusion"} if cfg.retrieval == "hybrid" else set()
    expected_stages |= {"rerank"} if cfg.rerank else set()
    assert set(r.timings_ms) == expected_stages
    assert (r.reranked != []) == cfg.rerank and (r.fused_ranking != []) == (cfg.retrieval == "hybrid")
    if cfg.rerank:
        pre = r.fused_ranking or r.bm25_ranking or r.dense_ranking[: cfg.candidates]
        assert sorted(x for x, _ in r.reranked) == sorted(x for x, _ in pre)  # permutation, no new chunks
    assert a.status == "answered" and a.citations.all_valid and len(llm.calls) == 1
    assert all(c.chunk_id in llm.calls[0][1] for c in r.context)  # every context chunk is in the prompt


def test_retrieval_modes_differ_as_expected(env):
    q = "What is KeyVault?"
    rankings = {
        m: make(env, RAGv2Config("v1_paragraph_800_150", m)).retrieve(q).final_ranking
        for m in RETRIEVAL_MODES
    }
    dense = make(env, RAGv2Config("v1_paragraph_800_150", "dense")).retrieve(q)
    assert rankings["dense"] == [c for c, _ in dense.dense_ranking[:10]]
    assert rankings["bm25"][0].startswith("IT-001")  # keyword found by BM25


def test_rerank_disabled_vs_enabled_changes_only_the_order(env):
    q = "Can I use public Wi-Fi when working from home?"
    off = make(env, RAGv2Config("recursive_800_100", "hybrid", False)).retrieve(q)
    on = make(env, RAGv2Config("recursive_800_100", "hybrid", True)).retrieve(q)
    assert sorted(off.final_ranking) == sorted(on.final_ranking)
    assert on.final_ranking == [c for c, _ in on.reranked]


# ------------------------------------------------------------------ v1 equivalence
def test_v1_equivalent_sends_identical_prompts_to_v1(env, tmp_path):
    s, embedder, indexes, _ = env
    v1_llm = FakeLLM(reply=REFUSAL_MESSAGE)
    v1 = RAGPipeline(dataclasses.replace(s, chroma_dir=tmp_path / "v1"), llm=v1_llm)
    v1.ingest()
    v2_llm = FakeLLM(reply=REFUSAL_MESSAGE)
    v2 = make(env, v1_equivalent(), v2_llm)
    for q in SMOKE:
        assert v1.ask(q).status == v2.ask(q).status
    assert v1_llm.calls == v2_llm.calls and len(v1_llm.calls) == 15


# ------------------------------------------------------------------ refusal semantics (dense only)
def test_dense_gate_refuses_without_llm_even_when_bm25_and_rerank_find_chunks(env):
    strict = dataclasses.replace(env[0], min_similarity=0.99)
    llm = FakeLLM()
    a = make(env, RAGv2Config("fixed_500", "hybrid", True), llm, settings=strict).ask("What is KeyVault?")
    assert a.retrieval.bm25_ranking and a.retrieval.reranked  # keyword methods did find candidates
    assert a.status == "refused_no_context" and a.answer == REFUSAL_MESSAGE and llm.calls == []
    assert a.llm_usage == {"llm_called": False} and a.estimated_cost_usd == 0.0


def test_per_chunk_dense_filter_drops_low_similarity_context(env):
    mid = dataclasses.replace(env[0], min_similarity=0.40)
    a = make(env, RAGv2Config("v1_paragraph_800_150", "bm25"), settings=mid).ask("What is KeyVault?")
    sim = dict(a.retrieval.dense_ranking)
    assert all(sim[c] < 0.40 for c in a.retrieval.dropped_by_dense_filter)
    assert all(c.similarity >= 0.40 for c in a.retrieval.context)


def test_llm_refusal_and_invalid_citations(env):
    refused = make(env, v1_equivalent(), FakeLLM(reply=REFUSAL_MESSAGE)).ask("Does the company insure pets?")
    assert refused.status == "refused" and refused.is_refusal and not refused.citations.has_citations
    fake = make(env, v1_equivalent(), FakeLLM(reply="20 days. [Source: HR-001, Chunk: HR-001-CH99]"))
    a = fake.ask("How many days of annual leave do I get?")
    assert a.status == "rejected_invalid_citations" and a.citations.invalid and "withheld" in a.answer


# ------------------------------------------------------------------ credentials and API failures
def test_missing_api_key_gives_clear_error_without_secret(env):
    s, embedder, indexes, _ = env
    pipe = RAGPipelineV2(s, v1_equivalent(), embedder=embedder, index=indexes["v1_paragraph_800_150"])
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY is not set"):
        pipe.ask("How many days of annual leave do I get?")
    assert pipe.retrieve("How many days of annual leave do I get?").context  # retrieval works without a key


class FailingLLM(FakeLLM):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def complete(self, system_prompt, user_prompt):
        raise self.exc


@pytest.mark.parametrize("exc", [GenerationError("Groq API unavailable"), QuotaExhaustedError("quota")])
def test_api_failures_propagate(env, exc):
    with pytest.raises(type(exc)):
        make(env, v1_equivalent(), FailingLLM(exc)).ask("How many days of annual leave do I get?")


def _rate_limit(retry_after):
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(
        429, headers={"retry-after": retry_after}, json={"error": {"message": "rate limited"}}, request=req
    )
    return groq.RateLimitError("rate limited", response=resp, body=None)


def test_retry_after_rate_limit_through_the_v2_pipeline(env, monkeypatch):
    gen = GroqGenerator(api_key="test-key-not-real", model="openai/gpt-oss-120b")
    outcomes = [_rate_limit("1"), "Answer. [Source: HR-001, Chunk: HR-001-CH02]"]

    def once(s, u):
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        gen.calls += 1
        return item

    slept = []
    monkeypatch.setattr(gen, "_complete_once", once)
    monkeypatch.setattr(generation.time, "sleep", slept.append)
    a = make(env, v1_equivalent(), gen).ask(
        "How many days of paid annual leave do full-time employees receive?"
    )
    assert slept == [2.0] and a.status == "answered"  # waited retry-after + 1 s, then succeeded


# ------------------------------------------------------------------ instrumentation
def test_tokens_cost_and_latency_recorded(env):
    a = make(env, RAGv2Config("v1_paragraph_800_150", "hybrid", True)).ask(
        "What is the hotel limit per night?"
    )
    assert a.llm_usage == {
        "cached": False,
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "llm_called": True,
    }
    assert a.estimated_cost_usd == pytest.approx(estimate_cost_usd("openai/gpt-oss-120b", 1000, 50))
    assert a.estimated_cost_usd == pytest.approx((1000 * 0.15 + 50 * 0.60) / 1e6)
    t = a.timings_ms
    assert all(v >= 0 for v in t.values())
    assert t["total"] >= t["retrieval_total"] + t["generation"] - 1e-6
    assert estimate_cost_usd("unknown-model", 1, 1) is None


def test_one_time_costs_are_kept_out_of_per_query_timings(env):
    s, embedder, _, chroma = env
    pipe = RAGPipelineV2(
        s,
        RAGv2Config("fixed_500", "hybrid", True),
        llm=FakeLLM(),
        embedder=embedder,
        reranker=stub_reranker(),
        chroma_dir=chroma,
    )
    assert {"dense index build", "BM25 index build"} <= set(pipe.one_time_ms)
    assert not set(pipe.one_time_ms) & set(pipe.ask("What is KeyVault?").timings_ms)


def test_caching_llm_avoids_repeated_calls(env, tmp_path):
    inner = FakeLLM()
    llm = CachingLLM(inner, tmp_path / "cache")
    pipe = make(env, v1_equivalent(), llm)
    first = pipe.ask("What is KeyVault?")
    second = pipe.ask("What is KeyVault?")
    assert len(inner.calls) == 1 and llm.real_calls == 1 and llm.cache_hits == 1
    assert first.raw_llm_answer == second.raw_llm_answer
    assert first.llm_usage["cached"] is False and second.llm_usage["cached"] is True
    assert second.llm_usage["prompt_tokens"] == 1000  # tokens of the original call, for cost accounting


def test_real_cross_encoder_in_pipeline_offline(env):
    try:
        rr = CrossEncoderReranker()
    except Exception as exc:
        pytest.skip(f"cross-encoder not cached ({exc.__class__.__name__})")
    a = make(env, RAGv2Config("v1_paragraph_800_150", "hybrid", True), reranker=rr).ask(
        "How far in advance do I have to request a two-week holiday?"
    )
    assert a.retrieval.reranked and a.status == "answered"
