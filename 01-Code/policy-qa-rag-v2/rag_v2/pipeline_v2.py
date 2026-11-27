"""RAG v2 pipeline: configurable chunking + dense / BM25 / hybrid retrieval + optional reranking,
with the frozen v1 generation, citation validation and refusal behaviour.

Flow for one question:

    dense    query embedding (v1 MiniLM) -> cosine similarity for EVERY chunk (Chroma)
    ranking  dense top-N | BM25 top-N | RRF(dense top-N, BM25 top-N) top-N        (N = candidates)
    rerank   optional cross-encoder re-ordering of that top-N list
    context  first context_k (4) chunks of the final ranking, KEEPING ONLY chunks whose DENSE
             similarity >= v1 MIN_SIMILARITY (0.25) - exactly v1's per-chunk relevance rule
    gate     no chunk left (or best dense similarity < 0.25) -> refused_no_context, LLM not called
    generate v1 SYSTEM_PROMPT + build_user_prompt + Groq (same model / temperature) -> validate
             citations against the context chunks -> v1 statuses (answered / refused /
             rejected_invalid_citations / refused_no_context)

BM25, RRF and cross-encoder scores only order candidates; they are never compared with the
0.25 threshold. With ``v1_equivalent()`` the context and the prompt are identical to RAG v1.

Every stage is timed separately (ms). One-time costs (index build, model loading) are recorded
in ``one_time_ms`` and never included in per-query timings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.citations import CitationReport, validate_citations
from src.config import Settings
from src.embeddings import LocalEmbedder
from src.generation import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    GroqGenerator,
    LLMClient,
    build_user_prompt,
    is_refusal,
)
from src.rag_pipeline import ANSWERED, REFUSED, REFUSED_NO_CONTEXT, REJECTED
from src.retrieval import RetrievedChunk

from .bm25 import BM25Index
from .chunking import load_corpus
from .config_v2 import RAGv2Config
from .fusion import rrf_fuse
from .indexing import V2_CHROMA_DIR, StrategyIndex, build_index
from .pricing import estimate_cost_usd
from .reranker import Candidate, CrossEncoderReranker

WITHHELD = (
    "Answer withheld: the generated answer failed citation validation "
    "(missing, invalid or malformed citations), so it cannot be shown as a grounded answer."
)


@dataclass
class V2Retrieval:
    query: str
    config: RAGv2Config
    dense_ranking: list[tuple[str, float]]  # (chunk_id, cosine similarity) for ALL chunks, best first
    bm25_ranking: list[tuple[str, float]]  # (chunk_id, BM25 score), top-N, positive scores only
    fused_ranking: list[tuple[str, float]]  # (chunk_id, RRF score), hybrid only
    reranked: list[tuple[str, float]]  # (chunk_id, cross-encoder logit), rerank only
    final_ranking: list[str]  # candidate list after the configured retrieval (+ rerank)
    context: list[RetrievedChunk]  # chunks given to the LLM
    dropped_by_dense_filter: list[str]  # in the top context_k but dense similarity < threshold
    best_dense_similarity: float | None
    gate_passes: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class V2Answer:
    question: str
    status: str
    answer: str
    raw_llm_answer: str | None
    retrieval: V2Retrieval
    citations: CitationReport | None
    llm_model: str | None
    llm_usage: dict = field(default_factory=dict)  # prompt/completion tokens, cached flag
    estimated_cost_usd: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def is_refusal(self) -> bool:
        return self.status in (REFUSED, REFUSED_NO_CONTEXT)


class RAGPipelineV2:
    def __init__(
        self,
        settings: Settings,
        config: RAGv2Config,
        *,
        llm: LLMClient | None = None,
        embedder: LocalEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        index: StrategyIndex | None = None,
        chroma_dir=V2_CHROMA_DIR,
    ) -> None:
        self.settings = settings
        self.config = config
        self.min_similarity = settings.min_similarity  # v1 threshold (0.25), dense similarity only
        self.one_time_ms: dict[str, float] = {}
        t0 = time.perf_counter()
        self.embedder = embedder or LocalEmbedder()
        if embedder is None:
            self.one_time_ms["embedding model load"] = (time.perf_counter() - t0) * 1000
        if index is None:
            t0 = time.perf_counter()
            index = build_index(
                config.chunking,
                load_corpus(settings.policy_dir),
                self.embedder,
                chroma_dir=chroma_dir,
                min_similarity=self.min_similarity,
            )
            self.one_time_ms["dense index build"] = (time.perf_counter() - t0) * 1000
        if index.strategy.name != config.chunking:
            raise ValueError(f"index is {index.strategy.name}, config wants {config.chunking}")
        self.index = index
        self.chunks = index.chunks_by_id
        self.bm25 = None
        if config.retrieval in ("bm25", "hybrid"):
            t0 = time.perf_counter()
            self.bm25 = BM25Index(index.chunks)
            self.one_time_ms["BM25 index build"] = (time.perf_counter() - t0) * 1000
        self.reranker = reranker
        if config.rerank and reranker is None:
            t0 = time.perf_counter()
            self.reranker = CrossEncoderReranker()
            self.one_time_ms["cross-encoder load"] = (time.perf_counter() - t0) * 1000
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        """Lazily create the real Groq client (raises a clear ConfigurationError without a key)."""
        if self._llm is None:
            self._llm = GroqGenerator(
                api_key=self.settings.require_groq_key(),
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
            )
        return self._llm

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str) -> V2Retrieval:
        if not query or not query.strip():
            raise ValueError("Question must not be empty")
        query = query.strip()
        cfg, timings = self.config, {}
        t0 = time.perf_counter()
        vector = self.embedder.embed_one(query)
        t1 = time.perf_counter()
        matches = self.index.store.query(vector, self.index.store.count())
        t2 = time.perf_counter()
        timings["query_embedding"], timings["vector_search"] = (t1 - t0) * 1000, (t2 - t1) * 1000
        dense = [(m.chunk_id, 1.0 - m.distance) for m in matches]
        dense_sim = dict(dense)
        dense_top = [cid for cid, _ in dense[: cfg.candidates]]

        bm25_ranking, fused = [], []
        if cfg.retrieval == "dense":
            ranking = dense_top
        else:
            t0 = time.perf_counter()
            hits = self.bm25.search(query, cfg.candidates)
            timings["bm25"] = (time.perf_counter() - t0) * 1000
            bm25_ranking = [(h.chunk_id, h.score) for h in hits]
            if cfg.retrieval == "bm25":
                ranking = [cid for cid, _ in bm25_ranking]
            else:
                t0 = time.perf_counter()
                fused_hits = rrf_fuse(
                    {"dense": dense_top, "bm25": [cid for cid, _ in bm25_ranking]},
                    k=cfg.rrf_k,
                    top_n=cfg.candidates,
                )
                timings["fusion"] = (time.perf_counter() - t0) * 1000
                fused = [(h.chunk_id, h.score) for h in fused_hits]
                ranking = [cid for cid, _ in fused]

        reranked = []
        if cfg.rerank:
            t0 = time.perf_counter()
            hits = self.reranker.rerank(
                query,
                [
                    Candidate(cid, self.chunks[cid].embedding_text, i, 0.0)
                    for i, cid in enumerate(ranking, start=1)
                ],
            )
            timings["rerank"] = (time.perf_counter() - t0) * 1000
            reranked = [(h.chunk_id, h.score) for h in hits]
            ranking = [cid for cid, _ in reranked]

        top = ranking[: cfg.context_k]
        kept = [cid for cid in top if dense_sim[cid] >= self.min_similarity]
        context = [self._retrieved(rank, cid, dense_sim[cid]) for rank, cid in enumerate(kept, start=1)]
        best = dense[0][1] if dense else None
        timings["retrieval_total"] = sum(timings.values())
        return V2Retrieval(
            query,
            cfg,
            dense,
            bm25_ranking,
            fused,
            reranked,
            ranking,
            context,
            [cid for cid in top if cid not in kept],
            best,
            bool(context) and best is not None and best >= self.min_similarity,
            timings,
        )

    def _retrieved(self, rank: int, chunk_id: str, similarity: float) -> RetrievedChunk:
        c = self.chunks[chunk_id]
        md = c.metadata
        return RetrievedChunk(
            rank=rank,
            chunk_id=chunk_id,
            policy_id=str(md["policy_id"]),
            title=str(md["title"]),
            section=str(md["section"]),
            source_file=str(md["source_file"]),
            text=c.text,
            distance=1.0 - similarity,
            similarity=similarity,
        )

    # ------------------------------------------------------------------ answering
    def ask(self, question: str) -> V2Answer:
        t_start = time.perf_counter()
        retrieval = self.retrieve(question)
        timings = dict(retrieval.timings_ms)
        if not retrieval.gate_passes:
            timings["generation"] = 0.0
            timings["total"] = (time.perf_counter() - t_start) * 1000
            return V2Answer(
                retrieval.query,
                REFUSED_NO_CONTEXT,
                REFUSAL_MESSAGE,
                None,
                retrieval,
                None,
                None,
                {"llm_called": False},
                0.0,
                timings,
            )
        llm = self.llm
        before = (getattr(llm, "prompt_tokens", None), getattr(llm, "completion_tokens", None))
        t0 = time.perf_counter()
        raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(retrieval.query, retrieval.context))
        timings["generation"] = (time.perf_counter() - t0) * 1000
        usage = dict(getattr(llm, "last_usage", {}) or {})
        if "prompt_tokens" not in usage and before[0] is not None:
            usage = {
                "cached": False,
                "prompt_tokens": llm.prompt_tokens - before[0],
                "completion_tokens": llm.completion_tokens - before[1],
            }
        usage["llm_called"] = True
        cost = (
            estimate_cost_usd(llm.model, usage["prompt_tokens"], usage["completion_tokens"])
            if "prompt_tokens" in usage
            else None
        )
        report = validate_citations(raw, retrieval.context)
        if is_refusal(raw):
            status, shown = REFUSED, REFUSAL_MESSAGE
        elif not report.all_valid:
            status, shown = REJECTED, WITHHELD
        else:
            status, shown = ANSWERED, raw
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return V2Answer(
            retrieval.query, status, shown, raw, retrieval, report, llm.model, usage, cost, timings
        )
