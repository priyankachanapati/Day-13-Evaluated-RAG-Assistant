"""Cross-encoder reranking of a candidate list (Activity C).

A **bi-encoder** (the dense retriever) embeds query and chunk *separately* and compares the
vectors - fast, so it can search the whole index. A **cross-encoder** reads the query and one
chunk *together* in a single transformer pass and outputs a relevance score - more accurate,
because every query token can attend to every chunk token, but it must run once per
(query, chunk) pair. So it is only applied to a short candidate list (here: the hybrid top 10).

Contract of ``CrossEncoderReranker.rerank``:
* input: the query and an ordered candidate list (chunk IDs + texts); output: the SAME chunks,
  re-ordered by cross-encoder score. No chunk is added, removed or duplicated.
* pair format: ``(query, title + chunk text)`` - the same text the dense and BM25 sides use.
* score = the model's raw relevance logit (``activation_fn`` = Identity): higher = more relevant.
  It is **only used to order candidates**. It is not a probability, is not comparable with
  cosine, BM25 or RRF scores, and does **not** replace the v1 dense-similarity refusal gate.
* ties: equal scores keep the incoming (hybrid) order -> deterministic.
* an empty candidate list returns an empty list without calling the model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

REQUESTED_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ID in the project plan (Hub alias)
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"  # canonical repository the alias resolves to
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"  # pinned Hub commit
ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "README.md",
]
MAX_LENGTH = 512  # model limit (max_position_embeddings); pairs are checked not to exceed it

Scorer = Callable[[list[tuple[str, str]]], Sequence[float]]


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    text: str  # text given to the cross-encoder (title + chunk)
    prior_rank: int  # rank in the incoming (hybrid) list
    prior_score: float  # incoming score (RRF) - kept for display, never mixed with CE scores


@dataclass(frozen=True)
class RerankedHit:
    rank: int
    chunk_id: str
    score: float  # cross-encoder logit (ordering only)
    prior_rank: int
    prior_score: float


def format_pairs(query: str, candidates: Sequence[Candidate]) -> list[tuple[str, str]]:
    """Cross-encoder input: one (query, passage) pair per candidate, in candidate order."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    return [(query.strip(), c.text) for c in candidates]


class CrossEncoderReranker:
    def __init__(
        self, scorer: Scorer | None = None, *, device: str = "cpu", local_files_only: bool = True
    ) -> None:
        """``scorer`` can be injected (tests). Otherwise the pinned model is loaded from the local
        cache only (run ``scripts/download_cross_encoder.py`` once first); CPU for determinism."""
        self.model = None
        if scorer is None:
            from sentence_transformers import CrossEncoder  # heavy import, only when needed

            self.model = CrossEncoder(
                MODEL_ID,
                revision=MODEL_REVISION,
                device=device,
                max_length=MAX_LENGTH,
                local_files_only=local_files_only,
            )
            model = self.model

            def scorer(pairs: list[tuple[str, str]]) -> Sequence[float]:
                return model.predict(
                    pairs, batch_size=max(1, len(pairs)), show_progress_bar=False, convert_to_numpy=True
                )

        self._scorer = scorer

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[RerankedHit]:
        if not candidates:
            return []
        ids = [c.chunk_id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate list contains duplicate chunk IDs")
        scores = [float(s) for s in self._scorer(format_pairs(query, candidates))]
        if len(scores) != len(candidates):
            raise RuntimeError("scorer returned a different number of scores than candidates")
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [
            RerankedHit(
                rank, candidates[i].chunk_id, scores[i], candidates[i].prior_rank, candidates[i].prior_score
            )
            for rank, i in enumerate(order, start=1)
        ]
