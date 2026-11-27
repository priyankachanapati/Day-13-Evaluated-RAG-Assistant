"""BM25 keyword retrieval over one chunking strategy (Activity B), using ``rank_bm25``.

* Indexed text = v1's ``Chunk.embedding_text`` (policy title + chunk text), i.e. exactly the
  text the dense index embeds, so dense and BM25 see the same content.
* Scorer: ``rank_bm25.BM25Okapi`` with its defaults k1=1.5, b=0.75, epsilon=0.25 (not tuned).
  IDF = log((N - n + 0.5) / (n + 0.5)); terms in more than half of the chunks would get a
  negative IDF and are floored at epsilon x mean IDF by the library.
* **BM25 scores are unbounded, non-negative and corpus-dependent.** They are only comparable
  between chunks for the same query and index - never with cosine similarities or RRF scores.
* A chunk that shares no term with the query (score 0) is **not returned**: it was not
  retrieved by keywords and must not receive rank credit in fusion.
* Ties are broken by the chunk's position in the index (deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingestion import Chunk

from .tokenizer import tokenize

BM25_PARAMS = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}


@dataclass(frozen=True)
class BM25Hit:
    rank: int
    chunk_id: str
    score: float  # BM25 score (unbounded; same query + index only)
    matched_terms: tuple[str, ...]  # distinct query tokens present in the chunk


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_tokens = [tokenize(c.embedding_text) for c in chunks]
        self._token_sets = [set(t) for t in self.doc_tokens]
        self.bm25 = BM25Okapi(self.doc_tokens, **BM25_PARAMS)

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        """Top-k chunks with a positive BM25 score, best first."""
        q_tokens = tokenize(query)
        if not q_tokens or top_k <= 0:
            return []
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        hits: list[BM25Hit] = []
        distinct = list(dict.fromkeys(q_tokens))
        for i in order:
            if scores[i] <= 0 or len(hits) == top_k:
                break
            hits.append(
                BM25Hit(
                    rank=len(hits) + 1,
                    chunk_id=self.chunk_ids[i],
                    score=float(scores[i]),
                    matched_terms=tuple(t for t in distinct if t in self._token_sets[i]),
                )
            )
        return hits
