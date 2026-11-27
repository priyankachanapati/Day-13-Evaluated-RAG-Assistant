"""RAG v2 configuration. Every Phase 2-4 option stays selectable; nothing is hard-coded as the winner.

    chunking   v1_paragraph_800_150 | fixed_500 | recursive_800_100
    retrieval  dense | bm25 | hybrid           (hybrid = RRF of dense top-N and BM25 top-N)
    rerank     False | True                    (cross-encoder re-orders the top-N candidate list)

Fixed (identical to v1 unless stated):
    context_k      4   -> the number of chunks given to the LLM (v1 TOP_K = 4)
    candidates    10   -> candidate depth for dense/BM25/RRF lists and the reranker (Phases 3-4)
    rrf_k         60
    refusal gate       v1 MIN_SIMILARITY (0.25) on DENSE cosine similarity only (from src.config)
    LLM, prompt, temperature, citation validation: the frozen v1 implementation (src/)

``v1_equivalent()`` is the configuration that must behave exactly like RAG v1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .chunking import STRATEGIES

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


@dataclass(frozen=True)
class RAGv2Config:
    chunking: str = "v1_paragraph_800_150"
    retrieval: str = "dense"
    rerank: bool = False
    context_k: int = 4
    candidates: int = 10
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.chunking not in STRATEGIES:
            raise ValueError(f"unknown chunking {self.chunking!r}; choose from {sorted(STRATEGIES)}")
        if self.retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"unknown retrieval {self.retrieval!r}; choose from {RETRIEVAL_MODES}")
        if not 1 <= self.context_k <= self.candidates:
            raise ValueError("need 1 <= context_k <= candidates")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")

    @property
    def name(self) -> str:
        return f"{self.chunking}+{self.retrieval}" + ("+rerank" if self.rerank else "")

    def as_dict(self) -> dict:
        return asdict(self) | {"name": self.name}

    @classmethod
    def from_name(cls, name: str) -> RAGv2Config:
        """Parse ``<chunking>+<retrieval>[+rerank]``."""
        parts = name.split("+")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "rerank"):
            raise ValueError(f"config name must be <chunking>+<retrieval>[+rerank], got {name!r}")
        return cls(chunking=parts[0], retrieval=parts[1], rerank=len(parts) == 3)


def v1_equivalent() -> RAGv2Config:
    return RAGv2Config("v1_paragraph_800_150", "dense", False)


def all_configs() -> list[RAGv2Config]:
    """Every chunking x retrieval x rerank combination (18)."""
    return [RAGv2Config(c, r, rr) for c in STRATEGIES for r in RETRIEVAL_MODES for rr in (False, True)]
