"""Per-strategy vector indexes (Chroma), built with the frozen v1 components.

Each chunking strategy gets its own collection in a separate persistent directory
(``.chroma-v2/``), so the Phase 1 v1 index in ``.chroma/`` is never touched. Embedding model,
embedded text (title + chunk text), distance metric and retriever are exactly v1's; only the
chunking differs between collections.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from src.embeddings import LocalEmbedder
from src.ingestion import Chunk
from src.retrieval import Retriever
from src.vector_store import PolicyVectorStore

from .chunking import STRATEGIES, ChunkingStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_CHROMA_DIR = PROJECT_ROOT / ".chroma-v2"


def collection_name(strategy: str) -> str:
    return f"d13_{strategy}"


def chunk_fingerprint(chunks: list[Chunk]) -> str:
    """SHA-256 over chunk IDs and texts, in order (identifies one exact chunking of the corpus)."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(json.dumps([c.chunk_id, c.text], ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


@dataclass
class StrategyIndex:
    strategy: ChunkingStrategy
    chunks: list[Chunk]
    store: PolicyVectorStore
    retriever: Retriever

    @property
    def chunks_by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}


def build_index(
    strategy_name: str,
    docs,
    embedder: LocalEmbedder,
    chroma_dir: Path = V2_CHROMA_DIR,
    min_similarity: float = 0.25,
) -> StrategyIndex:
    """Chunk, embed and (re)store one strategy's collection; return it with a v1 Retriever."""
    strategy = STRATEGIES[strategy_name]
    chunks = strategy.chunk_documents(docs)
    store = PolicyVectorStore(chroma_dir, collection_name(strategy_name))
    store.reset()  # rebuild from scratch: never mixes runs
    store.add_chunks(chunks, embedder.embed([c.embedding_text for c in chunks]))
    if store.count() != len(chunks):
        raise RuntimeError(f"{strategy_name}: stored {store.count()} of {len(chunks)} chunks")
    retriever = Retriever(embedder, store, top_k=3, min_similarity=min_similarity)
    return StrategyIndex(strategy, chunks, store, retriever)
