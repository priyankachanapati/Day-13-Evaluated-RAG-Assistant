"""Shared helpers for the four Day 13 activity scripts (activity-1 … activity-4).

Reuses the existing project code only: the Day 12 corpus and v1 retriever (``src/``), the chunking
strategies and indexes (``rag_v2/``) and the canonical evidence catalog (``evaluation/``).
Everything is local: no LLM, no API key, no network.

Relevance rule (same as the whole project, see ``evaluation/metrics.py``): a chunk is relevant to a
question iff it fully contains at least one of the question's required evidence sentences. The
"known relevant chunk IDs" of a question are therefore derived from its frozen fact labels, never
chosen by hand after seeing the retriever output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evidence import chunk_evidence, load_catalog  # noqa: E402
from rag_v2.chunking import STRATEGIES, load_corpus  # noqa: E402
from src.config import load_settings  # noqa: E402

DATASETS = PROJECT_ROOT / "evaluation" / "datasets"
DEV_QUERIES = DATASETS / "dev_queries.json"
EVAL_V11 = DATASETS / "eval_queries_v1_1.json"
PHASE7_RETRIEVAL = PROJECT_ROOT / "results" / "phase7" / "retrieval.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_hash(manifest: str, filename: str) -> str:
    """Hash recorded for ``filename`` in ``manifests/<manifest>``."""
    for line in (PROJECT_ROOT / "manifests" / manifest).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith(filename):
            return parts[0]
    raise KeyError(f"{filename} not in {manifest}")


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def corpus():
    return load_corpus(load_settings().policy_dir)


def chunk_facts(strategy: str, docs) -> dict[str, set[str]]:
    """chunk_id -> evidence IDs fully contained in that chunk, for one chunking strategy."""
    by_policy: dict = {}
    for unit in load_catalog().values():
        by_policy.setdefault(unit.policy_id, []).append(unit)
    return {
        c.chunk_id: set(chunk_evidence(c.metadata, by_policy))
        for c in STRATEGIES[strategy].chunk_documents(docs)
    }


def relevant_ids(facts: dict[str, set[str]], required_evidence: list[list[str]]) -> list[str]:
    """Known relevant chunk IDs: chunks that fully contain at least one required fact (sorted)."""
    required = {e for group in required_evidence for e in group}
    return sorted(cid for cid, f in facts.items() if f & required)


def write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
