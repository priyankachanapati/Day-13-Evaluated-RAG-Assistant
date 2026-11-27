"""RAG v2 command line (demo).

    .venv/bin/python -m rag_v2.cli retrieve "How many days of sick leave do I get?" --chunking recursive_800_100 --retrieval hybrid --rerank
    .venv/bin/python -m rag_v2.cli ask "How many days of sick leave do I get?" --retrieval hybrid --rerank   # needs GROQ_API_KEY
    .venv/bin/python -m rag_v2.cli configs

Defaults reproduce RAG v1 (v1_paragraph_800_150 + dense, no rerank). Run with HF_HUB_OFFLINE=1 when
--rerank is used and the model is already downloaded.
"""

from __future__ import annotations

import argparse
import sys

from src.config import ConfigurationError, load_settings
from src.embeddings import EmbeddingError
from src.generation import GenerationError

from .chunking import STRATEGIES
from .config_v2 import RETRIEVAL_MODES, RAGv2Config, all_configs
from .pipeline_v2 import RAGPipelineV2, V2Retrieval


def format_retrieval(r: V2Retrieval, threshold: float) -> str:
    sim = dict(r.dense_ranking)
    lines = [
        f"Config: {r.config.name} | context_k {r.config.context_k} | dense refusal threshold {threshold} "
        f"(dense cosine only) | best dense similarity {r.best_dense_similarity:.4f} -> "
        f"{'gate passes' if r.gate_passes else 'REFUSE (no LLM call)'}",
        f"  {'#':<3}{'chunk':<14}{'dense cos':>10}  in context",
    ]
    ctx = {c.chunk_id for c in r.context}
    for i, cid in enumerate(r.final_ranking[: r.config.context_k], start=1):
        flag = "yes" if cid in ctx else "no (dense similarity below threshold)"
        lines.append(f"  {i:<3}{cid:<14}{sim[cid]:>10.4f}  {flag}")
    lines.append("  timings ms: " + ", ".join(f"{k}={v:.1f}" for k, v in r.timings_ms.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Policy Q&A RAG v2")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("configs", help="list all configurations")
    for name in ("retrieve", "ask"):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--chunking", default="v1_paragraph_800_150", choices=sorted(STRATEGIES))
        p.add_argument("--retrieval", default="dense", choices=RETRIEVAL_MODES)
        p.add_argument("--rerank", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "configs":
        print("\n".join(c.name for c in all_configs()))
        return 0
    try:
        settings = load_settings()
        pipe = RAGPipelineV2(settings, RAGv2Config(args.chunking, args.retrieval, args.rerank))
        if args.command == "retrieve":
            print(format_retrieval(pipe.retrieve(args.question), settings.min_similarity))
            return 0
        a = pipe.ask(args.question)
        print(format_retrieval(a.retrieval, settings.min_similarity))
        print(f"\nLLM: {a.llm_model or '(not called)'} | status: {a.status.upper()}")
        print(f"Answer: {a.answer}")
        if a.citations:
            print(
                f"Valid citations: {', '.join(map(str, a.citations.valid)) or 'none'} | invalid: "
                f"{len(a.citations.invalid)} | malformed: {len(a.citations.malformed)}"
            )
        print(f"Tokens: {a.llm_usage} | estimated cost (list price): {a.estimated_cost_usd}")
        print("Timings ms: " + ", ".join(f"{k}={v:.1f}" for k, v in a.timings_ms.items()))
        return 0
    except (ConfigurationError, GenerationError, EmbeddingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
