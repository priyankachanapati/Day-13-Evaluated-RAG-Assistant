"""Phase 1: verify that the Day 13 copy of RAG v1 reproduces Day 12's original retrieval results.

Uses ONLY the frozen v1 code in ``src/`` (no v2 code) and compares against the original Day 12 run-1
results copied verbatim into ``reference/`` (hashes in manifests/day12-reference-results.sha256).

Checks
  1. Ingestion: document count, chunk count and chunks per policy equal the Day 12 run-1 log.
  2. Retrieval: for each of the 15 run-1 questions, top-k chunk IDs (and their order) are identical,
     and similarity/distance match the saved values (saved values are rounded to 4 decimals, so the
     tolerance is half a unit in the 4th decimal plus float noise). Strict check: our full-precision
     scores, rounded to 4 decimals, must equal the saved values exactly.
  3. Refusal gate: the v1 dense-similarity gate (best similarity >= min_similarity) takes the same
     decision as in run 1 (run 1 had no ``refused_no_context`` result, so the gate must pass for all 15).

Not reproduced here (and not claimed): the LLM answers. They need the Groq key and an LLM call, and
are not bit-for-bit deterministic even at temperature 0.

No API key, no network calls. Writes results/phase1-v1-baseline-reproduction.json.
Usage (project root):  .venv/bin/python scripts/verify_v1_baseline_reproduction.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.ingestion import chunk_documents, load_documents  # noqa: E402
from src.rag_pipeline import REFUSED_NO_CONTEXT, RAGPipeline  # noqa: E402
from src.smoke_eval import load_questions  # noqa: E402

REFERENCE_JSON = PROJECT_ROOT / "reference" / "day12-smoke-results-run1.json"
REFERENCE_LOG = PROJECT_ROOT / "reference" / "day12-smoke-test-log-run1.txt"
OUTPUT_JSON = PROJECT_ROOT / "results" / "phase1-v1-baseline-reproduction.json"
TOLERANCE = 0.5e-4 + 1e-6  # half a unit in the 4th decimal (saved rounding) + float noise


def reference_ingestion(log_text: str) -> dict:
    """Parse the ingestion summary printed in the Day 12 run-1 log."""
    docs = int(re.search(r"Source documents : (\d+)", log_text).group(1))
    chunks = int(re.search(r"Chunks generated : (\d+)", log_text).group(1))
    per_policy = {
        pid: int(n) for pid, n in re.findall(r"^\s+([A-Z]{2}-\d{3}): (\d+) chunks$", log_text, re.M)
    }
    return {"documents": docs, "chunks": chunks, "chunks_per_policy": per_policy}


def chunk_fingerprint(chunks) -> str:
    """SHA-256 over every chunk's ID, metadata and text, in order: identifies this exact v1 index."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(json.dumps([c.chunk_id, c.text, c.metadata], sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    started = datetime.now(UTC)
    settings = load_settings()
    reference = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    ref_ingest = reference_ingestion(REFERENCE_LOG.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool, str]] = []

    print("=" * 78)
    print("PHASE 1 - RAG v1 BASELINE REPRODUCTION (Day 13 copy vs Day 12 run 1)")
    print("=" * 78)
    print(f"Run started (UTC) : {started.isoformat(timespec='seconds')}")
    print(
        f"Python            : {platform.python_version()} | chromadb {metadata.version('chromadb')} | "
        f"onnxruntime {metadata.version('onnxruntime')} | numpy {metadata.version('numpy')}"
    )
    print(
        f"Reference         : {REFERENCE_JSON.relative_to(PROJECT_ROOT)} (Day 12 run 1, "
        f"started {reference['started_utc']})"
    )
    print(f"Chroma directory  : {settings.chroma_dir.relative_to(PROJECT_ROOT)} (Day 13's own, not Day 12's)")
    print()

    # --- Configuration must match the reference run ---
    cfg_ok = settings.top_k == reference["top_k"] and settings.min_similarity == reference["min_similarity"]
    checks.append(
        (
            "config top_k / min_similarity equal run 1",
            cfg_ok,
            f"top_k={settings.top_k} (ref {reference['top_k']}), min_similarity={settings.min_similarity} "
            f"(ref {reference['min_similarity']}), chunk_size={settings.chunk_size}, "
            f"overlap={settings.chunk_overlap}",
        )
    )

    # --- Questions must be the same as run 1 ---
    questions = {q.id: q.question for q in load_questions()}
    same_questions = [(r["id"], r["question"]) for r in reference["results"]] == list(questions.items())
    checks.append(("15 smoke questions identical to run 1", same_questions, f"{len(questions)} questions"))

    # --- 1. Ingestion ---
    pipeline = RAGPipeline(settings)  # no LLM is created: retrieval only
    summary = pipeline.ingest()
    print(summary.format())
    print()
    ingest_ok = (
        summary.document_count == ref_ingest["documents"]
        and summary.chunk_count == ref_ingest["chunks"]
        and summary.chunks_per_document == ref_ingest["chunks_per_policy"]
        and summary.stored_count == ref_ingest["chunks"]
    )
    checks.append(
        (
            "ingestion: documents / chunks / chunks per policy",
            ingest_ok,
            f"{summary.document_count} docs, {summary.chunk_count} chunks (ref {ref_ingest['documents']} "
            f"docs, {ref_ingest['chunks']} chunks); per policy {summary.chunks_per_document}",
        )
    )
    fingerprint = chunk_fingerprint(
        chunk_documents(load_documents(settings.policy_dir), settings.chunk_size, settings.chunk_overlap)
    )

    # --- 2 + 3. Retrieval and refusal gate ---
    rows = []
    max_sim_diff = max_dist_diff = 0.0
    print(
        f"{'ID':<7}{'IDs+order':<11}{'max|Δsim|':>11}{'max|Δdist|':>12}  {'gate':<6} top-{settings.top_k} chunk IDs"
    )
    for ref in reference["results"]:
        result = pipeline.retriever.retrieve(ref["question"])
        ours = [(c.chunk_id, c.policy_id, c.similarity, c.distance) for c in result.chunks]
        theirs = [(r["chunk_id"], r["policy_id"], r["similarity"], r["distance"]) for r in ref["retrieved"]]
        ids_equal = [o[:2] for o in ours] == [t[:2] for t in theirs]
        dsim = max((abs(o[2] - t[2]) for o, t in zip(ours, theirs, strict=False)), default=float("inf"))
        ddist = max((abs(o[3] - t[3]) for o, t in zip(ours, theirs, strict=False)), default=float("inf"))
        max_sim_diff, max_dist_diff = max(max_sim_diff, dsim), max(max_dist_diff, ddist)
        # Strict: our full-precision score, rounded like run 1 (4 decimals), must equal the saved value.
        exact = sum(
            round(o[2], 4) == t[2] and round(o[3], 4) == t[3] for o, t in zip(ours, theirs, strict=False)
        )
        gate_ours = result.has_relevant_context
        gate_ref = ref["status"] != REFUSED_NO_CONTEXT
        ok = (
            ids_equal
            and dsim <= TOLERANCE
            and ddist <= TOLERANCE
            and exact == len(theirs)
            and gate_ours == gate_ref
        )
        print(
            f"{ref['id']:<7}{'same' if ids_equal else 'DIFFERENT':<11}{dsim:>11.2e}{ddist:>12.2e}  "
            f"{'pass' if gate_ours else 'refuse':<6} {', '.join(o[0] for o in ours)}"
            + ("" if ok else "   <-- MISMATCH")
        )
        rows.append(
            {
                "id": ref["id"],
                "question": ref["question"],
                "ok": ok,
                "chunk_ids_and_order_equal": ids_equal,
                "max_abs_similarity_diff": dsim,
                "max_abs_distance_diff": ddist,
                "chunks_with_exact_4dp_scores": exact,
                "chunks_compared": len(theirs),
                "gate_passes": gate_ours,
                "reference_gate_passes": gate_ref,
                "reference_status": ref["status"],
                # full precision (no extra rounding, so later re-checks cannot suffer double rounding)
                "retrieved": [
                    {"chunk_id": o[0], "policy_id": o[1], "similarity": o[2], "distance": o[3]} for o in ours
                ],
            }
        )
    print()
    n_ok = sum(r["ok"] for r in rows)
    checks.append(
        (
            "retrieval: top-4 chunk IDs and order identical",
            all(r["chunk_ids_and_order_equal"] for r in rows),
            f"{sum(r['chunk_ids_and_order_equal'] for r in rows)}/{len(rows)} questions",
        )
    )
    checks.append(
        (
            f"retrieval: scores within {TOLERANCE:.1e} of saved values",
            max_sim_diff <= TOLERANCE and max_dist_diff <= TOLERANCE,
            f"max |Δsimilarity| = {max_sim_diff:.2e}, max |Δdistance| = {max_dist_diff:.2e}",
        )
    )
    n_exact = sum(r["chunks_with_exact_4dp_scores"] for r in rows)
    n_cmp = sum(r["chunks_compared"] for r in rows)
    checks.append(
        (
            "retrieval: scores rounded to 4 dp exactly equal saved values (strict)",
            n_exact == n_cmp,
            f"{n_exact}/{n_cmp} chunks (similarity and distance both exact)",
        )
    )
    checks.append(
        (
            "refusal gate (similarity >= 0.25) same decision as run 1",
            all(r["gate_passes"] == r["reference_gate_passes"] for r in rows),
            f"gate passes for {sum(r['gate_passes'] for r in rows)}/{len(rows)}; run 1 had "
            f"{sum(not r['reference_gate_passes'] for r in rows)} no-context refusals",
        )
    )

    print("CHECKS")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    overall = all(ok for _, ok, _ in checks) and n_ok == len(rows)
    print()
    print(f"v1 chunk-index fingerprint (SHA-256): {fingerprint}")
    print(
        "Not reproduced (not claimed): LLM answers - need GROQ_API_KEY and are not bit-for-bit deterministic."
    )
    print(f"OVERALL: {'PASS' if overall else 'FAIL'} ({n_ok}/{len(rows)} questions reproduced)")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "started_utc": started.isoformat(timespec="seconds"),
                "reference_file": str(REFERENCE_JSON.relative_to(PROJECT_ROOT)),
                "tolerance": TOLERANCE,
                "settings": {
                    "top_k": settings.top_k,
                    "min_similarity": settings.min_similarity,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "collection": settings.collection_name,
                    "llm_model": settings.llm_model,
                },
                "versions": {
                    p: metadata.version(p) for p in ("chromadb", "onnxruntime", "numpy", "tokenizers")
                },
                "ingestion": {
                    "documents": summary.document_count,
                    "chunks": summary.chunk_count,
                    "chunks_per_policy": summary.chunks_per_document,
                    "reference": ref_ingest,
                },
                "v1_chunk_fingerprint_sha256": fingerprint,
                "max_abs_similarity_diff": max_sim_diff,
                "max_abs_distance_diff": max_dist_diff,
                "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
                "overall_pass": overall,
                "questions": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
