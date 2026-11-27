"""Activity A - compare three chunking strategies on the frozen Day 12 corpus.

Strategies: v1_paragraph_800_150 (frozen Day 12 chunker), fixed_500, recursive_800_100.
For each: chunk count and lengths, boundary quality (sentences, words, headings), key-fact
integrity (canonical evidence units), representative samples, and the top-3 results of the
same 5 frozen development queries with evidence coverage.

No winner is selected here: 5 development queries are too few, and the strategy choice is
made later from the 30-query held-out evaluation. All three indexes stay available.

Usage (project root):
    .venv/bin/python activities/a-chunking-strategies/compare_chunking.py
Outputs: activities/a-chunking-strategies/results/chunking-comparison-report.md (generated)
         activities/a-chunking-strategies/results/chunking-comparison.json
No API key and no network calls (local MiniLM embeddings, local Chroma).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evidence import (  # noqa: E402
    CATALOG_PATH,
    build_catalog,
    chunk_evidence,
    load_catalog,
    satisfied_groups,
    sha256_file,
)
from rag_v2.chunk_analysis import boundary_report, evidence_integrity, length_stats, overlap_stats  # noqa: E402
from rag_v2.chunking import STRATEGIES, load_corpus  # noqa: E402
from rag_v2.indexing import V2_CHROMA_DIR, build_index, chunk_fingerprint, collection_name  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402
from src.ingestion import chunk_documents  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEV_QUERIES = PROJECT_ROOT / "evaluation" / "datasets" / "dev_queries.json"
MANIFESTS = PROJECT_ROOT / "manifests"
REFERENCE_RUN1 = PROJECT_ROOT / "reference" / "day12-smoke-results-run1.json"
TOP_K = 3
SAMPLE_DOC = "HR-003"  # boundary map + samples use this document for every strategy
SAMPLE_EVIDENCE = "HR-003-S04-E03"  # "Hotel accommodation is reimbursed up to USD 180 per night."


def manifest_hash(name: str) -> str:
    return (MANIFESTS / name).read_text().split()[0]


def integrity_checks(docs) -> list[tuple[str, bool, str]]:
    checks = []
    corpus = (
        subprocess.run(
            ["shasum", "-a", "256", "-c", "--quiet", "manifests/corpus.sha256"], cwd=PROJECT_ROOT
        ).returncode
        == 0
    )
    checks.append(("corpus matches manifests/corpus.sha256", corpus, "6 policy files"))
    cat_sha, dev_sha = sha256_file(CATALOG_PATH), sha256_file(DEV_QUERIES)
    checks.append(
        (
            "evidence catalog matches frozen hash",
            cat_sha == manifest_hash("evidence-catalog.sha256"),
            cat_sha[:16] + "…",
        )
    )
    checks.append(
        ("dev queries match frozen hash", dev_sha == manifest_hash("dev-queries.sha256"), dev_sha[:16] + "…")
    )
    rebuilt = build_catalog(docs)
    checks.append(
        (
            "evidence catalog == rebuild from corpus",
            list(load_catalog().values()) == rebuilt,
            f"{len(rebuilt)} units",
        )
    )
    v1 = [(c.chunk_id, c.text) for c in chunk_documents(docs, 800, 150)]
    wrapped = [(c.chunk_id, c.text) for c in STRATEGIES["v1_paragraph_800_150"].chunk_documents(docs)]
    checks.append(("v1 strategy == frozen v1 chunk_documents(800, 150)", v1 == wrapped, f"{len(v1)} chunks"))
    return checks


def v1_index_sanity(index) -> tuple[bool, str]:
    """The Day 13 v1-strategy collection must give run 1's top-3 for all 15 smoke questions."""
    ref = json.loads(REFERENCE_RUN1.read_text(encoding="utf-8"))["results"]
    same = 0
    for r in ref:
        got = index.retriever.retrieve(r["question"], top_k=TOP_K).chunks
        exp = r["retrieved"][:TOP_K]
        same += [c.chunk_id for c in got] == [e["chunk_id"] for e in exp] and all(
            round(c.similarity, 4) == e["similarity"] for c, e in zip(got, exp, strict=True)
        )
    return same == len(
        ref
    ), f"{same}/{len(ref)} smoke questions: top-3 IDs and 4-dp scores equal Day 12 run 1"


def run_dev_queries(indexes, catalog, units_by_policy) -> list[dict]:
    queries = json.loads(DEV_QUERIES.read_text(encoding="utf-8"))["queries"]
    out = []
    for q in queries:
        groups = q["required_evidence"]
        required = {e for g in groups for e in g}
        ok_policies = sorted({catalog[e].policy_id for e in required})
        per_strategy = {}
        for name, idx in indexes.items():
            result = idx.retriever.retrieve(q["query"], top_k=TOP_K)
            rows, found, first_rank = [], set(), None
            for c in result.chunks:
                ev = chunk_evidence(idx.chunks_by_id[c.chunk_id].metadata, units_by_policy)
                hits = [e for e in ev if e in required]
                found.update(hits)
                if hits and first_rank is None:
                    first_rank = c.rank
                rows.append(
                    {
                        "rank": c.rank,
                        "chunk_id": c.chunk_id,
                        "policy_id": c.policy_id,
                        "similarity": c.similarity,
                        "section": c.section,
                        "evidence": ev,
                        "required_evidence_found": hits,
                    }
                )
            sat = satisfied_groups(groups, found)
            per_strategy[name] = {
                "top": rows,
                "policy_match": any(r["policy_id"] in ok_policies for r in rows),
                "evidence_hit": any(sat),
                "groups_satisfied": sum(sat),
                "groups_total": len(groups),
                "first_evidence_rank": first_rank,
                "dense_gate_passes": result.has_relevant_context,  # v1 rule: best similarity >= 0.25
            }
        out.append({**q, "acceptable_policies": ok_policies, "results": per_strategy})
    return out


# ----------------------------------------------------------------------------- report
def fence(text: str) -> list[str]:
    return ["```text", text, "```"]


def show(text: str) -> str:
    return text.replace("\n", "⏎")


def render(run, checks, sanity, stats, dev, indexes, docs, catalog) -> str:
    L: list[str] = []
    names = list(STRATEGIES)
    L += [
        "# Activity A – Chunking Strategy Comparison (generated report)",
        "",
        "> Generated by `activities/a-chunking-strategies/compare_chunking.py` from a real run. "
        "Do not edit by hand; re-run the script instead.",
        "",
        f"- Run (UTC): {run['started_utc']} · Python {run['python']} · chromadb {run['chromadb']} · "
        f"onnxruntime {run['onnxruntime']}",
        "- Embeddings: all-MiniLM-L6-v2 (ONNX, local, v1's `LocalEmbedder`) · cosine · embedded text = "
        "title + chunk (v1's `embedding_text`) · retriever: v1 `Retriever`, top-3",
        f"- Indexes: `{V2_CHROMA_DIR.relative_to(PROJECT_ROOT)}/` collections "
        + ", ".join(f"`{collection_name(n)}`" for n in names),
        "",
        "## 0. Integrity checks",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    L += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    L += [
        f"| v1-strategy index reproduces Day 12 retrieval | {'PASS' if sanity[0] else 'FAIL'} | {sanity[1]} |",
        "",
    ]

    L += [
        "## 1. Strategies",
        "",
        "| Strategy | Splitter family | Size | Overlap | Description | Fingerprint |",
        "|---|---|---:|---:|---|---|",
    ]
    for n in names:
        s = STRATEGIES[n]
        L.append(
            f"| `{n}` | {s.family} | {s.chunk_size} | {s.chunk_overlap} | {s.description} | "
            f"`{stats[n]['fingerprint'][:12]}` |"
        )
    L += [
        "",
        "## 2. Chunk counts and lengths (characters)",
        "",
        "| Strategy | Chunks | Mean | Median | Min | Max | Per policy (HR-001…IT-002) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for n in names:
        ls = stats[n]["lengths"]
        per = " / ".join(str(ls["chunks_per_policy"].get(d, 0)) for d in docs)
        L.append(
            f"| `{n}` | {ls['chunk_count']} | {ls['mean_chars']} | {ls['median_chars']} | "
            f"{ls['min_chars']} | {ls['max_chars']} | {per} |"
        )
    L += [
        "",
        "## 3. Boundary quality: are headings, sentences and words split?",
        "",
        "Counts over all chunks. A boundary is *mid-sentence* if it falls strictly inside an evidence "
        "sentence; *mid-word* if it falls between two letters/digits; a heading is *broken* if a boundary "
        "falls inside the heading line; a *trailing heading* is a chunk that ends with a heading, separating "
        "it from the text it introduces.",
        "",
        "| Strategy | Chunks | Start mid-sentence | End mid-sentence | Mid-word cuts | Broken headings | "
        "Trailing headings | Multi-section chunks | Boundaries with overlap | Overlap chars mean (min–max) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for n in names:
        b, o = stats[n]["boundaries"], stats[n]["overlap"]
        L.append(
            f"| `{n}` | {b['chunks']} | {b['starts_mid_sentence']} | {b['ends_mid_sentence']} | "
            f"{b['mid_word_cuts']} | {b['broken_headings']} | {b['trailing_headings']} | "
            f"{b['multi_section_chunks']} | {o['with_overlap']}/{o['boundaries']} | "
            f"{o['mean_overlap_chars']} ({o['min_overlap_chars']}–{o['max_overlap_chars']}) |"
        )
    L += [
        "",
        "## 4. Key-fact integrity (102 canonical evidence sentences)",
        "",
        "*Intact* = the whole sentence is inside at least one chunk. *Split* = no chunk contains the whole "
        "sentence, so no retrieval result can ever deliver that fact intact. *Duplicated* = intact in more "
        "than one chunk (overlap).",
        "",
        "| Strategy | Intact | Split | Duplicated |",
        "|---|---:|---:|---:|",
    ]
    for n in names:
        e = stats[n]["evidence"]
        L.append(f"| `{n}` | {e['intact']}/{e['evidence_units']} | {e['split']} | {e['duplicated']} |")
    for n in names:
        e = stats[n]["evidence"]
        if e["split_ids"]:
            L += ["", f"**Split facts – `{n}`** ({e['split']}):", ""]
            L += [f"- `{i}` – {catalog[i].text}" for i in e["split_ids"]]

    L += [
        "",
        f"## 5. Representative samples (document {SAMPLE_DOC})",
        "",
        f"### 5.1 Boundary map of {SAMPLE_DOC} under each strategy",
        "",
        "`⏎` marks a newline. *Starts/ends with* shows the first/last 30 characters of each chunk.",
        "",
    ]
    for n in names:
        idx = indexes[n]
        L += [
            f"**`{n}`**",
            "",
            "| Chunk | Offsets | Chars | Section(s) | Starts with | Ends with |",
            "|---|---|---:|---|---|---|",
        ]
        for c in idx.chunks:
            if c.metadata["policy_id"] != SAMPLE_DOC:
                continue
            L.append(
                f"| `{c.chunk_id}` | {c.metadata['start_char']}–{c.metadata['end_char']} | {len(c.text)} | "
                f"{c.metadata['section']} | `{show(c.text[:30])}` | `{show(c.text[-30:])}` |"
            )
        L.append("")
    L += [f'### 5.2 The chunk that holds `{SAMPLE_EVIDENCE}` ("{catalog[SAMPLE_EVIDENCE].text}")', ""]
    for n in names:
        idx = indexes[n]
        holders = stats[n]["evidence"]["containing_chunks"][SAMPLE_EVIDENCE]
        if not holders:
            L += [f"**`{n}`**: no chunk contains the whole sentence (split).", ""]
            continue
        c = idx.chunks_by_id[holders[0]]
        body = docs[SAMPLE_DOC].text
        s, e = c.metadata["start_char"], c.metadata["end_char"]
        L += [
            f"**`{n}`** – `{c.chunk_id}` · policy {c.metadata['policy_id']} ({c.metadata['title']}) · "
            f"section(s): {c.metadata['section']} · {len(c.text)} chars · offsets {s}–{e}"
            + (f" · also in {', '.join(holders[1:])}" if holders[1:] else ""),
            "",
        ]
        L += fence(c.text)
        L += [
            f"Text just before the chunk: `…{show(body[max(0, s - 30) : s])}` · just after: "
            f"`{show(body[e : e + 30])}…`",
            "",
        ]
    L += ["### 5.3 First chunk with a boundary problem (per strategy)", ""]
    for n in names:
        idx = indexes[n]
        bad = next(
            (
                r
                for r in stats[n]["boundaries"]["per_chunk"]
                if r["ends_mid_sentence"]
                or r["mid_word_cut"]
                or r["trailing_heading"]
                or r["broken_headings"]
            ),
            None,
        )
        if bad is None:
            L += [
                f"**`{n}`**: no chunk has a mid-sentence end, mid-word cut, broken or trailing heading.",
                "",
            ]
            continue
        c = idx.chunks_by_id[bad["chunk_id"]]
        issues = [k for k in ("starts_mid_sentence", "ends_mid_sentence", "mid_word_cut") if bad[k]]
        if bad["trailing_heading"]:
            issues.append(f"trailing heading `{bad['trailing_heading']}`")
        if bad["broken_headings"]:
            issues.append("broken heading")
        L += [
            f"**`{n}`** – `{c.chunk_id}` · policy {c.metadata['policy_id']} · section(s): {c.metadata['section']} "
            f"· {len(c.text)} chars · issues: {', '.join(issues)}",
            "",
        ]
        L += fence(c.text)
        L.append("")

    L += [
        "## 6. Development queries – top-3 per strategy",
        "",
        "Similarity = 1 − cosine distance (v1 convention). *Evidence in chunk* lists every evidence unit "
        "fully contained in the chunk; required ones are marked ✅. Columns per query: **policy match** (a "
        "top-3 chunk comes from a policy that holds required evidence), **evidence hit** (≥ 1 required group "
        "satisfied), **coverage** (required groups satisfied / total, counting each fact once even if "
        "several chunks hold it), **first evidence rank**, and the v1 dense gate (best similarity ≥ 0.25).",
        "",
    ]
    for q in dev:
        L += [
            f"### {q['id']} – {q['query']}",
            "",
            f"Category: {q['category']} · Required evidence groups: "
            + " AND ".join("(" + " OR ".join(f"`{e}`" for e in g) + ")" for g in q["required_evidence"]),
            "",
        ]
        L += [f"- `{e}`: {catalog[e].text}" for g in q["required_evidence"] for e in g]
        L += [
            "",
            "| Strategy | Rank | Chunk | Similarity | Section(s) | Evidence in chunk |",
            "|---|---:|---|---:|---|---|",
        ]
        for n in names:
            for r in q["results"][n]["top"]:
                ev = (
                    ", ".join(f"{e} ✅" if e in r["required_evidence_found"] else e for e in r["evidence"])
                    or "—"
                )
                L.append(
                    f"| `{n}` | {r['rank']} | `{r['chunk_id']}` | {r['similarity']:.4f} | {r['section']} | {ev} |"
                )
        L += [
            "",
            "| Strategy | Policy match | Evidence hit | Coverage | First evidence rank | Dense gate |",
            "|---|---|---|---|---|---|",
        ]
        for n in names:
            m = q["results"][n]
            L.append(
                f"| `{n}` | {'yes' if m['policy_match'] else 'no'} | {'yes' if m['evidence_hit'] else 'no'} | "
                f"{m['groups_satisfied']}/{m['groups_total']} | {m['first_evidence_rank'] or '—'} | "
                f"{'pass' if m['dense_gate_passes'] else 'refuse'} |"
            )
        L.append("")

    L += [
        "## 7. Summary over the 5 development queries (top-3)",
        "",
        "| Strategy | Policy match | Evidence hit | Evidence groups covered | First-evidence ranks (DEV-01…05) |",
        "|---|---:|---:|---:|---|",
    ]
    for n in names:
        rs = [q["results"][n] for q in dev]
        L.append(
            f"| `{n}` | {sum(r['policy_match'] for r in rs)}/{len(rs)} | {sum(r['evidence_hit'] for r in rs)}/{len(rs)} | "
            f"{sum(r['groups_satisfied'] for r in rs)}/{sum(r['groups_total'] for r in rs)} | "
            + ", ".join(str(r["first_evidence_rank"] or "—") for r in rs)
            + " |"
        )
    L += [
        "",
        "**How to read this.** Policy match and evidence coverage can disagree: a strategy can return "
        "the right policy while the returned chunks do not contain the needed sentence. Five queries are a "
        "development sample, so differences of one query are not evidence that one strategy is better. "
        "**No strategy is selected in Activity A**; all three indexes are kept for the hybrid-retrieval and "
        "reranking experiments, and the choice for RAG v2 is made from the 30-query held-out evaluation.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    started = datetime.now(UTC)
    settings = load_settings()
    docs_list = load_corpus(settings.policy_dir)
    docs = {d.policy_id: d for d in docs_list}
    catalog = load_catalog()
    units_by_policy: dict = {}
    for u in catalog.values():
        units_by_policy.setdefault(u.policy_id, []).append(u)

    checks = integrity_checks(docs_list)
    if not all(ok for _, ok, _ in checks):
        for n, ok, d in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}")
        print("Integrity check failed - not running the comparison.", file=sys.stderr)
        return 1

    embedder = LocalEmbedder()
    indexes = {
        name: build_index(name, docs_list, embedder, min_similarity=settings.min_similarity)
        for name in STRATEGIES
    }
    sanity = v1_index_sanity(indexes["v1_paragraph_800_150"])

    stats = {}
    for name, idx in indexes.items():
        stats[name] = {
            "fingerprint": chunk_fingerprint(idx.chunks),
            "lengths": length_stats(idx.chunks),
            "boundaries": boundary_report(idx.chunks, docs, units_by_policy),
            "overlap": overlap_stats(idx.chunks),
            "evidence": evidence_integrity(idx.chunks, units_by_policy),
        }
    dev = run_dev_queries(indexes, catalog, units_by_policy)

    run = {
        "started_utc": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "chromadb": metadata.version("chromadb"),
        "onnxruntime": metadata.version("onnxruntime"),
    }
    results = {
        "inputs": {
            "corpus_manifest_sha256": sha256_file(MANIFESTS / "corpus.sha256"),
            "evidence_catalog_sha256": sha256_file(CATALOG_PATH),
            "dev_queries_sha256": sha256_file(DEV_QUERIES),
            "top_k": TOP_K,
        },
        "integrity_checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
        "v1_index_reproduces_day12": {"pass": sanity[0], "detail": sanity[1]},
        "strategies": {
            n: {
                "family": STRATEGIES[n].family,
                "chunk_size": STRATEGIES[n].chunk_size,
                "chunk_overlap": STRATEGIES[n].chunk_overlap,
                **stats[n],
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        **{
                            k: c.metadata[k]
                            for k in ("policy_id", "section", "start_char", "end_char", "char_count")
                        },
                        "text": c.text,
                    }
                    for c in indexes[n].chunks
                ],
            }
            for n in STRATEGIES
        },
        "dev_queries": dev,
    }
    results_sha = hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = render(run, checks, sanity, stats, dev, indexes, docs, catalog)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "chunking-comparison-report.md").write_text(report + "\n", encoding="utf-8")
    (RESULTS_DIR / "chunking-comparison.json").write_text(
        json.dumps({"run": run, "results_sha256": results_sha, **results}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Results SHA-256 (everything except run time/versions): {results_sha}")
    print(
        f"Saved: {(RESULTS_DIR / 'chunking-comparison-report.md').relative_to(PROJECT_ROOT)}, "
        f"{(RESULTS_DIR / 'chunking-comparison.json').relative_to(PROJECT_ROOT)}"
    )
    ok = all(c[1] for c in checks) and sanity[0]
    print(f"OVERALL: {'PASS' if ok else 'FAIL'} (integrity + v1 reproduction checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
