"""Objective, offset-based quality measures for a chunking of the corpus (Activity A).

Every chunk carries ``start_char``/``end_char`` offsets into its document body, and every
evidence unit (sentence) and heading has offsets too, so each property below is a precise,
reproducible count rather than a judgement:

* **mid-sentence boundary** - a chunk starts or ends strictly inside a sentence.
* **mid-word cut** - the boundary falls between two letters/digits of the same word.
* **broken heading** - a ``#`` heading line is cut by a chunk boundary.
* **trailing heading** - a chunk ends with a heading, i.e. the heading is separated from the
  text it introduces (that text starts in the next chunk).
* **split fact** - an evidence sentence that is not fully contained in ANY chunk: no single
  retrieved chunk can ever deliver that fact intact.
* **duplicated fact** - an evidence sentence fully contained in more than one chunk (caused
  by overlap). Relevant later: counting chunks instead of distinct facts would inflate recall.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from evaluation.evidence import EvidenceUnit, Heading, headings, span_contains
from src.ingestion import Chunk, PolicyDocument


def _span(c: Chunk) -> tuple[int, int]:
    return int(c.metadata["start_char"]), int(c.metadata["end_char"])


def _inside(pos: int, start: int, end: int) -> bool:
    return start < pos < end


def length_stats(chunks: list[Chunk]) -> dict:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "chunks_per_policy": dict(Counter(str(c.metadata["policy_id"]) for c in chunks)),
        "mean_chars": round(statistics.mean(lengths), 1),
        "median_chars": statistics.median(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "total_chars": sum(lengths),
    }


def boundary_report(
    chunks: list[Chunk], docs: dict[str, PolicyDocument], units: dict[str, list[EvidenceUnit]]
) -> dict:
    heads: dict[str, list[Heading]] = {pid: headings(d) for pid, d in docs.items()}
    per_chunk = []
    for c in chunks:
        pid = str(c.metadata["policy_id"])
        body = docs[pid].text
        start, end = _span(c)
        sents = units[pid]
        starts_mid = any(_inside(start, u.start, u.end) for u in sents)
        ends_mid = any(_inside(end, u.start, u.end) for u in sents)
        word_cut_start = 0 < start < len(body) and body[start - 1].isalnum() and body[start].isalnum()
        word_cut_end = 0 < end < len(body) and body[end - 1].isalnum() and body[end].isalnum()
        broken = [
            h.text for h in heads[pid] if _inside(start, h.start, h.end) or _inside(end, h.start, h.end)
        ]
        trailing = [h.text for h in heads[pid] if h.end == end and h.start > start]
        per_chunk.append(
            {
                "chunk_id": c.chunk_id,
                "starts_mid_sentence": starts_mid,
                "ends_mid_sentence": ends_mid,
                "mid_word_cut": word_cut_start or word_cut_end,
                "broken_headings": broken,
                "trailing_heading": trailing[0] if trailing else None,
                "sections_spanned": len(str(c.metadata["section"]).split("; ")),
            }
        )
    n = len(per_chunk)
    return {
        "chunks": n,
        "starts_mid_sentence": sum(r["starts_mid_sentence"] for r in per_chunk),
        "ends_mid_sentence": sum(r["ends_mid_sentence"] for r in per_chunk),
        "mid_word_cuts": sum(r["mid_word_cut"] for r in per_chunk),
        "broken_headings": sum(len(r["broken_headings"]) for r in per_chunk),
        "trailing_headings": sum(r["trailing_heading"] is not None for r in per_chunk),
        "multi_section_chunks": sum(r["sections_spanned"] > 1 for r in per_chunk),
        "per_chunk": per_chunk,
    }


def overlap_stats(chunks: list[Chunk]) -> dict:
    by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for c in chunks:
        by_doc[str(c.metadata["policy_id"])].append(_span(c))
    overlaps = [
        max(0, a[1] - b[0]) for spans in by_doc.values() for a, b in zip(spans, spans[1:], strict=False)
    ]
    return {
        "boundaries": len(overlaps),
        "with_overlap": sum(o > 0 for o in overlaps),
        "mean_overlap_chars": round(statistics.mean(overlaps), 1) if overlaps else 0,
        "min_overlap_chars": min(overlaps, default=0),
        "max_overlap_chars": max(overlaps, default=0),
    }


def evidence_integrity(chunks: list[Chunk], units: dict[str, list[EvidenceUnit]]) -> dict:
    containing: dict[str, list[str]] = {}
    for pid, sents in units.items():
        doc_chunks = [c for c in chunks if c.metadata["policy_id"] == pid]
        for u in sents:
            containing[u.evidence_id] = [c.chunk_id for c in doc_chunks if span_contains(*_span(c), u)]
    total = len(containing)
    split = sorted(e for e, cs in containing.items() if not cs)
    duplicated = sorted(e for e, cs in containing.items() if len(cs) > 1)
    return {
        "evidence_units": total,
        "intact": total - len(split),
        "split": len(split),
        "split_ids": split,
        "duplicated": len(duplicated),
        "duplicated_ids": duplicated,
        "containing_chunks": containing,
    }
