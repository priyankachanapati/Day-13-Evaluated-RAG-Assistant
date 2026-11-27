"""Chunking strategies for the Day 13 comparison (Activity A).

Three strategies, all applied to the same document bodies that RAG v1 uses
(``PolicyDocument.text`` from the frozen ``src.ingestion.load_documents``):

``v1_paragraph_800_150``  (document-specific / structure-aware)
    The frozen Day 12 chunker, called unchanged: paragraphs packed into chunks of at most
    800 characters; headings are glued to the paragraph that follows; each chunk starts
    with a ~150-character, word-aligned tail of the previous chunk.

``fixed_500``  (character splitting)
    A new chunk every 500 characters, no overlap, ignoring words, sentences and headings.
    This is the simplest possible splitter and the reference point for "no structure".

``recursive_800_100``  (recursive character splitting)
    The algorithm of LangChain's ``RecursiveCharacterTextSplitter``, implemented here
    without the LangChain dependency, with its default separators
    ``["\\n\\n", "\\n", " ", ""]``: split on the first separator that occurs; pieces that are
    still longer than ``chunk_size`` are split again with the next separator; neighbouring
    pieces are then merged greedily up to ``chunk_size``, and each new chunk starts with the
    trailing *whole pieces* of the previous chunk that fit in ``chunk_overlap``.
    Consequence worth knowing: overlap is at most 100 characters and can be 0 when the
    pieces at that level (here: paragraphs) are longer than 100 characters.

Every chunk is a v1 ``Chunk`` (same ID format ``HR-001-CH01`` so the v1 citation validator
accepts it, same metadata keys, same ``embedding_text`` = title + text), plus three keys:
``strategy``, ``start_char`` and ``end_char`` (offsets into the document body, so boundary
quality can be measured exactly). All strategies produce contiguous substrings of the body.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.ingestion import Chunk, PolicyDocument, chunk_document, load_documents

Span = tuple[int, int]  # [start, end) character offsets into a document body

RECURSIVE_SEPARATORS = ("\n\n", "\n", " ", "")
SECTION_HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
DEFAULT_SECTION = "Introduction"  # v1's name for text before the first "## " heading


# --------------------------------------------------------------------------- span builders
def fixed_spans(text: str, chunk_size: int) -> list[Span]:
    """Character splitting: consecutive windows of ``chunk_size`` characters, no overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    spans = [(i, min(i + chunk_size, len(text))) for i in range(0, len(text), chunk_size)]
    return [s for s in spans if text[s[0] : s[1]].strip()]  # never emit a whitespace-only chunk


def _pieces(text: str, start: int, end: int, separator: str) -> list[Span]:
    """Split text[start:end] on ``separator``, keeping the separator at the end of each piece.

    Keeping separators makes the split lossless: the pieces concatenate back to the input.
    """
    if separator == "":
        return [(i, i + 1) for i in range(start, end)]
    pieces, pos = [], start
    while pos < end:
        hit = text.find(separator, pos, end)
        if hit == -1:
            pieces.append((pos, end))
            break
        pieces.append((pos, hit + len(separator)))
        pos = hit + len(separator)
    return pieces


def _merge(pieces: list[Span], chunk_size: int, chunk_overlap: int) -> list[Span]:
    """Greedily merge contiguous pieces into chunks of at most ``chunk_size`` characters.

    When a chunk is emitted, pieces are dropped from its front until what remains is at most
    ``chunk_overlap`` characters long (and leaves room for the next piece); the remainder
    becomes the start of the next chunk (the overlap).
    """
    chunks: list[Span] = []
    current: list[Span] = []

    def length(parts: list[Span]) -> int:
        return parts[-1][1] - parts[0][0] if parts else 0

    for piece in pieces:
        piece_len = piece[1] - piece[0]
        if current and length(current) + piece_len > chunk_size:
            chunks.append((current[0][0], current[-1][1]))
            while current and (length(current) > chunk_overlap or length(current) + piece_len > chunk_size):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def _recursive(
    text: str, start: int, end: int, chunk_size: int, chunk_overlap: int, separators: tuple[str, ...]
) -> list[Span]:
    segment = text[start:end]
    index = next(i for i, sep in enumerate(separators) if sep == "" or sep in segment)
    separator, finer = separators[index], separators[index + 1 :]
    out: list[Span] = []
    fitting: list[Span] = []
    for piece in _pieces(text, start, end, separator):
        if piece[1] - piece[0] <= chunk_size:
            fitting.append(piece)
            continue
        if fitting:
            out.extend(_merge(fitting, chunk_size, chunk_overlap))
            fitting = []
        if finer:
            out.extend(_recursive(text, piece[0], piece[1], chunk_size, chunk_overlap, finer))
        else:
            out.append(piece)
    if fitting:
        out.extend(_merge(fitting, chunk_size, chunk_overlap))
    return out


def _strip_span(text: str, span: Span) -> Span | None:
    """Trim surrounding whitespace from a span; None if nothing is left."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def recursive_spans(
    text: str, chunk_size: int, chunk_overlap: int, separators: tuple[str, ...] = RECURSIVE_SEPARATORS
) -> list[Span]:
    """Recursive character splitting (LangChain ``RecursiveCharacterTextSplitter`` algorithm)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    spans = (
        _strip_span(text, s) for s in _recursive(text, 0, len(text), chunk_size, chunk_overlap, separators)
    )
    return [s for s in spans if s is not None]


# --------------------------------------------------------------------------- sections
def section_headings(text: str) -> list[tuple[int, str]]:
    """(offset, title) of every ``## `` heading in a document body."""
    return [(m.start(), m.group(1).strip()) for m in SECTION_HEADING.finditer(text)]


def sections_for_span(text: str, span: Span) -> str:
    """``; ``-joined titles of the sections a span overlaps (v1's metadata format)."""
    headings = section_headings(text)
    current = DEFAULT_SECTION
    for offset, title in headings:
        if offset <= span[0]:
            current = title
    names = [current] + [t for o, t in headings if span[0] < o < span[1]]
    return "; ".join(dict.fromkeys(names))


def locate(text: str, chunk_text: str) -> Span:
    """Offsets of a chunk inside its document body (every strategy yields contiguous substrings)."""
    start = text.find(chunk_text)
    if start == -1:
        raise ValueError("chunk text is not a contiguous substring of the document body")
    return (start, start + len(chunk_text))


# --------------------------------------------------------------------------- chunk objects
def _make_chunk(doc: PolicyDocument, index: int, span: Span, strategy: str, section: str) -> Chunk:
    chunk_id = f"{doc.policy_id}-CH{index:02d}"
    text = doc.text[span[0] : span[1]]
    metadata: dict[str, str | int] = {
        "chunk_id": chunk_id,
        "policy_id": doc.policy_id,
        "title": doc.title,
        "department": doc.department,
        "source_file": doc.source_file,
        "section": section,
        "chunk_index": index,
        "char_count": len(text),
        "strategy": strategy,
        "start_char": span[0],
        "end_char": span[1],
    }
    for key in ("version", "effective_date", "classification"):
        if key in doc.extra_metadata:
            metadata[key] = doc.extra_metadata[key]
    return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)


def _from_spans(doc: PolicyDocument, spans: list[Span], strategy: str) -> list[Chunk]:
    return [
        _make_chunk(doc, i, span, strategy, sections_for_span(doc.text, span))
        for i, span in enumerate(spans, start=1)
    ]


def chunk_v1_paragraph(doc: PolicyDocument) -> list[Chunk]:
    """Frozen v1 chunker (800/150), unchanged; adds strategy and offsets to the metadata only."""
    out = []
    for c in chunk_document(doc, 800, 150):
        span = locate(doc.text, c.text)
        out.append(
            _make_chunk(
                doc, int(c.metadata["chunk_index"]), span, "v1_paragraph_800_150", str(c.metadata["section"])
            )
        )  # keep v1's own section label
    return out


def chunk_fixed_500(doc: PolicyDocument) -> list[Chunk]:
    return _from_spans(doc, fixed_spans(doc.text, 500), "fixed_500")


def chunk_recursive_800_100(doc: PolicyDocument) -> list[Chunk]:
    return _from_spans(doc, recursive_spans(doc.text, 800, 100), "recursive_800_100")


@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    family: str  # splitter family, for the write-up
    description: str
    chunk_size: int
    chunk_overlap: int
    chunk_fn: Callable[[PolicyDocument], list[Chunk]]

    def chunk_documents(self, docs: list[PolicyDocument]) -> list[Chunk]:
        chunks = [c for d in docs for c in self.chunk_fn(d)]
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: duplicate chunk IDs")
        return chunks


STRATEGIES: dict[str, ChunkingStrategy] = {
    s.name: s
    for s in (
        ChunkingStrategy(
            "v1_paragraph_800_150",
            "document-specific (paragraph/Markdown-aware)",
            "Day 12 v1 chunker, unchanged",
            800,
            150,
            chunk_v1_paragraph,
        ),
        ChunkingStrategy(
            "fixed_500", "character", "fixed 500-character windows, no overlap", 500, 0, chunk_fixed_500
        ),
        ChunkingStrategy(
            "recursive_800_100",
            "recursive character",
            "recursive splitter, separators \\n\\n, \\n, space, char; 800 chars, 100 overlap",
            800,
            100,
            chunk_recursive_800_100,
        ),
    )
}


def load_corpus(policy_dir: Path) -> list[PolicyDocument]:
    """The frozen v1 loader (same documents, same bodies for every strategy)."""
    return load_documents(policy_dir)
