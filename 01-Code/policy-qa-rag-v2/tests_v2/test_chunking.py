"""Unit tests for the Day 13 chunking strategies (offline, no embeddings)."""

from __future__ import annotations

import math

import pytest

from rag_v2.chunking import (
    STRATEGIES,
    fixed_spans,
    load_corpus,
    recursive_spans,
    sections_for_span,
)
from src.citations import STRICT_CITATION
from src.config import load_settings
from src.ingestion import chunk_documents

V1_METADATA_KEYS = {
    "chunk_id",
    "policy_id",
    "title",
    "department",
    "source_file",
    "section",
    "chunk_index",
    "char_count",
    "version",
    "effective_date",
    "classification",
}


@pytest.fixture(scope="module")
def docs():
    return load_corpus(load_settings().policy_dir)


def texts(text, spans):
    return [text[s:e] for s, e in spans]


# ------------------------------------------------------------------ hand-calculated examples
def test_fixed_toy_example():
    # "abcdefghij" in windows of 4 -> [0,4) [4,8) [8,10)
    assert fixed_spans("abcdefghij", 4) == [(0, 4), (4, 8), (8, 10)]


def test_recursive_toy_example_with_overlap():
    # Pieces on " " (separator kept): "a " "b " "c " "d " "e " "f"
    # size 5, overlap 2:
    #   [a,b]=4; +c would be 6 > 5 -> emit "a b"; keep [b] (2 <= overlap) -> [b,c] ...
    #   last chunk [d,e,f] = 5 fits exactly.
    text = "a b c d e f"
    assert texts(text, recursive_spans(text, 5, 2)) == ["a b", "b c", "c d", "d e f"]


def test_recursive_overlap_is_whole_pieces_only():
    # Pieces: "aaa " "bbb " "ccc " (4 chars each) and "ddd" (3, no trailing separator).
    # A 4-char piece is longer than overlap 3, so no overlap can be kept between chunks.
    # "aaa "+"bbb " = 8 > 7 -> emit "aaa"; same for "bbb"; "ccc "+"ddd" = 7 <= 7 -> one chunk.
    text = "aaa bbb ccc ddd"
    assert texts(text, recursive_spans(text, 7, 3)) == ["aaa", "bbb", "ccc ddd"]


def test_recursive_prefers_paragraph_separator():
    text = "para one.\n\npara two."
    assert texts(text, recursive_spans(text, 50, 0)) == [text]  # fits: one chunk
    assert texts(text, recursive_spans(text, 12, 0)) == ["para one.", "para two."]  # split on \n\n


def test_recursive_splits_oversized_piece_with_next_separator():
    text = "short.\n\n" + "word " * 10  # second paragraph (50 chars) > size 20
    chunks = texts(text, recursive_spans(text, 20, 0))
    assert chunks[0] == "short."
    assert all(len(c) <= 20 for c in chunks)
    assert " ".join(chunks[1:]).split() == ["word"] * 10  # nothing lost


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        fixed_spans("abc", 0)
    with pytest.raises(ValueError):
        recursive_spans("abc", 10, 10)
    with pytest.raises(ValueError):
        recursive_spans("abc", 10, -1)


def test_sections_for_span():
    body = "# T\n\nintro\n\n## 1. One\n\naaa\n\n## 2. Two\n\nbbb"
    assert sections_for_span(body, (0, 8)) == "Introduction"
    one = body.index("## 1.")
    two = body.index("## 2.")
    assert sections_for_span(body, (one, one + 5)) == "1. One"
    assert sections_for_span(body, (one, two + 3)) == "1. One; 2. Two"


# ------------------------------------------------------------------ properties on the real corpus
def test_fixed_500_is_lossless_and_sized(docs):
    for d in docs:
        spans = fixed_spans(d.text, 500)
        assert "".join(texts(d.text, spans)) == d.text
        assert len(spans) == math.ceil(len(d.text) / 500)
        assert all(e - s <= 500 for s, e in spans)


def test_recursive_800_100_sizes_overlap_and_coverage(docs):
    for d in docs:
        spans = recursive_spans(d.text, 800, 100)
        assert all(e - s <= 800 for s, e in spans)
        assert spans == sorted(spans)
        for (_s1, e1), (s2, _) in zip(spans, spans[1:], strict=False):
            assert s2 <= e1 + 2, "gap larger than the stripped separator"
            assert max(0, e1 - s2) <= 100, "overlap exceeds 100"
        covered = set()
        for s, e in spans:
            covered.update(range(s, e))
        assert all(i in covered for i, ch in enumerate(d.text) if not ch.isspace()), "text lost"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_chunks_are_body_substrings_with_v1_compatible_ids_and_metadata(docs, name):
    chunks = STRATEGIES[name].chunk_documents(docs)
    by_policy = {d.policy_id: d for d in docs}
    for c in chunks:
        md = c.metadata
        assert by_policy[md["policy_id"]].text[md["start_char"] : md["end_char"]] == c.text
        # The v1 citation validator must accept citations to these chunks.
        assert STRICT_CITATION.fullmatch(f"[Source: {md['policy_id']}, Chunk: {c.chunk_id}]")
        assert c.chunk_id.startswith(md["policy_id"] + "-CH")
        assert set(md) >= V1_METADATA_KEYS and md["strategy"] == name
        assert c.embedding_text.startswith(f"{md['title']} ({md['policy_id']})\n")
        assert c.text.strip()


def test_v1_strategy_is_exactly_the_frozen_v1_chunker(docs):
    original = chunk_documents(docs, 800, 150)
    wrapped = STRATEGIES["v1_paragraph_800_150"].chunk_documents(docs)
    assert [(c.chunk_id, c.text, c.metadata["section"]) for c in original] == [
        (c.chunk_id, c.text, c.metadata["section"]) for c in wrapped
    ]
    assert len(wrapped) == 22


def test_chunking_is_deterministic(docs):
    for s in STRATEGIES.values():
        assert [(c.chunk_id, c.text) for c in s.chunk_documents(docs)] == [
            (c.chunk_id, c.text) for c in s.chunk_documents(docs)
        ]
