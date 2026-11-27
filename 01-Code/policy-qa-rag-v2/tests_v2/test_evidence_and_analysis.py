"""Tests for canonical evidence units, the frozen dev set and offset-based chunk analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from evaluation.evidence import (
    CATALOG_PATH,
    EvidenceUnit,
    build_catalog,
    build_units,
    catalog_document,
    dumps,
    load_catalog,
    normalise,
    satisfied_groups,
    sha256_file,
    span_contains,
)
from rag_v2.chunk_analysis import boundary_report, evidence_integrity, overlap_stats
from rag_v2.chunking import STRATEGIES, _from_spans, load_corpus
from src.config import load_settings
from src.ingestion import PolicyDocument

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "evaluation" / "datasets" / "dev_queries.json"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(load_settings().policy_dir)


def manifest(name):
    return (ROOT / "manifests" / name).read_text().split()[0]


# ------------------------------------------------------------------ evidence catalog
def test_catalog_matches_frozen_hash_and_rebuild(docs):
    assert sha256_file(CATALOG_PATH) == manifest("evidence-catalog.sha256")
    rebuilt = dumps(catalog_document(build_catalog(docs), sha256_file(ROOT / "manifests" / "corpus.sha256")))
    assert CATALOG_PATH.read_text(encoding="utf-8") == rebuilt


def test_units_are_verbatim_sentences_with_offsets(docs):
    bodies = {d.policy_id: d.text for d in docs}
    catalog = load_catalog()
    assert len(catalog) == 102
    for eid, u in catalog.items():
        assert re.fullmatch(r"[A-Z]{2}-\d{3}-S\d{2}-E\d{2}", eid)
        assert bodies[u.policy_id][u.start : u.end] == u.text
        assert not u.text.startswith(("#", ">")) and "SYNTHETIC" not in u.text
        assert u.text[-1] in ".!?"


def test_unit_ids_follow_section_numbers():
    doc = PolicyDocument(
        "XX-001",
        "T",
        "D",
        "x.md",
        "# T\n\n> note.\n\n## 1. Scope\n\nA one. A two.\n\n## 3. Later\n\nB one.\n\nB two.",
    )
    units = build_units(doc)
    assert [(u.evidence_id, u.text) for u in units] == [
        ("XX-001-S01-E01", "A one."),
        ("XX-001-S01-E02", "A two."),
        ("XX-001-S03-E01", "B one."),
        ("XX-001-S03-E02", "B two."),
    ]


def test_span_containment_is_whole_sentence_only():
    u = EvidenceUnit("XX-001-S01-E01", "XX-001", 1, "1. S", 1, "abc.", 10, 14)
    assert span_contains(10, 14, u) and span_contains(0, 100, u)
    assert not span_contains(11, 14, u) and not span_contains(10, 13, u)


def test_satisfied_groups_any_of_alternatives():
    groups = [["A", "B"], ["C"]]
    assert satisfied_groups(groups, {"B"}) == [True, False]
    assert satisfied_groups(groups, {"A", "C"}) == [True, True]
    assert satisfied_groups(groups, set()) == [False, False]


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_offset_containment_agrees_with_text_containment(docs, name):
    catalog = load_catalog()
    for c in STRATEGIES[name].chunk_documents(docs):
        for u in catalog.values():
            if u.policy_id != c.metadata["policy_id"]:
                continue
            by_offset = span_contains(c.metadata["start_char"], c.metadata["end_char"], u)
            by_text = normalise(u.text) in normalise(c.text)
            # offset containment implies text containment; text can only "contain" more when the same
            # sentence occurs twice in one document (it does not in this corpus)
            assert by_offset == by_text, (c.chunk_id, u.evidence_id)


# ------------------------------------------------------------------ dev set
def test_dev_queries_frozen_and_labels_valid():
    assert sha256_file(DEV) == manifest("dev-queries.sha256")
    data = json.loads(DEV.read_text(encoding="utf-8"))
    catalog = load_catalog()
    assert len(data["queries"]) == 5
    assert len({q["id"] for q in data["queries"]}) == 5
    for q in data["queries"]:
        assert q["required_evidence"] and all(q["required_evidence"])
        assert all(e in catalog for g in q["required_evidence"] for e in g)


# ------------------------------------------------------------------ analysis on a toy document
TOY = PolicyDocument(
    "XX-001", "T", "D", "x.md", "## 1. Alpha\n\nOne two three. Four five.\n\n## 2. Beta\n\nSix seven."
)


def toy_chunks(spans):
    return _from_spans(TOY, spans, "toy")


def test_boundary_report_detects_mid_sentence_word_cut_and_trailing_heading():
    units = {"XX-001": build_units(TOY)}
    body = TOY.text
    beta = body.index("## 2. Beta")
    cut = body.index("three") + 2  # inside "three"
    spans = [(0, cut), (cut, beta + len("## 2. Beta")), (body.index("Six"), len(body))]
    report = boundary_report(toy_chunks(spans), {"XX-001": TOY}, units)
    assert report["ends_mid_sentence"] == 1 and report["starts_mid_sentence"] == 1
    assert report["mid_word_cuts"] == 2  # chunk 1 end and chunk 2 start
    assert report["trailing_headings"] == 1  # chunk 2 ends with "## 2. Beta"
    assert report["broken_headings"] == 0


def test_evidence_integrity_split_and_duplicated():
    units = {"XX-001": build_units(TOY)}  # E01 "One two three." E02 "Four five." S02-E01 "Six seven."
    body = TOY.text
    e02 = body.index("Four")
    spans = [(0, body.index("three") + 2), (body.index("three"), len(body)), (e02, len(body))]
    integ = evidence_integrity(toy_chunks(spans), units)
    assert integ["split_ids"] == ["XX-001-S01-E01"]  # never whole in one chunk
    assert set(integ["duplicated_ids"]) == {"XX-001-S01-E02", "XX-001-S02-E01"}  # in chunks 2 and 3
    assert overlap_stats(toy_chunks(spans))["with_overlap"] == 2
