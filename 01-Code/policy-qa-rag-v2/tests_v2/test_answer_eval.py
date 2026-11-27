"""Deterministic answer checks (evaluation.answer_eval) - hand-worked cases."""

from __future__ import annotations

from evaluation.answer_eval import evaluate_answer, numbers_in
from src.citations import validate_citations
from src.retrieval import RetrievedChunk


def chunk(cid, pid, text):
    return RetrievedChunk(1, cid, pid, "T", "S", "f.md", text, 0.3, 0.7)


CTX = [
    chunk(
        "HR-001-CH02", "HR-001", "Full-time employees receive 20 days of paid annual leave per calendar year."
    ),
    chunk("HR-001-CH03", "HR-001", "Employees receive 10 days of paid sick leave per calendar year."),
]
FACTS = {"HR-001-CH02": {"HR-001-S02-E01"}, "HR-001-CH03": {"HR-001-S04-E01"}}
KEY = {
    "answer_terms": [["20 days"]],
    "required_evidence": [["HR-001-S02-E01"]],
    "expected_policy_ids": ["HR-001"],
}


def run(text, status="answered", answerable=True, key=KEY):
    return evaluate_answer(
        answerable=answerable,
        status=status,
        answer_text=text,
        citations=validate_citations(text, CTX),
        context_chunks=CTX,
        chunk_facts=FACTS,
        **key,
    )


def test_correct_answer():
    e = run("Full-time employees receive 20 days of paid annual leave. [Source: HR-001, Chunk: HR-001-CH02]")
    assert e["correct"] and e["citation_support"] == [True] and e["cited_fact_coverage"] == 1.0
    assert e["unsupported_numbers"] == [] and e["numbers_in_answer"] == ["20"]


def test_valid_citation_but_wrong_chunk_does_not_support_the_fact():
    e = run("Full-time employees receive 20 days of paid annual leave. [Source: HR-001, Chunk: HR-001-CH03]")
    assert e["citations_valid"] and e["answer_terms_found"] == [True]
    assert e["citation_support"] == [False] and e["cited_fact_coverage"] == 0.0 and not e["correct"]


def test_unsupported_number_is_flagged_as_ungrounded():
    e = run("You receive 20 days, or 25 days after five years. [Source: HR-001, Chunk: HR-001-CH02]")
    assert e["unsupported_numbers"] == ["25"] and not e["correct"]


def test_missing_fact_and_withheld_answer():
    assert not run("Annual leave is available. [Source: HR-001, Chunk: HR-001-CH02]")["correct"]
    e = run("20 days [Source: HR-001, Chunk: HR-001-CH99]", status="rejected_invalid_citations")
    assert not e["citations_valid"] and e["n_invalid_citations"] == 1 and not e["correct"]


def test_context_coverage_independent_of_citations():
    e = run("No citation here, 20 days.")
    assert e["context_fact_coverage"] == 1.0 and e["cited_fact_coverage"] == 0.0 and not e["correct"]


def test_refusal_cases():
    unans = {"answer_terms": [], "required_evidence": [], "expected_policy_ids": []}
    ok = run(
        "The policy documents do not contain enough information to answer this question.",
        "refused",
        False,
        unans,
    )
    assert ok["refusal_correct"] and ok["correct"]
    assert run("x", "refused_no_context", False, unans)["refusal_correct"]
    wrong = run("Pets are covered. [Source: HR-001, Chunk: HR-001-CH02]", "answered", False, unans)
    assert not wrong["refusal_correct"] and wrong["has_citations"]


def test_numbers_ignore_citations_and_policy_ids():
    assert numbers_in(
        "Per HR-003, claims above USD 1,000 need approval [Source: HR-003, Chunk: HR-003-CH02]."
    ) == ["1,000"]
    assert numbers_in("Core hours are 10:00 to 16:00.") == ["10:00", "16:00"]
