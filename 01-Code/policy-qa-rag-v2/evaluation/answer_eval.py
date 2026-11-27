"""Deterministic answer-quality checks for RAG answers (Phase 6+).

ALL checks here are deterministic (string / offset / ID comparisons). No LLM-as-judge is used.
Limitation stated up front: string checks cannot judge paraphrased meaning or detect every
unsupported non-numeric claim; the numeric groundedness check and the citation checks reduce,
but do not remove, that gap.

Answerable query (status must be ``answered``):
    answer_terms        every answer-term group appears in the answer (case-insensitive)
    citations_valid     >= 1 citation and every citation valid (v1 validator: cited chunk was in
                        the context given to the LLM and the policy ID matches); none malformed
    cites_expected      at least one valid citation points to an expected policy
    citation_support    every matched answer-term group also appears in the text of a VALIDLY
                        CITED chunk -> the cited evidence actually supports the stated facts
    cited_fact_coverage required canonical fact groups fully contained in validly cited chunks
    context_fact_cov.   required fact groups present in the context given to the LLM (retrieval)
    unsupported_numbers numbers / times / amounts in the answer that occur in no cited chunk
                        (deterministic groundedness proxy for hallucinated figures)
    uncited_sentences   informational (v1 validator)
    correct             status answered AND all answer terms AND citations_valid AND cites_expected
                        AND all citation_support AND no unsupported_numbers
    (Valid citations alone never make an answer correct.)

Unanswerable query:
    refusal_correct     status is ``refused`` or ``refused_no_context`` AND no citations
"""

from __future__ import annotations

import re
import unicodedata

from src.citations import LOOSE_CITATION

POLICY_OR_CHUNK_ID = re.compile(r"\b[A-Z]{2,5}-\d{3}(?:-CH\d{2})?\b")
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).lower()


def numbers_in(text: str) -> list[str]:
    """Numbers stated in the prose of an answer (citations and policy/chunk IDs removed)."""
    prose = POLICY_OR_CHUNK_ID.sub(" ", LOOSE_CITATION.sub(" ", text))
    return [n.rstrip(".,:") for n in NUMBER.findall(prose)]


def _number_supported(number: str, evidence: str) -> bool:
    variants = {number, number.replace(",", "")}
    return any(re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d])", evidence) for v in variants)


def evaluate_answer(
    *,
    answerable: bool,
    status: str,
    answer_text: str,
    citations,
    context_chunks,
    answer_terms: list[list[str]],
    required_evidence: list[list[str]],
    expected_policy_ids: list[str],
    chunk_facts: dict[str, set[str]],
) -> dict:
    """``citations`` is the v1 CitationReport (or None); ``context_chunks`` are the chunks given to the LLM."""
    valid = list(citations.valid) if citations else []
    has_citations = bool(citations and citations.has_citations)
    if not answerable:
        refused = status in ("refused", "refused_no_context")
        return {
            "answerable": False,
            "status": status,
            "refusal_correct": refused and not has_citations,
            "has_citations": has_citations,
            "correct": refused and not has_citations,
        }

    text = _norm(answer_text)
    cited_ids = {c.chunk_id for c in valid}
    by_id = {c.chunk_id: c for c in context_chunks}
    cited_text = _norm(" ".join(by_id[cid].text for cid in cited_ids if cid in by_id))
    terms_found = [any(_norm(t) in text for t in g) for g in answer_terms]
    support = [any(_norm(t) in text and _norm(t) in cited_text for t in g) for g in answer_terms]
    cited_facts = set().union(*(chunk_facts.get(cid, set()) for cid in cited_ids)) if cited_ids else set()
    context_facts = (
        set().union(*(chunk_facts.get(c.chunk_id, set()) for c in context_chunks))
        if context_chunks
        else set()
    )
    cited_cov = [any(e in cited_facts for e in g) for g in required_evidence]
    context_cov = [any(e in context_facts for e in g) for g in required_evidence]
    nums = numbers_in(answer_text) if status == "answered" else []
    unsupported = [n for n in nums if not _number_supported(n, cited_text)]
    citations_valid = bool(citations and citations.all_valid)
    cites_expected = any(c.policy_id in expected_policy_ids for c in valid)
    answered = status == "answered"
    return {
        "answerable": True,
        "status": status,
        "answered": answered,
        "answer_terms_found": terms_found,
        "answer_term_coverage": sum(terms_found) / len(terms_found) if terms_found else 0.0,
        "citations_valid": citations_valid,
        "n_valid_citations": len(valid),
        "n_invalid_citations": len(citations.invalid) if citations else 0,
        "n_malformed_citations": len(citations.malformed) if citations else 0,
        "cites_expected_policy": cites_expected,
        "citation_support": support,
        "cited_fact_coverage": sum(cited_cov) / len(cited_cov) if cited_cov else 0.0,
        "context_fact_coverage": sum(context_cov) / len(context_cov) if context_cov else 0.0,
        "numbers_in_answer": nums,
        "unsupported_numbers": unsupported,
        "uncited_sentences": len(citations.uncited_sentences) if citations else 0,
        "correct": answered
        and all(terms_found)
        and citations_valid
        and cites_expected
        and all(support)
        and not unsupported,
    }
