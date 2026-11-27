"""BM25 tokenizer rules (rag_v2.tokenizer)."""

from __future__ import annotations

import pytest

from rag_v2.tokenizer import fold_plural, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HR-003", ["hr-003"]),  # policy code stays atomic, no parts
        ("See IT-001 and HR-004.", ["see", "it-001", "hr-004"]),
        ("USD 1,000", ["usd", "1000"]),  # thousands separator removed
        ("USD 0.40 per kilometre", ["usd", "0.40", "per", "kilometre"]),  # decimal kept whole
        ("between 10:00 and 16:00", ["between", "10:00", "16:00"]),  # times kept whole
        ("work-from-home days", ["work-from-home", "work", "home", "day"]),  # compound + parts
        ("public Wi-Fi", ["public", "wi-fi", "wi", "fi"]),
        ("hr-ops@brightlake.example.", ["hr-ops@brightlake.example", "hr", "ops", "brightlake", "example"]),
        ("IT Service Desk's extension", ["it", "service", "desk", "extension"]),  # acronym kept, 's removed
        ("It applies to all policies", ["apply", "policy"]),  # pronoun "it" removed
        ("What is KeyVault?", ["keyvault"]),
        ("SMS codes are not an approved MFA method", ["sms", "code", "not", "approved", "mfa", "method"]),
    ],
)
def test_tokenize_examples(text, expected):
    assert tokenize(text) == expected


def test_plural_folding_rules():
    assert [
        fold_plural(w) for w in ["days", "policies", "requests", "business", "status", "analysis", "has"]
    ] == [
        "day",
        "policy",
        "request",
        "business",
        "status",
        "analysis",
        "has",
    ]


def test_query_and_document_forms_meet():
    # The same rules on both sides make surface variants match.
    assert set(tokenize("work from home")) <= set(tokenize("Work-From-Home Policy"))
    assert "holiday" in tokenize("Public Holidays") and "holiday" in tokenize("a two-week holiday")
    assert "1000" in tokenize("claims above USD 1,000") and "1000" in tokenize("over 1000 dollars")


def test_empty_and_stopword_only_text():
    assert tokenize("") == []
    assert tokenize("what is the of and") == []


def test_tokenize_is_deterministic():
    text = "Passwords must be changed every 180 days (IT-001)."
    assert tokenize(text) == tokenize(text) == ["password", "changed", "180", "day", "it-001"]
