"""Deliberate tokenizer for BM25 keyword retrieval (Activity B).

BM25 only matches exact tokens, so tokenization decides what can match. The rules were
fixed before any BM25 result was seen:

1. Lowercase everything, but first remember which tokens were ALL-CAPS acronyms
   (IT, HR, MFA, VPN, USB, USD, ID): they are never removed as stopwords, so
   "IT Service Desk" keeps "it" while the pronoun in "it applies" is dropped.
2. Policy codes such as ``HR-003`` / ``IT-001`` stay ONE atomic token (``hr-003``).
3. Other hyphenated terms emit the compound and its parts:
   ``work-from-home`` -> ``work-from-home``, ``work``, ``home`` (``from`` is a stopword),
   so both "work-from-home" and "work from home" match.
4. Numbers: thousands separators removed (``1,000`` -> ``1000``); decimals (``0.40``) and
   times (``10:00``) stay single tokens.
5. E-mail addresses are kept whole and also split on ``@ . -`` into parts.
6. Possessive ``'s`` is removed (``desk's`` -> ``desk``).
7. Light plural folding for alphabetic words longer than 3 letters: ``-ies`` -> ``-y``
   (``policies`` -> ``policy``), trailing ``-s`` dropped unless the word ends in
   ``ss``/``us``/``is`` (``days`` -> ``day``, ``business`` unchanged). No stemming library.
8. A small explicit English stopword list (below) is removed. Negations (``not``, ``no``)
   are kept.

The same function tokenizes chunks and queries, so both sides are normalised identically.
"""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but of to in on at by for from with without as is are was were be been being
    it its this that these those i me my we our you your he she they them their his her
    do does did can could may might must shall should will would have has had
    what which who whom whose when where why how if then than so such there here
    any all each every also only into about up down out over under again very just
    get got am""".split()  # noqa: SIM905 - a readable word block is intended
)

POLICY_CODE = re.compile(r"^[a-z]{2,5}-\d{3}$")
TOKEN = re.compile(
    r"""
    (?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})
  | (?P<compound>[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)
  | (?P<time>\d{1,2}:\d{2})
  | (?P<thousands>\d{1,3}(?:,\d{3})+(?:\.\d+)?)
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)
    """,
    re.VERBOSE,
)


def fold_plural(word: str) -> str:
    """Light, rule-based plural folding (documented rule 7)."""
    if not word.isalpha() or len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _word_tokens(raw: str) -> list[str]:
    acronym = raw.isupper() and len(raw) >= 2
    word = raw.lower()
    if word.endswith("'s"):
        word = word[:-2]
    word = word.replace("'", "")
    if not word or (word in STOPWORDS and not acronym):
        return []
    return [word if acronym else fold_plural(word)]


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 according to the rules in the module docstring."""
    tokens: list[str] = []
    for m in TOKEN.finditer(text):
        kind, raw = m.lastgroup, m.group(0)
        if kind == "email":
            tokens.append(raw.lower())
            for part in re.split(r"[@.\-]", raw):
                tokens.extend(_word_tokens(part))
        elif kind == "compound":
            compound = raw.lower()
            tokens.append(compound)
            if not POLICY_CODE.match(compound):
                for part in raw.split("-"):
                    tokens.extend(_word_tokens(part) if part.isalpha() else [part.lower()])
        elif kind == "thousands":
            tokens.append(raw.replace(",", ""))
        elif kind in ("time", "number"):
            tokens.append(raw)
        else:
            tokens.extend(_word_tokens(raw))
    return tokens
