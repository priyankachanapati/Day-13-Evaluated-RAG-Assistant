"""Canonical evidence units: chunk-independent relevance labels.

Why: chunk IDs such as ``HR-001-CH02`` mean different text under different chunking
strategies, so a label like "the answer is in HR-001-CH02" cannot be used to compare
strategies. Instead, every sentence of every policy section is an **evidence unit** with a
stable ID derived only from the (frozen) corpus:

    HR-001-S02-E01   = policy HR-001, section "## 2. ...", sentence 1 of that section

Labels (queries) refer to evidence IDs. A chunk **contains** an evidence unit only if the
*entire* sentence lies inside the chunk (offset containment). A chunk holding half a sentence
does not count - the fact was split.

Not evidence units: the ``# Title`` line, the synthetic-policy disclaimer (``> ...``) and the
``## `` headings themselves (headings are tracked separately for boundary analysis).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ingestion import PolicyDocument

CATALOG_VERSION = "1.0"
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
HEADING = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)
SECTION_NUMBER = re.compile(r"^(\d+)\.\s")

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
CATALOG_PATH = DATASETS_DIR / "evidence_catalog.json"


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    policy_id: str
    section_number: int
    section_title: str
    sentence_index: int
    text: str
    start: int  # offsets into PolicyDocument.text (the body v1 chunks)
    end: int


@dataclass(frozen=True)
class Heading:
    policy_id: str
    level: int
    text: str
    start: int
    end: int


def headings(doc: PolicyDocument) -> list[Heading]:
    return [
        Heading(doc.policy_id, len(m.group(1)), m.group(0), m.start(), m.end())
        for m in HEADING.finditer(doc.text)
    ]


def build_units(doc: PolicyDocument) -> list[EvidenceUnit]:
    """Split each ``## N. Title`` section of a document body into sentence evidence units."""
    body = doc.text
    marks = [(m.start(), m.end(), m.group(2).strip()) for m in HEADING.finditer(body) if m.group(1) == "##"]
    units: list[EvidenceUnit] = []
    for i, (_, h_end, title) in enumerate(marks):
        number_match = SECTION_NUMBER.match(title)
        if not number_match:
            raise ValueError(f"{doc.policy_id}: section heading without a number: {title!r}")
        number = int(number_match.group(1))
        region_end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        cursor, sentence_index = h_end, 0
        for paragraph in re.split(r"\n\s*\n", body[h_end:region_end]):
            for sentence in SENTENCE_END.split(paragraph.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                start = body.find(sentence, cursor, region_end)
                if start == -1:
                    raise ValueError(f"{doc.policy_id}: sentence not found verbatim: {sentence!r}")
                cursor = start + len(sentence)
                sentence_index += 1
                units.append(
                    EvidenceUnit(
                        evidence_id=f"{doc.policy_id}-S{number:02d}-E{sentence_index:02d}",
                        policy_id=doc.policy_id,
                        section_number=number,
                        section_title=title,
                        sentence_index=sentence_index,
                        text=sentence,
                        start=start,
                        end=cursor,
                    )
                )
    return units


def build_catalog(docs: list[PolicyDocument]) -> list[EvidenceUnit]:
    units = [u for d in docs for u in build_units(d)]
    ids = [u.evidence_id for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence IDs")
    return units


def catalog_document(units: list[EvidenceUnit], corpus_manifest_sha256: str) -> dict:
    """The JSON document that is frozen on disk (no timestamps, so it is reproducible byte for byte)."""
    return {
        "version": CATALOG_VERSION,
        "description": "Canonical evidence units: one per sentence of every numbered policy section.",
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "unit_count": len(units),
        "units": [asdict(u) for u in units],
    }


def dumps(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, EvidenceUnit]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {u["evidence_id"]: EvidenceUnit(**u) for u in data["units"]}


# --------------------------------------------------------------------------- containment
def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def span_contains(chunk_start: int, chunk_end: int, unit: EvidenceUnit) -> bool:
    """True if the whole evidence sentence lies inside the chunk span (same document assumed)."""
    return chunk_start <= unit.start and unit.end <= chunk_end


def chunk_evidence(chunk_metadata: dict, units_by_policy: dict[str, list[EvidenceUnit]]) -> list[str]:
    """Evidence IDs fully contained in a chunk (uses the chunk's start_char/end_char offsets)."""
    start, end = int(chunk_metadata["start_char"]), int(chunk_metadata["end_char"])
    return [
        u.evidence_id
        for u in units_by_policy.get(str(chunk_metadata["policy_id"]), [])
        if span_contains(start, end, u)
    ]


def satisfied_groups(groups: list[list[str]], found: set[str]) -> list[bool]:
    """A required-evidence group is satisfied if ANY of its alternative evidence IDs was found."""
    return [any(e in found for e in group) for group in groups]
