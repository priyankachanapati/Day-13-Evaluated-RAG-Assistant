"""Guard: the frozen 30-query held-out questions must not appear in code, tests, scripts or results.

Added in Phase 6 after I used held-out question EV-06 (by its text) as a CLI demo query and
in two mocked tests (disclosed in DAILY_PROGRESS and the Phase 6 documentation). Only the dataset
itself, its validator and its documentation/evidence may contain the question texts.
Extended before Phase 7 to dataset v1.1 (EV-31) and the EV-06 exposure record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ["rag_v2", "evaluation", "scripts", "activities", "tests_v2", "results"]
DATASETS = ROOT / "evaluation" / "datasets"
ALLOWED = {
    DATASETS / "eval_queries_30.json",
    DATASETS / "eval_queries_v1_1.json",
    DATASETS / "eval_queries_exposed.json",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def test_no_held_out_question_outside_the_dataset():
    questions = {}
    for name in (
        "eval_queries_30.json",
        "eval_queries_v1_1.json",
    ):  # v1.0 (incl. EV-06) and v1.1 (incl. EV-31)
        for q in json.loads((DATASETS / name).read_text(encoding="utf-8"))["queries"]:
            questions[q["id"]] = norm(q["question"]).rstrip("?")
    offenders = []
    for folder in SCANNED:
        for path in (ROOT / folder).rglob("*"):
            if (
                not path.is_file()
                or path in ALLOWED
                or path.suffix not in {".py", ".json", ".jsonl", ".md", ".txt"}
            ):
                continue
            text = norm(path.read_text(encoding="utf-8", errors="ignore"))
            offenders += [(qid, str(path.relative_to(ROOT))) for qid, q in questions.items() if q in text]
    assert offenders == []
