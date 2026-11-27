"""Validate (and optionally freeze) the 30-query held-out evaluation dataset.

    .venv/bin/python scripts/validate_eval_dataset.py            # validate only
    .venv/bin/python scripts/validate_eval_dataset.py --freeze   # validate, then record SHA-256 + timestamp

Independence: this script imports NO retrieval code (no embeddings, BM25, RRF, reranker). It
checks labels only against the frozen corpus, the frozen evidence catalog and the dev queries.
Freezing refuses to overwrite an existing freeze record with a different hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT_ROOT / "evaluation" / "datasets"
DATASET = DATASETS / "eval_queries_30.json"
FREEZE_RECORD = DATASETS / "eval_dataset_freeze.json"
FREEZE_MANIFEST = PROJECT_ROOT / "manifests" / "eval-dataset.sha256"
CATALOG = DATASETS / "evidence_catalog.json"
DEV = DATASETS / "dev_queries.json"
SMOKE = PROJECT_ROOT / "tests" / "smoke_questions.json"
POLICY_DIR = PROJECT_ROOT / "data" / "policies"

REQUIRED_KEYS = {
    "id",
    "question",
    "answerable",
    "category",
    "difficulty",
    "expected_policy_ids",
    "required_evidence",
    "answer_terms",
    "reference_answer",
    "distractor_policy_ids",
    "notes",
}
POLICIES = {"HR-001", "HR-002", "HR-003", "HR-004", "IT-001", "IT-002"}
CATEGORY_MINIMUMS = {  # the mixture required by the assignment brief
    "paraphrase / synonym": 3,
    "exact keyword / identifier": 3,
    "numerical / date": 3,
    "multi-fact": 2,
    "multi-fact, cross-section": 2,
    "near-topic distractor": 2,
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def validate() -> tuple[list[tuple[str, bool, str]], dict]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    catalog = {u["evidence_id"]: u for u in json.loads(CATALOG.read_text(encoding="utf-8"))["units"]}
    dev = json.loads(DEV.read_text(encoding="utf-8"))["queries"]
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))["questions"]
    corpus = " ".join(p.read_text(encoding="utf-8") for p in sorted(POLICY_DIR.glob("*.md"))).lower()
    qs = data["queries"]
    checks: list[tuple[str, bool, str]] = []
    problems: list[str] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- schema and counts
    check("version present", data.get("version") == "1.0", data.get("version", "missing"))
    check("30 queries", len(qs) == 30, str(len(qs)))
    check(
        "IDs EV-01..EV-30, unique, in order", [q["id"] for q in qs] == [f"EV-{i:02d}" for i in range(1, 31)]
    )
    missing = [q["id"] for q in qs if not set(q) >= REQUIRED_KEYS]
    check("every query has all required fields", not missing, ", ".join(missing) or "all fields present")
    answerable = [q for q in qs if q["answerable"]]
    unanswerable = [q for q in qs if not q["answerable"]]
    check(
        "25 answerable / 5 unanswerable",
        (len(answerable), len(unanswerable)) == (25, 5),
        f"{len(answerable)} / {len(unanswerable)}",
    )
    cats = Counter(q["category"] for q in qs)
    short = {c: n for c, n in CATEGORY_MINIMUMS.items() if cats.get(c, 0) < n}
    check(
        "category mixture meets minimums", not short, "; ".join(f"{c}: {n}" for c, n in sorted(cats.items()))
    )
    check("questions unique", len({q["question"].strip().lower() for q in qs}) == 30)
    check(
        "difficulty labels valid",
        all(q["difficulty"] in {"easy", "medium", "hard"} for q in qs),
        str(dict(Counter(q["difficulty"] for q in qs))),
    )

    # --- answerable labels against the corpus
    for q in answerable:
        ids = [e for g in q["required_evidence"] for e in g]
        if not q["required_evidence"] or not all(q["required_evidence"]):
            problems.append(f"{q['id']}: empty evidence group")
        unknown = [e for e in ids if e not in catalog]
        if unknown:
            problems.append(f"{q['id']}: unknown evidence IDs {unknown}")
            continue
        if sorted(q["expected_policy_ids"]) != sorted({catalog[e]["policy_id"] for e in ids}):
            problems.append(f"{q['id']}: expected_policy_ids {q['expected_policy_ids']} != evidence policies")
        evidence_text = " ".join(catalog[e]["text"] for e in ids).lower()
        if not q["answer_terms"]:
            problems.append(f"{q['id']}: no answer terms")
        for group in q["answer_terms"]:
            if not any(t.lower() in evidence_text for t in group):
                problems.append(f"{q['id']}: answer term group {group} not found in required evidence text")
        if not q["reference_answer"].strip():
            problems.append(f"{q['id']}: empty reference answer")
    # --- unanswerable labels
    for q in unanswerable:
        if q["required_evidence"] or q["answer_terms"] or q["expected_policy_ids"]:
            problems.append(
                f"{q['id']}: unanswerable query must have no evidence/answer terms/expected policies"
            )
        if not q.get("absent_terms"):
            problems.append(f"{q['id']}: unanswerable query needs absent_terms")
        present = [t for t in q.get("absent_terms", []) if t.lower() in corpus]
        if present:
            problems.append(f"{q['id']}: 'absent' terms found in corpus: {present}")
        if not q["reference_answer"].lower().startswith("refusal"):
            problems.append(f"{q['id']}: reference answer should state the expected refusal")
    bad_distractors = [q["id"] for q in qs if not set(q["distractor_policy_ids"]) <= POLICIES]
    if bad_distractors:
        problems.append(f"unknown distractor policy IDs in {bad_distractors}")
    check(
        "answerable labels: evidence IDs exist, policies match, answer terms occur in evidence",
        not [p for p in problems if any(p.startswith(q["id"]) for q in answerable)],
        f"{len(answerable)} queries, {sum(len(q['required_evidence']) for q in answerable)} fact groups, "
        f"{len({e for q in answerable for g in q['required_evidence'] for e in g})} distinct evidence IDs",
    )
    check(
        "unanswerable labels: no evidence, absent topic terms not in corpus",
        not [p for p in problems if any(p.startswith(q["id"]) for q in unanswerable)],
        ", ".join(f"{q['id']}: {q['absent_terms']}" for q in unanswerable),
    )
    check("distractor policy IDs valid", not bad_distractors)

    # --- independence from development (tuning) data and Day 12 smoke questions
    dev_facts = {e for d in dev for g in d["required_evidence"] for e in g}
    shared = sorted(dev_facts & {e for q in qs for g in q["required_evidence"] for e in g})
    check("no evidence fact shared with the 5 dev queries", not shared, ", ".join(shared) or "0 shared")
    sim_dev = max((jaccard(q["question"], d["query"]), q["id"], d["id"]) for q in qs for d in dev)
    sim_smoke = max((jaccard(q["question"], s["question"]), q["id"], s["id"]) for q in qs for s in smoke)
    check(
        "no near-duplicate of a dev question (word Jaccard < 0.6)",
        sim_dev[0] < 0.6,
        f"max {sim_dev[0]:.2f} ({sim_dev[1]} vs {sim_dev[2]})",
    )
    check(
        "no near-duplicate of a Day 12 smoke question (word Jaccard < 0.6)",
        sim_smoke[0] < 0.6,
        f"max {sim_smoke[0]:.2f} ({sim_smoke[1]} vs {sim_smoke[2]})",
    )

    # informational: facts shared with Day 12 smoke questions (term -> evidence unit in the expected policy)
    smoke_overlap = {}
    for q in answerable:
        ids = {e for g in q["required_evidence"] for e in g}
        for s in smoke:
            if not s["answerable"]:
                continue
            units = {
                eid
                for eid, u in catalog.items()
                if u["policy_id"] == s["expected_policy_id"]
                and any(
                    any(len(t) >= 3 and t.lower() in u["text"].lower() for t in grp)
                    for grp in s["required_facts"]
                )
            }  # terms < 3 chars (e.g. "5") match too broadly
            if ids & units:
                smoke_overlap.setdefault(q["id"], []).append(s["id"])
    info = {
        "categories": dict(cats),
        "difficulty": dict(Counter(q["difficulty"] for q in qs)),
        "policies_covered": dict(Counter(p for q in answerable for p in q["expected_policy_ids"])),
        "smoke_fact_overlap": smoke_overlap,
        "problems": problems,
    }
    return checks, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    checks, info = validate()
    print("== Evaluation dataset validation ==")
    print(f"Dataset: {DATASET.relative_to(PROJECT_ROOT)}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    for p in info["problems"]:
        print(f"  PROBLEM: {p}")
    print(f"  Policies covered (answerable): {info['policies_covered']}")
    print(f"  Difficulty: {info['difficulty']}")
    print(f"  Facts shared with Day 12 smoke questions (informational): {info['smoke_fact_overlap']}")
    ok = all(c[1] for c in checks) and not info["problems"]
    sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    print(f"Dataset SHA-256: {sha}")
    print(f"VALIDATION: {'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1
    if args.freeze:
        if FREEZE_RECORD.exists():
            old = json.loads(FREEZE_RECORD.read_text())
            if old["sha256"] != sha:
                print(
                    "A freeze record with a DIFFERENT hash exists - refusing to re-freeze.", file=sys.stderr
                )
                return 1
            print(f"Already frozen at {old['frozen_utc']} with the same hash.")
            return 0
        data = json.loads(DATASET.read_text(encoding="utf-8"))
        record = {
            "dataset": str(DATASET.relative_to(PROJECT_ROOT)),
            "version": data["version"],
            "sha256": sha,
            "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "queries": len(data["queries"]),
            "answerable": sum(q["answerable"] for q in data["queries"]),
            "unanswerable": sum(not q["answerable"] for q in data["queries"]),
            "categories": info["categories"],
            "difficulty": info["difficulty"],
            "validation": "PASS (scripts/validate_eval_dataset.py)",
            "retrieval_systems_run_on_dataset_before_freeze": "none",
        }
        FREEZE_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        FREEZE_MANIFEST.write_text(f"{sha}  {DATASET.relative_to(PROJECT_ROOT)}\n", encoding="utf-8")
        print(
            f"FROZEN: version {record['version']} at {record['frozen_utc']} -> "
            f"{FREEZE_RECORD.relative_to(PROJECT_ROOT)}, {FREEZE_MANIFEST.relative_to(PROJECT_ROOT)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
