"""Validate (and optionally freeze) evaluation dataset v1.1 = v1.0 without exposed EV-06, plus EV-31.

    .venv/bin/python scripts/validate_eval_dataset_v1_1.py            # validate only
    .venv/bin/python scripts/validate_eval_dataset_v1_1.py --freeze   # validate, then record SHA-256 + timestamp

Independence: imports NO retrieval code. v1.0 (eval_queries_30.json) is only read and must still match
its frozen hash. Freezing refuses to overwrite an existing v1.1 freeze record with a different hash.
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
D = PROJECT_ROOT / "evaluation" / "datasets"
V10, V11, EXPOSED = D / "eval_queries_30.json", D / "eval_queries_v1_1.json", D / "eval_queries_exposed.json"
FREEZE_RECORD = D / "eval_dataset_v1_1_freeze.json"
FREEZE_MANIFEST = PROJECT_ROOT / "manifests" / "eval-dataset-v1_1.sha256"
V10_MANIFEST = PROJECT_ROOT / "manifests" / "eval-dataset.sha256"
CATALOG, DEV, SMOKE = (
    D / "evidence_catalog.json",
    D / "dev_queries.json",
    PROJECT_ROOT / "tests" / "smoke_questions.json",
)
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jaccard(a: str, b: str) -> float:
    wa, wb = (set(re.findall(r"[a-z0-9]+", x.lower())) for x in (a, b))
    return len(wa & wb) / len(wa | wb)


def validate() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    v10_sha = V10_MANIFEST.read_text().split()[0]
    check("v1.0 unchanged (matches manifests/eval-dataset.sha256)", sha(V10) == v10_sha, v10_sha[:12] + "…")
    v10 = {q["id"]: q for q in json.loads(V10.read_text(encoding="utf-8"))["queries"]}
    v11 = json.loads(V11.read_text(encoding="utf-8"))
    qs = v11["queries"]
    ids = [q["id"] for q in qs]
    catalog = {u["evidence_id"]: u for u in json.loads(CATALOG.read_text(encoding="utf-8"))["units"]}
    dev = json.loads(DEV.read_text(encoding="utf-8"))["queries"]
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))["questions"]

    check(
        "version 1.1 and supersedes v1.0 by hash",
        v11["version"] == "1.1" and v11["supersedes"]["sha256"] == v10_sha,
    )
    check("30 queries", len(qs) == 30, str(len(qs)))
    check(
        "IDs = v1.0 IDs minus EV-06 plus EV-31",
        ids == [i for i in v10 if i != "EV-06"] + ["EV-31"],
        f"{len(set(ids))} unique",
    )
    same = [i for i in ids if i in v10 and next(q for q in qs if q["id"] == i) == v10[i]]
    check("29 carried-over queries identical to v1.0 (field by field)", len(same) == 29, f"{len(same)}/29")
    check("EV-06 absent from the clean set", "EV-06" not in ids)
    exposed = json.loads(EXPOSED.read_text(encoding="utf-8"))["queries"]
    check(
        "exposure record holds EV-06 exactly as in v1.0",
        len(exposed) == 1
        and exposed[0]["query"] == v10["EV-06"]
        and exposed[0]["source"]["sha256"] == v10_sha,
    )
    check(
        "25 answerable / 5 unanswerable",
        (sum(q["answerable"] for q in qs), sum(not q["answerable"] for q in qs)) == (25, 5),
    )
    cats = Counter(q["category"] for q in qs)
    check(
        "category mixture unchanged vs v1.0",
        cats == Counter(q["category"] for q in v10.values()),
        f"exact keyword / identifier: {cats['exact keyword / identifier']}",
    )

    q = next(x for x in qs if x["id"] == "EV-31")
    ev_ids = [e for g in q["required_evidence"] for e in g]
    text = " ".join(catalog[e]["text"] for e in ev_ids if e in catalog).lower()
    check("EV-31 has all required fields", set(q) >= REQUIRED_KEYS)
    check(
        "EV-31 evidence IDs exist in the frozen catalog",
        ev_ids and all(e in catalog for e in ev_ids),
        ", ".join(ev_ids),
    )
    check(
        "EV-31 expected policies = evidence policies",
        sorted(q["expected_policy_ids"]) == sorted({catalog[e]["policy_id"] for e in ev_ids}),
    )
    check(
        "EV-31 every answer-term group occurs in the evidence text",
        all(any(t.lower() in text for t in g) for g in q["answer_terms"]),
        str(q["answer_terms"]),
    )
    corpus = " ".join(p.read_text(encoding="utf-8") for p in sorted(POLICY_DIR.glob("*.md")))
    check(
        "EV-31 evidence sentence occurs verbatim in the policy corpus",
        all(catalog[e]["text"] in corpus for e in ev_ids),
    )
    dev_facts = {e for d in dev for g in d["required_evidence"] for e in g}
    other_facts = {e for x in list(v10.values()) for g in x["required_evidence"] for e in g}
    check("EV-31 fact not used by dev queries", not set(ev_ids) & dev_facts)
    check("EV-31 fact not used by any v1.0 query (incl. EV-06)", not set(ev_ids) & other_facts)
    others = (
        [(x["id"], x["question"]) for x in v10.values()]
        + [(d["id"], d["query"]) for d in dev]
        + [(s["id"], s["question"]) for s in smoke]
    )
    worst = max((jaccard(q["question"], t), i) for i, t in others)
    check(
        "EV-31 not a near-duplicate of any dev / smoke / v1.0 question (word Jaccard < 0.6)",
        worst[0] < 0.6,
        f"max {worst[0]:.2f} vs {worst[1]}",
    )
    check("questions unique in v1.1", len({x["question"].lower() for x in qs}) == 30)
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    checks = validate()
    print("== Evaluation dataset v1.1 validation ==")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    ok = all(c[1] for c in checks)
    digest = sha(V11)
    print(f"v1.1 SHA-256: {digest}")
    print(f"VALIDATION: {'PASS' if ok else 'FAIL'}")
    if not ok:
        return 1
    if args.freeze:
        if FREEZE_RECORD.exists():
            old = json.loads(FREEZE_RECORD.read_text())
            if old["sha256"] != digest:
                print(
                    "A v1.1 freeze record with a DIFFERENT hash exists - refusing to re-freeze.",
                    file=sys.stderr,
                )
                return 1
            print(f"Already frozen at {old['frozen_utc']} with the same hash.")
            return 0
        record = {
            "dataset": str(V11.relative_to(PROJECT_ROOT)),
            "version": "1.1",
            "sha256": digest,
            "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "supersedes": {"version": "1.0", "sha256": V10_MANIFEST.read_text().split()[0]},
            "removed": ["EV-06 (exposed in Phase 6)"],
            "added": ["EV-31"],
            "exposure_record": str(EXPOSED.relative_to(PROJECT_ROOT)),
            "exposure_record_sha256": sha(EXPOSED),
            "queries": 30,
            "answerable": 25,
            "unanswerable": 5,
            "validation": "PASS (scripts/validate_eval_dataset_v1_1.py)",
            "retrieval_or_generation_run_on_EV-31": "none",
        }
        FREEZE_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        FREEZE_MANIFEST.write_text(
            f"{digest}  {V11.relative_to(PROJECT_ROOT)}\n"
            f"{sha(EXPOSED)}  {EXPOSED.relative_to(PROJECT_ROOT)}\n",
            encoding="utf-8",
        )
        print(
            f"FROZEN: v1.1 at {record['frozen_utc']} -> {FREEZE_RECORD.relative_to(PROJECT_ROOT)}, "
            f"{FREEZE_MANIFEST.relative_to(PROJECT_ROOT)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
