"""Evaluation dataset v1.1 (clean Phase 7 set): v1.0 immutable, EV-06 excluded, EV-31 added."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "evaluation" / "datasets"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_v1_0_is_unchanged_and_v1_1_is_frozen():
    assert (
        sha(D / "eval_queries_30.json") == (ROOT / "manifests" / "eval-dataset.sha256").read_text().split()[0]
    )
    lines = dict(
        reversed(line.split())
        for line in (ROOT / "manifests" / "eval-dataset-v1_1.sha256").read_text().splitlines()
    )
    assert sha(D / "eval_queries_v1_1.json") == lines["evaluation/datasets/eval_queries_v1_1.json"]
    assert sha(D / "eval_queries_exposed.json") == lines["evaluation/datasets/eval_queries_exposed.json"]
    record = json.loads((D / "eval_dataset_v1_1_freeze.json").read_text())
    assert record["sha256"] == sha(D / "eval_queries_v1_1.json") and record["version"] == "1.1"
    assert record["retrieval_or_generation_run_on_EV-31"] == "none"


def test_validator_v1_1_passes():
    spec = importlib.util.spec_from_file_location("v11", ROOT / "scripts" / "validate_eval_dataset_v1_1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    failed = [name for name, ok, _ in mod.validate() if not ok]
    assert failed == []


def test_clean_set_composition():
    v10 = {q["id"]: q for q in json.loads((D / "eval_queries_30.json").read_text())["queries"]}
    v11 = json.loads((D / "eval_queries_v1_1.json").read_text())["queries"]
    ids = [q["id"] for q in v11]
    assert len(ids) == 30 and "EV-06" not in ids and ids[-1] == "EV-31"
    assert all(q == v10[q["id"]] for q in v11 if q["id"] != "EV-31")  # 29 unchanged
    assert sum(q["answerable"] for q in v11) == 25
    ev31 = v11[-1]
    assert ev31["required_evidence"] == [["HR-003-S02-E02"]] and ev31["category"] == v10["EV-06"]["category"]
