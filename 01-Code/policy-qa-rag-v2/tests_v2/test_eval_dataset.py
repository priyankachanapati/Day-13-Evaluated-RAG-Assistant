"""The frozen 30-query evaluation dataset: freeze integrity, labels and independence."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "datasets" / "eval_queries_30.json"
FREEZE = ROOT / "evaluation" / "datasets" / "eval_dataset_freeze.json"
VALIDATOR = ROOT / "scripts" / "validate_eval_dataset.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_eval_dataset", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_matches_freeze_record_and_manifest():
    sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    record = json.loads(FREEZE.read_text())
    assert record["sha256"] == sha == (ROOT / "manifests" / "eval-dataset.sha256").read_text().split()[0]
    assert record["version"] == "1.0" and record["queries"] == 30
    assert record["retrieval_systems_run_on_dataset_before_freeze"] == "none"


def test_validator_passes():
    checks, info = load_validator().validate()
    failed = [name for name, ok, _ in checks if not ok]
    assert not failed and not info["problems"], (failed, info["problems"])


def test_composition():
    qs = json.loads(DATASET.read_text())["queries"]
    assert sum(q["answerable"] for q in qs) == 25 and sum(not q["answerable"] for q in qs) == 5
    cats = {q["category"] for q in qs}
    for needed in (
        "paraphrase / synonym",
        "exact keyword / identifier",
        "numerical / date",
        "multi-fact",
        "multi-fact, cross-section",
        "near-topic distractor",
    ):
        assert needed in cats
    assert any(len(q["expected_policy_ids"]) > 1 for q in qs)  # at least one cross-policy query


def test_labels_use_canonical_fact_ids_not_chunk_ids():
    text = DATASET.read_text()
    assert "-CH" not in text  # no chunk ID anywhere in the labels
    for q in json.loads(text)["queries"]:
        for group in q["required_evidence"]:
            assert all(len(e.split("-")) == 4 and "-S" in e and "-E" in e for e in group)


def test_dev_queries_are_separate():
    qs = json.loads(DATASET.read_text())["queries"]
    dev = json.loads((ROOT / "evaluation" / "datasets" / "dev_queries.json").read_text())["queries"]
    assert not {q["question"] for q in qs} & {d["query"] for d in dev}
    assert not (
        {e for q in qs for g in q["required_evidence"] for e in g}
        & {e for d in dev for g in d["required_evidence"] for e in g}
    )


def test_validator_imports_no_retrieval_code():
    tree = ast.parse(VALIDATOR.read_text())
    imported = {n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not {
        m
        for m in imported
        if m and m.split(".")[0] in {"rag_v2", "src", "chromadb", "rank_bm25", "sentence_transformers"}
    }
