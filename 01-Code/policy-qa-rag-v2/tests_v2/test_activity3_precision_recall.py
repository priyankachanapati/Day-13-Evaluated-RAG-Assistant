"""Unit tests for the Activity 3 Precision@5 / Recall@5 function (hand-worked examples)."""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

ACTIVITIES = Path(__file__).resolve().parents[1] / "activities"


def _load(folder: str, filename: str):
    spec = importlib.util.spec_from_file_location(filename[:-3], ACTIVITIES / folder / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A3 = _load("activity-3-precision-recall-at-5", "precision_recall_at_5.py")


def test_precision_recall_at_5_hand_example():
    m = A3.precision_recall(["a", "x", "b", "y", "z", "c"], {"a", "b", "c"}, 5)
    assert m["top_k"] == ["a", "x", "b", "y", "z"]
    assert m["intersection"] == ["a", "b"]
    assert (m["num"], m["p_den"], m["r_den"]) == (2, 5, 3)
    assert m["precision"] == Fraction(2, 5) and m["recall"] == Fraction(2, 3)


def test_precision_recall_no_relevant_retrieved():
    m = A3.precision_recall(["x", "y", "z", "w", "v"], {"a"}, 5)
    assert m["precision"] == 0 and m["recall"] == 0 and m["intersection"] == []
