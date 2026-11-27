"""Unit tests for the Activity 4 Hit Rate@K / MRR function (hand-worked examples)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ACTIVITIES = Path(__file__).resolve().parents[1] / "activities"


def _load(folder: str, filename: str):
    spec = importlib.util.spec_from_file_location(filename[:-3], ACTIVITIES / folder / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A4 = _load("activity-4-hit-rate-mrr", "hit_rate_mrr.py")


def test_hit_rate_and_mrr_hand_example():
    rankings = [["a", "x", "y"], ["x", "b", "y", "z"], ["x", "y", "z", "w", "c"], ["x", "y", "z"]]
    relevant = [{"a"}, {"b"}, {"c"}, {"d"}]
    # ranks of first relevant: 1, 2, 5, none
    assert A4.hit_rate_and_mrr(rankings, relevant, 3) == pytest.approx((2 / 4, (1 + 1 / 2) / 4))
    assert A4.hit_rate_and_mrr(rankings, relevant, 5) == pytest.approx((3 / 4, (1 + 1 / 2 + 1 / 5) / 4))


def test_hit_rate_and_mrr_counts_only_first_relevant():
    assert A4.hit_rate_and_mrr([["x", "a", "b"]], [{"a", "b"}], 3) == pytest.approx((1.0, 0.5))


def test_hit_rate_and_mrr_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        A4.hit_rate_and_mrr([["a"]], [], 3)
