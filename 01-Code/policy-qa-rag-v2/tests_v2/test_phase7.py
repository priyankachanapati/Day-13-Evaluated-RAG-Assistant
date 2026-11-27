"""Phase 7 harness: statistics and dataset handling (no retrieval, no LLM)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def p7():
    spec = importlib.util.spec_from_file_location("run_phase7_eval", ROOT / "scripts" / "run_phase7_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sign_test_hand_calculated(p7):
    assert p7.sign_test(5, 0) == pytest.approx(2 / 32)  # n=5: 2 * C(5,0)/2^5
    assert p7.sign_test(3, 1) == pytest.approx(2 * (1 + 4) / 16)  # n=4, k=1
    assert p7.sign_test(2, 2) == 1.0 and p7.sign_test(0, 0) == 1.0


def test_clean_set_excludes_exposed_query(p7):
    clean, exposed = p7.load_sets()
    assert (
        len(clean) == 30 and "EV-06" not in {q["id"] for q in clean} and "EV-31" in {q["id"] for q in clean}
    )
    assert [q["id"] for q in exposed] == ["EV-06"]
