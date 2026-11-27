"""The Phase 6 dev-query runner: mocked end-to-end run, resume, quota interruption, missing key, guards."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.generation import QuotaExhaustedError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_dev_answer_eval", ROOT / "scripts" / "run_dev_answer_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(runner, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_dev_answer_eval.py", *args])
    return runner.main()


def test_only_development_queries_and_probes(runner):
    qs = runner.load_queries()
    assert [q["id"] for q in qs] == [
        "DEV-01",
        "DEV-02",
        "DEV-03",
        "DEV-04",
        "DEV-05",
        "SM-13",
        "SM-14",
        "SM-15",
    ]
    assert not any(q["id"].startswith("EV-") for q in qs)


def test_mock_run_end_to_end_then_resume(runner, monkeypatch, tmp_path):
    root = tmp_path / "out"
    assert (
        run(
            runner,
            monkeypatch,
            "--run-name",
            "t",
            "--mock",
            "--results-root",
            str(root),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--configs",
            "fixed_500+dense",
            "recursive_800_100+hybrid",
        )
        == 0
    )
    lines = (root / "t" / "records.jsonl").read_text().splitlines()
    assert len(lines) == 16  # 2 configs x (5 dev + 3 probes), saved incrementally
    summary = json.loads((root / "t" / "summary.json").read_text())
    assert set(summary["summary"]) == {"fixed_500+dense", "recursive_800_100+hybrid"}
    assert "MOCK" in summary["meta"]["llm_kind"] and (root / "t" / "summary.md").exists()
    # resume: nothing left to do, no duplicates
    assert (
        run(
            runner,
            monkeypatch,
            "--run-name",
            "t",
            "--mock",
            "--resume",
            "--results-root",
            str(root),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--configs",
            "fixed_500+dense",
            "recursive_800_100+hybrid",
        )
        == 0
    )
    assert len((root / "t" / "records.jsonl").read_text().splitlines()) == 16


def test_quota_interruption_keeps_completed_work(runner, monkeypatch, tmp_path):
    calls = {"n": 0}
    original = runner.MockLLM.complete

    def flaky(self, s, u):
        calls["n"] += 1
        if calls["n"] > 2:
            raise QuotaExhaustedError("Groq quota exhausted: tokens per day")
        return original(self, s, u)

    monkeypatch.setattr(runner.MockLLM, "complete", flaky)
    root = tmp_path / "out"
    code = run(
        runner,
        monkeypatch,
        "--run-name",
        "q",
        "--mock",
        "--results-root",
        str(root),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--configs",
        "fixed_500+bm25",
    )
    assert code == 2
    records = (root / "q" / "records.jsonl").read_text().splitlines()
    assert len(records) == 2  # the two completed answers were saved before the interruption
    meta = json.loads((root / "q" / "summary.json").read_text())["meta"]
    assert meta["status"].startswith("incomplete: provider quota exhausted")


def test_missing_key_exits_cleanly_without_secret(runner, monkeypatch, tmp_path, capsys):
    import dataclasses

    real = runner.load_settings
    monkeypatch.setattr(runner, "load_settings", lambda: dataclasses.replace(real(), groq_api_key=None))
    assert run(runner, monkeypatch, "--run-name", "k", "--results-root", str(tmp_path)) == 3
    err = capsys.readouterr().err
    assert "GROQ_API_KEY is not set" in err and "gsk_" not in err
    assert not (tmp_path / "k" / "records.jsonl").exists()  # nothing attempted


def _auth_error():
    import groq
    import httpx

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(401, json={"error": {"message": "Invalid API Key"}}, request=req)
    return groq.AuthenticationError("invalid key", response=resp, body=None)


def test_authentication_failure_stops_after_first_attempt(runner, monkeypatch, tmp_path):
    from src.generation import GenerationError

    def reject(self, s, u):
        raise GenerationError("Groq rejected the API key (authentication error).") from _auth_error()

    monkeypatch.setattr(runner.MockLLM, "complete", reject)
    root = tmp_path / "out"
    code = run(
        runner,
        monkeypatch,
        "--run-name",
        "a",
        "--mock",
        "--results-root",
        str(root),
        "--cache-dir",
        str(tmp_path / "c"),
        "--configs",
        "fixed_500+dense",
        "fixed_500+bm25",
    )
    assert code == 3
    assert len((root / "a" / "records.jsonl").read_text().splitlines()) == 1  # stopped after the first item
    meta = json.loads((root / "a" / "summary.json").read_text())["meta"]
    assert meta["status"] == "incomplete: authentication failed (Groq rejected the API key)"


def test_errors_mean_incomplete_then_resume_retries_them(runner, monkeypatch, tmp_path):
    from src.generation import GenerationError

    original = runner.MockLLM.complete
    monkeypatch.setattr(
        runner.MockLLM, "complete", lambda self, s, u: (_ for _ in ()).throw(GenerationError("503"))
    )
    root, args = (
        tmp_path / "out",
        [
            "--mock",
            "--results-root",
            str(tmp_path / "out"),
            "--cache-dir",
            str(tmp_path / "c"),
            "--configs",
            "fixed_500+dense",
        ],
    )
    assert run(runner, monkeypatch, "--run-name", "e", *args) == 2  # errors are never "complete"
    meta = json.loads((root / "e" / "summary.json").read_text())["meta"]
    assert meta["status"].startswith("incomplete:") and "failed with errors" in meta["status"]
    errors_before = sum(
        "error" in json.loads(x) for x in (root / "e" / "records.jsonl").read_text().splitlines()
    )
    assert errors_before >= 5  # the dev queries reached the LLM and failed
    monkeypatch.setattr(runner.MockLLM, "complete", original)
    assert run(runner, monkeypatch, "--run-name", "e", "--resume", *args) == 0  # failed items retried
    lines = [json.loads(x) for x in (root / "e" / "records.jsonl").read_text().splitlines()]
    assert len(runner.latest(lines)) == 8 and not any("error" in r for r in runner.latest(lines))
    summary = json.loads((root / "e" / "summary.json").read_text())
    assert summary["meta"]["status"] == "complete" and summary["summary"]["fixed_500+dense"]["errors"] == 0


def test_summarize_only_makes_no_llm_or_api_client(runner, monkeypatch, tmp_path):
    root = tmp_path / "out"
    assert (
        run(
            runner,
            monkeypatch,
            "--run-name",
            "s",
            "--mock",
            "--results-root",
            str(root),
            "--cache-dir",
            str(tmp_path / "c"),
            "--configs",
            "fixed_500+dense",
        )
        == 0
    )
    monkeypatch.setattr(runner, "GroqGenerator", lambda *a, **k: pytest.fail("no client may be created"))
    monkeypatch.setattr(runner, "CachingLLM", lambda *a, **k: pytest.fail("no LLM may be created"))
    assert run(runner, monkeypatch, "--run-name", "s", "--summarize-only", "--results-root", str(root)) == 0
    meta = json.loads((root / "s" / "summary.json").read_text())["meta"]
    assert meta["status"] == "complete" and "no API call" in meta["summary_rebuilt_note"]
    (root / "s" / "summary.json").unlink()  # also works without an existing summary
    assert run(runner, monkeypatch, "--run-name", "s", "--summarize-only", "--results-root", str(root)) == 0
    assert json.loads((root / "s" / "summary.json").read_text())["meta"]["run_name"] == "s"
