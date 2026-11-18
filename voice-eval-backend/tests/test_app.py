"""Smoke tests for the Streamlit app: each tab renders without an exception, with and without data."""
import json
import shutil
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN = ROOT / "batch" / "results" / "zz_apptest"


@pytest.fixture()
def fake_run():
    RUN.mkdir(parents=True)
    row = {"scenario": "cost_quote", "voice": "clip:human:alice", "passed": False, "error": None, "tool_score": 0.5, "tools_missing": ["get_servicing_cost"],
           "forbidden_called": [], "outcome_why": "", "wer_errors": 2, "wer_words": 17, "turns": [], "reply": "Sure.",
           "conversation": [{"role": "caller", "text": "How much?"}, {"role": "agent", "text": "Sure."}], "judge": {"overall": 3.5, "issues": ["no price"]}}
    ok = {**row, "passed": True, "tool_score": 1.0, "tools_missing": []}
    (RUN / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in (row, ok)), encoding="utf-8")
    stats = {"name": "clip:human:alice", "runs": 2, "passed": 1, "pass_rate": 0.5, "tool_score": 0.75, "wer": 0.1, "judge": 3.5, "turns": 2,
             "early_turns": 0, "ttfa_p50": 1200, "ttfa_p95": 1500, "tool_ms": 40, "judge_pass": 0.5}
    (RUN / "summary.json").write_text(json.dumps({"meta": {"voices": ["clip:human:alice"], "scenarios": ["cost_quote"], "repeat": 1, "augment": "clean",
                                                          "noise_snr_db": None}, "by_voice": [stats], "by_scenario": [{**stats, "name": "cost_quote"}]}), encoding="utf-8")
    yield RUN
    shutil.rmtree(RUN, ignore_errors=True)


def run_app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    return at


def test_empty_state_renders():
    at = run_app()
    assert not at.exception
    assert len(at.tabs) == 2


def test_runs_tab_shows_a_batch_run(fake_run):
    at = run_app()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Pass rate"] == "50.0%"
    assert any("FAIL" in e.label for e in at.expander)          # the failed call is listed with its conversation
