"""Unit tests: spoken numbers and WER, the static and dynamic users, the judge, summaries, the report, and the stored clips."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from sim import user, clips, judge, tts  # noqa: E402
from sim.report import render_report  # noqa: E402
from sim.run import aggregate, summarize  # noqa: E402
from sim.text import normalize, spoken, wer  # noqa: E402
from fake_agent import FakeStrandsAgent  # noqa: E402

SCENARIO = user.load_scenarios()["ac_not_cooling_book"]


# ------------------------------------------------------------------- text ---
def test_spoken_turns_digit_runs_into_words():
    assert spoken("My number is 555-0100 and the room is 42.") == "My number is five five five zero one zero zero and the room is four two."
    assert spoken("No digits here") == "No digits here"
    assert spoken("five five five") == "five five five"


def test_wer_treats_spoken_and_written_digits_alike():
    assert normalize("five five five, zero one zero zero") == ["5550100"]
    assert wer("my phone is five five five zero one zero zero", "my phone is 5550100")["wer"] == 0
    r = wer("book an inspection for my air conditioner", "book an inspection for my heater")
    assert r["errors"] == 2 and r["words"] == 7                        # "air conditioner" -> "heater": 1 substitution + 1 deletion
    assert wer("hello there", "")["wer"] == 1.0 and wer("", "x")["wer"] is None


# ---------------------------------------------------------------- users ---
def test_static_user_plays_its_lines_then_hangs_up():
    c = user.StaticUser(SCENARIO)
    n = len(SCENARIO["static_turns"])
    assert c.next_line([]) == SCENARIO["static_turns"][0]
    assert c.next_line([{"user": "x", "agent": "y"}]) == SCENARIO["static_turns"][1]
    assert c.next_line([{"user": "x", "agent": "y"}] * n) is None


def attach(agent, **kw):
    u = user.DynamicUser(SCENARIO, agent=agent, **kw)
    agent.user = u
    return u


def test_dynamic_user_feeds_the_agent_and_stops_when_it_hangs_up():
    agent = FakeStrandsAgent(["Hi, my air conditioner is not cooling.", "My number is 555-0100.", "HANGUP: Great, thank you."])
    u = attach(agent)
    history = []
    line = u.next_line(history)
    assert line == "Hi, my air conditioner is not cooling." and agent.prompts[0] == user.FIRST
    history.append({"user": line, "agent": "Sure. What is your phone number?"})
    line = u.next_line(history)
    assert line == "My number is five five five zero one zero zero."                      # digits are spoken, not read as a number
    assert agent.prompts[1] == "The assistant says: Sure. What is your phone number?"       # only the newest agent reply is passed in
    history.append({"user": line, "agent": ""})
    assert u.next_line(history) == "Great, thank you."                                      # the tool's farewell is the last line
    assert "(the assistant said nothing)" in agent.prompts[2]
    assert u.next_line(history) is None and len(agent.prompts) == 3                         # hung up: the agent is not asked again


def test_dynamic_user_respects_max_turns_and_empty_output():
    capped = user.DynamicUser({**SCENARIO, "max_turns": 1}, agent=FakeStrandsAgent(["Hello."]))
    assert capped.next_line([{"user": "a", "agent": "b"}]) is None
    assert user.DynamicUser(SCENARIO, agent=FakeStrandsAgent(["   "])).next_line([]) is None


def test_system_prompt_carries_the_scenario_and_the_hangup_tool():
    p = user.system_prompt(SCENARIO)
    assert "Sam Rivera" in p and "five five five, zero one zero zero" in p and SCENARIO["goal"] in p and "end_conversation" in p


def test_default_agent_is_a_strands_agent_with_the_hangup_tool(monkeypatch):
    from sim import config
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-key")
    config.get_settings.cache_clear()
    try:
        u = user.DynamicUser(SCENARIO)                       # builds the Strands Agent; no network until it is called
        assert type(u._agent).__module__.startswith("strands") and "end_conversation" in u._agent.tool_names
        u._agent.tool.end_conversation(farewell="Bye now.")  # the tool the model would call
        assert u._farewell == "Bye now."
    finally:
        config.get_settings.cache_clear()


def test_default_agent_needs_the_bedrock_key(monkeypatch):
    from sim import config
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
    config.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
            user.DynamicUser(SCENARIO)
    finally:
        config.get_settings.cache_clear()


def test_scenarios_file_is_valid():
    sc = user.load_scenarios()
    assert len(sc) >= 8 and all(s["static_turns"] and s["expect"] for s in sc.values())


# ------------------------------------------------------------------ judge ---
TURNS = [{"said": "Book it.", "heard": "book it", "reply": "Booked, request SR-1001.",
          "tools": [{"name": "create_service_request", "args": {"item_id": "AC-100"}, "result": {"request_id": "SR-1001"}, "ms": 9}]}]
GOOD = json.dumps({"scores": {k: 5 for k in judge.CRITERIA}, "issues": [], "reason": "Fine."})


def test_judge_sees_tool_results_and_scores_a_pass():
    seen = {}

    def fake(system, messages, model, *a, **k):
        seen["user"] = messages[0]["content"][0]["text"]
        return "Here you go: " + GOOD

    r = judge.evaluate(SCENARIO, TURNS, complete=fake, model="m")
    assert r["status"] == "ok" and r["pass"] is True and r["overall"] == 5.0
    assert "SR-1001" in seen["user"] and "create_service_request" in seen["user"] and SCENARIO["expect"] in seen["user"]


def test_judge_fails_a_session_with_one_low_score_and_clamps_wild_values():
    bad = json.loads(GOOD)
    bad["scores"]["accuracy"] = 2
    bad["scores"]["tone"] = 99
    bad["issues"] = ["invented a price"]
    r = judge.evaluate(SCENARIO, TURNS, complete=lambda *a, **k: json.dumps(bad))
    assert r["pass"] is False and r["scores"]["accuracy"] == 2 and r["scores"]["tone"] == 5 and r["issues"] == ["invented a price"]


def test_judge_never_raises():
    assert judge.evaluate(SCENARIO, TURNS, complete=lambda *a, **k: "not json")["status"] == "error"
    assert judge.evaluate(SCENARIO, TURNS, complete=lambda *a, **k: json.dumps({"scores": {"tone": 5}}))["status"] == "error"
    assert judge.evaluate(SCENARIO, [])["status"] == "error"

    def boom(*a, **k):
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set in .env")

    assert "AWS_BEARER_TOKEN_BEDROCK" in judge.evaluate(SCENARIO, TURNS, complete=boom)["reason"]


# -------------------------------------------------------- summary + report ---
def result(scenario, mode, passed, overall=None, error=None, reply="Hello <script>alert(1)</script>"):
    j = None if overall is None else {"status": "ok", "scores": {k: overall for k in judge.CRITERIA}, "overall": overall, "pass": passed,
                                      "issues": ["too long"], "reason": "ok"}
    t = {"n": 0, "said": "hi", "heard": "hi", "reply": reply, "tools": [], "ttfa_ms": 1200, "wer": {"wer": 0.0, "errors": 0, "words": 1}}
    return {"scenario": scenario, "title": scenario, "mode": mode, "voice": "v", "error": error, "turns": [t], "judge": j, "passed": passed,
            "dir": f"01_{scenario}_{mode}", "duration_s": 12.0}


def test_summary_counts_passes_errors_and_means():
    rs = [result("a", "static", True, 5.0), result("a", "dynamic", False, 3.0), result("b", "static", False, error="boom")]
    s = summarize(rs)
    assert s["overall"]["sessions"] == 3 and s["overall"]["passed"] == 1 and s["overall"]["errors"] == 1
    assert s["overall"]["judge_overall"] == 4.0 and s["overall"]["ttfa_p50_ms"] == 1200
    assert s["by_mode"]["static"]["sessions"] == 2 and s["by_scenario"]["a"]["pass_rate"] == 0.5
    assert aggregate([])["pass_rate"] is None


def test_report_escapes_transcripts_and_links_the_audio():
    rs = [result("a", "static", True, 5.0), result("b", "dynamic", False, 3.0, error="the agent never answered")]
    meta = {"created": "2026-11-18T10:00:00", "url": "ws://x/ws", "mode": "both", "voice": "v", "repeat": 1, "judge_model": "m", "user_model": "m"}
    html = render_report(rs, {"meta": meta, **summarize(rs)})
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "01_a_static/user.wav" in html and "01_b_dynamic/agent.wav" in html and "never answered" in html and "1/2 passed" in html


# ------------------------------------------------------------ stored clips ---
def test_every_static_line_has_a_verified_stored_clip():
    for s in user.load_scenarios().values():
        for i, text in enumerate(s["static_turns"]):
            pcm = clips.get(s["id"], i, text, clips.DEFAULT_VOICE)        # also verifies the checksum
            assert 0.5 < len(pcm) / 2 / 16000 < 30


def test_a_changed_line_is_refused_with_the_fix():
    with pytest.raises(LookupError, match="older line"):
        clips.get("ac_not_cooling_book", 0, "a completely different sentence", clips.DEFAULT_VOICE)
    with pytest.raises(LookupError, match="no stored clip"):
        clips.get("ac_not_cooling_book", 0, SCENARIO["static_turns"][0], "en_GB-nobody-medium")


@pytest.mark.skipif(tts.available(clips.DEFAULT_VOICE) is not None, reason="Piper voice not downloaded")
def test_piper_speaks_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "CACHE", tmp_path)
    pcm = tts.synthesize(clips.DEFAULT_VOICE, "Hello, this is a test.")
    assert 1.0 < len(pcm) / 2 / 16000 < 6 and list(tmp_path.glob("*.pcm"))
    assert tts.synthesize(clips.DEFAULT_VOICE, "Hello, this is a test.") == pcm
