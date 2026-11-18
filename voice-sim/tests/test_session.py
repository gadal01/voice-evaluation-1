"""The simulated user against a fake agent: a full session, its recordings, and the ways a session can go wrong."""
import asyncio
import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_agent import API_KEY, FakeStrandsAgent, make_agent, serve, tone  # noqa: E402

from sim import judge  # noqa: E402
from sim.user import load_scenarios  # noqa: E402
from sim.run import run_session, summarize  # noqa: E402

SCENARIO = load_scenarios()["ac_not_cooling_book"]
GOOD_JUDGE = json.dumps({"scores": {k: 5 for k in judge.CRITERIA}, "issues": [], "reason": "Fine."})
BOOKING = {"name": "create_service_request", "args": {"item_id": "AC-100"}, "result": {"request_id": "SR-1001"}, "ms": 7}


def script(heard_first=None):
    return [{"heard": heard_first or SCENARIO["static_turns"][0], "reply": "That comes to 53 dollars. Shall I book it?", "audio_s": 1.0},
            {"heard": SCENARIO["static_turns"][1], "reply": "Booked, request SR-1001.", "tool": BOOKING, "audio_s": 1.0}]


@pytest.fixture()
def agent():
    servers = []

    def start(script_, silent=False):
        server, port = serve(make_agent(script_, silent))
        servers.append(server)
        return f"ws://127.0.0.1:{port}/ws"

    yield start
    for s in servers:
        s.should_exit = True


def speaker(i, text):
    return tone(1.0, 16000)                       # a one-second tone stands in for Piper: the agent only needs audio


def open_session(tmp_path, url, **kw):
    kw = {"url": url, "api_key": API_KEY, "settle": 0.5, "speaker": speaker, **kw}
    return asyncio.run(run_session(SCENARIO, "static", "v", tmp_path / "01_session", **kw))


def judge_with(monkeypatch, reply=GOOD_JUDGE):
    monkeypatch.setattr(judge.llm, "complete", lambda *a, **k: reply)


def test_full_static_session_is_recorded_and_judged(agent, tmp_path, monkeypatch):
    judge_with(monkeypatch)
    r = open_session(tmp_path, agent(script()))
    assert r["error"] is None and r["passed"] is True and r["judge"]["status"] == "ok"
    t0, t1 = r["turns"]
    assert t0["reply"].startswith("That comes to 53") and t0["heard"] == SCENARIO["static_turns"][0] and t0["wer"]["wer"] == 0
    assert t1["tools"][0]["name"] == "create_service_request" and t1["tools"][0]["result"] == {"request_id": "SR-1001"}
    assert all(t["answered"] and t["ttfa_ms"] is not None and 400 < t["ttfa_ms"] < 3000 for t in r["turns"])   # 0.6 s of user silence + agent
    assert t0["agent_audio_s"] == pytest.approx(1.0, abs=0.05)
    d = tmp_path / "01_session"
    with wave.open(str(d / "user.wav")) as c, wave.open(str(d / "agent.wav")) as a:
        assert c.getframerate() == 16000 and a.getframerate() == 24000
        assert c.getnframes() / 16000 == pytest.approx(a.getnframes() / 24000, abs=0.02)      # one shared timeline
        assert c.getnframes() / 16000 == pytest.approx(r["duration_s"], abs=0.05)
    saved = json.loads((d / "session.json").read_text())
    assert saved["passed"] is True and saved["turns"][1]["reply"] == "Booked, request SR-1001."
    assert summarize([r])["overall"]["passed"] == 1


def test_recognition_errors_show_up_as_heard_wer(agent, tmp_path, monkeypatch):
    judge_with(monkeypatch)
    r = open_session(tmp_path, agent(script(heard_first="book an inspection for my heater")))
    assert r["turns"][0]["wer"]["wer"] > 0.5 and r["turns"][1]["wer"]["wer"] == 0
    assert summarize([r])["overall"]["heard_wer"] > 0


def test_a_judge_failure_fails_the_session(agent, tmp_path, monkeypatch):
    bad = json.dumps({"scores": {**{k: 5 for k in judge.CRITERIA}, "accuracy": 1}, "issues": ["invented price"], "reason": "x"})
    judge_with(monkeypatch, bad)
    r = open_session(tmp_path, agent(script()))
    assert r["error"] is None and r["passed"] is False and r["judge"]["issues"] == ["invented price"]


def test_no_judge_leaves_the_verdict_open(agent, tmp_path):
    r = open_session(tmp_path, agent(script()), judge_on=False)
    assert r["error"] is None and r["passed"] is None and r["judge"] is None and len(r["turns"]) == 2


def test_an_agent_that_never_answers_is_an_error(agent, tmp_path, monkeypatch):
    judge_with(monkeypatch)
    import sim.session
    monkeypatch.setattr(sim.session.Session, "listen", _quick_listen(sim.session.Session.listen))
    r = open_session(tmp_path, agent([], silent=True))
    assert r["passed"] is False and r["error"] == "the agent never answered"


def _quick_listen(real):
    async def listen(self, timeout=60.0):
        return await real(self, timeout=2.0)       # do not wait a minute for a silent fake
    return listen


def test_wrong_api_key_says_how_to_fix_it(agent, tmp_path):
    r = open_session(tmp_path, agent(script()), api_key="wrong", judge_on=False)
    assert r["passed"] is False and "401" in r["error"] and "restarted" in r["error"] and r["turns"] == []


def test_agent_not_running_is_a_clear_error(tmp_path):
    r = open_session(tmp_path, "ws://127.0.0.1:9/ws", judge_on=False)
    assert r["passed"] is False and "cannot reach the agent" in r["error"]


def test_dynamic_session_uses_the_strands_user_and_stops_when_it_hangs_up(agent, tmp_path, monkeypatch):
    judge_with(monkeypatch)
    fake = FakeStrandsAgent(["Hi, my air conditioner is not cooling.", "HANGUP: Yes please book it."])
    from sim.user import DynamicUser
    import sim.run
    monkeypatch.setattr(sim.run, "DynamicUser", lambda scenario, user_agent=None: _attach(DynamicUser(scenario, agent=fake), fake))
    r = asyncio.run(run_session(SCENARIO, "dynamic", "v", tmp_path / "02_session", url=agent(script()), api_key=API_KEY, settle=0.5,
                                speaker=speaker))
    assert r["error"] is None and [t["said"] for t in r["turns"]] == ["Hi, my air conditioner is not cooling.", "Yes please book it."]
    assert r["turns"][0]["reply"].startswith("That comes to") and r["mode"] == "dynamic"
    assert fake.prompts[1] == f"The assistant says: {r['turns'][0]['reply']}"         # the voice agent's reply reached the user agent


def _attach(user, fake):
    fake.user = user
    return user
