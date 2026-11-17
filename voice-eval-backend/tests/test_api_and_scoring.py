"""Tests for the QA console: scoring maths on synthetic sessions, and the HTTP API end to end (no AWS, no agent needed)."""
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voice_eval import scoring  # noqa: E402

PRICING = json.loads((ROOT / "voice_eval" / "pricing.json").read_text())


def tone(seconds, rate, amp=0.3, f=220):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * f * t))


def timeline(rate, total, spans):
    x = np.zeros(int(total * rate), dtype=np.float32)
    for a, b in spans:
        x[int(a * rate): int(b * rate)] = tone(b - a, rate)
    return (x * 32767).astype("<i2").tobytes()


def booking_events():
    return [
        {"t": 1.0, "type": "transcript", "text": "book an inspection for my air conditioner not cooling"},
        {"t": 2.0, "type": "transcript", "text": "my name is sam rivera phone five five five zero one zero zero"},
        {"t": 4.0, "type": "reply", "text": "That comes to 53 dollars. Shall I book it?"},
        {"t": 6.0, "type": "transcript", "text": "yes go ahead"},
        {"t": 7.0, "type": "tool", "name": "create_service_request", "ms": 12,
         "args": {"customer_name": "Sam Rivera", "phone": "5550100", "item_id": "AC-100", "issue_description": "not cooling", "service_type": "inspection"},
         "result": {"request_id": "SR-1001", "customer_name": "Sam Rivera", "phone": "5550100", "quote": {"item_id": "AC-100", "service_type": "inspection"}}},
        {"t": 8.0, "type": "reply", "text": "Booked, request SR-1001."},
    ]


GOLD = {"outcome": "booked", "intent": "book_service", "script": ["book an inspection for my air conditioner not cooling my name is sam rivera phone five five five zero one zero zero yes go ahead"],
        "slots": {"customer_name": "Sam Rivera", "phone": "5550100", "item_id": "AC-100", "issue_description": "~cool", "service_type": "inspection", "brand": "Samsung"}}


def test_outcome_slots_turns_and_wer():
    ev = booking_events()
    caller = timeline(16000, 10, [(0.5, 3.0), (5.5, 6.5)])
    agent = timeline(24000, 10, [(3.5, 5.0), (7.5, 9.0)])
    r = scoring.score_session({}, ev, caller, agent, GOLD, PRICING)
    assert r["outcome"]["actual"] == "booked" and r["outcome"]["task_success"] is True
    # brand was never given to the booking: 5 of 6 slots correct, and the report says which one failed
    assert r["slots"]["score"] == pytest.approx(5 / 6)
    assert r["slots"]["slots"]["brand"]["ok"] is False
    assert r["turns_to_resolution"]["turns"] == 2 and r["turns_to_resolution"]["resolved"]
    assert r["wer"]["wer"] == 0.0
    assert r["intent"]["predicted_trace"] == "book_service" and r["intent"]["trace_correct"] is True
    assert r["latency"]["values_ms"] == [500, 1000]


def test_booking_when_golden_says_decline_is_a_failure():
    ev = booking_events()
    r = scoring.score_session({}, ev, b"", b"", {"outcome": "declined", "intent": "out_of_scope", "slots": {}}, PRICING)
    assert r["outcome"]["task_success"] is False and r["outcome"]["forbidden_booking"] is True


def test_quote_and_declined_outcomes():
    q = [{"t": 1, "type": "tool", "name": "get_servicing_cost", "args": {}, "result": {"total": 100}}]
    assert scoring.session_outcome(scoring.tool_calls(q)) == "quote_given"
    assert scoring.session_outcome([]) == "declined"
    failed = [{"t": 1, "type": "tool", "name": "create_service_request", "args": {}, "result": {"error": "x"}}]
    assert scoring.session_outcome(scoring.tool_calls(failed)) == "error"


def test_false_and_missed_barge_in():
    caller = [(1.0, 2.0), (6.0, 7.5)]
    agent = [(2.5, 5.0), (6.2, 9.0)]
    ev = [{"t": 4.5, "type": "interrupted"},          # caller went quiet at 2.0 -> nobody talking near 4.5 -> false
          ]
    b = scoring.barge_in(ev, caller, agent)
    assert b["false"] == 1 and b["missed"] == 1       # caller talked over the 2nd agent turn, agent never stopped
    ev2 = [{"t": 6.4, "type": "interrupted"}]
    b2 = scoring.barge_in(ev2, caller, agent)
    assert b2["false"] == 0 and b2["missed"] == 0


def test_mix_noise_hits_requested_snr():
    s = tone(1, 16000)
    n = np.random.default_rng(0).normal(size=16000).astype(np.float32)
    out = scoring.mix_noise(s, n, 10.0)
    resid = out - s
    snr = 10 * np.log10((s ** 2).mean() / (resid ** 2).mean())
    assert snr == pytest.approx(10.0, abs=0.6)


def test_cost_flags_placeholder_rates():
    c = scoring.cost(60, 60, PRICING)
    assert c["estimate"] is True and c["total_usd"] > 0


# ------------------------------------------------------------------- HTTP API ---
KEY = "k" * 20


def make_client(tmp_path, monkeypatch, ui_dist=None):
    monkeypatch.setenv("AUTH_API_KEY", KEY)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    if ui_dist:
        monkeypatch.setenv("UI_DIST", str(ui_dist))
    else:
        monkeypatch.delenv("UI_DIST", raising=False)
    from voice_eval import config
    config.get_settings.cache_clear()
    from fastapi.testclient import TestClient
    from voice_eval import router as r
    from voice_eval.app import create_app
    monkeypatch.setattr(r, "DATA", tmp_path / "sessions")
    return TestClient(create_app()), {"X-API-Key": KEY}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch)


def wav_bytes(pcm, rate):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return b.getvalue()


def test_api_requires_key_and_roundtrips_a_session(client):
    c, h = client
    assert c.get("/api/sessions").status_code == 401
    assert c.get("/api/sessions", headers={"X-API-Key": "wrong"}).status_code == 401
    meta = {"scenario_id": "book_ac_inspection", "condition": {"noise": "babble", "snr_db": 10}, "caller_mode": "live_mic", "duration_s": 10}
    files = {"caller": ("c.wav", wav_bytes(timeline(16000, 10, [(0.5, 3), (5.5, 6.5)]), 16000), "audio/wav"),
             "agent": ("a.wav", wav_bytes(timeline(24000, 10, [(3.5, 5), (7.5, 9)]), 24000), "audio/wav")}
    r = c.post("/api/sessions", headers=h, data={"meta": json.dumps(meta), "events": json.dumps(booking_events())}, files=files)
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    ev = c.post(f"/api/sessions/{sid}/evaluate?use_llm=false&use_mos=false", headers=h)
    assert ev.status_code == 200, ev.text
    body = ev.json()
    assert body["outcome"]["actual"] == "booked"
    assert body["intent"]["source"] == "trace" and body["judge"]["status"] == "skipped"
    assert "false" in body["barge_in"]            # aliased keys reach the wire as the UI expects them
    s = c.get("/api/summary", headers=h).json()
    assert "babble @ 10 dB" in s["by_condition"] and s["overall"]["sessions"] == 1
    assert c.post(f"/api/sessions/{sid}/review", headers=h, json={"naturalness": 4, "pronunciation": 9}).status_code == 400
    assert c.post(f"/api/sessions/{sid}/review", headers=h, json={"naturalness": 4, "pronunciation": 5}).status_code == 200
    assert c.get(f"/api/sessions/{sid}/audio/agent", headers=h).status_code == 200
    assert c.get("/api/sessions/../../etc", headers=h).status_code in (404, 422)


def test_rejects_wrong_audio_format(client):
    c, h = client
    files = {"caller": ("c.wav", wav_bytes(bytes(200), 44100), "audio/wav"), "agent": ("a.wav", wav_bytes(bytes(200), 24000), "audio/wav")}
    r = c.post("/api/sessions", headers=h, data={"meta": "{}", "events": "[]"}, files=files)
    assert r.status_code == 400


def test_clips_and_noise_mixing_from_the_library(client):
    c, h = client
    d = c.get("/api/clips", headers=h).json()
    assert d["lines"] and "babble" in d["noises"]
    clip_id = next(x for x in d["lines"] if x["clips"])["clips"][0]["id"]
    plain = c.get(f"/api/clips/{clip_id}/pcm", headers=h)
    noisy = c.get(f"/api/clips/{clip_id}/pcm?noise=babble&snr_db=10", headers=h)
    assert plain.status_code == noisy.status_code == 200 and len(plain.content) == len(noisy.content)
    assert plain.content != noisy.content
    assert c.get(f"/api/clips/{clip_id}/pcm?noise=../x", headers=h).status_code == 404


def test_ui_is_served_only_when_configured_and_confined(tmp_path, monkeypatch):
    c, _ = make_client(tmp_path / "a", monkeypatch)
    assert c.get("/").status_code == 404            # no UI_DIST: the API stands alone
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>ok</html>")
    (dist / "assets" / "app.js").write_text("1")
    (tmp_path / "secret.txt").write_text("nope")
    c, _ = make_client(tmp_path / "b", monkeypatch, ui_dist=dist)
    assert c.get("/").text == "<html>ok</html>"
    assert c.get("/assets/app.js").status_code == 200
    assert c.get("/assets/../../secret.txt").status_code == 404
    assert c.get("/assets/%2e%2e/index.html").status_code == 404


def test_openapi_documents_the_ui_contract(client):
    c, _ = client
    schemas = c.get("/openapi.json").json()["components"]["schemas"]
    assert {"Evaluation", "SessionDetail", "SummaryResponse", "ClipsResponse"} <= set(schemas)
