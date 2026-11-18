"""Scoring maths on synthetic sessions (no AWS, no agent needed)."""
import json
import sys
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
