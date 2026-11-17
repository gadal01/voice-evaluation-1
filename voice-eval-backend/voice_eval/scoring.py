"""Pure scoring functions for one recorded session (no network, no server): easy to unit-test.

A session is: `events` (timestamped, seconds from record start), `caller` PCM16 16 kHz and `agent` PCM16 24 kHz
(both on the same timeline, what was sent / what was actually played), a golden scenario, and a condition label.

Metric definitions (what the number means):
  wer              caller reference text vs the agent's own transcript of the caller (jiwer, normalised digits)
  slot_accuracy    expected slots found correctly in the booking the agent actually made / expected slots
  session_outcome  booked | quote_given | slots_given | declined | error, derived from real tool calls and results
  task_success     session_outcome == golden outcome AND no forbidden/unconfirmed booking
  turns_to_resolution  caller turns up to and including the one that produced the resolving tool call
  intent           predicted vs labelled intent (see predict_intent_from_trace; the LLM classifier is the primary source)
  ttfa             caller stops talking -> first agent audio is played (includes endpointing + network)
  false_barge_in   agent was cut off ("interrupted") while the caller was NOT speaking (echo / noise / cough)
  missed_barge_in  caller talked over the agent for > 0.6 s and the agent never stopped
  cost             estimated from audio seconds and judge tokens using pricing.json
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

import numpy as np

EVALS = Path(__file__).resolve().parent.parent / "batch"   # the batch harness: metrics.py (jiwer), clips.py, dialogue.py
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
import metrics as evals_metrics  # noqa: E402  (jiwer WER with digit normalisation, shared with the batch harness)

CALLER_RATE, AGENT_RATE = 16_000, 24_000
ACTION_TOOLS = {"create_service_request": "book_service", "get_servicing_cost": "get_quote",
                "get_available_slots": "check_availability", "cancel_service_request": "cancel_request",
                "get_service_request": "request_status"}
INTENT_PRIORITY = ["book_service", "cancel_request", "request_status", "get_quote", "check_availability"]
INTENTS = INTENT_PRIORITY + ["out_of_scope"]


# --------------------------------------------------------------------- audio ---
def to_float(pcm: bytes | np.ndarray) -> np.ndarray:
    x = np.frombuffer(pcm, dtype="<i2") if isinstance(pcm, (bytes, bytearray)) else pcm
    return x.astype(np.float32) / 32768.0


def vad_segments(pcm: bytes | np.ndarray, rate: int, frame_ms: int = 20, min_speech_s: float = 0.15,
                 merge_gap_s: float = 0.45) -> list[tuple[float, float]]:
    """Energy VAD with an adaptive threshold: (start_s, end_s) of speech. Good enough to time turns and barge-ins;
    it is not a speech/non-speech classifier, so loud steady noise needs a manual look (see the waveform)."""
    x = to_float(pcm)
    n = int(rate * frame_ms / 1000)
    if len(x) < n:
        return []
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    floor = np.percentile(db, 10)
    thr = max(-48.0, floor + 12.0)
    active = db > thr
    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        if not a and start is not None:
            segs.append([start * n / rate, i * n / rate])
            start = None
    if start is not None:
        segs.append([start * n / rate, len(active) * n / rate])
    merged: list[list[float]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_speech_s]


def audio_stats(pcm: bytes, rate: int) -> dict:
    x = to_float(pcm)
    if not len(x):
        return {"duration_s": 0.0, "peak_dbfs": None, "clipping_pct": 0.0, "noise_floor_dbfs": None, "snr_db": None}
    n = int(rate * 0.02)
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    floor, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    return {"duration_s": round(len(x) / rate, 2), "peak_dbfs": round(float(20 * np.log10(np.abs(x).max() + 1e-9)), 1),
            "clipping_pct": round(float((np.abs(x) > 0.99).mean() * 100), 3), "noise_floor_dbfs": round(floor, 1),
            "snr_db": round(speech - floor, 1)}


def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Float arrays in [-1, 1]; noise is looped/cut to the speech length and scaled to the requested SNR."""
    if not len(noise):
        return speech
    noise = np.resize(noise, len(speech))
    ps, pn = float((speech ** 2).mean()) + 1e-12, float((noise ** 2).mean()) + 1e-12
    out = speech + noise * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------- trace ---
def user_turns(events: list[dict]) -> list[dict]:
    """Groups the agent's transcripts of the caller into turns: consecutive `transcript` events with no agent
    reply/tool event in between are one turn. Returns [{text, t_start, t_end}]."""
    turns: list[dict] = []
    prev_user = False
    for e in sorted(events, key=lambda e: e["t"]):
        if e["type"] == "transcript":
            if prev_user and turns:
                turns[-1]["text"] += " " + e["text"].strip()
                turns[-1]["t_end"] = e["t"]
            else:
                turns.append({"text": e["text"].strip(), "t_start": e["t"], "t_end": e["t"]})
            prev_user = True
        elif e["type"] in ("reply", "tool"):
            prev_user = False
    return turns


def tool_calls(events: list[dict]) -> list[dict]:
    return [{"name": e["name"], "args": e.get("args") or {}, "result": e.get("result"), "t": e["t"], "ms": e.get("ms")}
            for e in sorted(events, key=lambda e: e["t"]) if e["type"] == "tool"]


def _booking(calls: list[dict]) -> dict | None:
    for c in reversed(calls):
        if c["name"] == "create_service_request" and isinstance(c["result"], dict) and "request_id" in c["result"]:
            return c
    return None


def session_outcome(calls: list[dict]) -> str:
    """What the session actually achieved, from tool results (not from what the agent claimed)."""
    if _booking(calls):
        return "booked"
    if any(c["name"] == "create_service_request" for c in calls) and not any(
            c["name"] in ("get_servicing_cost", "get_available_slots") for c in calls):
        return "error"  # tried to book and failed, never got as far as a quote
    if any(c["name"] == "get_servicing_cost" and isinstance(c["result"], dict) and "error" not in c["result"] for c in calls):
        return "quote_given"
    if any(c["name"] == "get_available_slots" and isinstance(c["result"], dict) and "error" not in c["result"] for c in calls):
        return "slots_given"
    return "declined"


def predict_intent_from_trace(calls: list[dict]) -> str:
    """Behavioural proxy: the highest-priority action tool the agent used. A cheap fallback; Nova Sonic exposes no NLU
    intent, so the real predicted-vs-labelled comparison uses the LLM classifier on the caller's words (llm.py)."""
    used = {ACTION_TOOLS[c["name"]] for c in calls if c["name"] in ACTION_TOOLS}
    return next((i for i in INTENT_PRIORITY if i in used), "out_of_scope")


# --------------------------------------------------------------------- slots ---
def _digits(s: object) -> str:
    return re.sub(r"\D", "", str(s or ""))


def _slot_ok(key: str, want: str, got: object) -> bool:
    if got is None or got == "":
        return False
    g, w = str(got).lower(), str(want).lower()
    if key == "phone":
        return _digits(got).endswith(_digits(want)[-7:]) and len(_digits(got)) >= 7
    if want.startswith("~"):                      # "~leak" -> substring match for free-text fields (symptom)
        return w[1:] in g
    if key == "customer_name":                    # every word of the expected name appears, any order
        return all(p in g for p in w.split())
    return g == w


def slot_accuracy(expected: dict, calls: list[dict]) -> dict:
    """Compares the golden slots with what create_service_request was actually called with (args) and returned."""
    if not expected:
        return {"score": None, "slots": {}, "note": "no slots expected"}
    booking = _booking(calls)
    attempt = booking or next((c for c in reversed(calls) if c["name"] == "create_service_request"), None)
    result: dict[str, dict] = {}
    for key, want in expected.items():
        got = None
        if attempt:
            args, res = attempt["args"], attempt["result"] if isinstance(attempt["result"], dict) else {}
            got = args.get(key, res.get(key))
            if key in ("service_type", "item_id", "urgency") and got is None:
                got = (res.get("quote") or {}).get(key)
            if key == "preferred_slot" and got is None:
                got = res.get("slot")
        result[key] = {"expected": want, "got": got, "ok": _slot_ok(key, str(want), got)}
    ok = sum(v["ok"] for v in result.values())
    return {"score": ok / len(result), "slots": result, "note": "" if attempt else "no booking attempted"}


# --------------------------------------------------------------- resolution ---
def turns_to_resolution(events: list[dict], outcome: str, goal: str | None) -> dict:
    """Number of caller turns until the resolving event. resolved=False when the outcome is not the golden one."""
    turns = user_turns(events)
    calls = tool_calls(events)
    resolving = {"booked": "create_service_request", "quote_given": "get_servicing_cost",
                 "slots_given": "get_available_slots"}.get(outcome)
    t_res = None
    if resolving:
        t_res = next((c["t"] for c in calls if c["name"] == resolving and isinstance(c["result"], dict)
                      and "error" not in c["result"]), None)
    n = sum(1 for t in turns if t_res is None or t["t_start"] <= t_res) if turns else 0
    return {"turns": n, "total_user_turns": len(turns), "resolved": goal is None or outcome == goal,
            "agent_replies": sum(1 for e in events if e["type"] == "reply")}


# ------------------------------------------------------------------- latency ---
def ttfa(caller_segs: list[tuple[float, float]], agent_segs: list[tuple[float, float]]) -> dict:
    """For each caller speech segment, ms from its end to the next agent audio start (if before the next caller segment)."""
    vals = []
    for i, (_, end) in enumerate(caller_segs):
        nxt = caller_segs[i + 1][0] if i + 1 < len(caller_segs) else float("inf")
        start = next((a for a, _ in agent_segs if end <= a < nxt), None)
        if start is not None:
            vals.append(round((start - end) * 1000))
    return {"values_ms": vals, "p50_ms": statistics.median(vals) if vals else None,
            "p95_ms": sorted(vals)[min(len(vals) - 1, round(0.95 * (len(vals) - 1)))] if vals else None}


# ---------------------------------------------------------------- barge-in ---
def barge_in(events: list[dict], caller_segs: list[tuple[float, float]], agent_segs: list[tuple[float, float]],
             lookback_s: float = 1.5, grace_s: float = 0.3) -> dict:
    """false: interrupted event with no caller speech in [t-lookback, t+grace].  missed: caller spoke over the agent
    for > 0.6 s and no interrupted event arrived within 1.5 s of the overlap start."""
    inter = sorted(e["t"] for e in events if e["type"] == "interrupted")
    false = [t for t in inter if not any(a <= t + grace_s and b >= t - lookback_s for a, b in caller_segs)]
    missed = 0
    for a, b in caller_segs:
        for s, e in agent_segs:
            ov0, ov1 = max(a, s), min(b, e)
            if ov1 - ov0 > 0.6 and not any(ov0 - 0.2 <= t <= ov0 + 1.5 for t in inter):
                missed += 1
                break
    turns = max(1, len(agent_segs))
    return {"interrupts": len(inter), "false": len(false), "false_times": [round(t, 2) for t in false],
            "false_rate": len(false) / turns, "missed": missed, "agent_turns": len(agent_segs)}


# ---------------------------------------------------------------------- cost ---
def cost(caller_s: float, agent_s: float, pricing: dict, judge_in: int = 0, judge_out: int = 0) -> dict:
    tps = pricing["speech_tokens_per_second"]
    inp = caller_s * tps / 1000 * pricing["speech_input_usd_per_1k_tokens"]
    out = agent_s * tps / 1000 * pricing["speech_output_usd_per_1k_tokens"]
    j = judge_in / 1000 * pricing["judge_input_usd_per_1k_tokens"] + judge_out / 1000 * pricing["judge_output_usd_per_1k_tokens"]
    agent_total = inp + out + pricing.get("per_session_fixed_usd", 0.0)
    return {"agent_usd": round(agent_total, 5), "judge_usd": round(j, 5), "total_usd": round(agent_total + j, 5),
            "speech_in_usd": round(inp, 5), "speech_out_usd": round(out, 5),
            "estimate": not pricing.get("verified", False)}


# ----------------------------------------------------------------------- WER ---
def wer(reference: str, events: list[dict]) -> dict:
    hyp = " ".join(e["text"] for e in sorted(events, key=lambda e: e["t"]) if e["type"] == "transcript")
    if not reference.strip():
        return {"wer": None, "note": "no reference text"}
    d = evals_metrics.word_error_detail(reference, hyp)
    return {"wer": d["errors"] / d["words"] if d["words"] else None, **d, "hypothesis": hyp}


# -------------------------------------------------------------- orchestration ---
def score_session(session: dict, events: list[dict], caller_pcm: bytes, agent_pcm: bytes, golden: dict | None,
                  pricing: dict) -> dict:
    """All the deterministic metrics. LLM and MOS metrics are added by the caller (router) because they may be slow."""
    calls = tool_calls(events)
    outcome = session_outcome(calls)
    caller_segs = vad_segments(caller_pcm, CALLER_RATE) if caller_pcm else []
    agent_segs = vad_segments(agent_pcm, AGENT_RATE) if agent_pcm else []
    golden = golden or {}
    goal = golden.get("outcome")
    # WER reference: what the caller actually said (typed in), else the golden script (clip replay / reading the script).
    reference = session.get("reference_text") or " ".join(golden.get("script") or [])
    slots = slot_accuracy(golden.get("slots") or {}, calls)
    forbidden_booking = goal not in (None, "booked") and outcome == "booked"
    success = None if goal is None else (outcome == goal and not forbidden_booking)
    ttr = turns_to_resolution(events, outcome, goal)
    intent_trace = predict_intent_from_trace(calls)
    caller_s = len(caller_pcm) / 2 / CALLER_RATE if caller_pcm else float(session.get("duration_s", 0))
    agent_s = sum(b - a for a, b in agent_segs) if agent_segs else 0.0
    return {
        "outcome": {"actual": outcome, "golden": goal, "task_success": success, "forbidden_booking": forbidden_booking},
        "slots": slots,
        "turns_to_resolution": ttr,
        "intent": {"labelled": golden.get("intent"), "predicted_trace": intent_trace,
                   "trace_correct": None if not golden.get("intent") else intent_trace == golden["intent"]},
        "wer": wer(reference, events),
        "latency": ttfa(caller_segs, agent_segs),
        "barge_in": barge_in(events, caller_segs, agent_segs),
        "cost": cost(caller_s, agent_s, pricing),
        "tools": {"calls": [{"name": c["name"], "args": c["args"], "ms": c["ms"]} for c in calls],
                  "errors": sum(1 for c in calls if isinstance(c["result"], dict) and "error" in c["result"])},
        "audio": {"caller": audio_stats(caller_pcm, CALLER_RATE) if caller_pcm else None,
                  "agent": audio_stats(agent_pcm, AGENT_RATE) if agent_pcm else None,
                  "caller_speech_s": round(sum(b - a for a, b in caller_segs), 2), "agent_speech_s": round(agent_s, 2)},
        "segments": {"caller": caller_segs, "agent": agent_segs},
    }
