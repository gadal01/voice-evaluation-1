"""FastAPI router for the evaluation API: /api/*, every route guarded by X-API-Key (see security.py).

Sessions land in <DATA_DIR>/sessions/<id>/ (session.json, events.jsonl, caller.wav, agent.wav, evaluation.json).
Baseline human clips go through batch/clips.py so the batch harness (run_evals.py --voices clip:human:<name>) can replay them.
"""
from __future__ import annotations

import io
import json
import re
import statistics
import tempfile
import uuid
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from voice_eval import llm, quality, scoring
from voice_eval.config import ROOT, get_settings
from voice_eval.security import require_key
from voice_eval.models import (ClipSaved, ClipsResponse, Evaluation, GoldenResponse, Review, SavedSession, SessionDetail,
                               SessionRow, SummaryResponse)

HERE = Path(__file__).resolve().parent
DATA = get_settings().data_dir / "sessions"
NOISE_DIR = ROOT / "batch" / "noise"
router = APIRouter()
api = APIRouter(prefix="/api", dependencies=[Depends(require_key)])


# ------------------------------------------------------------------ helpers ---
def golden_set() -> dict[str, dict]:
    return {s["id"]: s for s in json.loads((HERE / "golden.json").read_text(encoding="utf-8"))["scenarios"]}


def pricing() -> dict:
    return json.loads((HERE / "pricing.json").read_text(encoding="utf-8"))


def _dir(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", session_id):
        raise HTTPException(404, "no such session")
    d = DATA / session_id
    if not d.exists():
        raise HTTPException(404, "no such session")
    return d


def _wav_pcm(raw: bytes, want_rate: int) -> bytes:
    """PCM16 mono bytes from an uploaded WAV; refuses anything but 16-bit mono at the expected rate."""
    try:
        with wave.open(io.BytesIO(raw)) as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, want_rate):
                raise HTTPException(400, f"WAV must be mono 16-bit {want_rate} Hz")
            return w.readframes(w.getnframes())
    except wave.Error as exc:
        raise HTTPException(400, f"not a valid WAV: {exc}")


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def _read_pcm(path: Path) -> bytes:
    if not path.exists():
        return b""
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def _load(session_id: str) -> tuple[Path, dict, list[dict]]:
    d = _dir(session_id)
    meta = json.loads((d / "session.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return d, meta, events


# ----------------------------------------------------------------- sessions ---
@api.get("/golden", response_model=GoldenResponse)
def get_golden() -> dict:
    return {"scenarios": list(golden_set().values()), "pricing_verified": pricing().get("verified", False)}


@api.post("/sessions", response_model=SavedSession)
async def save_session(meta: str = Form(...), events: str = Form(...), caller: UploadFile = File(...),
                       agent: UploadFile = File(...)) -> dict:
    """Stores one recorded session. `meta` and `events` are JSON strings; caller is 16 kHz WAV, agent is 24 kHz WAV."""
    try:
        m, ev = json.loads(meta), json.loads(events)
        assert isinstance(m, dict) and isinstance(ev, list)
        for e in ev:
            assert isinstance(e.get("t"), (int, float)) and isinstance(e.get("type"), str)
    except (ValueError, AssertionError):
        raise HTTPException(400, "meta must be a JSON object and events a JSON list of {t, type, ...}")
    caller_pcm = _wav_pcm(await caller.read(), scoring.CALLER_RATE)
    agent_pcm = _wav_pcm(await agent.read(), scoring.AGENT_RATE)
    sid = uuid.uuid4().hex[:12]
    d = DATA / sid
    d.mkdir(parents=True)
    keep = ("scenario_id", "condition", "reference_text", "notes", "caller_mode", "conversation_id", "duration_s", "tester")
    record = {k: m.get(k) for k in keep}
    record.update(id=sid, created=datetime.now().isoformat(timespec="seconds"))
    (d / "session.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev), encoding="utf-8")
    _write_wav(d / "caller.wav", caller_pcm, scoring.CALLER_RATE)
    _write_wav(d / "agent.wav", agent_pcm, scoring.AGENT_RATE)
    return {"id": sid}


@api.get("/sessions", response_model=list[SessionRow])
def list_sessions() -> list[dict]:
    rows = []
    for d in sorted(DATA.glob("*/session.json"), key=lambda p: p.stat().st_mtime, reverse=True) if DATA.exists() else []:
        m = json.loads(d.read_text(encoding="utf-8"))
        ev = d.parent / "evaluation.json"
        e = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else None
        m["evaluated"] = bool(e)
        if e:
            m["task_success"] = e["outcome"]["task_success"]
            m["outcome"] = e["outcome"]["actual"]
        rows.append(m)
    return rows


@api.get("/sessions/{sid}", response_model=SessionDetail)
def get_session(sid: str) -> dict:
    d, meta, events = _load(sid)
    ev = d / "evaluation.json"
    return {"meta": meta, "events": events, "evaluation": json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else None,
            "review": json.loads((d / "review.json").read_text(encoding="utf-8")) if (d / "review.json").exists() else None}


@api.get("/sessions/{sid}/audio/{who}")
def get_audio(sid: str, who: str) -> FileResponse:
    d = _dir(sid)
    if who not in ("caller", "agent"):
        raise HTTPException(404)
    return FileResponse(d / f"{who}.wav", media_type="audio/wav")


@api.post("/sessions/{sid}/review", response_model=Review)
def save_review(sid: str, review: dict) -> dict:
    """Human spot check: {naturalness: 1-5, pronunciation: 1-5, mispronounced: "...", notes: "...", reviewer: "..."}."""
    d = _dir(sid)
    clean = {}
    for k in ("naturalness", "pronunciation"):
        v = review.get(k)
        if v is not None and not (isinstance(v, int) and 1 <= v <= 5):
            raise HTTPException(400, f"{k} must be an integer 1-5")
        clean[k] = v
    for k in ("mispronounced", "notes", "reviewer"):
        clean[k] = str(review.get(k, ""))[:1000]
    clean["at"] = datetime.now().isoformat(timespec="seconds")
    (d / "review.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


@api.post("/sessions/{sid}/evaluate", response_model=Evaluation)
def evaluate(sid: str, use_llm: bool = True, use_mos: bool = True) -> dict:
    d, meta, events = _load(sid)
    golden = golden_set().get(meta.get("scenario_id") or "")
    caller_pcm, agent_pcm = _read_pcm(d / "caller.wav"), _read_pcm(d / "agent.wav")
    result = scoring.score_session(meta, events, caller_pcm, agent_pcm, golden, pricing())

    if use_llm:
        result["judge"] = llm.judge(events, (golden or {}).get("title", ""))
        result["intent_llm"] = llm.classify_intent(events)
    else:
        result["judge"] = {"status": "skipped", "reason": "disabled"}
        result["intent_llm"] = {"status": "skipped", "reason": "disabled"}
    il = result["intent_llm"]
    labelled = result["intent"]["labelled"]
    result["intent"]["predicted"] = il["intent"] if il["status"] == "ok" else result["intent"]["predicted_trace"]
    result["intent"]["source"] = "llm" if il["status"] == "ok" else "trace"
    result["intent"]["correct"] = None if not labelled else result["intent"]["predicted"] == labelled
    jt = result["judge"].get("tokens", {})
    it = il.get("tokens", {})
    caller_s = len(caller_pcm) / 2 / scoring.CALLER_RATE
    result["cost"] = scoring.cost(caller_s, result["audio"]["agent_speech_s"], pricing(),
                                  jt.get("in", 0) + it.get("in", 0), jt.get("out", 0) + it.get("out", 0))

    if use_mos and agent_pcm:
        segs = [tuple(s) for s in result["segments"]["agent"]]
        result["mos"] = {"dnsmos": quality.dnsmos(agent_pcm, segs), "utmos": quality.utmos(agent_pcm, segs)}
    else:
        result["mos"] = {"dnsmos": {"status": "skipped", "reason": "disabled"}, "utmos": {"status": "skipped", "reason": "disabled"}}
    result["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    (d / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@api.get("/summary", response_model=SummaryResponse)
def summary() -> dict:
    """Aggregate over all evaluated sessions, grouped by scenario and by condition (clean / noisy ...)."""
    rows = []
    for p in DATA.glob("*/evaluation.json") if DATA.exists() else []:
        e = json.loads(p.read_text(encoding="utf-8"))
        m = json.loads((p.parent / "session.json").read_text(encoding="utf-8"))
        rows.append((m, e))

    def agg(group: list[tuple[dict, dict]]) -> dict:
        def vals(f):
            out = [f(e) for _, e in group]
            return [v for v in out if v is not None]

        def mean(v):
            return round(statistics.fmean(v), 4) if v else None

        succ = vals(lambda e: e["outcome"]["task_success"])
        words = sum(e["wer"].get("words", 0) for _, e in group if e["wer"].get("wer") is not None)
        errs = sum(e["wer"].get("errors", 0) for _, e in group if e["wer"].get("wer") is not None)
        ttfa = [v for _, e in group for v in e["latency"]["values_ms"]]
        return {
            "sessions": len(group), "task_success": mean([float(s) for s in succ]),
            "wer": round(errs / words, 4) if words else None,
            "slot_accuracy": mean(vals(lambda e: e["slots"]["score"])),
            "intent_accuracy": mean(vals(lambda e: None if e["intent"].get("correct") is None else float(e["intent"]["correct"]))),
            "turns_to_resolution": mean(vals(lambda e: e["turns_to_resolution"]["turns"] if e["turns_to_resolution"]["resolved"] else None)),
            "ttfa_p50_ms": statistics.median(ttfa) if ttfa else None,
            "false_barge_in_rate": mean(vals(lambda e: e["barge_in"]["false_rate"])),
            "cost_usd": mean(vals(lambda e: e["cost"]["total_usd"])),
            "tone": mean(vals(lambda e: (e["judge"].get("scores") or {}).get("tone"))),
            "clarity": mean(vals(lambda e: (e["judge"].get("scores") or {}).get("clarity"))),
            "empathy": mean(vals(lambda e: (e["judge"].get("scores") or {}).get("empathy"))),
            "utmos": mean(vals(lambda e: e["mos"]["utmos"].get("utmos"))),
            "dnsmos": mean(vals(lambda e: e["mos"]["dnsmos"].get("ovrl_mos"))),
        }

    def group_by(key) -> dict:
        groups: dict[str, list] = {}
        for m, e in rows:
            groups.setdefault(key(m), []).append((m, e))
        return {k: agg(v) for k, v in sorted(groups.items())}

    def cond(m: dict) -> str:
        c = m.get("condition") or {}
        if not c.get("noise") or c.get("noise") == "none":
            return "clean"
        return f"{c['noise']} @ {c.get('snr_db', '?')} dB"

    return {"overall": agg(rows) if rows else None, "by_scenario": group_by(lambda m: m.get("scenario_id") or "free_form"),
            "by_condition": group_by(cond), "pricing_verified": pricing().get("verified", False)}


# ---------------------------------------------------------- baseline clips ---
def _clips():
    import clips  # evals/clips.py (path added by scoring)

    return clips


@api.get("/clips", response_model=ClipsResponse)
def list_clips() -> dict:
    clips = _clips()
    idx = clips.load_index()["clips"]
    lines = []
    for s in clips._scenarios():
        from dialogue import Script

        for i, text in enumerate(Script(s).lines()):
            have = [c for c in idx.values() if c["scenario"] == s["id"] and c["turn"] == i]
            lines.append({"scenario": s["id"], "turn": i, "text": text, "reactive": "dialogue" in s,
                          "clips": [{"id": c["id"], "source": c["source"], "duration_s": c["duration_s"]} for c in have]})
    return {"lines": lines, "sources": clips.sources(), "noises": sorted(p.stem for p in NOISE_DIR.glob("*.wav"))}


@api.post("/clips", response_model=ClipSaved)
async def add_clip(scenario: str = Form(...), turn: int = Form(...), name: str = Form(...), wav: UploadFile = File(...)) -> dict:
    """Adds a human recording of a scripted line to the caller clip library (source human:<name>)."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")[:32]
    if not name:
        raise HTTPException(400, "name is required (letters, digits, - _)")
    raw = await wav.read()
    pcm = _wav_pcm(raw, scoring.CALLER_RATE)
    st = scoring.audio_stats(pcm, scoring.CALLER_RATE)
    problems = []
    if st["duration_s"] < 0.5:
        problems.append("recording shorter than 0.5 s")
    if st["clipping_pct"] > 0.5:
        problems.append(f"clipped ({st['clipping_pct']}% of samples): lower the mic gain and re-record")
    if st["peak_dbfs"] is not None and st["peak_dbfs"] < -30:
        problems.append("very quiet: speak closer to the mic")
    if problems:
        raise HTTPException(422, "; ".join(problems))
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "clip.wav"
        f.write_bytes(raw)
        try:
            _clips().add_human(scenario, f, name, turn)
        except SystemExit as exc:
            raise HTTPException(404, str(exc))
    entry = next(c for c in _clips().load_index()["clips"].values()
                 if c["scenario"] == scenario and c["turn"] == turn and c["source"] == f"human:{name}")
    return {"id": entry["id"], "duration_s": entry["duration_s"], "audio": st}


@api.get("/clips/{clip_id}/pcm")
def clip_pcm(clip_id: str, noise: str = Query("none"), snr_db: float = Query(15.0, ge=-5, le=40)) -> Response:
    """The stored clip as raw PCM16 16 kHz, optionally mixed with a noise recording at the given SNR (WER-under-noise replay)."""
    clips = _clips()
    entry = clips.load_index()["clips"].get(clip_id)
    if not entry:
        raise HTTPException(404, "no such clip")
    pcm = clips._read(entry)
    if noise != "none":
        nf = NOISE_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '', noise)}.wav"
        if not nf.exists():
            raise HTTPException(404, "no such noise")
        import soundfile as sf
        import soxr

        n, rate = sf.read(str(nf), dtype="float32", always_2d=True)
        n = n.mean(axis=1)
        if rate != scoring.CALLER_RATE:
            n = soxr.resample(n, rate, scoring.CALLER_RATE)
        mixed = scoring.mix_noise(scoring.to_float(pcm), n, snr_db)
        pcm = (mixed * 32767).astype("<i2").tobytes()
    return Response(pcm, media_type="application/octet-stream", headers={"X-Text": entry["text"][:200].encode("ascii", "ignore").decode()})


@api.get("/clips/{clip_id}/audio")
def clip_audio(clip_id: str) -> FileResponse:
    clips = _clips()
    entry = clips.load_index()["clips"].get(clip_id)
    if not entry:
        raise HTTPException(404, "no such clip")
    return FileResponse(clips.DIR / entry["file"], media_type="audio/wav")


router.include_router(api)
