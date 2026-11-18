"""QA app for the voice agent.   streamlit run app.py

  Runs            results of `python batch/run_evals.py` (synthetic and human clips against the live agent)
  Baseline clips  record human voices for the clip library the runner replays
"""
import io
import json
import tempfile
import wave
from pathlib import Path

import pandas as pd
import soundfile as sf
import soxr
import streamlit as st
import streamlit.components.v1 as components

from voice_eval import scoring   # importing scoring puts batch/ on sys.path
import clips                                # batch/clips.py: the caller clip library
from dialogue import Script                 # batch/dialogue.py

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "batch" / "results"
st.set_page_config(page_title="Voice agent QA", layout="wide")


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def num(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


# =================================================================== runs ===
def run_dirs() -> list[Path]:
    return sorted((p for p in RESULTS.glob("*") if (p / "runs.jsonl").exists()), reverse=True) if RESULTS.exists() else []


def load_runs(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def stats_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": r["name"], "runs": r["runs"], "pass rate": pct(r["pass_rate"]), "tools": pct(r["tool_score"]), "WER": pct(r["wer"]),
        "judge": num(r["judge"], 1), "TTFA p50 ms": num(r["ttfa_p50"], 0), "TTFA p95 ms": num(r["ttfa_p95"], 0),
        "tool ms": num(r["tool_ms"], 0), "early replies": r["early_turns"]} for r in rows])


def runs_tab() -> None:
    dirs = run_dirs()
    if not dirs:
        st.info("No runs yet. Start one (agent running on :8000):")
        st.code("python batch/run_evals.py --voices clip:piper:en_US-lessac-medium --scenarios cost_quote,book_ac_inspection --repeat 3", "bash")
        return
    d = st.selectbox("Run", dirs, format_func=lambda p: p.name)
    runs = load_runs(d)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}
    meta = summary.get("meta") or (json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {})

    passed = sum(bool(r["passed"]) for r in runs)
    c = st.columns(4)
    c[0].metric("Pass rate", pct(passed / len(runs)), f"{passed}/{len(runs)} calls")
    words = sum(r.get("wer_words", 0) for r in runs)
    c[1].metric("WER", pct(sum(r.get("wer_errors", 0) for r in runs) / words if words else None))
    c[2].metric("Condition", meta.get("augment", "clean") + (f" @ {meta['noise_snr_db']} dB" if meta.get("noise_snr_db") is not None else ""))
    c[3].metric("Repeats", meta.get("repeat", "–"))
    st.caption(f"voices: {', '.join(meta.get('voices', []))} · scenarios: {', '.join(meta.get('scenarios', []))} · judge: {meta.get('judge_model', '–')}")

    cmp = meta.get("baseline_cmp")
    if cmp:
        k = cmp["counts"]
        st.subheader(f"Against baseline “{cmp['name']}”")
        st.write(f"❌ {k['regression']} regression(s) · ✅ {k['improved']} improved · {k['same']} unchanged · {k['new']} not in baseline")
        bad = [r for r in cmp["rows"] if r["status"] in ("regression", "improved")]
        if bad:
            st.dataframe(pd.DataFrame([{"status": r["status"], "scenario": r["scenario"], "voice": r["voice"],
                                        "what changed": "; ".join(n["text"] for n in r["notes"])} for r in bad]), hide_index=True)

    if summary.get("by_voice"):
        st.subheader("By voice")
        st.dataframe(stats_frame(summary["by_voice"]), hide_index=True)
        st.subheader("By scenario")
        st.dataframe(stats_frame(summary["by_scenario"]), hide_index=True)

    st.subheader("Calls")
    st.dataframe(pd.DataFrame([{
        "scenario": r["scenario"], "voice": r["voice"], "result": "PASS" if r["passed"] else "FAIL",
        "tools": pct(r.get("tool_score")), "WER": pct(r["wer_errors"] / r["wer_words"] if r.get("wer_words") else None),
        "judge": num((r.get("judge") or {}).get("overall"), 1), "error": r.get("error") or ""} for r in runs]), hide_index=True)

    for i, r in enumerate(runs):
        if r["passed"]:
            continue
        with st.expander(f"FAIL · {r['scenario']} · {r['voice']}"):
            why = [x for x in (r.get("error"), r.get("outcome_why"), r.get("tools_missing") and f"missing tools: {r['tools_missing']}",
                               r.get("forbidden_called") and f"forbidden tools called: {r['forbidden_called']}") if x]
            st.write("; ".join(map(str, why)) or "see conversation")
            for line in r.get("conversation", []):
                role = line["role"]
                if role == "tool":
                    st.code(f"{line['name']}({json.dumps(line['args'])}) -> {json.dumps(line['result'])[:300]}", "text")
                else:
                    st.write(f"**{role}**: {line.get('text', '')}")
            j = r.get("judge") or {}
            if j.get("issues"):
                st.write("Judge issues: " + " · ".join(j["issues"]))
            audio = d / r["audio_file"] if r.get("audio_file") else None
            if audio and audio.exists():
                st.audio(str(audio))

    report = d / "report.html"
    if report.exists():
        st.download_button("Download report.html", report.read_bytes(), "report.html", "text/html")
        if st.toggle("Show the full HTML report here (audio players need the file download)"):
            components.html(report.read_text(encoding="utf-8"), height=900, scrolling=True)


# =========================================================== baseline clips ===
def clip_lines() -> list[dict]:
    out = []
    for s in clips._scenarios():
        for i, text in enumerate(Script(s).lines()):
            out.append({"scenario": s["id"], "turn": i, "text": text, "reactive": "dialogue" in s})
    return out


def to_16k_pcm(raw: bytes) -> bytes:
    x, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != scoring.CALLER_RATE:
        x = soxr.resample(x, rate, scoring.CALLER_RATE)
    return (x.clip(-1, 1) * 32767).astype("<i2").tobytes()


def baseline_tab() -> None:
    st.caption("Read the line aloud, naturally, in a quiet room. Saved to the clip library as `human:<speaker>`; replay with "
               "`run_evals.py --voices clip:human:<speaker>`. The browser microphone applies its own noise suppression: for "
               "raw voice, record in an external recorder and use the upload box.")
    speaker = st.text_input("Speaker id", placeholder="e.g. alice-us-f")
    lines = clip_lines()
    index = clips.load_index()["clips"]
    mine = f"human:{clips._slug(speaker)}" if speaker.strip() else None
    done = {(c["scenario"], c["turn"]) for c in index.values() if c["source"] == mine}
    st.write(f"{len(done)} / {len(lines)} lines recorded" + (f" for **{speaker.strip()}**" if mine else ""))

    pick = st.selectbox("Line", range(len(lines)), format_func=lambda i: f"{'✅' if (lines[i]['scenario'], lines[i]['turn']) in done else '⬜'} "
                                                                       f"{lines[i]['scenario']} · line {lines[i]['turn'] + 1}")
    line = lines[pick]
    st.info(f"Say exactly: **{line['text']}**")
    shot = st.audio_input("Record") or st.file_uploader("…or upload a WAV/FLAC", type=["wav", "flac"])
    if shot is not None and mine:
        raw = shot.getvalue()
        pcm = to_16k_pcm(raw)
        stats = scoring.audio_stats(pcm, scoring.CALLER_RATE)
        problems = []
        if stats["duration_s"] < 0.5:
            problems.append("shorter than 0.5 s")
        if stats["clipping_pct"] > 0.5:
            problems.append(f"clipped ({stats['clipping_pct']}% of samples): lower the mic gain")
        if stats["peak_dbfs"] is not None and stats["peak_dbfs"] < -30:
            problems.append("very quiet: speak closer to the mic")
        st.caption(f"{stats['duration_s']} s · peak {stats['peak_dbfs']} dBFS · SNR ≈ {stats['snr_db']} dB")
        if problems:
            st.error("; ".join(problems))
        elif st.button("Save to the clip library", type="primary"):
            with tempfile.TemporaryDirectory() as tmp:
                f = Path(tmp) / "clip.wav"
                with wave.open(str(f), "wb") as w:   # the converted mono 16 kHz audio, whatever format was recorded or uploaded
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(scoring.CALLER_RATE); w.writeframes(pcm)
                clips.add_human(line["scenario"], f, clips._slug(speaker), line["turn"])
            st.success("Saved.")
            st.rerun()
    elif shot is not None:
        st.warning("Enter a speaker id first.")
    existing = [c for c in index.values() if c["scenario"] == line["scenario"] and c["turn"] == line["turn"]]
    if existing:
        st.write("**Stored clips for this line**")
        for c in sorted(existing, key=lambda c: c["source"]):
            st.write(c["source"]); st.audio(str(clips.DIR / c["file"]))


tabs = st.tabs(["Runs", "Baseline clips"])
with tabs[0]:
    runs_tab()
with tabs[1]:
    baseline_tab()
