"""Static user clips: the scenario lines rendered ONCE with Piper and stored as WAV, so every run replays exactly the same audio
(no TTS drift, no model needed to replay). Each clip is checksummed in clips/index.json and tied to the line it says.

    python -m sim.clips render [--voices en_US-lessac-medium] [--scenarios id,id]
    python -m sim.clips list
    python -m sim.clips verify
"""
import argparse
import hashlib
import json
import sys
import wave

from sim import tts
from sim.user import load_scenarios
from sim.config import ROOT
from sim.text import normalize

DIR = ROOT / "clips"
INDEX = DIR / "index.json"
DEFAULT_VOICE = "en_US-lessac-medium"


def _key(scenario: str, turn: int, voice: str) -> str:
    return f"{scenario}.t{turn}.{voice}"


def load_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {}


def _read(file: str) -> bytes:
    with wave.open(str(DIR / file)) as w:
        return w.readframes(w.getnframes())


def get(scenario: str, turn: int, text: str, voice: str) -> bytes:
    """The stored PCM16 16 kHz clip for this scenario line, checksum-verified. Raises with the fix if missing or stale."""
    entry = load_index().get(_key(scenario, turn, voice))
    if entry is None:
        raise LookupError(f"no stored clip for {scenario} turn {turn} in voice {voice}: python -m sim.clips render --voices {voice}")
    if normalize(entry["text"]) != normalize(text):
        raise LookupError(f"the clip for {scenario} turn {turn} says an older line: python -m sim.clips render --scenarios {scenario}")
    pcm = _read(entry["file"])
    if hashlib.sha256(pcm).hexdigest() != entry["sha256"]:
        raise ValueError(f"{entry['file']} no longer matches its checksum (the file changed): re-render it")
    return pcm


def render(voices: list[str], scenario_ids: list[str] | None = None) -> int:
    DIR.mkdir(exist_ok=True)
    index, n = load_index(), 0
    for voice in voices:
        why = tts.available(voice)
        if why:
            raise SystemExit(why)
        for s in load_scenarios().values():
            if scenario_ids and s["id"] not in scenario_ids:
                continue
            for i, text in enumerate(s["static_turns"]):
                pcm = tts.synthesize(voice, text)
                file = f"{_key(s['id'], i, voice)}.wav"
                with wave.open(str(DIR / file), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(tts.RATE)
                    w.writeframes(pcm)
                index[_key(s["id"], i, voice)] = {"file": file, "text": text, "voice": voice, "scenario": s["id"], "turn": i,
                                                  "sha256": hashlib.sha256(pcm).hexdigest(), "duration_s": round(len(pcm) / 2 / tts.RATE, 2)}
                n += 1
                print(f"  {file:<60} {index[_key(s['id'], i, voice)]['duration_s']:>5.1f}s")
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return n


def verify() -> int:
    bad = 0
    for k, e in load_index().items():
        if not (DIR / e["file"]).exists():
            print(f"MISSING  {k}")
            bad += 1
        elif hashlib.sha256(_read(e["file"])).hexdigest() != e["sha256"]:
            print(f"CHANGED  {k}")
            bad += 1
    print("all clips intact" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--voices", default=DEFAULT_VOICE)
    r.add_argument("--scenarios", default="")
    sub.add_parser("list")
    sub.add_parser("verify")
    a = ap.parse_args()
    if a.cmd == "render":
        print(f"rendered {render(a.voices.split(','), [s for s in a.scenarios.split(',') if s] or None)} clip(s) -> {DIR}")
        return 0
    if a.cmd == "list":
        for k, e in sorted(load_index().items()):
            print(f"{k:<58} {e['duration_s']:>5.1f}s  {e['text'][:50]}")
        return 0
    return verify()


if __name__ == "__main__":
    sys.exit(main())
