"""Piper text-to-speech for the simulated user (open source, local ONNX). Piper is the only voice engine.

    python -m piper.download_voices --download-dir models en_US-lessac-medium     # one-time, ~60 MB per voice

A voice is the model file stem in models/ (en_US-lessac-medium, en_GB-alba-medium, ...). Output is raw PCM16 mono 16 kHz at a
common speech level, which is what the voice agent expects from a microphone. Dynamic lines are cached on disk by (voice, text)
so a repeated line costs nothing; static lines come from stored clips (see clips.py).
"""
import hashlib
from pathlib import Path

import numpy as np

from sim.config import ROOT

MODELS = ROOT / "models"
CACHE = ROOT / "cache"
RATE = 16_000
TARGET_RMS, TARGET_PEAK = 3000, 16000   # speech RMS after normalisation (~ -21 dBFS) and a peak ceiling (~ -6 dBFS)

_voices: dict = {}


def available(voice: str) -> str | None:
    """None if the voice can be used, else the reason and what to do about it."""
    if not (MODELS / f"{voice}.onnx").exists():
        return f"models/{voice}.onnx is missing: python -m piper.download_voices --download-dir models {voice}"
    try:
        import piper  # noqa: F401
    except ImportError:
        return "piper-tts is not installed: pip install piper-tts"
    return None


def _to_16k(x: np.ndarray, rate: int) -> bytes:
    import soxr

    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    if rate != RATE:
        x = soxr.resample(x, rate, RATE)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def normalize_level(pcm: bytes) -> bytes:
    """Scale to a common speech level so voices of different loudness are comparable."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    speech = x[np.abs(x) > 300]
    if speech.size == 0:
        return pcm
    gain = min(TARGET_RMS / np.sqrt(np.mean(speech ** 2)), TARGET_PEAK / max(np.abs(x).max(), 1.0))
    return np.clip(x * gain, -32768, 32767).astype("<i2").tobytes()


def _render(voice: str, text: str) -> bytes:
    if voice not in _voices:
        from piper import PiperVoice

        _voices[voice] = PiperVoice.load(MODELS / f"{voice}.onnx")
    chunks = list(_voices[voice].synthesize(text))
    if not chunks:
        raise RuntimeError(f"piper produced no audio for {text[:40]!r}")
    x = np.concatenate([c.audio_int16_array for c in chunks])
    return _to_16k(x, chunks[0].sample_rate)


def synthesize(voice: str, text: str) -> bytes:
    """PCM16 16 kHz of `text` in `voice`, level-normalised. Rendered once, then served from the cache."""
    why = available(voice)
    if why:
        raise RuntimeError(why)
    path = CACHE / f"{voice}_{hashlib.sha1(text.encode()).hexdigest()[:16]}.pcm"
    if path.exists():
        return normalize_level(path.read_bytes())
    pcm = _render(voice, text)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pcm)
    return normalize_level(pcm)
