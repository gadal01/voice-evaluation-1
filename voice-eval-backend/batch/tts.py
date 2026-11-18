"""Text-to-speech for the simulated caller: Piper and Kokoro (open source, local ONNX), with an on-disk cache.

Voice spec strings:
  piper:en_US-lessac-medium   a Piper voice; the voice name is the model file stem in batch/models/
  kokoro:af_heart             Kokoro; any voice id from voices-v1.0.bin, e.g. af_heart, am_michael, bf_emma
  clip:kokoro:af_heart        replay a stored caller recording (batch/caller_clips, see clips.py); clip:piper:..., clip:human:alice, clip:*
No cloud TTS (Polly, ElevenLabs, ...) is supported on purpose: synthetic callers are Piper or Kokoro, real callers are
recorded human clips.
Local models live in batch/models/ (git-ignored): Piper voices via  python -m piper.download_voices --download-dir batch/models en_US-lessac-medium
and kokoro-v1.0.onnx + voices-v1.0.bin from the kokoro-onnx GitHub release "model-files-v1.0".
Every clip is raw PCM16 mono 16 kHz, which is what the voice agent expects from a mic.
"""
import hashlib
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CACHE = Path(__file__).resolve().parent / "cache"
MODELS = Path(__file__).resolve().parent / "models"  # local TTS model files (Piper voices, Kokoro)
SAMPLE_RATE = 16_000
TARGET_RMS = 3000   # speech RMS after normalization (~ -21 dBFS)
TARGET_PEAK = 16000  # peak ceiling, ~ -6 dBFS

# Characters rendered on a cache miss, for the run report.
chars_rendered: dict[str, int] = {"kokoro": 0, "piper": 0}

_local: dict = {}  # loaded local models, reused across clips


def _cache_path(spec: str, text: str) -> Path:
    key = hashlib.sha1(f"{spec}|{text}".encode()).hexdigest()[:16]
    return CACHE / f"{spec.replace(':', '_')}_{key}.pcm"


def _to_16k(x: np.ndarray, rate: int) -> bytes:
    """float32 [-1, 1] or int16 samples at `rate` -> raw PCM16 mono 16 kHz."""
    import soxr

    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    if rate != SAMPLE_RATE:
        x = soxr.resample(x, rate, SAMPLE_RATE)
    return (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _kokoro(voice: str, text: str) -> bytes:
    if "kokoro" not in _local:
        from kokoro_onnx import Kokoro

        _local["kokoro"] = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    lang = "en-gb" if voice[:2] in ("bf", "bm") else "en-us"
    samples, rate = _local["kokoro"].create(text, voice=voice, speed=1.0, lang=lang)
    return _to_16k(samples, rate)


def _piper(voice: str, text: str) -> bytes:
    if voice not in _local:
        from piper import PiperVoice

        _local[voice] = PiperVoice.load(MODELS / f"{voice}.onnx")
    chunks = list(_local[voice].synthesize(text))
    if not chunks:
        raise RuntimeError("piper produced no audio")
    x = np.concatenate([c.audio_int16_array for c in chunks])
    return _to_16k(x, chunks[0].sample_rate)


def available(spec: str) -> str | None:
    """Returns a reason the voice can't be used, or None if it looks usable."""
    if spec.startswith("clip:"):
        return None  # a missing clip is reported per scenario when it is looked up
    provider, _, voice = spec.partition(":")
    if provider not in ("piper", "kokoro"):
        return f"unsupported voice {spec!r}: only piper:<voice>, kokoro:<voice> and clip:<source> are supported"
    if provider == "kokoro":
        if not (MODELS / "kokoro-v1.0.onnx").exists():
            return "batch/models/kokoro-v1.0.onnx + voices-v1.0.bin missing (see tts.py docstring)"
        return None
    if not (MODELS / f"{voice}.onnx").exists():
        return f"batch/models/{voice}.onnx missing (python -m piper.download_voices --download-dir batch/models {voice})"
    return None


def _normalize(pcm: bytes) -> bytes:
    """Scale to a common speech level so voices of different loudness are comparable."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    speech = x[np.abs(x) > 300]
    if speech.size == 0:
        return pcm
    gain = min(TARGET_RMS / np.sqrt(np.mean(speech**2)), TARGET_PEAK / max(np.abs(x).max(), 1.0))
    return np.clip(x * gain, -32768, 32767).astype("<i2").tobytes()


def synthesize(spec: str, text: str) -> bytes:
    """Render `text` with the voice `spec` (level-normalized), using the cache when possible.
    The cache holds the engine's raw output; normalization is applied on the way out."""
    if spec.startswith("clip:"):  # stored, checksummed recording: replayed exactly as saved (see clips.py)
        import clips

        return clips.get(spec[5:], text)
    return _normalize(_render(spec, text))


def _render(spec: str, text: str) -> bytes:
    why = available(spec)
    if why:
        raise ValueError(why)
    path = _cache_path(spec, text)
    if path.exists():
        return path.read_bytes()
    provider, _, voice = spec.partition(":")
    pcm = _kokoro(voice, text) if provider == "kokoro" else _piper(voice, text)
    chars_rendered[provider] += len(text)
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(pcm)
    return pcm
