"""Synthetic household / street background noises, generated locally (no downloads) into batch/noise/*.wav.

These are procedural approximations, good for stressing the agent's speech recognition and turn-taking, not
field recordings. Drop real recordings (16 kHz mono WAV) into batch/noise/ under the same names to override them.

    real-*    recordings from the DEMAND corpus (CC BY 4.0), fetched by fetch_noise.py: real-kitchen, real-sink
              (washing dishes), real-living, real-cafe, real-traffic, real-office
    laundry   washing machine: motor hum, drum thumps, water swish
    vacuum    vacuum cleaner: broadband roar plus motor whine
    kitchen   clinking dishes, sizzle, fan hum
    traffic   low rumble with passing cars
    babble    TV / crowd chatter, made by mixing several of our own TTS voices
"""
import wave
from pathlib import Path

import numpy as np
from scipy import signal

RATE = 16_000
SECONDS = 30
DIR = Path(__file__).resolve().parent / "noise"
SYNTHETIC = ("laundry", "vacuum", "kitchen", "traffic", "babble")
REAL = ("real-kitchen", "real-sink", "real-living", "real-cafe", "real-traffic", "real-office")  # DEMAND, see fetch_noise.py
NAMES = SYNTHETIC + REAL


def _filt(x, kind, freq, order=4):
    sos = signal.butter(order, freq, btype=kind, fs=RATE, output="sos")
    return signal.sosfilt(sos, x)


def _impulse(x, at, tone, decay_s, amp):
    n = int(decay_s * 6 * RATE)
    t = np.arange(n) / RATE
    seg = amp * np.sin(2 * np.pi * tone * t) * np.exp(-t / decay_s)
    i = int(at * RATE)
    x[i:i + n] += seg[: len(x) - i]


def _laundry(rng, t):
    hum = sum(np.sin(2 * np.pi * 55 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 6))
    drum = 0.55 + 0.45 * np.sin(2 * np.pi * 1.1 * t)                    # drum rotation
    swish = _filt(rng.standard_normal(t.size), "bandpass", [200, 1800], 2)
    swish *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.35 * t + 1.0) * np.sin(2 * np.pi * 0.9 * t)
    x = 0.5 * hum * drum + 0.8 * swish
    for at in np.arange(0.3, SECONDS - 1, 0.9):                          # clothes tumbling
        _impulse(x, at + rng.uniform(-0.08, 0.08), rng.uniform(70, 110), 0.07, rng.uniform(1.5, 3.5))
    return x


def _vacuum(rng, t):
    roar = _filt(rng.standard_normal(t.size), "bandpass", [350, 6500], 2)
    whine = sum(np.sin(2 * np.pi * 240 * k * t) / k for k in range(1, 5)) * (1 + 0.05 * np.sin(2 * np.pi * 0.4 * t))
    return roar + 0.6 * whine


def _kitchen(rng, t):
    x = 0.25 * _filt(np.cumsum(rng.standard_normal(t.size)), "highpass", 80, 2)
    x /= np.abs(x).max() + 1e-9
    x += 0.15 * np.sin(2 * np.pi * 120 * t)                              # extractor fan
    gate = (_filt(rng.standard_normal(t.size), "lowpass", 2, 2) > 0.3).astype(float)
    x += 0.3 * _filt(rng.standard_normal(t.size), "highpass", 4000, 2) * _filt(gate, "lowpass", 20, 2)  # sizzle
    for at in np.cumsum(rng.exponential(1.1, 60)):                       # clinks
        if at < SECONDS - 1:
            _impulse(x, at, rng.uniform(2000, 5200), rng.uniform(0.01, 0.04), rng.uniform(0.5, 1.5))
    return x


def _traffic(rng, t):
    x = _filt(np.cumsum(rng.standard_normal(t.size)), "lowpass", 400, 2)
    x /= np.abs(x).max() + 1e-9
    for at in np.arange(2, SECONDS, rng.uniform(4, 7)):
        env = np.exp(-0.5 * ((t - at) / rng.uniform(0.7, 1.4)) ** 2)
        x += 1.2 * env * _filt(rng.standard_normal(t.size), "bandpass", [150, 2200], 2)
    return x


def _babble(rng, t):
    import tts  # our own voices, mixed so no single word stands out

    lines = ["The weather forecast says it will be cloudy with a chance of rain through the weekend.",
             "And in other news the council has approved the new plan for the town centre.",
             "I told him we should have left earlier but he never listens to what I say.",
             "Please remember to pick up some milk and bread on the way back from work.",
             "Welcome back to the show, we have a great lineup of guests for you tonight.",
             "Can you pass the salt, and did you see what happened at the match yesterday."]
    voices = [v for v in ("kokoro:af_heart", "kokoro:am_michael", "kokoro:bf_emma", "piper:en_US-lessac-medium",
                          "piper:en_GB-alba-medium") if not tts.available(v)]
    if not voices:
        raise RuntimeError("babble needs at least one available TTS voice")
    x = np.zeros(t.size)
    for k, line in enumerate(lines):
        s = np.frombuffer(tts.synthesize(voices[k % len(voices)], line), "<i2").astype(float) / 32768.0
        s = np.tile(np.concatenate([s, np.zeros(int(rng.uniform(0.2, 1.0) * RATE))]), SECONDS * RATE // max(len(s), 1) + 2)
        off = int(rng.uniform(0, 2) * RATE)
        x += s[off:off + t.size]
    return x


_MAKERS = {"laundry": _laundry, "vacuum": _vacuum, "kitchen": _kitchen, "traffic": _traffic, "babble": _babble}


def path(name: str) -> Path:
    """batch/noise/<name>.wav, generated on first use."""
    if name not in NAMES:
        raise ValueError(f"unknown noise {name!r}; choose from {', '.join(NAMES)}")
    p = DIR / f"{name}.wav"
    if not p.exists() and name in REAL:
        raise FileNotFoundError(f"{p} missing; run: python batch/fetch_noise.py")
    if not p.exists():
        rng = np.random.default_rng(sum(map(ord, name)))
        t = np.arange(SECONDS * RATE) / RATE
        x = _MAKERS[name](rng, t)
        x = x / (np.sqrt(np.mean(x**2)) + 1e-9) * 0.1                      # RMS 0.1; SNR is set at mix time
        x = np.clip(x, -1, 1)
        DIR.mkdir(exist_ok=True)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes((x * 32767).astype("<i2").tobytes())
    return p
