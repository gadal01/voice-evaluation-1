"""Caller audio degradation with audiomentations, so the agent is tested on more than studio-clean TTS.

    clean   untouched
    noisy   pink/brown background noise, 10-20 dB SNR (cafe / office / street)
    phone   300-3400 Hz telephone band, 8 kHz round trip, light noise
    distant caller far from the mic: quieter, band-limited, noisier (5-12 dB SNR)
    hurried faster speech (1.15-1.3x) over light noise
    bg-laundry / bg-vacuum / bg-kitchen / bg-traffic / bg-babble   household and street backgrounds at 8-12 dB SNR

Each clip is seeded from its text, so a run is reproducible and the same words always get the same degradation.
"""
import random
import zlib

import numpy as np
import soxr
import noises
from audiomentations import AddBackgroundNoise, AddColorNoise, Compose, Gain, HighPassFilter, LowPassFilter, TimeStretch

SAMPLE_RATE = 16_000
PROFILES = ("clean", "noisy", "phone", "distant", "hurried") + tuple(f"bg-{n}" for n in noises.NAMES)
BG_SNR_DB = (8.0, 12.0)  # background noise 8-12 dB below the voice: clearly audible, still intelligible


def _pipeline(profile: str, snr_db: float | None) -> Compose | None:
    noise = lambda lo, hi: AddColorNoise(min_snr_db=lo, max_snr_db=hi, min_f_decay=-3.0, max_f_decay=0.0, p=1.0)  # noqa: E731
    if profile == "clean":
        steps = []
    elif profile == "noisy":
        steps = [noise(10, 20)]
    elif profile == "phone":
        steps = [HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=300, p=1.0),
                 LowPassFilter(min_cutoff_freq=3400, max_cutoff_freq=3400, p=1.0),
                 noise(20, 30)]
    elif profile == "distant":
        steps = [Gain(min_gain_db=-12, max_gain_db=-8, p=1.0), LowPassFilter(min_cutoff_freq=4500, max_cutoff_freq=5500, p=1.0),
                 noise(5, 12)]
    elif profile == "hurried":
        steps = [TimeStretch(min_rate=1.15, max_rate=1.3, leave_length_unchanged=False, p=1.0), noise(20, 30)]
    elif profile.startswith("bg-"):  # bg-laundry, bg-vacuum, bg-kitchen, bg-traffic, bg-babble (see noises.py)
        steps = [AddBackgroundNoise(sounds_path=[str(noises.path(profile[3:]))], min_snr_db=BG_SNR_DB[0], max_snr_db=BG_SNR_DB[1], p=1.0)]
    else:
        raise ValueError(f"unknown augment profile {profile!r}; choose from {', '.join(PROFILES)}")
    if snr_db is not None:  # --noise-snr: a fixed-SNR noise layer on top of whatever the profile does
        steps.append(AddColorNoise(min_snr_db=snr_db, max_snr_db=snr_db, min_f_decay=-3.0, max_f_decay=0.0, p=1.0))
    return Compose(steps) if steps else None


def apply(pcm: bytes, profile: str = "clean", snr_db: float | None = None, key: str = "") -> bytes:
    """PCM16 mono 16 kHz in, PCM16 mono 16 kHz out."""
    pipe = _pipeline(profile, snr_db)
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    if pipe is None or x.size == 0:
        return pcm
    seed = zlib.crc32(f"{profile}|{key}".encode())
    random.seed(seed)
    np.random.seed(seed)
    y = x.astype(np.float32) / 32768.0
    if profile == "phone":  # 16 -> 8 -> 16 kHz round trip (audiomentations' Resample would change the pitch/speed instead)
        y = soxr.resample(soxr.resample(y, SAMPLE_RATE, 8000), 8000, SAMPLE_RATE)
    y = pipe(y, sample_rate=SAMPLE_RATE)
    return (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()
