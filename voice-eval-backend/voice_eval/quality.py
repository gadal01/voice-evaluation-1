"""TTS naturalness of the agent's audio: UTMOS and DNSMOS, both OPTIONAL (heavy dependencies).

    pip install speechmos onnxruntime            # DNSMOS (P.808 / OVRL / SIG / BAK), CPU, ~seconds per clip
    pip install torch torchaudio                 # UTMOS22 via torch.hub (tarepan/SpeechMOS), downloads weights on first use

Each metric returns {"status": "ok", ...} or {"status": "skipped", "reason": ...}. They are no-reference predictors
trained on listening tests: use them to rank/track, and keep the human spot check (rating form in the console) as the
ground truth. Agent audio is resampled 24 -> 16 kHz, and only the agent's speech regions are scored.
"""
from __future__ import annotations

import numpy as np

from . import scoring

_utmos = None


def _agent_speech_16k(agent_pcm: bytes, segs: list[tuple[float, float]]) -> np.ndarray:
    import soxr

    x = scoring.to_float(agent_pcm)
    parts = [x[int(a * scoring.AGENT_RATE): int(b * scoring.AGENT_RATE)] for a, b in segs]
    x = np.concatenate(parts) if parts else x
    return soxr.resample(x, scoring.AGENT_RATE, 16_000).astype(np.float32)


def dnsmos(agent_pcm: bytes, segs: list[tuple[float, float]]) -> dict:
    try:
        from speechmos import dnsmos as dm
    except Exception as exc:
        return {"status": "skipped", "reason": f"speechmos not installed ({type(exc).__name__}); pip install speechmos onnxruntime"}
    x = _agent_speech_16k(agent_pcm, segs)
    if len(x) < 16_000:
        return {"status": "skipped", "reason": "less than 1 s of agent speech"}
    try:
        r = dm.run(x, sr=16_000)
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    return {"status": "ok", **{k: round(float(r[k]), 2) for k in ("ovrl_mos", "sig_mos", "bak_mos", "p808_mos") if k in r}}


def utmos(agent_pcm: bytes, segs: list[tuple[float, float]]) -> dict:
    global _utmos
    try:
        import torch
    except Exception as exc:
        return {"status": "skipped", "reason": f"torch not installed ({type(exc).__name__}); pip install torch torchaudio"}
    x = _agent_speech_16k(agent_pcm, segs)
    if len(x) < 16_000:
        return {"status": "skipped", "reason": "less than 1 s of agent speech"}
    try:
        if _utmos is None:
            _utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_remote_code=True)
        with torch.no_grad():
            score = _utmos(torch.from_numpy(x).unsqueeze(0), 16_000)
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    return {"status": "ok", "utmos": round(float(score.mean()), 2)}
