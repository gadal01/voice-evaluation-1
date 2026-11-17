"""API-key check for the evaluation API. Sessions contain call recordings, so every /api route requires it."""
import hmac

from fastapi import Header, HTTPException

from voice_eval.config import get_settings


def check_api_key(candidate: str | None) -> bool:
    expected = get_settings().auth_api_key
    return bool(expected) and bool(candidate) and hmac.compare_digest(candidate.encode(), expected.encode())


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not check_api_key(x_api_key):
        raise HTTPException(401, "invalid or missing API key")
