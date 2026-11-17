"""Settings for the evaluation backend, read once and typed. Real environment variables win over the nearest `.env`.

    AUTH_API_KEY           key the UI sends as X-API-Key (use the same value as the agent so one key works for both)
    AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN     Bedrock, for the LLM judge / intent classifier
    QA_JUDGE_MODEL         Bedrock model id used by the judge (default Nova Pro; any Converse-capable model works)
    DATA_DIR               where recorded sessions are stored (default ./data)
    CORS_ORIGINS           comma-separated browser origins allowed to call this API (empty = none)
    UI_DIST                path to the built voice-eval-ui (its dist/ folder); if set, served at /
    PORT                   listening port for `python -m voice_eval` (default 8001)
"""
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    auth_api_key: str = ""

    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    qa_judge_model: str = "us.amazon.nova-pro-v1:0"

    data_dir: Path = ROOT / "data"
    cors_origins: str = ""
    ui_dist: str | None = None
    port: int = Field(8001, ge=1, le=65535)


@lru_cache
def get_settings() -> Settings:
    """Loaded once per process. In tests, call get_settings.cache_clear() after changing the environment."""
    return Settings(_env_file=find_dotenv() or None)
