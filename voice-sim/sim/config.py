"""Settings, read once and typed. Real environment variables win over the nearest `.env`.

    AUTH_API_KEY               the voice agent's AUTH_API_KEY: used to get a token from the agent
    AGENT_URL                  the agent's websocket (default ws://127.0.0.1:8000/ws)
    AWS_BEARER_TOKEN_BEDROCK   Bedrock API key for the simulated user (dynamic mode) and the LLM judge
    AWS_REGION                 Bedrock region (default us-east-1)
    JUDGE_MODEL / USER_MODEL Bedrock model ids (default Nova Pro; any Converse-capable model works)
"""
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    auth_api_key: str = ""
    agent_url: str = "ws://127.0.0.1:8000/ws"
    aws_region: str = "us-east-1"
    aws_bearer_token_bedrock: str | None = None
    judge_model: str = "us.amazon.nova-pro-v1:0"
    user_model: str = "us.amazon.nova-pro-v1:0"


@lru_cache
def get_settings() -> Settings:
    """Loaded once per process. In tests, call get_settings.cache_clear() after changing the environment."""
    return Settings(_env_file=find_dotenv() or None)
