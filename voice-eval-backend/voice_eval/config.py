"""Settings, read once and typed. Real environment variables win over the nearest `.env`.

    AWS_BEARER_TOKEN_BEDROCK   Bedrock API key for the LLM judge / intent classifier (the only AWS credential used)
    AWS_REGION                 Bedrock region (default us-east-1)
    QA_JUDGE_MODEL             Bedrock model id for the judge (default Nova Pro; any Converse-capable model works)
"""
from functools import lru_cache

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    aws_region: str = "us-east-1"
    aws_bearer_token_bedrock: str | None = None
    qa_judge_model: str = "us.amazon.nova-pro-v1:0"


@lru_cache
def get_settings() -> Settings:
    """Loaded once per process. In tests, call get_settings.cache_clear() after changing the environment."""
    return Settings(_env_file=find_dotenv() or None)
