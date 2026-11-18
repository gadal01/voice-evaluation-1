"""One thin wrapper over Bedrock Converse, used by the dynamic user and the judge. Auth is the Bedrock API key only."""
import os

from sim.config import get_settings


def _client():
    import boto3

    s = get_settings()
    if not s.aws_bearer_token_bedrock:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set in .env")
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = s.aws_bearer_token_bedrock   # boto3 reads the key from the environment
    return boto3.client("bedrock-runtime", region_name=s.aws_region)


def msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"text": text}]}


def complete(system: str, messages: list[dict], model: str, max_tokens: int = 300, temperature: float = 0.7) -> str:
    """The model's reply text. `messages` alternate user/assistant and start with a user message."""
    resp = _client().converse(modelId=model, system=[{"text": system}], messages=messages,
                              inferenceConfig={"maxTokens": max_tokens, "temperature": temperature})
    return "".join(b.get("text", "") for b in resp["output"]["message"]["content"]).strip()
