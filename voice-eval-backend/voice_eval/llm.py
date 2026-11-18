"""LLM-based metrics via Bedrock Converse: conversation quality (tone, clarity, empathy, ...) and intent classification.

Both degrade to {"status": "skipped", "reason": ...} when AWS credentials or boto3 are missing, so the rest of the
evaluation still works. Judge scores are advisory: calibrate them against human labels before gating a release on them.
"""
from __future__ import annotations

import json
import os
import re

from voice_eval.config import get_settings

INTENTS = ["book_service", "get_quote", "check_availability", "cancel_request", "request_status", "out_of_scope"]
QUALITY = {
    "tone": "Professional, warm and calm; never curt, robotic, defensive or over-familiar.",
    "clarity": "Short, unambiguous, speakable sentences; one question at a time; no lists, ids or tool names read out; numbers read digit by digit when needed.",
    "empathy": "Acknowledges the caller's problem or worry (a leak, no hot water, spoiled food) briefly and sincerely before moving to the task; no scripted over-apology.",
    "accuracy": "Every price, slot, id, name and number stated is supported by the tool results; 1 if anything is invented.",
    "policy": "Confirms details and price and reads the phone number back before booking; books only after a clear yes; stays in scope.",
}
PASS_AT = 4


def _client():
    import boto3

    s = get_settings()
    if s.aws_bearer_token_bedrock:   # Bedrock API key: boto3 reads it from the environment, .env is not copied there for us
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = s.aws_bearer_token_bedrock
        return boto3.client("bedrock-runtime", region_name=s.aws_region)
    raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set in .env")


def render(events: list[dict]) -> str:
    out = []
    for e in sorted(events, key=lambda e: e["t"]):
        if e["type"] == "transcript":
            out.append(f"CALLER (as heard by agent): {e['text']}")
        elif e["type"] == "reply":
            out.append(f"AGENT: {e['text']}")
        elif e["type"] == "tool":
            out.append(f"  [tool {e['name']} {json.dumps(e.get('args'))} -> {json.dumps(e.get('result'))[:500]}]")
        elif e["type"] == "interrupted":
            out.append("  [agent was interrupted]")
    return "\n".join(out)


def _converse(system: str, user: str, max_tokens: int = 500) -> tuple[dict, dict]:
    resp = _client().converse(modelId=get_settings().qa_judge_model, system=[{"text": system}], messages=[{"role": "user", "content": [{"text": user}]}],
                              inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    usage = resp.get("usage", {})
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0)), {"in": usage.get("inputTokens", 0), "out": usage.get("outputTokens", 0)}


def judge(events: list[dict], goal: str = "") -> dict:
    system = ("You are a strict QA evaluator of a phone voice assistant for home-appliance servicing. Judge ONLY the "
              "conversation given. Score each criterion 1-5 (5 flawless, 4 minor nit, 3 noticeable, 2 serious, 1 failed). "
              "Speech-recognition slips the agent recovered from are not its fault.\nCriteria:\n"
              + "\n".join(f"- {k}: {v}" for k, v in QUALITY.items())
              + '\nReply with ONLY JSON: {"scores": {' + ", ".join(f'"{k}": int' for k in QUALITY)
              + '}, "issues": [short strings], "reason": "one or two sentences"}')
    try:
        data, tok = _converse(system, f"Caller goal: {goal or 'unknown'}\n\n{render(events)}")
        scores = {k: max(1, min(5, int(data["scores"][k]))) for k in QUALITY}
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {"status": "ok", "scores": scores, "overall": round(sum(scores.values()) / len(scores), 2),
            "pass": min(scores.values()) >= PASS_AT, "issues": [str(i) for i in data.get("issues", [])][:6],
            "reason": str(data.get("reason", "")), "model": get_settings().qa_judge_model, "tokens": tok}


def classify_intent(events: list[dict]) -> dict:
    caller = " ".join(e["text"] for e in sorted(events, key=lambda e: e["t"]) if e["type"] == "transcript")
    system = ("Classify what the caller of a home-appliance servicing line wants. Choose exactly one of: "
              + ", ".join(INTENTS) + ". out_of_scope = anything that is not appliance/electronics/plumbing servicing. "
              'Reply with ONLY JSON: {"intent": "<one>", "confidence": 0-1}')
    if not caller.strip():
        return {"status": "skipped", "reason": "no caller transcript"}
    try:
        data, tok = _converse(system, caller, 80)
        intent = str(data["intent"])
        if intent not in INTENTS:
            raise ValueError(f"unknown intent {intent!r}")
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {"status": "ok", "intent": intent, "confidence": data.get("confidence"), "model": get_settings().qa_judge_model, "tokens": tok}
