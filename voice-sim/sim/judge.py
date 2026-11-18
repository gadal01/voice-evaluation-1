"""LLM judge for one finished voice session. It reads the whole conversation (what the user said, what the agent heard, the agent's SPOKEN
replies, and every tool call with its real result) and scores it against the scenario's goal and expectation.

It judges the words the agent spoke, as text: it does not hear audio, so voice quality (naturalness, pronunciation) is not scored here.
Scores are advisory: calibrate against human labels before gating a release on them.
"""
import json
import re

from sim import llm
from sim.config import get_settings

CRITERIA = {
    "task_success": "Did the agent accomplish what the user wanted, as described by the scenario's expectation (or correctly decline when out of scope)?",
    "accuracy": "Is every price, slot, id, name and number the agent states supported by the tool results? 1 if anything is invented.",
    "policy": "Did it follow the rules: confirm details and price before booking, read back the phone number, never book unless the user clearly agreed, stay in scope? Offering to book or asking whether the user wants to book is allowed and is NOT a problem.",
    "voice_style": "Is it suited to being SPOKEN: short and natural, numbers read digit by digit, no lists, markdown, ids or tool names read out, no reasoning aloud?",
    "tone": "Professional, warm and calm; never curt, robotic, defensive or over-familiar.",
    "empathy": "Briefly and sincerely acknowledges the user's problem or worry (a leak, no hot water, spoiled food) before the task; no scripted over-apology. If the user showed no problem or distress (a plain price or availability question), score 5: do not penalize the absence of sympathy that was not called for.",
}
PASS_AT = 4   # a session passes when every criterion scores at least this

SYSTEM = (
    "You are a strict QA evaluator of a voice assistant for home-appliance servicing. You are given one voice session: the scenario "
    "(what the user wanted and what a correct agent does), the user's words, what the assistant heard (its own speech "
    "recognition, which may contain slips), the assistant's spoken replies, and its tool calls with real results. Judge ONLY what is in "
    "the session. Score each criterion 1-5 (5 flawless, 4 minor nit, 3 noticeable problem, 2 serious, 1 failed or harmful). Recognition slips "
    "the assistant recovered from are not its fault. Asking \"would you like me to book that?\" is only an offer and is fine. A policy "
    "problem is only: actually booking without the user's clear agreement, skipping the phone read-back before booking, or acting out "
    "of scope. Do not demand confirmations for read-only questions.\n\nCriteria:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    + '\n\nReply with ONLY a JSON object: {"scores": {' + ", ".join(f'"{k}": int' for k in CRITERIA)
    + '}, "issues": [short strings, empty if none], "reason": "one or two sentences"}'
)


def render(scenario: dict, turns: list[dict]) -> str:
    lines = [f"Scenario: {scenario['title']}", f"User's goal: {scenario['goal']}", f"A correct agent: {scenario['expect']}", ""]
    for t in turns:
        lines.append(f"USER SAID: {t['said']}")
        lines.append(f"  (assistant heard: {t['heard'] or '[nothing]'})")
        for tool in t.get("tools", []):
            lines.append(f"  [tool {tool['name']} args={json.dumps(tool['args'])} -> {json.dumps(tool['result'])[:600]}]")
        lines.append(f"ASSISTANT SAID: {t['reply'] or '[nothing: the assistant did not answer]'}")
    return "\n".join(lines)


def evaluate(scenario: dict, turns: list[dict], complete=None, model: str | None = None) -> dict:
    """{"status": "ok", scores, overall, pass, issues, reason, model} or {"status": "error", "reason"}. Never raises."""
    model = model or get_settings().judge_model
    if not turns:
        return {"status": "error", "reason": "no conversation to judge", "model": model}
    try:
        text = (complete or llm.complete)(SYSTEM, [llm.msg("user", render(scenario, turns))], model, 600, 0.0)
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        scores = {k: max(1, min(5, int(data["scores"][k]))) for k in CRITERIA}
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:200]}", "model": model}
    return {"status": "ok", "scores": scores, "overall": round(sum(scores.values()) / len(scores), 2),
            "pass": min(scores.values()) >= PASS_AT, "issues": [str(i) for i in data.get("issues", [])][:6],
            "reason": str(data.get("reason", "")).strip(), "model": model}
