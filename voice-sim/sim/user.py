"""The simulated user: decides WHAT to say next. HOW it sounds (Piper) is decided elsewhere.

  StaticUser   a fixed list of lines (the scenario's static_turns), the same every run: regression testing
  DynamicUser  a Strands agent (Amazon Bedrock model) that plays the customer from the scenario's goal, facts and persona,
               remembers the conversation, reacts to what the voice agent just said, and hangs up by calling its
               `end_conversation` tool

Both implement next_line(history) -> str | None, where history is [{"user": str, "agent": str}, ...] for the turns so far.
None means "hang up".
"""
import json
from pathlib import Path

from sim.config import ROOT, get_settings
from sim.text import spoken

FIRST = "The voice session is open and the assistant is listening. Say your first line."


def load_scenarios(path: Path | None = None) -> dict[str, dict]:
    items = json.loads((path or ROOT / "scenarios.json").read_text(encoding="utf-8"))
    for s in items:
        missing = {"id", "title", "goal", "static_turns", "expect"} - set(s)
        if missing:
            raise ValueError(f"scenario {s.get('id')!r} is missing {sorted(missing)}")
    return {s["id"]: s for s in items}


class StaticUser:
    mode = "static"

    def __init__(self, scenario: dict) -> None:
        self.lines = list(scenario["static_turns"])

    def next_line(self, history: list[dict]) -> str | None:
        return self.lines[len(history)] if len(history) < len(self.lines) else None


def system_prompt(scenario: dict) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in (scenario.get("facts") or {}).items() if v) or "- (nothing special)"
    return (
        "You are role-playing a CUSTOMER of a home-servicing company. You are speaking by voice to the company's automated voice assistant.\n"
        f"Your goal: {scenario['goal']}\n"
        f"Who you are and what you know:\n{facts}\n"
        f"Your manner: {scenario.get('persona') or 'ordinary and polite'}\n\n"
        "Rules:\n"
        "- Say only what you would say ALOUD: one or two short, natural sentences. No stage directions, no quotes, no markdown.\n"
        "- Answer what the assistant asks. Give your details (name, phone number, appliance, problem) when asked or when it is natural; "
        "do not recite everything at once unless that is natural.\n"
        '- Say phone numbers as words, digit by digit ("five five five, zero one zero zero").\n'
        "- If the assistant reads details back and they are correct, confirm clearly. If something is wrong, correct it.\n"
        "- Never invent facts that are not listed above. If you do not know something, say you are not sure.\n"
        "- When your goal is complete, or the assistant cannot help, do NOT reply normally: call the end_conversation tool with your "
        "brief thanks as the farewell."
    )


class DynamicUser:
    mode = "dynamic"

    def __init__(self, scenario: dict, agent=None, model: str | None = None) -> None:
        """`agent` is anything callable with a prompt that returns the reply text (a Strands Agent; a fake in tests).
        By default a Strands Agent is built on the Bedrock model USER_MODEL, authenticated with the Bedrock API key."""
        self.scenario = scenario
        self.max_turns = scenario.get("max_turns", 6)
        self._done = False
        self._farewell: str | None = None
        self._agent = agent if agent is not None else self._build_agent(model or get_settings().user_model)

    def end_conversation(self, farewell: str) -> str:
        """What the agent's `end_conversation` tool does: remember the last words and stop after them."""
        self._farewell = farewell
        return "The session is closed. Do not say anything else."

    def _build_agent(self, model_id: str):
        from strands import Agent, tool
        from strands.models import BedrockModel

        s = get_settings()
        if not s.aws_bearer_token_bedrock:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set in .env")

        @tool
        def end_conversation(farewell: str) -> str:
            """Hang up. Call this once your goal is complete or the assistant cannot help. `farewell` is the last thing you say aloud."""
            return self.end_conversation(farewell)

        model = BedrockModel(model_id=model_id, region_name=s.aws_region, api_key=s.aws_bearer_token_bedrock, temperature=0.7, max_tokens=300)
        return Agent(model=model, system_prompt=system_prompt(self.scenario), tools=[end_conversation], callback_handler=None)

    def next_line(self, history: list[dict]) -> str | None:
        """The Strands agent keeps the conversation itself, so only the newest thing the voice agent said is passed in."""
        if self._done or len(history) >= self.max_turns:
            return None
        prompt = FIRST if not history else f"The assistant says: {history[-1]['agent'].strip() or '(the assistant said nothing)'}"
        text = str(self._agent(prompt)).strip()
        if self._farewell is not None:               # the agent hung up: its farewell is the last line, then no more
            text, self._done = self._farewell, True
        text = spoken(text.strip().strip('"').strip())
        return text or None
