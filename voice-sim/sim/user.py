"""The simulated user: decides WHAT to say next. HOW it sounds (Piper) is decided elsewhere.

  StaticUser   a fixed list of lines (the scenario's static_turns), the same every run: regression testing
  DynamicUser  an LLM plays the customer from the scenario's goal, facts and persona and reacts to what the agent just said

Both implement next_line(history) -> str | None, where history is [{"user": str, "agent": str}, ...] for the turns so far.
None means "hang up".
"""
import json
from pathlib import Path
from typing import Callable

from sim import llm
from sim.config import ROOT, get_settings
from sim.text import spoken

END = "[END]"


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
        f"- When your goal is complete, or the assistant cannot help, say a brief thanks and end your message with {END}."
    )


class DynamicUser:
    mode = "dynamic"

    def __init__(self, scenario: dict, complete: Callable[..., str] | None = None, model: str | None = None) -> None:
        self.scenario = scenario
        self.max_turns = scenario.get("max_turns", 6)
        self._complete = complete or llm.complete
        self._model = model or get_settings().user_model
        self._system = system_prompt(scenario)
        self._done = False

    def _messages(self, history: list[dict]) -> list[dict]:
        out = [llm.msg("user", "The voice session is open and the assistant is listening. Say your first line.")]
        for turn in history:
            out.append(llm.msg("assistant", turn["user"]))
            said = turn["agent"].strip() or "(the assistant said nothing)"
            out.append(llm.msg("user", f"The assistant says: {said}"))
        return out

    def next_line(self, history: list[dict]) -> str | None:
        if self._done or len(history) >= self.max_turns:
            return None
        text = self._complete(self._system, self._messages(history), self._model)
        if END in text:
            self._done = True
            text = text.replace(END, "")
        text = spoken(text.strip().strip('"').strip())
        return text or None
