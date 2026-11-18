"""Run the simulated user against the live voice agent (websocket + token auth) and judge every session.

    python -m sim.run                                        # static clips, every scenario
    python -m sim.run --mode dynamic --scenarios ac_not_cooling_book,fridge_leaking_book
    python -m sim.run --mode both --repeat 3 --voice en_US-lessac-medium --concurrency 2
    python -m sim.run --no-judge                             # only talk to the agent, skip the LLM judge

  static   the scenario's fixed lines, replayed from stored Piper clips (python -m sim.clips render): identical audio every run
  dynamic  an LLM plays the customer (goal, facts, persona), reacts to what the agent says, and each line is spoken by Piper

Start the agent first (voice-agent, port 8000). Every session is a real Bedrock session and costs money.
Each run writes results/<timestamp>/: report.html, summary.json and one folder per session (session.json, user.wav, agent.wav).
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import statistics
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from sim import clips, judge, tts
from sim.session import Session, SessionError, get_token, write_wav
from sim.user import DynamicUser, StaticUser, load_scenarios
from sim.config import get_settings
from sim.report import render_report

MAX_SILENT_TURNS = 2   # hang up after this many unanswered lines in a row


def default_speaker(mode: str, scenario: dict, voice: str) -> Callable[[int, str], bytes]:
    if mode == "static":
        return lambda i, text: clips.get(scenario["id"], i, text, voice)
    return lambda i, text: tts.synthesize(voice, text)


async def run_session(scenario: dict, mode: str, voice: str, out_dir: Path, *, url: str, api_key: str = "", judge_on: bool = True,
                   settle: float = 2.5, token: str | None = None, speaker: Callable[[int, str], bytes] | None = None,
                   user_agent=None) -> dict:
    """One session from start to judged result. Never raises: a failure is recorded in result["error"]."""
    user = StaticUser(scenario) if mode == "static" else DynamicUser(scenario, user_agent)
    speak = speaker or default_speaker(mode, scenario, voice)
    out = {"scenario": scenario["id"], "title": scenario["title"], "mode": mode, "voice": voice, "error": None, "traceback": None,
           "turns": [], "judge": None, "passed": None, "dir": out_dir.name, "duration_s": 0.0}
    session = Session(url, "", settle)
    history: list[dict] = []
    try:
        session.token = token or await get_token(url, api_key)
        async with session:
            silent = 0
            for i in itertools.count():
                line = await asyncio.to_thread(user.next_line, history)       # the dynamic user waits on an LLM
                if line is None:
                    break
                pcm = await asyncio.to_thread(speak, i, line)                   # Piper (or a stored clip)
                await session.say(line, pcm)
                reply = await session.listen()
                history.append({"user": line, "agent": reply})
                silent = silent + 1 if not session.turns[-1]["answered"] else 0
                if silent >= MAX_SILENT_TURNS:
                    break
    except Exception as exc:
        out["error"] = str(exc) if isinstance(exc, (SessionError, LookupError, ValueError, RuntimeError)) else f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    if session.turns:
        data = session.result()
        out.update(turns=data["turns"], duration_s=data["duration_s"], events=data["events"])
        out_dir.mkdir(parents=True, exist_ok=True)
        write_wav(out_dir / "user.wav", data["user_pcm"], 16_000)
        write_wav(out_dir / "agent.wav", data["agent_pcm"], 24_000)
        if out["error"] is None and not any(t.get("answered") for t in out["turns"]):
            out["error"] = "the agent never answered"
    if judge_on and out["turns"]:
        out["judge"] = await asyncio.to_thread(judge.evaluate, scenario, out["turns"])
    if judge_on:
        out["passed"] = out["error"] is None and bool(out["judge"] and out["judge"]["status"] == "ok" and out["judge"]["pass"])
    elif out["error"]:
        out["passed"] = False
    if out["turns"] or out["error"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "session.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


# ------------------------------------------------------------------ summary ---
def aggregate(group: list[dict]) -> dict:
    judged = [r["judge"] for r in group if r.get("judge") and r["judge"]["status"] == "ok"]
    turns = [t for r in group for t in r["turns"]]
    ttfa = [t["ttfa_ms"] for t in turns if t.get("ttfa_ms") is not None]
    words = sum(t["wer"]["words"] for t in turns if t.get("wer"))
    errors = sum(t["wer"]["errors"] for t in turns if t.get("wer"))
    criteria = sorted({k for j in judged for k in j["scores"]})
    return {
        "sessions": len(group), "passed": sum(1 for r in group if r["passed"]), "errors": sum(1 for r in group if r["error"]),
        "pass_rate": (sum(1 for r in group if r["passed"]) / len(group)) if group else None,
        "judge_overall": round(statistics.fmean(j["overall"] for j in judged), 2) if judged else None,
        "scores": {k: round(statistics.fmean(j["scores"][k] for j in judged), 2) for k in criteria},
        "ttfa_p50_ms": statistics.median(ttfa) if ttfa else None,
        "heard_wer": round(errors / words, 4) if words else None,
    }


def summarize(results: list[dict]) -> dict:
    def by(key: str) -> dict:
        names = sorted({r[key] for r in results})
        return {n: aggregate([r for r in results if r[key] == n]) for n in names}

    return {"overall": aggregate(results), "by_mode": by("mode"), "by_scenario": by("scenario")}


# ------------------------------------------------------------------- engine ---
class RunError(Exception):
    """A run cannot start: a bad option or a missing prerequisite. The message says what to fix."""


@dataclass
class RunOptions:
    mode: str = "static"                       # static | dynamic | both
    scenarios: list[str] = field(default_factory=list)   # empty = all
    voice: str = clips.DEFAULT_VOICE
    repeat: int = 1
    concurrency: int = 1
    url: str = ""                              # agent websocket; empty = AGENT_URL
    settle: float = 2.5
    judge: bool = True


def preflight(scenarios: list[dict], modes: list[str], voice: str, judge_on: bool) -> None:
    """Fail before spending anything if a prerequisite is missing."""
    s = get_settings()
    if not s.auth_api_key:
        raise RunError("AUTH_API_KEY is not set in .env (it must match the agent's)")
    needs_llm = judge_on or "dynamic" in modes
    if needs_llm and not s.aws_bearer_token_bedrock:
        raise RunError("AWS_BEARER_TOKEN_BEDROCK is not set in .env (needed for the judge and the dynamic user; use --no-judge with --mode static to skip)")
    if "dynamic" in modes and (why := tts.available(voice)):
        raise RunError(why)
    if "static" in modes:
        for sc in scenarios:
            for i, text in enumerate(sc["static_turns"]):
                try:
                    clips.get(sc["id"], i, text, voice)
                except (LookupError, ValueError) as exc:
                    raise RunError(str(exc))


def plan(opts: RunOptions) -> tuple[list[dict], list[str]]:
    """Validates the options and the prerequisites. Returns (scenarios, modes) or raises RunError."""
    everything = load_scenarios()
    wanted = opts.scenarios or list(everything)
    unknown = [x for x in wanted if x not in everything]
    if unknown:
        raise RunError(f"unknown scenario(s) {unknown}; available: {', '.join(everything)}")
    if opts.mode not in ("static", "dynamic", "both"):
        raise RunError(f"mode must be static, dynamic or both, not {opts.mode!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", opts.voice):
        raise RunError(f"invalid voice name {opts.voice!r}")
    if opts.repeat < 1 or opts.concurrency < 1:
        raise RunError("repeat and concurrency must be at least 1")
    scenarios = [everything[x] for x in wanted]
    modes = ["static", "dynamic"] if opts.mode == "both" else [opts.mode]
    preflight(scenarios, modes, opts.voice, opts.judge)
    return scenarios, modes


async def execute(opts: RunOptions, run_dir: Path, on_result: Callable[[dict, int, int], None] | None = None) -> dict:
    """Runs every (scenario, mode, repeat) session against the agent, judges them, writes summary.json and report.html into
    `run_dir`, and returns the summary. `on_result(result, done, total)` is called as each session finishes. Used by the CLI
    and by the FastAPI layer."""
    s = get_settings()
    scenarios, modes = plan(opts)
    url = opts.url or s.agent_url
    jobs = [(sc, m) for sc in scenarios for m in modes for _ in range(opts.repeat)]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({"options": asdict(opts), "created": datetime.now().isoformat(timespec="seconds"),
                                                  "total": len(jobs)}, indent=2), encoding="utf-8")
    sem = asyncio.Semaphore(opts.concurrency)
    results: list[dict] = []

    async def one(n: int, sc: dict, mode: str) -> None:
        async with sem:
            res = await run_session(sc, mode, opts.voice, run_dir / f"{n:02d}_{sc['id']}_{mode}", url=url, api_key=s.auth_api_key,
                                    judge_on=opts.judge, settle=opts.settle)
        results.append(res)
        if on_result:
            on_result(res, len(results), len(jobs))

    await asyncio.gather(*(one(i + 1, sc, m) for i, (sc, m) in enumerate(jobs)))
    results.sort(key=lambda r: r["dir"])
    summary = {"meta": {"created": datetime.now().isoformat(timespec="seconds"), "url": url, "mode": opts.mode, "voice": opts.voice,
                        "repeat": opts.repeat, "judge_model": s.judge_model if opts.judge else None,
                        "user_model": s.user_model if "dynamic" in modes else None},
               **summarize(results)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(render_report(results, summary), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------- cli ---
async def main_async(args: argparse.Namespace) -> int:
    opts = RunOptions(mode=args.mode, scenarios=[x for x in args.scenarios.split(",") if x], voice=args.voice, repeat=args.repeat,
                      concurrency=args.concurrency, url=args.url, settle=args.settle, judge=not args.no_judge)
    run_dir = Path(args.out) if args.out else get_settings().results_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        scenarios, modes = plan(opts)
    except RunError as exc:
        raise SystemExit(str(exc))
    print(f"{len(scenarios) * len(modes) * opts.repeat} session(s) -> {run_dir}   mode={opts.mode} voice={opts.voice} judge={'on' if opts.judge else 'off'}")

    def show(res: dict, done: int, total: int) -> None:
        j = res["judge"]
        mark = "PASS" if res["passed"] else ("FAIL" if res["passed"] is False else "done")
        detail = res["error"] or (f"judge {j['overall']}" if j and j["status"] == "ok" else (j or {}).get("reason", ""))
        print(f"  {mark}  {res['scenario']:<28} {res['mode']:<8} {res['duration_s']:>5.1f}s  {detail}")

    try:
        summary = await execute(opts, run_dir, show)
    except RunError as exc:
        raise SystemExit(str(exc))
    o = summary["overall"]
    print("\n" + f"{o['passed']}/{o['sessions']} passed" + (f", judge {o['judge_overall']}" if o["judge_overall"] else "") + f"   report: {run_dir / 'report.html'}")
    return 1 if args.fail_on_fail and o["passed"] < o["sessions"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["static", "dynamic", "both"], default="static")
    ap.add_argument("--scenarios", default="", help="comma list of scenario ids (default: all)")
    ap.add_argument("--voice", default=clips.DEFAULT_VOICE, help="Piper voice (model file stem in models/)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--url", default="", help="agent websocket (default AGENT_URL or ws://127.0.0.1:8000/ws)")
    ap.add_argument("--settle", type=float, default=2.5, help="seconds of quiet that mean the agent finished its turn")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--fail-on-fail", action="store_true", help="exit 1 if any session failed (for CI)")
    ap.add_argument("--out", default="", help="output folder (default results/<timestamp>)")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
