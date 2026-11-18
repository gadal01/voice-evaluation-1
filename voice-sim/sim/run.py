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
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

from sim import clips, judge, tts
from sim.session import Session, SessionError, get_token, write_wav
from sim.user import DynamicUser, StaticUser, load_scenarios
from sim.config import ROOT, get_settings
from sim.report import render_report

MAX_SILENT_TURNS = 2   # hang up after this many unanswered lines in a row


def default_speaker(mode: str, scenario: dict, voice: str) -> Callable[[int, str], bytes]:
    if mode == "static":
        return lambda i, text: clips.get(scenario["id"], i, text, voice)
    return lambda i, text: tts.synthesize(voice, text)


async def run_session(scenario: dict, mode: str, voice: str, out_dir: Path, *, url: str, api_key: str = "", judge_on: bool = True,
                   settle: float = 2.5, token: str | None = None, speaker: Callable[[int, str], bytes] | None = None,
                   complete: Callable[..., str] | None = None) -> dict:
    """One session from start to judged result. Never raises: a failure is recorded in result["error"]."""
    user = StaticUser(scenario) if mode == "static" else DynamicUser(scenario, complete)
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


# ---------------------------------------------------------------------- cli ---
def preflight(scenarios: list[dict], modes: list[str], voice: str, judge_on: bool) -> None:
    """Fail before spending anything if a prerequisite is missing."""
    s = get_settings()
    if not s.auth_api_key:
        raise SystemExit("AUTH_API_KEY is not set in .env (it must match the agent's)")
    needs_llm = judge_on or "dynamic" in modes
    if needs_llm and not s.aws_bearer_token_bedrock:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK is not set in .env (needed for the judge and the dynamic user; use --no-judge with --mode static to skip)")
    if "dynamic" in modes and (why := tts.available(voice)):
        raise SystemExit(why)
    if "static" in modes:
        for sc in scenarios:
            for i, text in enumerate(sc["static_turns"]):
                try:
                    clips.get(sc["id"], i, text, voice)
                except (LookupError, ValueError) as exc:
                    raise SystemExit(str(exc))


async def main_async(args: argparse.Namespace) -> int:
    s = get_settings()
    everything = load_scenarios()
    wanted = [x for x in args.scenarios.split(",") if x] or list(everything)
    unknown = [x for x in wanted if x not in everything]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; available: {', '.join(everything)}")
    scenarios = [everything[x] for x in wanted]
    modes = ["static", "dynamic"] if args.mode == "both" else [args.mode]
    judge_on = not args.no_judge
    preflight(scenarios, modes, args.voice, judge_on)

    run_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs = [(sc, m, r) for sc in scenarios for m in modes for r in range(args.repeat)]
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    print(f"{len(jobs)} session(s) -> {run_dir}   mode={args.mode} voice={args.voice} judge={'on' if judge_on else 'off'}")

    async def one(n: int, sc: dict, mode: str) -> None:
        async with sem:
            res = await run_session(sc, mode, args.voice, run_dir / f"{n:02d}_{sc['id']}_{mode}", url=args.url or s.agent_url,
                                 api_key=s.auth_api_key, judge_on=judge_on, settle=args.settle)
        results.append(res)
        j = res["judge"]
        mark = "PASS" if res["passed"] else ("FAIL" if res["passed"] is False else "done")
        detail = res["error"] or (f"judge {j['overall']}" if j and j["status"] == "ok" else (j or {}).get("reason", ""))
        print(f"  {mark}  {sc['id']:<28} {mode:<8} {res['duration_s']:>5.1f}s  {detail}")

    await asyncio.gather(*(one(i + 1, sc, m) for i, (sc, m, _) in enumerate(jobs)))
    results.sort(key=lambda r: r["dir"])
    summary = {"meta": {"created": datetime.now().isoformat(timespec="seconds"), "url": args.url or s.agent_url, "mode": args.mode,
                        "voice": args.voice, "repeat": args.repeat, "judge_model": s.judge_model if judge_on else None,
                        "user_model": s.user_model if "dynamic" in modes else None},
               **summarize(results)}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(render_report(results, summary), encoding="utf-8")
    o = summary["overall"]
    print(f"\n{o['passed']}/{o['sessions']} passed" + (f", judge {o['judge_overall']}" if o["judge_overall"] else "") + f"   report: {run_dir / 'report.html'}")
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
