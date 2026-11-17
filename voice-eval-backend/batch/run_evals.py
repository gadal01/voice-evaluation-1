"""Voice-agent eval harness: synthetic callers (Polly / ElevenLabs) talk to the live WebSocket agent.

    python batch/run_evals.py                                  # default voices, all scenarios
    python batch/run_evals.py --voices polly:Joanna,eleven:Rachel --scenarios cost_quote,book_ac_inspection
    python batch/run_evals.py --repeat 3 --augment phone --concurrency 2
    python batch/run_evals.py --augment noisy --noise-snr 12 --no-judge

Start the agent first, WITHOUT auto-reload:   python -m uvicorn main:app --port 8000
Each call is a real Bedrock session (~30-90 s), so the run costs money and real time.

Per call we measure, against what the scenario expects:
  WER            caller words vs the agent's own transcript of them (jiwer; speech recognition quality)
  tools          expected MCP tool calls, in order, with argument checks; forbidden calls
  outcome        what the booking actually contained (read from the create_service_request result)
  TTFA           caller stops talking -> first agent audio byte (includes Sonic's end-of-speech wait)
  tool ms        time the MCP tool call itself took
  replied early the agent began answering before the caller finished (possible cut-off at a pause)
  judge          an LLM (Bedrock, batch/judge.py) scores task success, accuracy, policy and voice style 1-5; advisory,
                 it does not change PASS/FAIL. Disable with --no-judge.
Baselines: --save-baseline NAME keeps this run (metrics + recordings) as baselines/NAME/. Every run is compared with
baselines/default/ (or --baseline NAME) and regressions are flagged in the console and report; see batch/baseline.py.
Caller audio can be degraded with --augment {clean,noisy,phone,distant,hurried} (audiomentations, batch/augment.py).

Each run writes to batch/results/<timestamp>/ : report.html, transcripts/*.txt, audio/*.wav (stereo: caller left,
agent right), runs.jsonl, summary.json, meta.json.
Rebuild the report for any past run with:  python batch/report.py batch/results/<timestamp>
"""
import argparse
import asyncio
import collections
import json
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).parent))
import augment  # noqa: E402
import baseline  # noqa: E402
import clips as clip_library  # noqa: E402
from dialogue import Script  # noqa: E402
import judge  # noqa: E402
import metrics  # noqa: E402
import report  # noqa: E402
import tts  # noqa: E402

HERE = Path(__file__).parent
FRAME_BYTES, FRAME_S = 960, 0.03          # 30 ms of 16 kHz PCM16
OUT_BYTES_PER_S = 24_000 * 2              # agent audio is 24 kHz PCM16
DEFAULT_VOICES = "polly:Joanna,polly:Matthew,polly:Amy,polly:Kajal,eleven:Rachel,eleven:Adam"
SETTLE_S = 3.0                            # quiet time that means "the agent finished this turn"
TURN_TIMEOUT_S = 60
# Signs the model is reading its own reasoning / tool names aloud instead of just answering the caller.
THINKING_ALOUD = re.compile(
    r"\bthe user\b|\blet's (?:see|break|think)\b|\blet me think\b|\bfunction call\b"
    r"|\b(?:list|get|create|cancel)_ ?(?:serviceable|servicing|available|service)", re.I)


# ------------------------------------------------------------ one live call ---
class Call:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.events: list[dict] = []
        self.turn = -1
        self.ready = asyncio.Event()
        self.error: str | None = None
        self.last_activity = 0.0
        self.last_audio = 0.0
        self.audio_secs: dict[int, float] = collections.defaultdict(float)
        self.first_audio: dict[int, float] = {}
        self.user_start: dict[int, float] = {}
        self.user_end: dict[int, float] = {}
        self.early_audio: dict[int, float] = {}  # agent audio that began while the caller was still talking
        self.said: list[str] = []  # the caller lines actually spoken, in order (reactive scenarios vary per call)
        self.queue: collections.deque = collections.deque()
        self.caller_audio: list[tuple[float, bytes]] = []  # (time sent, 16 kHz PCM16 frame)
        self.agent_audio: list[tuple[float, bytes]] = []   # (time received, 24 kHz PCM16 chunk)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    async def receive(self, ws) -> None:
        try:
            async for msg in ws:
                t = self.now()
                if isinstance(msg, bytes):
                    self.last_activity = self.last_audio = t
                    if t >= self.user_end.get(self.turn, float("inf")):
                        self.first_audio.setdefault(self.turn, t)
                    elif self.turn in self.user_start:  # the agent started talking before the caller finished
                        self.early_audio.setdefault(self.turn, t)
                    self.audio_secs[self.turn] += len(msg) / OUT_BYTES_PER_S
                    self.agent_audio.append((t, msg))
                    continue
                ev = json.loads(msg)
                ev.update(t=t, turn=self.turn)
                self.events.append(ev)
                if ev["type"] == "ready":
                    self.ready.set()
                elif ev["type"] == "error":
                    self.error = ev["message"]
                    self.ready.set()
                else:
                    self.last_activity = t
        except websockets.ConnectionClosed:
            pass
        finally:
            self.ready.set()

    async def send_loop(self, ws) -> None:
        """Streams 30 ms frames at real time; silence whenever the caller isn't speaking."""
        nxt = time.perf_counter()
        silence = b"\x00" * FRAME_BYTES
        while True:
            from_queue = bool(self.queue)
            frame, done = self.queue.popleft() if from_queue else (silence, None)
            await ws.send(frame)
            if from_queue:
                self.caller_audio.append((self.now(), frame))
            if done:
                done.set_result(self.now())
            nxt += FRAME_S
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    async def say(self, pcm: bytes) -> float:
        """Queue a clip and wait until its last frame has been sent; returns that time."""
        done = asyncio.get_running_loop().create_future()
        frames = [pcm[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00") for i in range(0, len(pcm), FRAME_BYTES)]
        for k, f in enumerate(frames):
            self.queue.append((f, done if k == len(frames) - 1 else None))
        return await done

    def turn_done(self, i: int) -> bool:
        t = self.now()
        tools = [e["t"] for e in self.events if e["turn"] == i and e["type"] == "tool"]
        spoke = i in self.first_audio or any(e["turn"] == i and e["type"] == "reply" for e in self.events)
        if not spoke or (tools and self.last_audio <= max(tools)):  # nothing yet, or a tool result not answered yet
            return False
        playback_end = self.first_audio.get(i, t) + self.audio_secs[i]
        return t >= max(self.last_activity + SETTLE_S, playback_end + 0.5)


async def get_token(ws_url: str) -> str:
    """Exchanges AUTH_API_KEY for a JWT at the agent's /auth/token endpoint (a fresh one per call, they are short-lived)."""
    import os

    import httpx

    base = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1).rsplit("/ws", 1)[0]
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{base}/auth/token", headers={"X-API-Key": os.getenv("AUTH_API_KEY", "")})
    if r.status_code != 200:
        raise RuntimeError(f"could not get a token from {base}/auth/token (HTTP {r.status_code}): set AUTH_API_KEY in .env")
    return r.json()["access_token"]


async def run_call(url: str, script: Script, audio: dict[str, bytes]) -> Call:
    call = Call()
    async with websockets.connect(url, max_size=None, additional_headers={"Authorization": f"Bearer {await get_token(url)}"}) as ws:
        rx = asyncio.create_task(call.receive(ws))
        try:
            await asyncio.wait_for(call.ready.wait(), 30)
            if call.error:
                return call
            tx = asyncio.create_task(call.send_loop(ws))
            text, i = script.first(), 0
            while text is not None:
                await asyncio.sleep(0.5)  # a beat of silence before the caller speaks
                call.turn = i
                call.said.append(text)
                call.user_start[i] = call.now()
                call.user_end[i] = await call.say(audio[text])
                deadline = time.perf_counter() + TURN_TIMEOUT_S
                while not call.turn_done(i) and not call.error and time.perf_counter() < deadline and not rx.done():
                    await asyncio.sleep(0.1)
                if call.turn_done(i) is False:
                    call.events.append({"type": "timeout", "turn": i, "t": call.now()})
                    break
                # reactive scenarios choose the next line from what the agent just said (fixed ones just go in order)
                agent_text = " ".join(e["text"] for e in call.events if e["turn"] == i and e["type"] == "reply")
                text, i = script.next(agent_text), i + 1
            tx.cancel()
        finally:
            rx.cancel()
    return call


# -------------------------------------------------------------- audio export ---
def write_conversation_wav(call: Call, path: Path) -> float:
    """Stereo 16 kHz WAV on the call's real timeline: caller on the left, agent on the right.
    Agent chunks arrive faster than real time, so they are queued like a speaker would play them."""
    rate = 16_000
    caller = [(t, np.frombuffer(f, dtype="<i2")) for t, f in call.caller_audio]
    agent, cursor = [], 0.0
    for t, chunk in call.agent_audio:
        cursor = max(cursor, t)
        x = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype="<i2").astype(np.float64)
        n_out = int(len(x) * rate / 24_000)  # 24 kHz -> 16 kHz (linear; fine for listening back)
        agent.append((cursor, np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype("<i2") if len(x) > 1 else x.astype("<i2")))
        cursor += len(x) / 24_000
    end = max([t + len(x) / rate for t, x in caller] + [t + len(x) / rate for t, x in agent] + [0.0]) + 0.5
    left, right = np.zeros(int(end * rate), "<i2"), np.zeros(int(end * rate), "<i2")
    for track, items in ((left, caller), (right, agent)):
        for t, x in items:
            i = int(t * rate)
            track[i:i + len(x)] = x[: len(track) - i]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.column_stack([left, right]).tobytes())
    return end


# -------------------------------------------------------------------- score ---
def score(call: Call, scenario: dict, voice: str) -> dict:
    exp = scenario.get("expect", {})
    tools = [e for e in call.events if e["type"] == "tool"]
    tool_match = metrics.match_tools(exp.get("tools", []), tools)
    forbidden = [t["name"] for t in tools if t["name"] in exp.get("forbidden_tools", [])]
    tool_errors = [t["name"] for t in tools if isinstance(t["result"], dict) and "error" in t["result"]]
    outcome_ok, outcome_why = metrics.check_outcome(exp.get("outcome"), tools)

    reply_text = " ".join(e["text"] for e in call.events if e["type"] == "reply")
    missing_mentions = [p for p in exp.get("reply_mentions", []) if not re.search(p, reply_text, re.I)]
    thinking_aloud = bool(THINKING_ALOUD.search(reply_text))

    # Digits (e.g. the phone number) must be read back to the caller BEFORE the booking is made.
    first_create = next((t["t"] for t in tools if t["name"] == "create_service_request"), float("inf"))
    spoken = metrics.normalize(" ".join(e["text"] for e in call.events if e["type"] == "reply" and e["t"] < first_create))
    missing_readback = [d for d in exp.get("readback_digits", []) if not any(d in tok for tok in spoken)]

    said = call.said or Script(scenario).planned()
    turns, errs, words = [], 0, 0
    for i, ref in enumerate(said):
        heard = " ".join(e["text"] for e in call.events if e["turn"] == i and e["type"] == "transcript")
        wd = metrics.word_error_detail(ref, heard)
        e_, n_ = wd["errors"], wd["words"]
        errs, words = errs + e_, words + n_
        end = call.user_end.get(i)
        first_audio = call.first_audio.get(i)
        first_text = next((e["t"] for e in call.events if e["turn"] == i and e["type"] == "reply" and end and e["t"] >= end), None)
        turn_tools = [e for e in tools if e["turn"] == i]
        early = ([call.early_audio[i]] if i in call.early_audio else []) + [
            e["t"] for e in call.events if e["turn"] == i and e["type"] in ("reply", "tool") and end and e["t"] < end]
        early_ms = round((end - min(early)) * 1000) if early and end else None
        turns.append({
            "turn": i, "wer": e_ / n_ if n_ else 0.0, "heard": heard,
            "wer_sub": wd["sub"], "wer_del": wd["del"], "wer_ins": wd["ins"],
            "ttfa_ms": round((first_audio - end) * 1000) if end and first_audio else None,
            "first_text_ms": round((first_text - end) * 1000) if first_text else None,
            "tool_ms": sum(t.get("ms", 0) for t in turn_tools) if turn_tools else None,
            "agent_audio_s": round(call.audio_secs.get(i, 0.0), 1), "early_ms": early_ms,
        })

    conversation = []
    for i, ref in enumerate(said):
        conversation.append({"role": "caller", "turn": i, "t": call.user_start.get(i), "t_end": call.user_end.get(i), "text": ref})
        items = []
        heard_evs = [e for e in call.events if e["turn"] == i and e["type"] == "transcript"]
        if heard_evs:
            items.append({"role": "heard", "turn": i, "t": heard_evs[0]["t"], "text": turns[i]["heard"], "wer": turns[i]["wer"]})
        for e in call.events:
            if e["turn"] != i:
                continue
            if e["type"] == "reply":
                items.append({"role": "agent", "turn": i, "t": e["t"], "text": e["text"].strip()})
            elif e["type"] == "tool":
                items.append({"role": "tool", "turn": i, "t": e["t"], "name": e["name"], "args": e["args"],
                              "result": e["result"], "ms": e.get("ms")})
            elif e["type"] in ("timeout", "error"):
                items.append({"role": "system", "turn": i, "t": e["t"], "text": e.get("message") or "timed out waiting for the agent"})
        conversation += sorted(items, key=lambda x: x["t"])

    timed_out = any(e["type"] == "timeout" for e in call.events)
    passed = (not call.error and not timed_out and tool_match["score"] == 1.0 and not forbidden
              and not tool_errors and outcome_ok and not missing_mentions and not thinking_aloud and not missing_readback)
    return {
        "scenario": scenario["id"], "voice": voice, "passed": passed, "error": call.error, "timed_out": timed_out,
        "tool_score": tool_match["score"], "tools_missing": [m["name"] for m in tool_match["missing"]],
        "tools_called": [t["name"] for t in tools], "forbidden_called": forbidden, "tool_errors": tool_errors,
        "outcome_ok": outcome_ok, "outcome_why": outcome_why, "thinking_aloud": thinking_aloud, "missing_readback": missing_readback, "missing_mentions": missing_mentions,
        "wer_errors": errs, "wer_words": words, "turns": turns, "reply": reply_text.strip(),
        "conversation": conversation,
    }


# ------------------------------------------------------------------ summary ---
def fmt(v, unit="") -> str:
    return "-" if v is None else (f"{v:.0f}{unit}" if isinstance(v, (int, float)) and abs(v) >= 10 else f"{v:.2f}{unit}")


def summarize(results: list[dict], key: str) -> list[list[str]]:
    rows = [[key, "runs", "pass", "tools", "WER", "judge", "TTFA p50", "TTFA p95", "tool ms"]]
    for g in metrics.aggregate(results, key):
        rows.append([g["name"], str(g["runs"]), f"{100 * g['pass_rate']:.0f}%", fmt(g["tool_score"]),
                     f"{100 * g['wer']:.1f}%" if g["wer"] is not None else "-", fmt(g["judge"]),
                     fmt(g["ttfa_p50"], " ms"), fmt(g["ttfa_p95"], " ms"), fmt(g["tool_ms"], " ms")])
    return rows


def print_table(rows: list[list[str]]) -> None:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for k, r in enumerate(rows):
        print("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(r)))
        if k == 0:
            print("  ".join("-" * w for w in widths))


# --------------------------------------------------------------------- main ---
async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--voices", default=DEFAULT_VOICES, help="comma list, e.g. polly:Joanna,eleven:Rachel")
    ap.add_argument("--scenarios", default="all", help="comma list of scenario ids")
    ap.add_argument("--repeat", type=int, default=1, help="runs per scenario/voice (the model is non-deterministic)")
    ap.add_argument("--augment", default="clean", choices=augment.PROFILES, help="degrade the caller audio (audiomentations)")
    ap.add_argument("--noise-snr", type=float, default=None, help="add a fixed-SNR noise layer in dB on top of --augment (e.g. 15)")
    ap.add_argument("--baseline", default="default", help="baseline to compare against (baselines/<name>/); skipped if it does not exist")
    ap.add_argument("--no-baseline", action="store_true", help="do not compare against a baseline")
    ap.add_argument("--save-baseline", metavar="NAME", help="after the run, save it as baseline NAME")
    ap.add_argument("--force", action="store_true", help="with --save-baseline: replace an existing baseline")
    ap.add_argument("--baseline-audio", action="store_true", help="with --save-baseline: also keep a call recording per key")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any result regressed against the baseline")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--judge-model", default=judge.MODEL, help="Bedrock model/inference-profile id for the judge")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--no-audio", action="store_true", help="do not save the conversation audio (stereo WAV per call)")
    args = ap.parse_args()

    scenarios = json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))
    if args.scenarios != "all":
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s["id"] in wanted]
    voices = []
    for v in [x for spec in args.voices.split(",") for x in clip_library.expand(spec)]:
        why = tts.available(v)
        print(f"skipping {v}: {why}") if why else voices.append(v)
    if not scenarios or not voices:
        print("nothing to run")
        return 2

    print(f"rendering caller audio for {len(voices)} voice(s) x {len(scenarios)} scenario(s)...")
    clips: dict[tuple[str, str], dict[str, bytes]] = {}  # (voice, scenario) -> caller line -> audio
    for v in voices:
        for s in scenarios:
            try:
                clips[v, s["id"]] = {t: await asyncio.to_thread(tts.synthesize, v, t) for t in Script(s).lines()}
            except Exception as exc:
                print(f"  ! {v} / {s['id']}: {type(exc).__name__}: {str(exc)[:160]}")
    if args.augment != "clean" or args.noise_snr is not None:
        clips = {(v, sid): {t: augment.apply(c, args.augment, args.noise_snr, key=f"{sid}|{n}|{t}") for n, (t, c) in enumerate(cs.items())}
                 for (v, sid), cs in clips.items()}
    print(f"  characters sent to providers (cache misses): {tts.chars_rendered}")

    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs = [(v, s) for v in voices for s in scenarios for _ in range(args.repeat) if (v, s["id"]) in clips]
    if not jobs:
        print("no audio could be rendered; fix the errors above")
        return 2
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def one(i: int, v: str, s: dict) -> None:
        async with sem:
            try:
                call = await run_call(args.url, Script(s), clips[v, s["id"]])
                r = score(call, s, v)
                if not args.no_judge:
                    r["judge"] = await asyncio.to_thread(judge.evaluate, s, r, args.judge_model)
                if not args.no_audio and call.agent_audio:
                    name = f"{len(results) + 1:02d}_{report.slug(s['id'])}_{report.slug(v)}.wav"
                    r["audio_s"] = round(write_conversation_wav(call, out / "audio" / name), 1)
                    r["audio_file"] = f"audio/{name}"
            except Exception as exc:
                r = {"scenario": s["id"], "voice": v, "passed": False, "error": f"{type(exc).__name__}: {exc}",
                     "tool_score": 0.0, "wer_errors": 0, "wer_words": 0, "turns": []}
            results.append(r)
            mark = "PASS" if r["passed"] else "FAIL"
            ttfa = [t["ttfa_ms"] for t in r["turns"] if t.get("ttfa_ms") is not None]
            why = r["error"] or ", ".join(filter(None, [
                f"missing tools {r['tools_missing']}" if r.get("tools_missing") else "",
                f"forbidden {r['forbidden_called']}" if r.get("forbidden_called") else "",
                f"tool errors {r['tool_errors']}" if r.get("tool_errors") else "",
                f"outcome: {r['outcome_why']}" if not r.get("outcome_ok", True) else "",
                f"reply lacks {r['missing_mentions']}" if r.get("missing_mentions") else "",
                "agent narrated its reasoning aloud" if r.get("thinking_aloud") else "",
                f"no read-back of {r['missing_readback']} before booking" if r.get("missing_readback") else "",
                "timed out" if r.get("timed_out") else ""]))
            j = r.get("judge")
            jtxt = "" if not j else (f" judge={j['overall']:.1f}{'' if j['pass'] else '!'}" if "scores" in j else f" judge=ERR({j['error'][:60]})")
            print(f"[{len(results)}/{len(jobs)}] {mark} {v:<16} {s['id']:<20} ttfa={ttfa}{jtxt} {why}", flush=True)

    await asyncio.gather(*(one(i, v, s) for i, (v, s) in enumerate(jobs)))

    meta = {"created": datetime.now().isoformat(timespec="seconds"), "url": args.url, "voices": voices,
            "scenarios": [s["id"] for s in scenarios], "repeat": args.repeat, "augment": args.augment, "noise_snr_db": args.noise_snr,
            "judge_model": None if args.no_judge else args.judge_model,
            "concurrency": args.concurrency, "chars_rendered": tts.chars_rendered}
    cmp = None if args.no_baseline else baseline.compare(results, meta, args.baseline, out)
    if cmp:
        meta["baseline_cmp"] = cmp
    report_path = report.write_all(results, out, meta)  # also numbers the runs and writes transcripts/
    (out / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for key in ("voice", "scenario"):
        print(f"\nBy {key}")
        print_table(summarize(results, key))
    if cmp:
        baseline.print_comparison(cmp)
    elif not args.no_baseline and args.baseline != "default":
        print(f"\nno baseline named {args.baseline!r} (save one with --save-baseline)")
    if args.save_baseline:
        try:
            print(f"\nBaseline    : saved '{args.save_baseline}' -> {baseline.save(args.save_baseline, [out], args.force, args.baseline_audio)}")
        except FileExistsError as exc:
            print(f"\nBaseline    : NOT saved: {exc}")
    print(f"\nHTML report : {report_path}")
    print(f"Transcripts : {out / 'transcripts'}")
    print(f"Audio       : {out / 'audio'}" if not args.no_audio else "Audio       : skipped (--no-audio)")
    print(f"Raw data    : {out / 'runs.jsonl'}")
    if args.fail_on_regression and cmp and cmp["counts"]["regression"]:
        return 1
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
