"""One simulated voice session with the voice agent under test. The agent is a websocket service protected by a token:

    1. POST {agent}/auth/token with header X-API-Key: <AUTH_API_KEY>  ->  a short-lived JWT        (get_token)
    2. open ws://{agent}/ws with  Authorization: Bearer <JWT>  and wait for the agent's `ready` event
    3. stream 16 kHz PCM16 audio up; receive JSON events (transcript, reply, tool, interrupted, error) and 24 kHz PCM16 audio back

The user's audio is streamed in real time in 30 ms frames (silence while the user is not speaking), exactly like a
microphone, so the agent's own voice-activity detection and end-of-speech timing are exercised. A session is a sequence of turns:

    await session.say(text, pcm)   speak one line (PCM16 16 kHz); returns when the last frame has been sent
    reply = await session.listen() wait until the agent has answered and gone quiet; returns what it said

Per turn we keep what the user said, what the AGENT HEARD (its own transcript of the user), the agent's reply text, its tool
calls, time to first audio, and the word error of heard vs said. Both audio tracks are kept on one timeline: the agent's audio
is placed where a listener would hear it (queued back to back as it arrives, cut at an `interrupted` event).
"""
from __future__ import annotations

import asyncio
import collections
import json
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import websockets

from sim.text import wer

IN_RATE, OUT_RATE = 16_000, 24_000
FRAME = 480                      # 30 ms of 16 kHz PCM16
FRAME_S = FRAME / IN_RATE
SILENCE = bytes(FRAME * 2)


class SessionError(Exception):
    """The session could not be opened or the agent ended it. The message says why."""


async def get_token(ws_url: str, api_key: str) -> str:
    """Exchanges AUTH_API_KEY for the short-lived JWT the agent's websocket requires."""
    base = ws_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1).removesuffix("/ws")
    try:
        async with httpx.AsyncClient() as http:
            r = await http.post(f"{base}/auth/token", headers={"X-API-Key": api_key}, timeout=15)
    except httpx.HTTPError as exc:
        raise SessionError(f"cannot reach the agent at {base} ({type(exc).__name__}): is it running?") from exc
    if r.status_code == 401:
        raise SessionError("the agent rejected AUTH_API_KEY (401): it must match the agent's own AUTH_API_KEY, and the agent must be restarted after editing its .env")
    if r.status_code != 200:
        raise SessionError(f"could not get a token from {base}/auth/token (HTTP {r.status_code})")
    return r.json()["access_token"]


class Timeline:
    """Both tracks on one clock. The user track is the frames actually sent; the agent track is what a listener hears."""

    def __init__(self) -> None:
        self.user_pad_s = 0.0
        self.user: list[bytes] = []
        self.agent: list[tuple[float, np.ndarray]] = []
        self.play_at = 0.0

    def agent_chunk(self, t: float, pcm: bytes) -> None:
        x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
        start = max(t, self.play_at)           # agent audio arrives in bursts; a listener hears it queued back to back
        self.play_at = start + len(x) / OUT_RATE
        self.agent.append((start, x))

    def interrupt(self, t: float) -> None:
        kept = []
        for start, x in self.agent:
            if start >= t:
                continue
            keep = int((t - start) * OUT_RATE)
            kept.append((start, x[:keep] if keep < len(x) else x))
        self.agent, self.play_at = kept, 0.0

    def build(self, duration: float) -> tuple[bytes, bytes]:
        user = np.frombuffer(b"".join(self.user), dtype="<i2")
        user_full = np.zeros(int(np.ceil(duration * IN_RATE)), dtype="<i2")
        pad = int(self.user_pad_s * IN_RATE)
        n = max(0, min(len(user), len(user_full) - pad))
        user_full[pad: pad + n] = user[:n]
        agent_full = np.zeros(int(np.ceil(duration * OUT_RATE)), dtype="<i2")
        for start, x in self.agent:
            at = round(start * OUT_RATE)
            n = max(0, min(len(x), len(agent_full) - at))
            agent_full[at: at + n] = x[:n]
        return user_full.tobytes(), agent_full.tobytes()


class Session:
    def __init__(self, url: str, token: str, settle: float = 2.5, ready_timeout: float = 20.0) -> None:
        self.url, self.token, self.settle, self.ready_timeout = url, token, settle, ready_timeout
        self.events: list[dict] = []
        self.turns: list[dict] = []
        self.error: str | None = None
        self.timeline = Timeline()
        self._queue: collections.deque[bytes] = collections.deque()
        self._ready = asyncio.Event()
        self._cur: dict | None = None
        self._started = False          # the agent has begun answering the current turn
        self._last_activity = 0.0
        self._tasks: list[asyncio.Task] = []

    def now(self) -> float:
        return time.perf_counter() - self._t0

    async def __aenter__(self) -> "Session":
        self._t0 = time.perf_counter()
        try:
            self.ws = await websockets.connect(self.url, max_size=None, additional_headers={"Authorization": f"Bearer {self.token}"})
        except Exception as exc:
            raise SessionError(f"the agent refused or dropped the connection ({type(exc).__name__}: {exc})") from exc
        self._tasks.append(asyncio.create_task(self._receive()))
        try:
            await asyncio.wait_for(self._ready.wait(), self.ready_timeout)
        except asyncio.TimeoutError:
            await self.__aexit__(None, None, None)
            raise SessionError(f"no `ready` from the agent within {self.ready_timeout:.0f} s (Bedrock credentials or region wrong?)")
        if self.error:
            await self.__aexit__(None, None, None)
            raise SessionError(f"the agent reported an error: {self.error}")
        self._tasks.append(asyncio.create_task(self._send_loop()))
        return self

    async def __aexit__(self, *exc) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        try:
            await self.ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ turns
    async def say(self, text: str, pcm: bytes) -> None:
        turn = {"n": len(self.turns), "said": text, "t_start": round(self.now(), 3), "heard": "", "reply": "", "tools": [],
                "agent_audio_s": 0.0}
        self.turns.append(turn)
        self._cur, self._started = turn, False
        pcm = pcm + bytes(-len(pcm) % (FRAME * 2))
        self._queue.extend(pcm[i: i + FRAME * 2] for i in range(0, len(pcm), FRAME * 2))
        while self._queue:
            self._check()
            await asyncio.sleep(0.02)
        turn["t_end"] = round(self.now(), 3)
        self._last_activity = self.now()

    async def listen(self, timeout: float = 60.0) -> str:
        """Waits for the agent to start answering and then go quiet (no audio or events, and its queued speech finished)."""
        turn = self._cur
        assert turn is not None, "say() first"
        deadline = self.now() + timeout
        while self.now() < deadline:
            await asyncio.sleep(0.1)
            self._check()
            if self._started and self.now() - max(self._last_activity, self.timeline.play_at) >= self.settle:
                break
        turn["heard"] = turn["heard"].strip()
        turn["reply"] = turn["reply"].strip()
        first = turn.pop("t_first_audio", None)
        turn["ttfa_ms"] = round((first - turn["t_end"]) * 1000) if first is not None else None
        turn["wer"] = wer(turn["said"], turn["heard"])
        turn["answered"] = self._started
        return turn["reply"]

    def _check(self) -> None:
        if self.error:
            raise SessionError(f"the agent reported an error: {self.error}")
        if any(t.done() for t in self._tasks):
            raise SessionError(self.error or "the agent closed the connection")

    def result(self) -> dict:
        duration = max(self.now(), self.timeline.play_at, self.timeline.user_pad_s + len(self.timeline.user) * FRAME_S)
        user, agent = self.timeline.build(duration)
        return {"events": self.events, "turns": self.turns, "error": self.error, "duration_s": round(duration, 2),
                "user_pcm": user, "agent_pcm": agent}

    # ------------------------------------------------------------- background
    async def _receive(self) -> None:
        try:
            async for msg in self.ws:
                t = self.now()
                cur = self._cur
                answering = cur is not None and "t_end" in cur and t >= cur["t_end"]
                if isinstance(msg, bytes):
                    self.timeline.agent_chunk(t, msg)
                    self._last_activity = t
                    if answering:
                        self._started = True
                        cur.setdefault("t_first_audio", t)
                        cur["agent_audio_s"] += len(msg) / 2 / OUT_RATE
                    continue
                ev = json.loads(msg)
                ev.update(t=round(t, 3), turn=cur["n"] if cur else -1)
                self.events.append(ev)
                kind = ev.get("type")
                if kind == "ready":
                    self._ready.set()
                elif kind == "error":
                    self.error = ev.get("message", "unknown error")
                    self._ready.set()
                elif kind == "interrupted":
                    self.timeline.interrupt(t)
                elif cur is not None:
                    if kind == "transcript":
                        cur["heard"] += " " + ev["text"]
                    elif kind == "reply":
                        cur["reply"] += " " + ev["text"]
                    elif kind == "tool":
                        cur["tools"].append({k: ev.get(k) for k in ("name", "args", "result", "ms")})
                    if kind in ("reply", "tool") and answering:
                        self._started = True
                        self._last_activity = t
        except websockets.ConnectionClosed as exc:
            self.error = self.error or f"the agent closed the connection ({exc.rcvd.code if exc.rcvd else 'no close frame'})"
        finally:
            self._ready.set()

    async def _send_loop(self) -> None:
        """30 ms frames on the wall clock: user audio if queued, silence otherwise."""
        self.timeline.user_pad_s = self.now()
        t_next = time.perf_counter()
        try:
            while True:
                frame = self._queue.popleft() if self._queue else SILENCE
                self.timeline.user.append(frame)
                await self.ws.send(frame)
                t_next += FRAME_S
                delay = t_next - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -1.0:             # fell far behind (a stall): resync rather than burst-send
                    t_next = time.perf_counter()
        except websockets.ConnectionClosed:
            return


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
