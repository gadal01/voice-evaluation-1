"""A fake voice agent speaking the real protocol, so the simulator can be tested without AWS or the real agent.

POST /auth/token (X-API-Key) -> JWT. WS /ws (Bearer): sends `ready`, then listens to 16 kHz PCM; after the user has spoken and gone
quiet for 0.6 s it answers the next scripted turn: `transcript` (what it "heard"), optional `tool`, `reply`, then 24 kHz audio.
"""
import socket
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException, WebSocket

TOKEN = "good-token"
API_KEY = "secret"


def tone(seconds: float, rate: int, amp: float = 0.3) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app) -> tuple[uvicorn.Server, int]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            return server, port
        time.sleep(0.05)
    raise RuntimeError("server did not start")


def make_agent(script: list[dict], silent: bool = False) -> FastAPI:
    """script[i] = {"heard": str, "reply": str, "tool": {...} | None, "audio_s": float} for the agent's i-th answer."""
    app = FastAPI()

    @app.post("/auth/token")
    async def auth(x_api_key: str | None = Header(default=None)):
        if x_api_key != API_KEY:
            raise HTTPException(401)
        return {"access_token": TOKEN}

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        if ws.headers.get("authorization", "") != f"Bearer {TOKEN}":
            await ws.close(code=1008)
            return
        await ws.accept()
        await ws.send_json({"type": "ready", "in_rate": 16000, "out_rate": 24000, "tools": [], "conversation_id": "c1"})
        speaking, quiet, turn = False, 0, 0
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            frame = np.frombuffer(msg["bytes"], "<i2") if msg.get("bytes") else None
            if frame is None or silent:
                continue
            if np.abs(frame).max() > 1000:
                speaking, quiet = True, 0
            elif speaking:
                quiet += 1
                if quiet >= 20 and turn < len(script):          # 0.6 s of silence ends the user's turn
                    speaking, step = False, script[turn]
                    turn += 1
                    await ws.send_json({"type": "transcript", "text": step["heard"]})
                    if step.get("tool"):
                        await ws.send_json({"type": "tool", **step["tool"]})
                    await ws.send_json({"type": "reply", "text": step["reply"]})
                    await ws.send_bytes(tone(step.get("audio_s", 1.0), 24000))

    return app


class FakeStrandsAgent:
    """Stands in for the Strands agent that plays the user: returns scripted lines, and a reply starting with "HANGUP:"
    simulates the model calling its end_conversation tool. Set `.user` to the DynamicUser it is attached to."""

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts, self.user = iter(replies), [], None

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = next(self.replies)
        if reply.startswith("HANGUP:"):
            self.user.end_conversation(reply.removeprefix("HANGUP:").strip())
            return "(ignored: the farewell argument wins)"
        return reply
