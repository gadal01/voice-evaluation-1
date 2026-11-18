"""FastAPI layer for the simulated user: start runs, follow them, read the results.

    python -m sim.api                      # http://127.0.0.1:8100   interactive docs at /docs
    curl -H "X-API-Key: <key>" http://127.0.0.1:8100/scenarios

A run is the same thing the CLI does (`python -m sim.run`): sessions of the Piper-voiced simulated user (static clips or the
Strands agent) against the voice agent's websocket, each judged by the LLM judge. Runs execute in the background: POST /runs returns
at once with a run id; poll GET /runs/{id}. Results land in RESULTS_DIR, the same folders the CLI writes.

Every route except /health needs the X-API-Key header (SIM_API_KEY, or AUTH_API_KEY when that is not set): a run opens real, paid
Bedrock sessions. The agent to test is always AGENT_URL from .env: a client cannot point the simulator at another host, because the
simulator sends the agent's API key to whatever host it is told to test.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from sim import clips, tts
from sim.config import get_settings
from sim.run import RunError, RunOptions, execute, plan
from sim.user import load_scenarios

RUN_ID = re.compile(r"[0-9A-Za-z_-]{1,48}")
SESSION_NAME = re.compile(r"\d{2,}_[A-Za-z0-9_.-]+")
SERVABLE = {".html", ".json", ".wav", ".txt"}
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(_key_header)) -> None:
    s = get_settings()
    expected = s.sim_api_key or s.auth_api_key
    if not expected:
        raise HTTPException(503, "SIM_API_KEY (or AUTH_API_KEY) is not set in .env: the API refuses every request until it is")
    if not key or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(401, "invalid or missing X-API-Key")


# ------------------------------------------------------------------- models ---
class RunRequest(BaseModel):
    """Options for a run. `url` is deliberately not accepted: the agent under test is AGENT_URL."""
    model_config = ConfigDict(extra="forbid")

    mode: Literal["static", "dynamic", "both"] = "static"
    scenarios: list[str] = Field(default_factory=list, description="scenario ids; empty means all")
    voice: str = Field(clips.DEFAULT_VOICE, pattern=r"^[A-Za-z0-9_.-]+$", description="Piper voice (model file stem)")
    repeat: int = Field(1, ge=1, le=20)
    concurrency: int = Field(1, ge=1, le=4)
    settle: float = Field(2.5, ge=0.5, le=10, description="seconds of quiet that mean the agent finished its turn")
    judge: bool = True


class SessionSummary(BaseModel):
    name: str
    scenario: str
    mode: str
    passed: bool | None = None
    error: str | None = None
    judge_overall: float | None = None
    duration_s: float = 0.0


class RunStatus(BaseModel):
    id: str
    status: Literal["running", "done", "failed", "cancelled", "interrupted"]
    created: str | None = None
    options: dict = {}
    total: int | None = None
    done: int = 0
    error: str | None = None
    summary: dict | None = None
    sessions: list[SessionSummary] = []


def compact(res: dict) -> SessionSummary:
    j = res.get("judge") or {}
    return SessionSummary(name=res["dir"], scenario=res["scenario"], mode=res["mode"], passed=res.get("passed"), error=res.get("error"),
                          judge_overall=j.get("overall") if j.get("status") == "ok" else None, duration_s=res.get("duration_s", 0.0))


# -------------------------------------------------------------------- state ---
@dataclass
class RunState:
    id: str
    dir: Path
    options: dict
    created: str
    total: int
    status: str = "running"
    done: int = 0
    error: str | None = None
    summary: dict | None = None
    sessions: list[SessionSummary] = field(default_factory=list)
    task: asyncio.Task | None = None

    def record(self, res: dict, done: int, total: int) -> None:
        self.done = done
        self.sessions.append(compact(res))

    def status_model(self, with_sessions: bool = True) -> RunStatus:
        return RunStatus(id=self.id, status=self.status, created=self.created, options=self.options, total=self.total, done=self.done,
                         error=self.error, summary=self.summary, sessions=sorted(self.sessions, key=lambda s: s.name) if with_sessions else [])


STATE: dict[str, RunState] = {}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_dir(run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(404, "no such run")
    d = get_settings().results_dir / run_id
    if not d.is_dir():
        raise HTTPException(404, "no such run")
    return d


def status_from_disk(d: Path, with_sessions: bool = True) -> RunStatus:
    """A run this process did not start (the CLI, or before a restart), described from the files it left."""
    info, summary = _read_json(d / "run.json") or {}, _read_json(d / "summary.json")
    status = "done" if summary else info.get("status", "interrupted")
    sessions = []
    if with_sessions:
        for f in sorted(d.glob("*/session.json")):
            res = _read_json(f)
            if res:
                sessions.append(compact(res))
    created = info.get("created") or (summary or {}).get("meta", {}).get("created")
    options = info.get("options") or {k: v for k, v in (summary or {}).get("meta", {}).items() if k in ("mode", "voice", "repeat")}
    total = info.get("total") or (summary["overall"]["sessions"] if summary else None)
    return RunStatus(id=d.name, status=status, created=created, options=options, total=total, done=len(sessions) if with_sessions else 0,
                     error=info.get("error"), summary=summary, sessions=sessions)


def persist(state: RunState) -> None:
    f = state.dir / "run.json"
    info = _read_json(f) or {"options": state.options, "created": state.created, "total": state.total}
    f.write_text(json.dumps({**info, "status": state.status, "error": state.error}, indent=2), encoding="utf-8")


async def drive(state: RunState, opts: RunOptions) -> None:
    try:
        state.summary = await execute(opts, state.dir, on_result=state.record)
        state.status = "done"
    except asyncio.CancelledError:
        state.status = "cancelled"
    except RunError as exc:
        state.status, state.error = "failed", str(exc)
    except Exception as exc:
        state.status, state.error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        persist(state)


# ---------------------------------------------------------------------- app ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for st in STATE.values():              # do not leave paid sessions running after the server stops
        if st.task and not st.task.done():
            st.task.cancel()
    await asyncio.gather(*(st.task for st in STATE.values() if st.task), return_exceptions=True)


app = FastAPI(title="voice-sim", version="1.0", lifespan=lifespan,
              description="Simulated user (Piper voice, static clips or a Strands agent) that tests a voice agent over its websocket, plus an LLM judge.")
router = APIRouter(dependencies=[Depends(require_key)])


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


@router.get("/scenarios", tags=["catalog"])
async def scenarios() -> list[dict]:
    """Every scenario: goal, facts, persona, the fixed lines and what a correct agent does."""
    return list(load_scenarios().values())


@router.get("/voices", tags=["catalog"])
async def voices() -> dict:
    """Piper voices that can render new lines (dynamic mode), and voices that have stored static clips."""
    models = sorted(p.name.removesuffix(".onnx") for p in tts.MODELS.glob("*.onnx")) if tts.MODELS.exists() else []
    return {"piper_models": models, "static_clip_voices": sorted({e["voice"] for e in clips.load_index().values()})}


@router.post("/runs", status_code=202, response_model=RunStatus, tags=["runs"])
async def start_run(req: RunRequest) -> RunStatus:
    """Starts a run in the background and returns at once. Checks every prerequisite first, so a bad request fails here, not later."""
    s = get_settings()
    opts = RunOptions(mode=req.mode, scenarios=req.scenarios, voice=req.voice, repeat=req.repeat, concurrency=req.concurrency,
                      settle=req.settle, judge=req.judge)           # url stays empty: AGENT_URL
    try:
        chosen, modes = await asyncio.to_thread(plan, opts)
    except RunError as exc:
        raise HTTPException(400, str(exc))
    if sum(1 for st in STATE.values() if st.status == "running") >= s.max_active_runs:
        raise HTTPException(409, f"{s.max_active_runs} run(s) already in progress (MAX_ACTIVE_RUNS): wait for it or POST /runs/{{id}}/cancel")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    state = RunState(id=run_id, dir=s.results_dir / run_id, options=asdict(opts), created=datetime.now().isoformat(timespec="seconds"),
                     total=len(chosen) * len(modes) * opts.repeat)
    STATE[run_id] = state
    state.task = asyncio.create_task(drive(state, opts))
    return state.status_model()


@router.get("/runs", response_model=list[RunStatus], tags=["runs"])
async def list_runs(limit: int = 50) -> list[RunStatus]:
    """Newest first, without the per-session lists (use GET /runs/{id})."""
    root = get_settings().results_dir
    on_disk = [p for p in root.iterdir() if p.is_dir() and RUN_ID.fullmatch(p.name)] if root.exists() else []
    names = {p.name for p in on_disk if (p / "run.json").exists() or (p / "summary.json").exists()} | set(STATE)
    return [STATE[n].status_model(False) if n in STATE else status_from_disk(root / n, False) for n in sorted(names, reverse=True)[:max(1, limit)]]


@router.get("/runs/{run_id}", response_model=RunStatus, tags=["runs"])
async def get_run(run_id: str) -> RunStatus:
    """Status, progress (done of total), the sessions finished so far and, when the run is done, the summary."""
    if run_id in STATE:
        return STATE[run_id].status_model()
    return status_from_disk(run_dir(run_id))


@router.post("/runs/{run_id}/cancel", response_model=RunStatus, tags=["runs"])
async def cancel_run(run_id: str) -> RunStatus:
    state = STATE.get(run_id)
    if state is None:
        raise HTTPException(404, "no active run with that id in this server")
    if state.status != "running" or state.task is None:
        raise HTTPException(409, f"the run is {state.status}, not running")
    state.task.cancel()
    await asyncio.wait({state.task}, timeout=10)
    return state.status_model()


@router.get("/runs/{run_id}/sessions/{name}", tags=["runs"])
async def get_session(run_id: str, name: str) -> dict:
    """One session in full: every turn (said, heard, reply, tool calls), the judge's scores, the events."""
    if not SESSION_NAME.fullmatch(name):
        raise HTTPException(404, "no such session")
    res = _read_json(run_dir(run_id) / name / "session.json")
    if res is None:
        raise HTTPException(404, "no such session")
    return res


@router.get("/runs/{run_id}/{path:path}", tags=["runs"])
async def get_file(run_id: str, path: str) -> FileResponse:
    """report.html, summary.json and each session's user.wav / agent.wav. Never leaves the run's folder."""
    root = run_dir(run_id).resolve()
    try:
        f = (root / path).resolve()
        ok = root in f.parents and f.is_file() and f.suffix in SERVABLE
    except (OSError, ValueError):                      # a NUL byte or another character the filesystem rejects
        ok = False
    if not ok:
        raise HTTPException(404, "no such file")
    return FileResponse(f)


app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("sim.api:app", host="127.0.0.1", port=get_settings().api_port)
