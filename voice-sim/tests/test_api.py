"""The FastAPI layer: auth, catalog, validation, a full background run against the fake voice agent, files, cancel and listing."""
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_agent import make_agent, serve  # noqa: E402

import sim.api  # noqa: E402
from sim import config  # noqa: E402

KEY = {"X-API-Key": "api-key-123"}
SCRIPT = [{"heard": "can you fix the engine of my car", "reply": "Sorry, I only service appliances.", "audio_s": 1.0}]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Settings for a throwaway results folder; the voice agent's key is `secret` (what the fake agent accepts)."""
    def apply(agent_url="ws://127.0.0.1:9/ws", **extra):
        values = {"AUTH_API_KEY": "secret", "SIM_API_KEY": "api-key-123", "AGENT_URL": agent_url, "RESULTS_DIR": str(tmp_path / "results"),
                  "AWS_BEARER_TOKEN_BEDROCK": "", **extra}
        for k, v in values.items():
            monkeypatch.setenv(k, v)
        config.get_settings.cache_clear()
    apply()
    sim.api.STATE.clear()
    yield apply
    sim.api.STATE.clear()
    config.get_settings.cache_clear()


@pytest.fixture()
def live(env):
    """A client whose server tests a running fake voice agent."""
    servers = []

    def start(silent=False):
        server, port = serve(make_agent(SCRIPT, silent))
        servers.append(server)
        env(agent_url=f"ws://127.0.0.1:{port}/ws")
        return TestClient(sim.api.app)

    yield start
    for s in servers:
        s.should_exit = True


def wait_for(client, run_id, until=("done", "failed", "cancelled"), timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers=KEY).json()
        if r["status"] in until:
            return r
        time.sleep(0.3)
    raise AssertionError(f"run still {r['status']} after {timeout}s")


# --------------------------------------------------------------------- auth ---
def test_health_is_open_and_everything_else_needs_the_key(env):
    with TestClient(sim.api.app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/scenarios").status_code == 401
        assert c.get("/scenarios", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/runs", json={}).status_code == 401
        assert c.get("/runs").status_code == 401
        assert c.get("/scenarios", headers=KEY).status_code == 200


def test_the_agent_key_is_the_fallback_and_no_key_at_all_refuses_everything(env):
    env(SIM_API_KEY="")
    with TestClient(sim.api.app) as c:
        assert c.get("/scenarios", headers={"X-API-Key": "secret"}).status_code == 200        # AUTH_API_KEY stands in
    env(SIM_API_KEY="", AUTH_API_KEY="")
    with TestClient(sim.api.app) as c:
        r = c.get("/scenarios", headers={"X-API-Key": "anything"})
        assert r.status_code == 503 and "SIM_API_KEY" in r.json()["detail"]


# ------------------------------------------------------------------ catalog ---
def test_catalog(env):
    with TestClient(sim.api.app) as c:
        sc = c.get("/scenarios", headers=KEY).json()
        assert len(sc) >= 8 and {"id", "title", "goal", "static_turns", "expect"} <= set(sc[0])
        v = c.get("/voices", headers=KEY).json()
        assert "en_US-lessac-medium" in v["static_clip_voices"] and isinstance(v["piper_models"], list)


# --------------------------------------------------------------- validation ---
def test_bad_requests_are_refused_before_anything_runs(env):
    with TestClient(sim.api.app) as c:
        post = lambda body: c.post("/runs", json=body, headers=KEY)  # noqa: E731
        assert post({"mode": "sideways"}).status_code == 422
        assert post({"url": "ws://evil.example/ws"}).status_code == 422                  # the agent under test is AGENT_URL only
        assert post({"voice": "../../etc/passwd"}).status_code == 422
        assert post({"repeat": 0}).status_code == 422 and post({"concurrency": 99}).status_code == 422
        r = post({"scenarios": ["nope"], "judge": False})
        assert r.status_code == 400 and "unknown scenario" in r.json()["detail"] and "ac_not_cooling_book" in r.json()["detail"]
        r = post({"mode": "static", "judge": True})                                      # no Bedrock key configured
        assert r.status_code == 400 and "AWS_BEARER_TOKEN_BEDROCK" in r.json()["detail"]
        r = post({"mode": "dynamic", "judge": False})
        assert r.status_code == 400 and "AWS_BEARER_TOKEN_BEDROCK" in r.json()["detail"]
        assert sim.api.STATE == {}                                                       # nothing was started


# --------------------------------------------------------------- a full run ---
def test_a_run_goes_to_the_agent_and_its_results_are_served(live):
    with live() as c:
        r = c.post("/runs", json={"mode": "static", "scenarios": ["out_of_scope_car"], "judge": False, "settle": 0.5}, headers=KEY)
        assert r.status_code == 202 and r.json()["status"] == "running" and r.json()["total"] == 1
        run_id = r.json()["id"]
        run = wait_for(c, run_id)
        assert run["status"] == "done" and run["done"] == 1 and run["summary"]["overall"]["sessions"] == 1
        (s,) = run["sessions"]
        assert s["scenario"] == "out_of_scope_car" and s["mode"] == "static" and s["error"] is None and s["passed"] is None   # judge off

        full = c.get(f"/runs/{run_id}/sessions/{s['name']}", headers=KEY).json()
        assert full["turns"][0]["reply"] == "Sorry, I only service appliances." and full["turns"][0]["wer"]["wer"] == 0

        assert "out_of_scope_car" in c.get(f"/runs/{run_id}/report.html", headers=KEY).text
        wav = c.get(f"/runs/{run_id}/{s['name']}/user.wav", headers=KEY)
        assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
        assert c.get(f"/runs/{run_id}/summary.json", headers=KEY).json()["overall"]["sessions"] == 1
        assert c.get(f"/runs/{run_id}/report.html").status_code == 401                    # files need the key too

        assert [x["id"] for x in c.get("/runs", headers=KEY).json()] == [run_id]
        sim.api.STATE.clear()                                                            # as after a server restart: from the files
        again = c.get(f"/runs/{run_id}", headers=KEY).json()
        assert again["status"] == "done" and again["summary"]["overall"]["sessions"] == 1 and again["sessions"][0]["name"] == s["name"]


def test_files_cannot_leave_the_run_folder(live, tmp_path):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "secret.txt").write_text("nope")
    with live() as c:
        run_id = c.post("/runs", json={"scenarios": ["out_of_scope_car"], "judge": False, "settle": 0.5}, headers=KEY).json()["id"]
        wait_for(c, run_id)
        for path in ("%2e%2e/secret.txt", "%2e%2e/%2e%2e/secret.txt", "run.json%00.wav", "01_out_of_scope_car_static/session.json.bak"):
            assert c.get(f"/runs/{run_id}/{path}", headers=KEY).status_code == 404, path
        assert c.get("/runs/..%2f..", headers=KEY).status_code == 404
        assert c.get(f"/runs/{run_id}/sessions/..%2f..%2fsecret.txt", headers=KEY).status_code == 404
        assert c.get("/runs/nope", headers=KEY).status_code == 404


def test_one_run_at_a_time_and_a_run_can_be_cancelled(live):
    with live(silent=True) as c:                                                         # an agent that never answers: the run stays busy
        first = c.post("/runs", json={"scenarios": ["out_of_scope_car"], "judge": False}, headers=KEY)
        assert first.status_code == 202
        second = c.post("/runs", json={"scenarios": ["out_of_scope_car"], "judge": False}, headers=KEY)
        assert second.status_code == 409 and "already in progress" in second.json()["detail"]
        run_id = first.json()["id"]
        cancelled = c.post(f"/runs/{run_id}/cancel", headers=KEY)
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        assert c.post(f"/runs/{run_id}/cancel", headers=KEY).status_code == 409           # no longer running
        assert c.post("/runs/20200101-000000-abcd/cancel", headers=KEY).status_code == 404
        sim.api.STATE.clear()
        assert c.get(f"/runs/{run_id}", headers=KEY).json()["status"] == "cancelled"      # remembered on disk
        ok = c.post("/runs", json={"scenarios": ["out_of_scope_car"], "judge": False}, headers=KEY)
        assert ok.status_code == 202                                                      # the slot is free again
        c.post(f"/runs/{ok.json()['id']}/cancel", headers=KEY)


def test_a_run_whose_agent_is_down_fails_cleanly(env):
    with TestClient(sim.api.app) as c:                                                    # AGENT_URL points at a closed port
        run_id = c.post("/runs", json={"scenarios": ["out_of_scope_car"], "judge": False}, headers=KEY).json()["id"]
        run = wait_for(c, run_id, until=("done", "failed"))
        assert run["status"] == "done" and run["sessions"][0]["error"] and "cannot reach the agent" in run["sessions"][0]["error"]
        assert run["summary"]["overall"]["errors"] == 1


def test_runs_made_by_the_cli_or_interrupted_are_listed_from_disk(env, tmp_path):
    results = tmp_path / "results"
    cli = results / "20261101-120000"
    cli.mkdir(parents=True)
    (cli / "summary.json").write_text(json.dumps({"meta": {"created": "2026-11-01T12:00:00", "mode": "static", "voice": "v", "repeat": 1},
                                                  "overall": {"sessions": 3, "passed": 2}, "by_mode": {}, "by_scenario": {}}))
    dead = results / "20261102-120000"
    dead.mkdir()
    (dead / "run.json").write_text(json.dumps({"options": {"mode": "both"}, "created": "2026-11-02T12:00:00", "total": 4}))
    (results / "not-a-run").mkdir()
    with TestClient(sim.api.app) as c:
        runs = {r["id"]: r for r in c.get("/runs", headers=KEY).json()}
        assert set(runs) == {"20261101-120000", "20261102-120000"}                       # newest first, folders that are not runs ignored
        assert runs["20261101-120000"]["status"] == "done" and runs["20261101-120000"]["total"] == 3
        assert runs["20261102-120000"]["status"] == "interrupted" and runs["20261102-120000"]["total"] == 4
        assert c.get("/runs/20261101-120000", headers=KEY).json()["summary"]["overall"]["passed"] == 2
