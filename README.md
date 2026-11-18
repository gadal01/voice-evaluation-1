# voice-evaluation

A voice agent (a websocket service with token auth) and a simulated user that tests it: two projects in one repo.

| Folder | Role |
|---|---|
| [voice-agent](voice-agent) | Sample voice agent (Amazon Nova Sonic + MCP tools): the system under test |
| [voice-sim](voice-sim) | Simulated user (Piper voice; static clips or an LLM-driven dynamic user) and an LLM judge that scores each session |

```
voice-sim ── token request, then audio over the websocket (Piper voice) ──▶ voice-agent :8000 ──▶ speech + transcripts + tool calls ──▶ LLM judge ──▶ report.html
```

## Quick start (PowerShell, two terminals)
```powershell
cd voice-agent; .\.venv\Scripts\python.exe main.py                  # agent :8000   (needs its .env: AWS credentials, JWT_SECRET, AUTH_API_KEY)
cd voice-sim;   .\.venv\Scripts\python.exe -m sim.run               # static clips, every scenario, judged -> results\<timestamp>\report.html
```
Dynamic (LLM) user: `python -m sim.run --mode dynamic`. Setup, all options and limits are in each folder's README.
