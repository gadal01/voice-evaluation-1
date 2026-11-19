# voice-sim

A **simulated user** that talks to the voice agent under test over its **websocket** and an **LLM judge** that scores each session.
There is no telephony: the agent is a websocket service protected by a token, and the simulated user plays the part of a person speaking into a microphone.

## How a session works
```
voice-sim                                   voice-agent (system under test)
   │  POST /auth/token   X-API-Key ───────▶ checks AUTH_API_KEY, returns a short-lived JWT
   │  WS /ws   Authorization: Bearer JWT ─▶ checks the JWT before accepting; sends {"type": "ready"}
   │  16 kHz PCM16 audio, 30 ms frames ───▶ the simulated user's speech (Piper), silence between lines
   │◀─ JSON events: transcript · reply · tool · interrupted · error
   │◀─ 24 kHz PCM16 audio ────────────────  the agent's speech
   ▼
   judge reads the conversation (what the user said, what the agent heard, its spoken replies, its tool calls and results)
```
`AUTH_API_KEY` in voice-sim's `.env` must equal the agent's own `AUTH_API_KEY`; the agent must be restarted after editing its `.env`.

## Where the user's speech comes from (always a Piper voice)
- **static**: the scenario's fixed lines, rendered once and stored as WAV clips in `clips/`. Every run replays identical audio, so this is the
  regression test.
- **dynamic**: a **Strands agent** (Amazon Bedrock model) plays the customer from the scenario's goal, facts and persona. It remembers the conversation,
  reacts to what the agent just said, and hangs up by calling its `end_conversation` tool; each line is spoken by Piper on the fly (cached by text).
  It finds problems a fixed script never reaches.

## The judge
An LLM on Bedrock reads the whole conversation and scores task success, accuracy, policy, voice style, tone and empathy (1-5). A session passes when every
score is at least 4. It judges the words the agent spoke, as text: it cannot hear audio.

```
sim/
  session.py    one session over the websocket: token, real-time 30 ms audio frames, turns, both audio tracks on one timeline
  user.py       StaticUser and DynamicUser, the Strands agent (decide WHAT to say next)
  tts.py        Piper voices with a text cache         clips.py   static clips: render / list / verify (checksummed)
  judge.py      the LLM judge                          llm.py     Bedrock Converse wrapper for the judge (API key auth)
  run.py        the engine (plan, execute) and the CLI      report.py   HTML report     text.py  spoken digits, WER
scenarios.json  goal, facts, persona, static lines and what a correct agent does (8 scenarios; add your own)
clips/          stored static clips (en_US-lessac-medium)         tests/   pytest, no AWS and no agent needed (a fake voice agent is included)
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m piper.download_voices --download-dir models en_US-lessac-medium      # ~60 MB, once
copy .env.example .env      # AUTH_API_KEY (the agent's), AWS_BEARER_TOKEN_BEDROCK (dynamic user + judge)
```
Python 3.12 recommended. The stored static clips replay without the Piper model; the model is needed for dynamic mode and to render new clips.

## Run
Start the agent (`voice-agent`, port 8000), then:
```powershell
.\.venv\Scripts\python.exe -m sim.run                                           # static clips, all scenarios, judged
.\.venv\Scripts\python.exe -m sim.run --mode dynamic --scenarios ac_not_cooling_book,fridge_leaking_book
.\.venv\Scripts\python.exe -m sim.run --mode both --repeat 3 --concurrency 2    # the model is non-deterministic: repeat and look at rates
.\.venv\Scripts\python.exe -m sim.run --mode static --no-judge                  # only talk to the agent
.\.venv\Scripts\python.exe -m sim.run --url ws://host:8000/ws                   # another agent (default AGENT_URL / ws://127.0.0.1:8000/ws)
```
Output goes to `results/<timestamp>/`: `report.html` (summary tables, every conversation, judge scores and issues, audio players for both
sides), `summary.json`, and one folder per session (`session.json`, `user.wav`, `agent.wav`). Open the report from that folder so the audio links work.
Use `--fail-on-fail` to make the exit code non-zero when any session fails (CI). Every session is a real Bedrock session and costs money, and dynamic
mode and the judge add LLM requests.

## Static clips
```powershell
.\.venv\Scripts\python.exe -m sim.clips render                  # (re)render all scenario lines with Piper, after editing scenarios.json
.\.venv\Scripts\python.exe -m sim.clips render --voices en_GB-alba-medium --scenarios cost_quote_ac     # another voice (download it first)
.\.venv\Scripts\python.exe -m sim.clips verify                  # checksums
```
A clip is tied to the line it says: if you change a line in `scenarios.json`, the run stops and tells you to re-render.

## What is measured per session
Judge scores and verdict · time to first agent audio after the user stops speaking (TTFA) · **heard WER**: the agent's transcript of the user against
what the user actually said, a direct read on speech recognition of the Piper voice (with `--mode both` you can compare static and dynamic) ·
the agent's tool calls and replies · errors (agent unreachable, wrong key, never answered).

## Limits worth knowing
- The judge reads **text**. It cannot hear the agent, so voice quality (naturalness, pronunciation) is not scored; the audio is saved for a human to listen to.
- Piper is a clean synthetic voice: expect better recognition than real users, accents or noise. Add human recordings or noise later if you need that.
- Judge scores are advisory; calibrate them against a few human-scored sessions before gating releases on them. `--repeat` shows how much they move.
- The simulated user never interrupts the agent (no barge-in testing) and waits for the agent to go quiet (`--settle`, default 2.5 s) before speaking.
- Dynamic users are non-deterministic by design; use static mode when you need an identical input.
