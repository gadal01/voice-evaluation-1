# voice-agent

Sample voice agent: the **system under test** for the simulated user in `voice-sim`.
Amazon Nova Sonic (speech-to-speech on Bedrock) with tools served over MCP, behind a WebSocket protected by short-lived JWTs.

```
main.py          FastAPI app: /auth/token, /ws (voice), /mcp (the sample tools), / (index.html demo page)
sonic.py         one Nova Sonic bidirectional stream per session (STT + LLM + TTS + barge-in handled by the model)
mcp_server.py    sample "service desk" tools: list items, quote cost, available slots, create/get/cancel request
auth.py          API key -> JWT, JWT check on the WebSocket handshake
agent_settings.py  typed settings from environment / .env
index.html       minimal browser client to talk to the agent by hand
test_auth.py     auth checks against a running server
```

## Run
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # fill in AWS credentials, JWT_SECRET, AUTH_API_KEY
python main.py                # http://127.0.0.1:8000
```
Python 3.12+ is required by the AWS Bedrock streaming SDK. Each session is a real Bedrock session and costs money.

## Protocol (what the eval tools talk to)
1. `POST /auth/token` with header `X-API-Key: <AUTH_API_KEY>` returns a JWT (default 5 min).
2. Open `ws://host/ws` presenting the JWT as header `Authorization: Bearer <jwt>` or subprotocol `bearer, <jwt>` (browsers).
3. Client -> server: binary PCM16 mono 16 kHz. Server -> client: binary PCM16 mono 24 kHz, plus JSON events
   `ready | transcript | reply | interrupted | tool | error`.

## Related
`../voice-sim` tests this agent: a simulated user (Piper voice, static or LLM-driven) talks to it over the protocol above and an LLM judge scores the session.
