# voice-eval-ui

QA console for a voice agent: **record** live sessions (caller audio, agent audio as played, events, transcripts, tool calls),
**evaluate** them (metrics from `voice-eval-backend`), and **record human baseline clips**. Vite + React + TypeScript.

It talks to two services:
- the voice agent (`voice-agent`): `POST /auth/token` and the `/ws` audio WebSocket
- the evaluation backend (`voice-eval-backend`): `/api/*`

## Run (development)
Start the agent (`:8000`) and the backend (`:8001`), then:
```
npm install
npm run dev          # http://localhost:5173 , proxies /api -> :8001 and /auth, /ws -> :8000 (one origin, no CORS)
```
Paste the shared `AUTH_API_KEY` into "Access key" (the agent and backend use the same value).

## Build / serve
```
npm run build        # dist/
```
Either set `UI_DIST=../voice-eval-ui/dist` in the backend's `.env` (backend serves the UI at `/`, but the agent must then be reachable
at the same origin through a reverse proxy, or see below), or host `dist/` anywhere.
For services on different origins set `VITE_EVAL_API_URL` and `VITE_AGENT_URL` before building (see `.env.example`) and set `CORS_ORIGINS`
on both services to this UI's origin.

## Layout
```
src/audio/       framework-free audio code: worklet, LiveCall recorder, baseline recorder, WAV maths (vitest-tested)
src/api/         fetch client, generated types.ts (never edit), friendly aliases in models.ts
src/tabs/        Live session, Sessions & metrics, Baseline clips
src/components/  EvalView (metric cards, targets are starting values: tune them)
```

## Types come from the backend
`openapi.json` is a committed copy of the backend's schema. When the backend's response models change:
```
npm run sync:types   # fetches http://127.0.0.1:8001/openapi.json and regenerates src/api/types.ts
npm run typecheck    # lists everything in the UI that must follow
```
(or, offline: `python ../voice-eval-backend/scripts/export_openapi.py openapi.json` then `npm run gen:types`).

## Scripts
`dev`, `build` (typecheck + bundle), `typecheck`, `test` (vitest), `gen:types`, `sync:types`.
