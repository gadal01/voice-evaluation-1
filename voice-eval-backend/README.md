# voice-eval-backend

Python backend that **evaluates a voice agent**: stores recorded sessions, scores them, runs the LLM judge, keeps the human
baseline clip library, and (separately) runs synthetic-caller batch evaluations. It talks to the agent only over the agent's
public protocol, so it can test any agent that speaks it (see `voice-agent`).

```
voice_eval/              the evaluation API (FastAPI)
  app.py                 create_app(): CORS, router, optional serving of the built UI
  router.py              /api/*: sessions, evaluate, summary, review, baseline clips        (X-API-Key on every route)
  models.py              Pydantic response models = the contract the UI's TypeScript types are generated from
  scoring.py             deterministic metrics (pure functions): WER, slots, outcome, turns, latency, barge-in, cost
  llm.py                 LLM-as-judge (tone, clarity, empathy, accuracy, policy) and intent classifier, via Bedrock Converse
  quality.py             UTMOS / DNSMOS (optional deps)
  golden.json            labelled scenarios: intent, outcome, slots, script
  pricing.json           PLACEHOLDER cost rates
batch/                   the script harness: synthetic callers (Piper and Kokoro voices, stored human clips) talk to the live agent
  run_evals.py  clips.py  metrics.py  judge.py  augment.py  noises.py  baseline.py  report.py  tts.py  dialogue.py
  scenarios.json  caller_clips/  noise/  baselines/
tests/                   pytest, no AWS and no agent needed
scripts/export_openapi.py
```

## Run the API
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # add requirements-mos.txt for UTMOS/DNSMOS, requirements-batch.txt for the batch runner
copy .env.example .env                   # AUTH_API_KEY (same as the agent's); AWS_BEARER_TOKEN_BEDROCK (Bedrock API key) for the judge
python -m voice_eval                     # http://127.0.0.1:8001   docs at /docs, schema at /openapi.json
python -m pytest tests -q                # pip install -r requirements-dev.txt first
```
Set `UI_DIST=../voice-eval-ui/dist` to serve the built UI from this process. Otherwise run the UI from its own repo.

## API (all under /api, header `X-API-Key`)
| | |
|---|---|
| `GET /golden` | golden scenarios |
| `POST /sessions` | multipart: `meta` + `events` (JSON), `caller` (WAV 16 kHz), `agent` (WAV 24 kHz) |
| `GET /sessions`, `GET /sessions/{id}`, `GET /sessions/{id}/audio/{caller\|agent}` | list / detail / audio |
| `POST /sessions/{id}/evaluate?use_llm=&use_mos=` | all metrics; judge and MOS degrade to `skipped` when unavailable |
| `POST /sessions/{id}/review` | human naturalness / pronunciation spot check |
| `GET /summary` | aggregates by scenario and by noise condition (WER under noise) |
| `GET /clips`, `POST /clips`, `GET /clips/{id}/pcm?noise=&snr_db=`, `GET /clips/{id}/audio` | baseline human clips, with server-side noise mixing |

## Metrics
WER (jiwer, digit-normalised), WER under noise (grouped by condition), slot accuracy, session outcome and task success
(from real tool results vs golden), turns to resolution, intent accuracy (LLM classifier vs label; tool-trace fallback),
tone / clarity / empathy / accuracy / policy (LLM judge, 1-5), latency p50/p95 (energy VAD), false and missed barge-in,
cost per session, UTMOS and DNSMOS (optional) plus the human spot check. Definitions are in the docstring of `scoring.py`.

## Batch runner (synthetic callers)
Run from this folder with the agent up on :8000 and `AUTH_API_KEY` / AWS credentials in `.env`:
```
pip install -r requirements-batch.txt
python batch/run_evals.py --voices clip:human:alice --scenarios book_ac_inspection --repeat 3 --augment noisy --noise-snr 10
python batch/clips.py list
```
Results go to `batch/results/<timestamp>/` (ignored by git). **Synthetic callers are Piper and Kokoro** (open source, local); real callers are
recorded human clips. No cloud TTS (Polly, ElevenLabs) is used or supported. Model files are not in the repo: Piper voices via
`python -m piper.download_voices --download-dir batch/models en_US-lessac-medium en_GB-alba-medium`, and Kokoro's `kokoro-v1.0.onnx` +
`voices-v1.0.bin` from the kokoro-onnx release "model-files-v1.0" into `batch/models/`. The stored Piper, Kokoro and human clips in
`batch/caller_clips/` replay without them; the engines are only needed to render new clips (`python batch/clips.py record`).
`*.wav` files are tracked with Git LFS. The only AWS credential used here is `AWS_BEARER_TOKEN_BEDROCK` (Bedrock API key), for the LLM judge and intent classifier.

## Before you trust the numbers
- `pricing.json` rates are placeholders (`"verified": false`); the UI says so.
- The judge and MOS scores are advisory: calibrate against human labels before gating releases on them.
- WER measures the agent's own transcript of the caller, not an independent ASR.
- Sessions (`data/`) are call recordings: keep them out of git and apply your retention policy.
