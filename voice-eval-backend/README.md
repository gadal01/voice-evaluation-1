# voice-eval-backend

Evaluates a voice agent. Two pieces, all Python:

1. **Runs** (`batch/run_evals.py`): synthetic callers (Piper, Kokoro) and recorded human clips talk to the live agent; every call is scored
   and compared with a saved baseline. This is the main workflow.
2. **Streamlit app** (`app.py`): view runs and record human baseline clips.

Recording live calls (a proxy between any client and the agent) was removed for now; `scoring.py`, `llm.py`, `golden.json` and `pricing.json`
are kept as the library for evaluating recorded sessions when it comes back (see the git history).

```
app.py                     Streamlit app
voice_eval/
  scoring.py               deterministic metrics for a recorded session (pure functions): WER, slots, outcome, turns, latency, barge-in, cost
  llm.py                   LLM-as-judge (tone, clarity, empathy, accuracy, policy) and intent classifier, via Bedrock Converse
  config.py                typed settings from env / .env
  golden.json              labelled scenarios: intent, outcome, slots, script
  pricing.json             PLACEHOLDER cost rates
batch/                     the runner and everything it needs
  run_evals.py             run scenarios against the agent, score, report, compare with a baseline
  clips.py  scenarios.json caller clip library and the scenarios the clips are for
  caller_clips/ noise/ baselines/    data (*.wav in Git LFS)
  metrics.py judge.py baseline.py report.py augment.py noises.py tts.py dialogue.py     run_evals' dependencies
tests/                     pytest, no AWS and no agent needed
```

## Setup
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt        # runner + app + tests (requirements.txt = app only)
copy .env.example .env                     # AUTH_API_KEY (the agent's), AWS_BEARER_TOKEN_BEDROCK for the judge
python -m pytest tests -q
```

## 1. Runs (the main workflow)
Start the agent (`voice-agent`, port 8000), then from this folder:
```
python batch/run_evals.py --voices clip:piper:en_US-lessac-medium,clip:human:alice --scenarios cost_quote,book_ac_inspection --repeat 3
python batch/run_evals.py --voices clip:human:alice --augment noisy --noise-snr 10       # WER under noise
python batch/run_evals.py --save-baseline good                                          # keep this run as the baseline
python batch/run_evals.py --baseline good --fail-on-regression                          # CI gate
```
Each run writes `batch/results/<timestamp>/` (report.html, runs.jsonl, summary.json, transcripts, call audio). Open the **Runs** tab of the app
to browse it: pass rate, WER, tool accuracy, judge score, TTFA, regressions against the baseline, failed calls with their conversation and audio.
Every call is a real Bedrock session and costs money. See `python batch/run_evals.py --help`.

Voices: `clip:<source>` replays a stored recording exactly (`clip:piper:...`, `clip:kokoro:...`, `clip:human:<speaker>`); `piper:<voice>` and
`kokoro:<voice>` synthesize on the fly. No cloud TTS is used. Piper/Kokoro model files are not in the repo, and are only needed to render
new clips: `python -m piper.download_voices --download-dir batch/models en_US-lessac-medium`, and Kokoro's `kokoro-v1.0.onnx` +
`voices-v1.0.bin` from the kokoro-onnx release "model-files-v1.0".

## 2. Baseline clips
**Baseline clips** tab: enter a speaker id, read each line, save. Clips go to `batch/caller_clips/` as `human:<speaker>` and replay in runs. The browser
microphone applies its own noise suppression; for raw voice record in an external recorder and use the upload box. Command line equivalent:
`python batch/clips.py add-human <scenario> <wav> --name <speaker> --turn N`.

## Metrics
Per run: pass rate, WER (jiwer, digit-normalised; WER under noise with `--augment`), tool-call accuracy, judge score, latency. The session metrics in `scoring.py` are: WER · slot accuracy · session outcome and task success (from real tool results vs golden) · turns to
resolution · intent accuracy (LLM classifier vs label; tool-trace fallback) · tone, clarity, empathy, accuracy, policy (LLM judge, 1-5) · latency
p50/p95 · false and missed barge-in · cost per session. Definitions are in the docstring of `voice_eval/scoring.py`.

## Before you trust the numbers
- `pricing.json` rates are placeholders (`"verified": false`).
- The judge is advisory: calibrate it against human labels before gating releases on it.
- WER measures the agent's own transcript of the caller, not an independent ASR.
- `batch/judge.py` (used by runs) and `voice_eval/llm.py` (not wired in now) are two judges with different rubrics.
