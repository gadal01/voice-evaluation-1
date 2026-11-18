# voice-evaluation

Evaluation tooling for a voice agent: two projects in one repo.

| Folder | Role |
|---|---|
| [voice-agent](voice-agent) | Sample voice agent (Amazon Nova Sonic + MCP tools): the system under test |
| [voice-eval-backend](voice-eval-backend) | Evaluation: batch runner, LLM-as-judge, Streamlit app, clip library |

```
runs   voice-eval-backend/batch/run_evals.py ──▶ voice-agent :8000          (main workflow; results viewed in the app)
app    streamlit run app.py                                                   (Runs · Baseline clips)
```

## Quick start (two terminals)
```
cd voice-agent && python main.py                                    # agent  :8000  (needs its own .env: AWS credentials, JWT_SECRET, AUTH_API_KEY)
cd voice-eval-backend && streamlit run app.py                       # app    :8501
```
To run an evaluation, from `voice-eval-backend`: `python batch/run_evals.py --voices clip:human:alice --scenarios book_ac_inspection --repeat 3`,
then open the **Runs** tab. Setup, commands and caveats are in each folder's README.

`voice-eval-backend` tracks `*.wav` with Git LFS, so install Git LFS before cloning.
