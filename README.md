# voice-evaluation

Evaluation tooling for a voice agent, as three independent projects in one repo:

| Folder | Role |
|---|---|
| [voice-agent](voice-agent) | Sample voice agent (Amazon Nova Sonic + MCP tools): the system under test |
| [voice-eval-backend](voice-eval-backend) | Python evaluation API: scoring, LLM-as-judge, clip library, batch runner |
| [voice-eval-ui](voice-eval-ui) | TypeScript/React QA console: record sessions, evaluate, record baseline clips |

Each folder has its own README, requirements and `.env.example`. Wire-up: the UI talks to the agent (`/auth/token`, `/ws`) and to the
backend (`/api/*`); use the same `AUTH_API_KEY` in the agent's and the backend's `.env`.
Note: `voice-eval-backend` tracks `*.wav` with Git LFS, so install Git LFS before cloning.
