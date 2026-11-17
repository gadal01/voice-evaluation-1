"""Response models for the QA console API. They are the single source of truth for the web UI's TypeScript types:
`python scripts/export_openapi.py` writes openapi.json, and the voice-eval-ui repo runs `npm run gen:types` on it to
produce src/api/types.ts. Change a shape here, regenerate, and the compiler shows every place in the UI that must follow.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Loose(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ------------------------------------------------------------------ golden ---
class GoldenScenario(BaseModel):
    id: str
    title: str
    item: str = ""
    symptom: str = ""
    intent: str | None = None
    outcome: str | None = None
    slots: dict[str, str] = {}
    script: list[str] = []
    max_turns: int = 4


class GoldenResponse(BaseModel):
    scenarios: list[GoldenScenario]
    pricing_verified: bool


# ------------------------------------------------------------------ events ---
class SessionEvent(Loose):
    """One WebSocket event stamped with seconds since Record. Extra fields depend on `type`:
    transcript/reply -> text; tool -> name, args, result, ms; error -> message; ready -> conversation_id."""
    t: float
    type: str
    text: str | None = None
    name: str | None = None
    args: dict[str, Any] | None = None
    result: Any = None
    ms: float | None = None
    message: str | None = None
    conversation_id: str | None = None


class Condition(BaseModel):
    noise: str | None = None
    snr_db: float | None = None


class SessionMeta(BaseModel):
    id: str
    created: str
    scenario_id: str | None = None
    condition: Condition | None = None
    reference_text: str | None = None
    notes: str | None = None
    caller_mode: str | None = None
    conversation_id: str | None = None
    duration_s: float | None = None
    tester: str | None = None


class SessionRow(SessionMeta):
    evaluated: bool = False
    task_success: bool | None = None
    outcome: str | None = None


class SavedSession(BaseModel):
    id: str


# -------------------------------------------------------------- evaluation ---
class OutcomeResult(BaseModel):
    actual: str
    golden: str | None = None
    task_success: bool | None = None
    forbidden_booking: bool = False


class SlotDetail(BaseModel):
    expected: str
    got: Any = None
    ok: bool


class SlotsResult(BaseModel):
    score: float | None = None
    slots: dict[str, SlotDetail] = {}
    note: str = ""


class TurnsResult(BaseModel):
    turns: int
    total_user_turns: int
    resolved: bool
    agent_replies: int


class IntentResult(BaseModel):
    labelled: str | None = None
    predicted_trace: str
    trace_correct: bool | None = None
    predicted: str | None = None
    source: str | None = None          # "llm" or "trace"
    correct: bool | None = None


class WerResult(Loose):
    wer: float | None = None
    note: str | None = None
    errors: int | None = None
    words: int | None = None
    sub: int | None = None
    del_: int | None = Field(None, alias="del")
    ins: int | None = None
    hypothesis: str | None = None


class LatencyResult(BaseModel):
    values_ms: list[int]
    p50_ms: float | None = None
    p95_ms: float | None = None


class BargeInResult(BaseModel):
    interrupts: int
    false_: int = Field(alias="false")
    false_times: list[float]
    false_rate: float
    missed: int
    agent_turns: int
    model_config = ConfigDict(populate_by_name=True)


class CostResult(BaseModel):
    agent_usd: float
    judge_usd: float
    total_usd: float
    speech_in_usd: float
    speech_out_usd: float
    estimate: bool


class ToolCallSummary(BaseModel):
    name: str
    args: dict[str, Any]
    ms: float | None = None


class ToolsResult(BaseModel):
    calls: list[ToolCallSummary]
    errors: int


class AudioStats(BaseModel):
    duration_s: float
    peak_dbfs: float | None = None
    clipping_pct: float
    noise_floor_dbfs: float | None = None
    snr_db: float | None = None


class AudioResult(BaseModel):
    caller: AudioStats | None = None
    agent: AudioStats | None = None
    caller_speech_s: float
    agent_speech_s: float


class Segments(BaseModel):
    caller: list[list[float]]
    agent: list[list[float]]


class JudgeTokens(BaseModel):
    in_: int = Field(0, alias="in")
    out: int = 0
    model_config = ConfigDict(populate_by_name=True)


class JudgeResult(BaseModel):
    status: str                         # "ok" | "skipped"
    reason: str | None = None
    scores: dict[str, int] | None = None
    overall: float | None = None
    pass_: bool | None = Field(None, alias="pass")
    issues: list[str] = []
    model: str | None = None
    tokens: JudgeTokens | None = None
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class IntentLlmResult(BaseModel):
    status: str
    reason: str | None = None
    intent: str | None = None
    confidence: float | None = None
    model: str | None = None
    tokens: JudgeTokens | None = None
    model_config = ConfigDict(protected_namespaces=())


class MosScore(Loose):
    status: str
    reason: str | None = None
    utmos: float | None = None
    ovrl_mos: float | None = None
    sig_mos: float | None = None
    bak_mos: float | None = None
    p808_mos: float | None = None


class MosResult(BaseModel):
    dnsmos: MosScore
    utmos: MosScore


class Evaluation(BaseModel):
    outcome: OutcomeResult
    slots: SlotsResult
    turns_to_resolution: TurnsResult
    intent: IntentResult
    wer: WerResult
    latency: LatencyResult
    barge_in: BargeInResult
    cost: CostResult
    tools: ToolsResult
    audio: AudioResult
    segments: Segments
    judge: JudgeResult
    intent_llm: IntentLlmResult
    mos: MosResult
    evaluated_at: str


class Review(BaseModel):
    naturalness: int | None = None
    pronunciation: int | None = None
    mispronounced: str = ""
    notes: str = ""
    reviewer: str = ""
    at: str | None = None


class SessionDetail(BaseModel):
    meta: SessionMeta
    events: list[SessionEvent]
    evaluation: Evaluation | None = None
    review: Review | None = None


# ----------------------------------------------------------------- summary ---
class GroupStats(BaseModel):
    sessions: int
    task_success: float | None = None
    wer: float | None = None
    slot_accuracy: float | None = None
    intent_accuracy: float | None = None
    turns_to_resolution: float | None = None
    ttfa_p50_ms: float | None = None
    false_barge_in_rate: float | None = None
    cost_usd: float | None = None
    tone: float | None = None
    clarity: float | None = None
    empathy: float | None = None
    utmos: float | None = None
    dnsmos: float | None = None


class SummaryResponse(BaseModel):
    overall: GroupStats | None = None
    by_scenario: dict[str, GroupStats]
    by_condition: dict[str, GroupStats]
    pricing_verified: bool


# ------------------------------------------------------------------- clips ---
class ClipRef(BaseModel):
    id: str
    source: str
    duration_s: float


class ClipLine(BaseModel):
    scenario: str
    turn: int
    text: str
    reactive: bool
    clips: list[ClipRef]


class ClipsResponse(BaseModel):
    lines: list[ClipLine]
    sources: list[str]
    noises: list[str]


class ClipSaved(BaseModel):
    id: str
    duration_s: float
    audio: AudioStats
