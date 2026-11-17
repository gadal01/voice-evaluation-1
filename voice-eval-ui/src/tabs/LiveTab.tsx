import { useCallback, useEffect, useRef, useState } from "react";
import { api, fetchToken } from "../api/client";
import type { ClipsResponse, Evaluation, GoldenScenario, ServerMsg } from "../api/models";
import { LiveCall, type RecordedSession } from "../audio/liveCall";
import { CALLER_RATE, encodeWav } from "../audio/wav";
import { EvalView } from "../components/EvalView";

interface LogLine {
  cls: "u" | "a" | "t" | "i";
  text: string;
}

function describe(m: ServerMsg): LogLine | null {
  switch (m.type) {
    case "transcript":
      return { cls: "u", text: `Caller (heard): ${m.text}` };
    case "reply":
      return { cls: "a", text: `Agent: ${m.text}` };
    case "tool":
      return { cls: "t", text: `🔧 ${m.name}(${JSON.stringify(m.args)}) → ${JSON.stringify(m.result)} [${m.ms} ms]` };
    case "interrupted":
      return { cls: "i", text: "⏸ agent interrupted" };
    case "error":
      return { cls: "i", text: `error: ${m.message}` };
    default:
      return null;
  }
}

const mmss = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function LiveTab({ keyVersion, onOpenSession }: { keyVersion: number; onOpenSession: (id: string) => void }) {
  const [golden, setGolden] = useState<GoldenScenario[]>([]);
  const [clips, setClips] = useState<ClipsResponse>({ lines: [], sources: [], noises: [] });
  const [scenario, setScenario] = useState("");
  const [mode, setMode] = useState<"mic" | "clip">("mic");
  const [clipSource, setClipSource] = useState("");
  const [noise, setNoise] = useState("none");
  const [snr, setSnr] = useState(15);
  const [tester, setTester] = useState("");
  const [reference, setReference] = useState("");
  const [status, setStatus] = useState("idle");
  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [log, setLog] = useState<LogLine[]>([]);
  const [sent, setSent] = useState<Set<number>>(new Set());
  const [savedId, setSavedId] = useState<string | null>(null);
  const [useLlm, setUseLlm] = useState(true);
  const [useMos, setUseMos] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [result, setResult] = useState<Evaluation | string | null>(null);
  const call = useRef<LiveCall | null>(null);
  const logEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.golden().then((g) => { setGolden(g.scenarios); setScenario((s) => s || g.scenarios[0]?.id || ""); }).catch((e: Error) => setStatus(`${e.message}: enter the access key`));
    api.clips().then((c) => { setClips(c); setClipSource((s) => s || c.sources[0] || ""); }).catch(() => undefined);
  }, [keyVersion]);

  useEffect(() => logEnd.current?.scrollIntoView({ block: "end" }), [log]);
  useEffect(() => {
    if (!recording) return;
    const id = window.setInterval(() => setElapsed(call.current?.now() ?? 0), 500);
    return () => window.clearInterval(id);
  }, [recording]);
  useEffect(() => () => call.current?.end(), []);

  const golds = golden.find((g) => g.id === scenario);
  const clipLines = clips.lines.filter((l) => l.scenario === scenario && !l.reactive).sort((a, b) => a.turn - b.turn);

  const save = useCallback(async (rec: RecordedSession) => {
    setStatus("saving…");
    const fd = new FormData();
    fd.append("meta", JSON.stringify({
      scenario_id: scenario, tester, conversation_id: rec.conversationId, duration_s: rec.durationS,
      caller_mode: mode === "clip" ? `clip:${clipSource}` : "live_mic",
      condition: { noise: noise === "none" ? "none" : noise, snr_db: noise === "none" ? null : snr },
      reference_text: reference.trim() || (mode === "clip" ? rec.clipTexts.join(" ") : ""),
    }));
    fd.append("events", JSON.stringify(rec.events));
    fd.append("caller", encodeWav(rec.callerPcm, CALLER_RATE), "caller.wav");
    fd.append("agent", encodeWav(rec.agentPcm, rec.outRate), "agent.wav");
    try {
      const { id } = await api.saveSession(fd);
      setSavedId(id);
      setResult(null);
      setStatus("saved");
    } catch (e) {
      setStatus(`save failed: ${(e as Error).message}`);
    }
  }, [scenario, tester, mode, clipSource, noise, snr, reference]);

  const endCall = useCallback(() => {
    const c = call.current;
    if (!c) return;
    c.end();
    const rec = c.takeResult();
    call.current = null;
    setConnected(false);
    setRecording(false);
    setLevel(0);
    if (rec) void save(rec);
    else setStatus((s) => (s === "recording…" || s.startsWith("connect") ? "idle" : s));
  }, [save]);

  const startCall = async () => {
    setLog([]); setSent(new Set()); setSavedId(null); setStatus("connecting…");
    try {
      const token = await fetchToken();
      call.current = await LiveCall.connect(mode, token, {
        onMessage: (m) => {
          if (m.type === "ready") { setConnected(true); setStatus("connected · press ● Record"); }
          const l = describe(m);
          if (l) setLog((x) => [...x, l]);
        },
        onLevel: setLevel,
        onClosed: (wasOpen) => { if (!wasOpen) setStatus("connection refused (token rejected or server down)"); endCall(); },
      });
    } catch (e) {
      setStatus((e as Error).message);
      call.current?.end(); call.current = null;
    }
  };

  const record = () => { call.current?.startRecording(); setRecording(true); setElapsed(0); setStatus("recording…"); };
  const stopAndSave = () => {
    const c = call.current;
    if (!c) return;
    const rec = c.finishRecording();
    setRecording(false);
    if (rec) void save(rec);
    c.takeResult();
    endCall();
  };

  const sendTurn = async (turn: number, clipId: string, text: string) => {
    if (!call.current?.isRecording) { alert("Press ● Record first so this turn is captured."); return; }
    const pcm = await api.clipPcm(clipId, noise === "other" ? "none" : noise, snr);
    call.current.queueClip(pcm, text);
    setSent((s) => new Set(s).add(turn));
    setLog((x) => [...x, { cls: "u", text: `Caller (clip ${clipSource}): “${text}”` }]);
  };

  const evaluate = async () => {
    if (!savedId) return;
    setEvaluating(true); setResult("Evaluating…");
    try { setResult(await api.evaluate(savedId, { llm: useLlm, mos: useMos })); }
    catch (e) { setResult(`Evaluation failed: ${(e as Error).message}`); }
    setEvaluating(false);
  };

  return (
    <section>
      <div className="card">
        <h2>Agent under test</h2>
        <div className="row">
          <label>Golden scenario
            <select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={connected}>
              {golden.map((g) => <option key={g.id} value={g.id}>{g.title}</option>)}
            </select>
          </label>
          <label>Caller is
            <select value={mode} onChange={(e) => setMode(e.target.value as "mic" | "clip")} disabled={connected}>
              <option value="mic">Me, on the microphone</option>
              <option value="clip">A stored baseline clip</option>
            </select>
          </label>
          {mode === "clip" && (
            <label>Clip source
              <select value={clipSource} onChange={(e) => setClipSource(e.target.value)} disabled={connected}>
                {clips.sources.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
          )}
          <label>Noise
            <select value={noise} onChange={(e) => setNoise(e.target.value)}>
              <option value="none">none (clean)</option>
              {clips.noises.map((n) => <option key={n}>{n}</option>)}
              <option value="other">other (label only)</option>
            </select>
          </label>
          <label>SNR dB <input type="number" min={-5} max={40} value={snr} style={{ minWidth: 70 }} onChange={(e) => setSnr(Number(e.target.value))} /></label>
          <label>Tester <input value={tester} placeholder="your name" style={{ minWidth: 110 }} onChange={(e) => setTester(e.target.value)} /></label>
        </div>
        <p className="small mute" style={{ margin: "8px 0 0" }}>
          {mode === "clip"
            ? "The stored clip is streamed into the agent in real time (noise mixed in at the chosen SNR). Send each turn with its button once the agent has finished replying."
            : `Speak to the agent. Noise/SNR here only label the condition you create (e.g. play a noise recording nearby); they group WER.${golds?.script.length ? ` Suggested script: “${golds.script.join(" … ")}”` : ""}`}
        </p>
        <details style={{ marginTop: 8 }}>
          <summary>Reference text for WER (optional)</summary>
          <textarea value={reference} onChange={(e) => setReference(e.target.value)}
            placeholder="What the caller actually says, word for word. Leave empty to use the golden script (or the clip text when replaying clips)." />
        </details>
      </div>

      <div className="card">
        <div className="row">
          <button className="btn" onClick={() => (connected || call.current ? endCall() : void startCall())}>{connected ? "End call" : "Start call"}</button>
          <button className="btn rec" disabled={!connected || recording} onClick={record}>● Record</button>
          <button className="btn" disabled={!recording} onClick={stopAndSave}>■ Stop &amp; save</button>
          <span className="mute">{status}</span>
          <span className="meter" title="mic level"><i style={{ width: `${Math.min(100, level * 100)}%` }} /></span>
          {recording && <span className="pill">{mmss(elapsed)}</span>}
        </div>
        {mode === "clip" && connected && (
          <div className="row" style={{ marginTop: 10 }}>
            {clipLines.length === 0 && <span className="mute">No stored clips for this scenario: record some in the Baseline clips tab or pick another scenario.</span>}
            {clipLines.map((l) => {
              const c = l.clips.find((x) => x.source === clipSource);
              return (
                <button key={l.turn} className="btn" disabled={!c || sent.has(l.turn)} title={c ? l.text : `no ${clipSource} clip for this line`}
                  onClick={() => c && void sendTurn(l.turn, c.id, l.text)}>▶ Send turn {l.turn + 1}</button>
              );
            })}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Conversation (live)</h2>
        <div className="log">
          {log.map((l, i) => <p key={i} className={l.cls}>{l.text}</p>)}
          <div ref={logEnd} />
        </div>
      </div>

      {savedId && (
        <div className="card">
          <h2>Recorded session</h2>
          <div className="row">
            <span>Saved as <b>{savedId}</b></span>
            <label className="inline"><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} /> LLM judge + intent (needs AWS)</label>
            <label className="inline"><input type="checkbox" checked={useMos} onChange={(e) => setUseMos(e.target.checked)} /> UTMOS / DNSMOS (optional deps)</label>
            <button className="btn primary" disabled={evaluating} onClick={() => void evaluate()}>Evaluate</button>
            <a href="#sessions" className="small" onClick={(e) => { e.preventDefault(); onOpenSession(savedId); }}>open in Sessions tab</a>
          </div>
          <div style={{ marginTop: 12 }}>
            {typeof result === "string" ? result : result ? <EvalView e={result} /> : null}
          </div>
        </div>
      )}
    </section>
  );
}
