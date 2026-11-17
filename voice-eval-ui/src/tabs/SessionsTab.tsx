import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Evaluation, GroupStats, SessionDetail, SessionRow, SummaryResponse } from "../api/models";
import { EvalView, fmt, pct } from "../components/EvalView";

type Col = { key: keyof GroupStats; label: string; f: (v: number | null | undefined) => string };
const COLS: Col[] = [
  { key: "sessions", label: "n", f: (v) => String(v ?? "–") },
  { key: "task_success", label: "Task success", f: pct },
  { key: "wer", label: "WER", f: pct },
  { key: "slot_accuracy", label: "Slot acc.", f: pct },
  { key: "intent_accuracy", label: "Intent acc.", f: pct },
  { key: "turns_to_resolution", label: "Turns", f: (v) => fmt(v, 1) },
  { key: "ttfa_p50_ms", label: "TTFA p50 ms", f: (v) => fmt(v, 0) },
  { key: "false_barge_in_rate", label: "False barge-in", f: pct },
  { key: "tone", label: "Tone", f: (v) => fmt(v, 1) },
  { key: "clarity", label: "Clarity", f: (v) => fmt(v, 1) },
  { key: "empathy", label: "Empathy", f: (v) => fmt(v, 1) },
  { key: "utmos", label: "UTMOS", f: (v) => fmt(v, 2) },
  { key: "dnsmos", label: "DNSMOS", f: (v) => fmt(v, 2) },
  { key: "cost_usd", label: "Cost $", f: (v) => fmt(v, 4) },
];

function StatsTable({ title, rows }: { title: string; rows: Record<string, GroupStats> }) {
  return (
    <>
      <h3>{title}</h3>
      <div style={{ overflow: "auto" }}>
        <table>
          <thead>
            <tr><th />{COLS.map((c) => <th key={c.key}>{c.label}</th>)}</tr>
          </thead>
          <tbody>
            {Object.entries(rows).map(([name, g]) => (
              <tr key={name}>
                <td><b>{name}</b></td>
                {COLS.map((c) => <td key={c.key}>{c.f(g[c.key] as number | null | undefined)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Summary({ s }: { s: SummaryResponse }) {
  if (!s.overall) return <span className="mute">No evaluated sessions yet.</span>;
  return (
    <>
      <StatsTable title="Overall" rows={{ all: s.overall }} />
      <StatsTable title="WER under noise: by condition" rows={s.by_condition} />
      <StatsTable title="By scenario (item / symptom)" rows={s.by_scenario} />
      {!s.pricing_verified && <p className="small">Cost uses placeholder rates: edit voice_eval/pricing.json in voice-eval-backend.</p>}
    </>
  );
}

function Detail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const [s, setS] = useState<SessionDetail | null>(null);
  const [urls, setUrls] = useState<{ caller: string; agent: string } | null>(null);
  const [ev, setEv] = useState<Evaluation | string | null>(null);
  const [llm, setLlm] = useState(true);
  const [mos, setMos] = useState(true);
  const [busy, setBusy] = useState(false);
  const [nat, setNat] = useState("");
  const [pron, setPron] = useState("");
  const [misp, setMisp] = useState("");
  const [rev, setRev] = useState("");
  const [notes, setNotes] = useState("");
  const [rvMsg, setRvMsg] = useState("");

  useEffect(() => {
    let alive = true;
    let made: string[] = [];
    setS(null); setUrls(null);
    (async () => {
      const [d, caller, agent] = await Promise.all([api.session(id), api.audioUrl(id, "caller"), api.audioUrl(id, "agent")]);
      made = [caller, agent];
      if (!alive) return URL.revokeObjectURL(caller), URL.revokeObjectURL(agent);
      setS(d); setUrls({ caller, agent }); setEv(d.evaluation ?? null);
      setNat(d.review?.naturalness?.toString() ?? ""); setPron(d.review?.pronunciation?.toString() ?? "");
      setMisp(d.review?.mispronounced ?? ""); setRev(d.review?.reviewer ?? ""); setNotes(d.review?.notes ?? "");
      setRvMsg(d.review?.at ? `saved ${d.review.at}` : "");
    })().catch((e: Error) => alive && setEv(e.message));
    return () => { alive = false; made.forEach(URL.revokeObjectURL); };
  }, [id]);

  if (!s || !urls) return <div className="card">{typeof ev === "string" ? ev : "Loading…"}</div>;

  const run = async () => {
    setBusy(true); setEv("Evaluating…");
    try { setEv(await api.evaluate(id, { llm, mos })); onChanged(); }
    catch (e) { setEv(`Evaluation failed: ${(e as Error).message}`); }
    setBusy(false);
  };
  const num = (v: string) => (v === "" ? null : parseInt(v, 10));
  const saveReview = async () => {
    try {
      await api.review(id, { naturalness: num(nat), pronunciation: num(pron), mispronounced: misp, notes, reviewer: rev });
      setRvMsg("saved");
    } catch (e) { setRvMsg((e as Error).message); }
  };

  return (
    <div className="card" id="detail">
      <h2>Session {id}</h2>
      <p className="small mute">{s.meta.created} · {s.meta.scenario_id} · {s.meta.caller_mode} · tester {s.meta.tester || "–"}</p>
      <div className="row">
        <div><div className="small mute">Caller</div><audio controls src={urls.caller} /></div>
        <div><div className="small mute">Agent (as played)</div><audio controls src={urls.agent} /></div>
      </div>
      <div className="row" style={{ margin: "12px 0" }}>
        <label className="inline"><input type="checkbox" checked={llm} onChange={(e) => setLlm(e.target.checked)} /> LLM judge</label>
        <label className="inline"><input type="checkbox" checked={mos} onChange={(e) => setMos(e.target.checked)} /> MOS</label>
        <button className="btn primary" disabled={busy} onClick={() => void run()}>Evaluate</button>
      </div>
      <div>{typeof ev === "string" ? ev : ev ? <EvalView e={ev} /> : <span className="mute">Not evaluated yet.</span>}</div>

      <h2 style={{ marginTop: 16 }}>Human spot check (TTS naturalness &amp; pronunciation)</h2>
      <div className="row">
        <label>Naturalness 1–5 <input type="number" min={1} max={5} value={nat} style={{ minWidth: 70 }} onChange={(e) => setNat(e.target.value)} /></label>
        <label>Pronunciation 1–5 <input type="number" min={1} max={5} value={pron} style={{ minWidth: 70 }} onChange={(e) => setPron(e.target.value)} /></label>
        <label>Mispronounced words <input value={misp} onChange={(e) => setMisp(e.target.value)} /></label>
        <label>Reviewer <input value={rev} style={{ minWidth: 100 }} onChange={(e) => setRev(e.target.value)} /></label>
        <button className="btn" onClick={() => void saveReview()}>Save review</button>
        <span className="small mute">{rvMsg}</span>
      </div>
      <label style={{ marginTop: 8 }}>Notes <textarea value={notes} onChange={(e) => setNotes(e.target.value)} /></label>

      <h2 style={{ marginTop: 16 }}>Transcript, events &amp; tool calls</h2>
      <div className="log" style={{ maxHeight: 360 }}>
        {s.events.filter((e) => ["transcript", "reply", "tool", "interrupted", "error"].includes(e.type)).map((e, i) => {
          const t = <span className="mute">{e.t.toFixed(1)}s </span>;
          if (e.type === "transcript") return <p key={i} className="u">{t}Caller (heard): {e.text}</p>;
          if (e.type === "reply") return <p key={i} className="a">{t}Agent: {e.text}</p>;
          if (e.type === "tool") return <p key={i} className="t">{t}🔧 {e.name}({JSON.stringify(e.args)}) → {JSON.stringify(e.result)} [{e.ms} ms]</p>;
          return <p key={i} className="i">{t}{e.type} {e.message}</p>;
        })}
      </div>
    </div>
  );
}

export function SessionsTab({ keyVersion, openId }: { keyVersion: number; openId: string | null }) {
  const [rows, setRows] = useState<SessionRow[]>([]);
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(openId);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    try {
      const [r, s] = await Promise.all([api.sessions(), api.summary()]);
      setRows(r); setSummary(s); setErr("");
    } catch (e) { setErr((e as Error).message); }
  }, []);
  useEffect(() => { void load(); }, [load, keyVersion]);
  useEffect(() => { if (openId) setSelected(openId); }, [openId]);

  const cond = (r: SessionRow) => (!r.condition?.noise || r.condition.noise === "none" ? "clean" : `${r.condition.noise} @ ${r.condition.snr_db} dB`);

  return (
    <section>
      <div className="card">
        <div className="row"><h2 style={{ margin: 0 }}>Aggregate (all evaluated sessions)</h2><button className="btn" style={{ marginLeft: "auto" }} onClick={() => void load()}>Refresh</button></div>
        <div className="small" style={{ marginTop: 8 }}>{summary ? <Summary s={summary} /> : err || "Loading…"}</div>
      </div>
      <div className="card">
        <h2>Sessions</h2>
        {rows.length === 0 ? <span className="mute">{err || "No sessions recorded yet. Record one in the Live session tab."}</span> : (
          <table>
            <thead><tr><th>When</th><th>Session</th><th>Scenario</th><th>Caller</th><th>Condition</th><th>Outcome</th><th>Success</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="click" onClick={() => setSelected(r.id)}>
                  <td>{r.created}</td><td>{r.id}</td><td>{r.scenario_id}</td><td>{r.caller_mode}</td><td>{cond(r)}</td>
                  <td>{r.evaluated ? r.outcome : <span className="mute">not evaluated</span>}</td>
                  <td>{r.task_success == null ? "–" : <span className={`pill ${r.task_success ? "ok" : "bad"}`}>{r.task_success ? "pass" : "fail"}</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {selected && <Detail id={selected} onChanged={() => void load()} />}
    </section>
  );
}
