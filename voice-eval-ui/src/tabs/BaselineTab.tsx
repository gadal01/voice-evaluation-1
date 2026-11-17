import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ClipLine, ClipsResponse } from "../api/models";
import { BaselineRecorder } from "../audio/baselineRecorder";

const slug = (s: string) => s.trim().replace(/[^A-Za-z0-9_-]+/g, "-");

interface Take {
  blob: Blob | null;
  msg: string;
}

export function BaselineTab({ keyVersion }: { keyVersion: number }) {
  const [data, setData] = useState<ClipsResponse>({ lines: [], sources: [], noises: [] });
  const [speaker, setSpeaker] = useState("");
  const [filter, setFilter] = useState("");
  const [err, setErr] = useState("");
  const [level, setLevel] = useState(0);
  const [active, setActive] = useState<string | null>(null); // "scenario#turn" being recorded
  const [takes, setTakes] = useState<Record<string, Take>>({});
  const rec = useRef<BaselineRecorder | null>(null);

  const load = useCallback(() => api.clips().then((d) => { setData(d); setErr(""); }).catch((e: Error) => setErr(e.message)), []);
  useEffect(() => { void load(); }, [load, keyVersion]);

  const keyOf = (l: ClipLine) => `${l.scenario}#${l.turn}`;
  const me = `human:${slug(speaker)}`;
  const mine = (l: ClipLine) => speaker.trim() !== "" && l.clips.some((c) => c.source === me);
  const done = data.lines.filter(mine).length;
  const scenarios = [...new Set(data.lines.map((l) => l.scenario))];
  const lines = filter ? data.lines.filter((l) => l.scenario === filter) : data.lines;
  const setTake = (k: string, t: Take) => setTakes((x) => ({ ...x, [k]: t }));

  const start = async (l: ClipLine) => {
    const k = keyOf(l);
    if (!speaker.trim()) { alert("Enter a speaker id first."); return; }
    try {
      rec.current = new BaselineRecorder();
      await rec.current.start((v) => setLevel(v));
      setActive(k);
      setTake(k, { blob: null, msg: "recording… say the line, then Stop" });
    } catch (e) { setTake(k, { blob: null, msg: (e as Error).message }); }
  };
  const stop = (l: ClipLine) => {
    const k = keyOf(l);
    const blob = rec.current?.stop() ?? null;
    rec.current = null; setActive(null); setLevel(0);
    setTake(k, { blob, msg: blob ? `${(blob.size / 32000).toFixed(1)} s captured: play it back, then save or re-record` : "nothing captured" });
  };
  const upload = async (l: ClipLine) => {
    const k = keyOf(l);
    const blob = takes[k]?.blob;
    if (!blob) return;
    const fd = new FormData();
    fd.append("scenario", l.scenario); fd.append("turn", String(l.turn)); fd.append("name", speaker.trim()); fd.append("wav", blob, "clip.wav");
    try {
      const r = await api.addClip(fd);
      setTake(k, { blob: null, msg: `saved ${r.id} (${r.duration_s}s, SNR≈${r.audio.snr_db} dB)` });
      await load();
    } catch (e) { setTake(k, { blob, msg: `✗ ${(e as Error).message}` }); }
  };

  return (
    <section>
      <div className="card">
        <h2>Record human baseline clips</h2>
        <p className="small mute" style={{ marginTop: 0 }}>
          Read each line aloud, naturally, in a quiet room. Clips are stored 16 kHz mono in <code>evals/caller_clips/</code> as source <code>human:&lt;speaker&gt;</code> and
          replay through the agent here or with <code>run_evals.py --voices clip:human:&lt;speaker&gt;</code>. Echo cancellation, noise suppression and auto-gain are OFF so the clip is
          your raw voice. Aim for a variety of speakers (accent, age, gender, pace).
        </p>
        <div className="row">
          <label>Speaker id <input value={speaker} placeholder="e.g. alice-us-f" onChange={(e) => setSpeaker(e.target.value)} /></label>
          <label>Scenario filter
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="">all</option>
              {scenarios.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <span className="meter"><i style={{ width: `${Math.min(100, level * 100)}%` }} /></span>
          <span className="small mute">{err}</span>
        </div>
      </div>
      <div className="card">
        <h2>Lines to record {speaker.trim() && `· ${done}/${data.lines.length} done for “${speaker.trim()}”`}</h2>
        {lines.map((l) => {
          const k = keyOf(l);
          const t = takes[k];
          const recording = active === k;
          const humans = l.clips.filter((c) => c.source.startsWith("human:")).map((c) => c.source);
          return (
            <div className="line" key={k}>
              <div className="txt">
                <span className="mute small">{l.scenario} · line {l.turn + 1}{l.reactive ? " (reactive)" : ""}</span><br />
                {l.text}
                <div className="small mute">{humans.length ? humans.join(", ") : "no human clips yet"}</div>
              </div>
              <span className={`pill ${mine(l) ? "ok" : ""}`}>{mine(l) ? "✓ recorded" : "todo"}</span>
              <button className="btn rec" disabled={active !== null} onClick={() => void start(l)}>● Record</button>
              <button className="btn" disabled={!recording} onClick={() => stop(l)}>■ Stop</button>
              <button className="btn" disabled={!t?.blob} onClick={() => t?.blob && void new Audio(URL.createObjectURL(t.blob)).play()}>▶ Play</button>
              <button className="btn primary" disabled={!t?.blob} onClick={() => void upload(l)}>Save to library</button>
              <span className="small">{t?.msg}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
