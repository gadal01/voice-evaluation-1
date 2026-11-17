import type { ReactNode } from "react";
import type { Evaluation } from "../api/models";

// Starting targets (green/red outline). These are opinions: tune them for your product.
export const TARGETS = { wer: 0.1, slot: 0.95, ttfaMs: 1500, falseBargeIn: 0.05, utmos: 3.5, judge: 4 };

export const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
export const fmt = (v: number | null | undefined, d = 2) => (v == null ? "–" : String(+v.toFixed(d)));

function Metric(p: { name: string; value: ReactNode; sub?: ReactNode; good?: boolean | null }) {
  const cls = p.good === true ? "ok" : p.good === false ? "bad" : "";
  return (
    <div className={`metric ${cls}`}>
      <div className="n">{p.name}</div>
      <div className="v">{p.value}</div>
      <div className="s">{p.sub}</div>
    </div>
  );
}

const JUDGE_KEYS = ["tone", "clarity", "empathy", "accuracy", "policy"] as const;

export function EvalView({ e }: { e: Evaluation }) {
  const { outcome: o, slots: sl, wer: w, latency: lat, barge_in: bi, turns_to_resolution: tr, judge: jd, mos, intent, cost } = e;
  const utmos = mos.utmos;
  const dns = mos.dnsmos;
  return (
    <>
      <div className="grid">
        <Metric
          name="Task success"
          value={o.task_success == null ? "–" : o.task_success ? "PASS" : "FAIL"}
          sub={`outcome ${o.actual} vs golden ${o.golden ?? "none"}${o.forbidden_booking ? " · booked when it should not" : ""}`}
          good={o.task_success}
        />
        <Metric
          name="Slot accuracy"
          value={pct(sl.score)}
          sub={sl.note || Object.entries(sl.slots).map(([k, v]) => `${v.ok ? "✓" : "✗"} ${k}`).join(" · ")}
          good={sl.score == null ? null : sl.score >= TARGETS.slot}
        />
        <Metric
          name="WER"
          value={pct(w.wer)}
          sub={w.wer == null ? w.note : `${w.sub} sub · ${w.del} del · ${w.ins} ins / ${w.words} words`}
          good={w.wer == null ? null : w.wer <= TARGETS.wer}
        />
        <Metric
          name="Intent"
          value={intent.correct == null ? intent.predicted : intent.correct ? "✓ correct" : "✗ wrong"}
          sub={`predicted ${intent.predicted} vs ${intent.labelled ?? "unlabelled"} (${intent.source})`}
          good={intent.correct}
        />
        <Metric
          name="Turns to resolution"
          value={tr.turns}
          sub={`${tr.total_user_turns} caller turns total · ${tr.resolved ? "resolved" : "NOT resolved"}`}
          good={tr.resolved}
        />
        <Metric
          name="Latency p50 / p95"
          value={`${fmt(lat.p50_ms, 0)} / ${fmt(lat.p95_ms, 0)} ms`}
          sub={`caller stops → agent audio plays · ${lat.values_ms.length} turns`}
          good={lat.p50_ms == null ? null : lat.p50_ms <= TARGETS.ttfaMs}
        />
        <Metric
          name="False barge-in"
          value={`${bi.false} / ${bi.agent_turns} turns`}
          sub={`${bi.interrupts} interrupts · ${bi.missed} missed barge-in(s)`}
          good={bi.false_rate <= TARGETS.falseBargeIn}
        />
        <Metric
          name="Cost / session"
          value={`$${fmt(cost.total_usd, 4)}`}
          sub={
            <>
              agent ${fmt(cost.agent_usd, 4)} · judge ${fmt(cost.judge_usd, 4)}
              {cost.estimate && (
                <>
                  {" "}
                  · <b>placeholder rates</b> (pricing.json)
                </>
              )}
            </>
          }
        />
        {jd.status === "ok" && jd.scores ? (
          JUDGE_KEYS.map((k) => (
            <Metric
              key={k}
              name={k[0]!.toUpperCase() + k.slice(1)}
              value={`${jd.scores![k]} / 5`}
              sub={k === "tone" ? jd.reason : ""}
              good={(jd.scores![k] ?? 0) >= TARGETS.judge}
            />
          ))
        ) : (
          <Metric name="LLM judge" value="–" sub={`skipped: ${jd.reason}`} />
        )}
        <Metric
          name="UTMOS"
          value={utmos.status === "ok" ? utmos.utmos : "–"}
          sub={utmos.status === "ok" ? "TTS naturalness (1–5)" : `skipped: ${utmos.reason}`}
          good={utmos.status === "ok" && utmos.utmos != null ? utmos.utmos >= TARGETS.utmos : null}
        />
        <Metric
          name="DNSMOS"
          value={dns.status === "ok" ? dns.ovrl_mos : "–"}
          sub={dns.status === "ok" ? `sig ${dns.sig_mos} · bg ${dns.bak_mos} · p808 ${dns.p808_mos}` : `skipped: ${dns.reason}`}
        />
      </div>
      {jd.status === "ok" && jd.issues.length > 0 && (
        <p className="small">
          <b>Judge issues:</b> {jd.issues.join(" · ")}
        </p>
      )}
      <details>
        <summary>
          Tool calls ({e.tools.calls.length}
          {e.tools.errors ? `, ${e.tools.errors} error(s)` : ""})
        </summary>
        <div className="t">
          {e.tools.calls.map((c, i) => (
            <div key={i}>
              {c.name} {JSON.stringify(c.args)} {c.ms ?? ""}ms
            </div>
          ))}
        </div>
      </details>
      <p className="small mute">
        Targets (green/red outline) are starting values in EvalView.tsx; judge and MOS scores are advisory: calibrate with human labels.
      </p>
    </>
  );
}
