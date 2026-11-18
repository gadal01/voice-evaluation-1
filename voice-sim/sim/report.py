"""A self-contained HTML report for one run: open results/<timestamp>/report.html (audio players need it opened from that folder)."""
import json
from html import escape as e

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--mute:#667085;--line:#e3e6ea;--ok:#0f8a4f;--bad:#c62828;--accent:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#14171b;--card:#1c2025;--ink:#e8eaed;--mute:#9aa3af;--line:#2d333b;--ok:#4cc38a;--bad:#ff7b72;--accent:#6ea0ff}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:20px}h2{font-size:15px;margin:22px 0 8px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}th{color:var(--mute);font-weight:600}
.pass{color:var(--ok);font-weight:600}.fail{color:var(--bad);font-weight:600}.mute{color:var(--mute)}.u{color:var(--accent)}.a{color:var(--ok)}
.t{font:12px ui-monospace,Consolas,monospace;color:var(--mute);word-break:break-all}summary{cursor:pointer;font-weight:600}p{margin:4px 0}audio{height:32px}
"""


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


def num(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def stats_table(title: str, rows: dict, criteria: list[str]) -> str:
    head = "".join(f"<th>{e(c)}</th>" for c in criteria)
    body = "".join(
        f"<tr><td><b>{e(name)}</b></td><td>{a['passed']}/{a['sessions']}</td><td>{pct(a['pass_rate'])}</td><td>{num(a['judge_overall'])}</td>"
        + "".join(f"<td>{num(a['scores'].get(c))}</td>" for c in criteria)
        + f"<td>{num(a['ttfa_p50_ms'], 0)}</td><td>{pct(a['heard_wer'])}</td><td>{a['errors']}</td></tr>" for name, a in rows.items())
    return (f"<h2>{e(title)}</h2><div class='card'><table><tr><th></th><th>passed</th><th>rate</th><th>judge</th>{head}"
            f"<th>TTFA p50 ms</th><th>heard WER</th><th>errors</th></tr>{body}</table></div>")


def session_block(r: dict) -> str:
    verdict = {True: "<span class='pass'>PASS</span>", False: "<span class='fail'>FAIL</span>", None: "<span class='mute'>not judged</span>"}[r["passed"]]
    turns = []
    for t in r["turns"]:
        w = t.get("wer") or {}
        tools = "".join(f"<p class='t'>tool {e(x['name'])}({e(json.dumps(x['args']))}) -> {e(json.dumps(x['result'])[:300])} [{x.get('ms')} ms]</p>" for x in t["tools"])
        turns.append(
            f"<p class='u'><b>User:</b> {e(t['said'])}</p>"
            f"<p class='mute'>heard as: {e(t['heard'] or '[nothing]')} &nbsp;·&nbsp; word error {pct(w.get('wer'))} &nbsp;·&nbsp; "
            f"first audio {num(t.get('ttfa_ms'), 0)} ms</p>{tools}"
            f"<p class='a'><b>Agent:</b> {e(t['reply'] or '[no answer]')}</p>")
    j = r.get("judge")
    if j and j["status"] == "ok":
        scores = " &nbsp; ".join(f"{e(k)} <b>{v}</b>" for k, v in j["scores"].items())
        judge = (f"<p>{scores}</p><p>{e(j['reason'])}</p>" + (f"<p><b>Issues:</b> {e(' · '.join(j['issues']))}</p>" if j["issues"] else ""))
    else:
        judge = f"<p class='mute'>judge: {e((j or {}).get('reason', 'off'))}</p>"
    audio = (f"<p>User <audio controls preload='none' src='{e(r['dir'])}/user.wav'></audio> &nbsp; "
             f"Agent <audio controls preload='none' src='{e(r['dir'])}/agent.wav'></audio></p>") if r["turns"] else ""
    err = f"<p class='fail'>{e(r['error'])}</p>" if r["error"] else ""
    return (f"<details class='card' {'open' if r['passed'] is False else ''}><summary>{verdict} &nbsp; {e(r['scenario'])} · {e(r['mode'])} "
            f"<span class='mute'>· {r['duration_s']:.0f}s</span></summary>{err}{audio}{judge}<hr>{''.join(turns)}</details>")


def render_report(results: list[dict], summary: dict) -> str:
    m, o = summary["meta"], summary["overall"]
    criteria = list(o["scores"]) or []
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Simulated user run</title>"
            f"<style>{CSS}</style><main><h1>Simulated user run</h1>"
            f"<p class='mute'>{e(m['created'])} · mode {e(m['mode'])} · Piper voice {e(m['voice'])} · agent {e(m['url'])} · "
            f"judge {e(str(m['judge_model']))} · user model {e(str(m['user_model']))}</p>"
            f"<div class='card'><b>{o['passed']}/{o['sessions']} passed</b> · pass rate {pct(o['pass_rate'])} · judge {num(o['judge_overall'])} / 5 · "
            f"TTFA p50 {num(o['ttfa_p50_ms'], 0)} ms · heard WER {pct(o['heard_wer'])} · {o['errors']} session error(s)</div>"
            + stats_table("By mode", summary["by_mode"], criteria) + stats_table("By scenario", summary["by_scenario"], criteria)
            + "<h2>Sessions</h2>" + "".join(session_block(r) for r in results) + "</main>")
