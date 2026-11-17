"""Writes the human-readable outputs of an eval run: transcripts/*.txt, report.html, summary.json.

    python batch/report.py                    # rebuild from the newest folder in batch/results
    python batch/report.py batch/results/20261016-114914

Also called by run_evals.py at the end of every run. Reads runs.jsonl (+ meta.json if present).
The HTML is one self-contained file: no scripts, no external assets.
"""
import difflib
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402

HERE = Path(__file__).parent
esc = html.escape


# ------------------------------------------------------------- conversation ---
def _scripts() -> dict[str, list[str]]:
    try:
        return {s["id"]: s.get("turns") or [s["dialogue"]["start"]]
                for s in json.loads((HERE / "scenarios.json").read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return {}


def conversation_of(run: dict) -> list[dict]:
    """The ordered conversation. Runs saved before conversations were captured get a rebuilt, coarser one."""
    if run.get("conversation"):
        return run["conversation"]
    script = _scripts().get(run["scenario"], [])
    conv = []
    for t in run.get("turns", []):
        i = t["turn"]
        conv.append({"role": "caller", "turn": i, "t": None, "text": script[i] if i < len(script) else ""})
        if t.get("heard"):
            conv.append({"role": "heard", "turn": i, "t": None, "text": t["heard"], "wer": t["wer"]})
    if run.get("reply"):
        conv.append({"role": "agent", "turn": len(run.get("turns", [])) - 1, "t": None, "text": run["reply"]})
    return conv


def _fmt_t(t) -> str:
    return f"{t:5.1f}s" if isinstance(t, (int, float)) else "     -"


def _short(obj, n=160) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def verdict_reasons(r: dict) -> list[str]:
    out = []
    if r.get("error"):
        out.append(f"error: {r['error']}")
    if r.get("timed_out"):
        out.append("timed out waiting for the agent")
    if r.get("tools_missing"):
        out.append("missing tool calls: " + ", ".join(r["tools_missing"]))
    if r.get("forbidden_called"):
        out.append("forbidden tool called: " + ", ".join(r["forbidden_called"]))
    if r.get("tool_errors"):
        out.append("tool errors: " + ", ".join(r["tool_errors"]))
    if not r.get("outcome_ok", True):
        out.append("outcome: " + r.get("outcome_why", ""))
    if r.get("missing_mentions"):
        out.append("reply never matched: " + ", ".join(r["missing_mentions"]))
    if r.get("missing_readback"):
        out.append("did not read back before booking: " + ", ".join(r["missing_readback"]))
    if r.get("thinking_aloud"):
        out.append("agent narrated its reasoning aloud (spoke its chain of thought or tool names)")
    return out


# --------------------------------------------------------------- text files ---
def transcript_text(run: dict) -> str:
    wer_pct = f"{100 * run['wer_errors'] / run['wer_words']:.1f}%" if run.get("wer_words") else "-"
    lines = [f"Scenario : {run['scenario']}", f"Voice    : {run['voice']}",
             f"Result   : {'PASS' if run['passed'] else 'FAIL'}", f"WER      : {wer_pct}"]
    lines += [f"Problem  : {why}" for why in verdict_reasons(run)]
    j = run.get("judge")
    if j and "scores" in j:
        lines.append(f"Judge    : {j['overall']:.1f}/5 ({', '.join(f'{k} {v}' for k, v in j['scores'].items())}) - {j.get('reason', '')}")
        lines += [f"Judge !  : {i}" for i in j.get("issues", [])]
    if run.get("audio_file"):
        lines.append(f"Audio    : {run['audio_file']} (stereo: caller left, agent right)")
    lines.append("")
    for c in conversation_of(run):
        t = _fmt_t(c.get("t"))
        if c["role"] == "caller":
            lines.append(f"[{t}] CALLER (script) : {c['text']}")
        elif c["role"] == "heard":
            lines.append(f"{'':9} AGENT HEARD    : {c['text']}   (WER {100 * c.get('wer', 0):.0f}%)")
        elif c["role"] == "tool":
            lines.append(f"[{t}] TOOL           : {c['name']}({_short(c['args'])}) -> {_short(c['result'])}"
                         + (f"  [{c['ms']} ms]" if c.get("ms") is not None else ""))
        elif c["role"] == "agent":
            lines.append(f"[{t}] AGENT          : {c['text']}")
        else:
            lines.append(f"[{t}] !! {c['text']}")
    return "\n".join(lines) + "\n"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def write_transcripts(runs: list[dict], out: Path) -> None:
    (out / "transcripts").mkdir(exist_ok=True)
    for r in runs:
        name = f"{r['run_id']}_{slug(r['scenario'])}_{slug(r['voice'])}.txt"
        r["transcript_file"] = f"transcripts/{name}"
        (out / "transcripts" / name).write_text(transcript_text(r), encoding="utf-8")


# --------------------------------------------------------------------- html ---
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2230;--muted:#5d6678;--line:#e1e5ec;--accent:#2f5fd0;
--pass:#17794a;--pass-bg:#dff3e7;--fail:#b3261e;--fail-bg:#fbe3e1;--warn:#8a5a00;--warn-bg:#fdf0d0;--bar:#9db7ee;--bar2:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b202b;--ink:#e8ebf2;--muted:#9aa3b5;--line:#2d3444;--accent:#8fb0ff;
--pass:#7fdca6;--pass-bg:#173a29;--fail:#ff9d96;--fail-bg:#472120;--warn:#f2c36b;--warn-bg:#3f3213;--bar:#3a4f86;--bar2:#8fb0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{margin:0 0 4px;font-size:1.6rem}h2{margin:32px 0 10px;font-size:1.15rem}
.sub{color:var(--muted);margin:0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:1.5rem}
.card span{color:var(--muted);font-size:.8rem}.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:.78rem;font-weight:700;white-space:nowrap}
.pass{background:var(--pass-bg);color:var(--pass)}.fail{background:var(--fail-bg);color:var(--fail)}.part{background:var(--warn-bg);color:var(--warn)}
.bar{position:relative;height:10px;min-width:120px;background:var(--line);border-radius:5px}.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--bar)}
.bar u{position:absolute;left:0;top:0;bottom:0;border-radius:5px;background:var(--bar2);text-decoration:none}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0}summary{cursor:pointer;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
summary .meta{color:var(--muted);font-size:.85rem}.body{padding:4px 14px 14px;border-top:1px solid var(--line)}
.why{color:var(--fail);margin:8px 0 0;padding-left:18px}.chat{display:flex;flex-direction:column;gap:6px;margin-top:10px}
.msg{max-width:85%;padding:7px 11px;border-radius:12px;border:1px solid var(--line);white-space:pre-wrap;overflow-wrap:anywhere}
.msg small{display:block;color:var(--muted);font-size:.72rem;margin-bottom:1px}.caller{align-self:flex-end;background:var(--bar);border-color:transparent}
.heard{align-self:flex-end;font-size:.85rem;color:var(--muted);border-style:dashed}.agent{align-self:flex-start}
.tool{align-self:stretch;max-width:100%;font:12px/1.4 ui-monospace,Consolas,monospace;background:var(--bg)}.system{align-self:center;color:var(--fail)}
mark{background:var(--fail-bg);color:var(--fail);border-radius:3px;padding:0 2px}.mono{font-family:ui-monospace,Consolas,monospace;font-size:.82rem}
a{color:var(--accent)}.hint{color:var(--muted);font-size:.82rem;margin:6px 2px 0}
"""


def _pill(ok: bool) -> str:
    return f'<span class="pill {"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span>'


def _ms(v) -> str:
    return "–" if v is None else f"{v:,.0f} ms"


def _pct(v) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def _judge(v) -> str:
    return "–" if v is None else f"{v:.1f}/5"


def _judge_block(r: dict) -> str:
    j = r.get("judge")
    if not j:
        return ""
    if "scores" not in j:
        return f'<p class="hint">LLM judge unavailable: {esc(j.get("error", ""))}</p>'
    chips = " ".join(f'<span class="pill {"pass" if v >= 4 else "fail"}">{esc(k)} {v}</span>' for k, v in j["scores"].items())
    issues = "".join(f"<li>{esc(i)}</li>" for i in j.get("issues", []))
    return (f'<p class="hint"><b>LLM judge</b> ({esc(j["model"])}) · overall {j["overall"]:.1f}/5 · {"pass" if j["pass"] else "needs attention"}</p>'
            f'<p>{chips}</p><p>{esc(j.get("reason", ""))}</p>' + (f'<ul class="why">{issues}</ul>' if issues else ""))


def _early(t: dict) -> str:
    return f"<mark>{t['early_ms'] / 1000:.1f} s early</mark>" if t.get("early_ms") else "–"


def _diff_heard(ref: str, heard: str) -> str:
    """Heard text (normalised) with the words that differ from the script highlighted."""
    r, h = metrics.normalize(ref), metrics.normalize(heard)
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=r, b=h).get_opcodes():
        words = esc(" ".join(h[j1:j2]))
        if words:
            out.append(words if op == "equal" else f"<mark>{words}</mark>")
    return " ".join(out) or "<i>(nothing heard)</i>"


def _group_table(rows: list[dict], label: str) -> str:
    top = max([g["ttfa_p95"] or 0 for g in rows] + [1])
    body = []
    for g in rows:
        p50, p95 = g["ttfa_p50"] or 0, g["ttfa_p95"] or 0
        bar = f'<div class="bar" title="p50 {_ms(g["ttfa_p50"])} / p95 {_ms(g["ttfa_p95"])}"><i style="width:{100 * p95 / top:.0f}%"></i><u style="width:{100 * p50 / top:.0f}%"></u></div>'
        cls = "pass" if g["passed"] == g["runs"] else ("fail" if g["passed"] == 0 else "part")
        body.append(f'<tr><td>{esc(g["name"])}</td><td class="num">{g["runs"]}</td>'
                    f'<td><span class="pill {cls}">{g["passed"]}/{g["runs"]} · {100 * g["pass_rate"]:.0f}%</span></td>'
                    f'<td class="num">{g["tool_score"]:.2f}</td><td class="num">{_pct(g["wer"])}</td><td class="num">{_judge(g.get("judge"))}</td>'
                    f'<td class="num">{_ms(g["ttfa_p50"])}</td><td class="num">{_ms(g["ttfa_p95"])}</td><td>{bar}</td>'
                    f'<td class="num">{_ms(g["tool_ms"])}</td></tr>')
    return (f'<div class="wrap"><table><tr><th>{label}</th><th class="num">Calls</th><th>Passed</th><th class="num">Tools</th>'
            f'<th class="num">WER</th><th class="num">Judge</th><th class="num">TTFA p50</th><th class="num">TTFA p95</th><th>p50 / p95</th>'
            f'<th class="num">Tool call</th></tr>{"".join(body)}</table></div>')


def _matrix(runs: list[dict]) -> str:
    voices = sorted({r["voice"] for r in runs})
    scenarios = sorted({r["scenario"] for r in runs})
    head = "".join(f"<th>{esc(v)}</th>" for v in voices)
    rows = []
    for s in scenarios:
        cells = []
        for v in voices:
            rs = [r for r in runs if r["scenario"] == s and r["voice"] == v]
            if not rs:
                cells.append("<td>–</td>")
                continue
            ok = sum(r["passed"] for r in rs)
            cls = "pass" if ok == len(rs) else ("fail" if ok == 0 else "part")
            cells.append(f'<td><span class="pill {cls}">{ok}/{len(rs)}</span></td>')
        rows.append(f"<tr><td>{esc(s)}</td>{''.join(cells)}</tr>")
    return f'<div class="wrap"><table><tr><th>Scenario</th>{head}</tr>{"".join(rows)}</table></div>'


def _chat(run: dict) -> str:
    out = []
    for c in conversation_of(run):
        t = f' · {c["t"]:.1f}s' if isinstance(c.get("t"), (int, float)) else ""
        if c["role"] == "caller":
            span = f' · speaking {c["t"]:.1f}–{c["t_end"]:.1f}s' if isinstance(c.get("t"), (int, float)) and c.get("t_end") else ""
            out.append(f'<div class="msg caller"><small>Caller (script){span}</small>{esc(c["text"])}</div>')
        elif c["role"] == "heard":
            out.append(f'<div class="msg heard"><small>Agent heard · WER {100 * c.get("wer", 0):.0f}%</small>{esc(c["text"])}</div>')
        elif c["role"] == "tool":
            ms = f' · {c["ms"]} ms' if c.get("ms") is not None else ""
            out.append(f'<div class="msg tool"><small>Tool call{t}{ms}</small>{esc(c["name"])}({esc(_short(c["args"], 300))})<br>→ {esc(_short(c["result"], 400))}</div>')
        elif c["role"] == "agent":
            out.append(f'<div class="msg agent"><small>Agent{t}</small>{esc(c["text"])}</div>')
        else:
            out.append(f'<div class="msg system"><small>Harness{t}</small>{esc(c["text"])}</div>')
    # merge consecutive agent bubbles (the model streams a reply in sentences)
    merged = re.sub(r"</div>\s*<div class=\"msg agent\"><small>Agent[^<]*</small>", " ", "".join(out))
    return f'<div class="chat">{merged}</div>'


def _run_block(r: dict, scripts: dict[str, list[str]]) -> str:
    ttfa = [t["ttfa_ms"] for t in r["turns"] if t.get("ttfa_ms") is not None]
    wer = r["wer_errors"] / r["wer_words"] if r.get("wer_words") else None
    reasons = verdict_reasons(r)
    script = {c["turn"]: c["text"] for c in conversation_of(r) if c["role"] == "caller"} or dict(enumerate(scripts.get(r["scenario"], [])))
    turn_rows = "".join(
        f'<tr><td class="num">{t["turn"] + 1}</td><td>{esc(script.get(t["turn"], ""))}</td>'
        f'<td>{_diff_heard(script.get(t["turn"], ""), t["heard"]) if t.get("heard") is not None else ""}</td>'
        f'<td class="num">{_pct(t["wer"])}</td><td class="num">{_ms(t.get("ttfa_ms"))}</td>'
        f'<td class="num">{_ms(t.get("first_text_ms"))}</td><td class="num">{_ms(t.get("tool_ms"))}</td>'
        f'<td class="num">{_early(t)}</td>'
        f'<td class="num">{t.get("agent_audio_s", 0)} s</td></tr>' for t in r["turns"])
    called = " → ".join(r.get("tools_called", [])) or "none"
    link = f' · <a href="{esc(r["transcript_file"])}">transcript .txt</a>' if r.get("transcript_file") else ""
    player = (f'<p class="hint">Conversation audio (caller left, agent right):</p>'
              f'<audio controls preload="none" src="{esc(r["audio_file"])}" style="width:100%"></audio>') if r.get("audio_file") else ""
    return (
        f'<details id="run-{esc(r["run_id"])}"><summary>{_pill(r["passed"])} <b>{esc(r["scenario"])}</b>'
        f'<span class="meta">{esc(r["voice"])} · TTFA {", ".join(_ms(x) for x in ttfa) or "–"} · WER {_pct(wer)} · tools {r["tool_score"]:.2f}{" · judge " + _judge(r["judge"]["overall"]) if r.get("judge", {}).get("scores") else ""}</span></summary>'
        f'<div class="body">'
        + (f'<ul class="why">{"".join(f"<li>{esc(x)}</li>" for x in reasons)}</ul>' if reasons else "")
        + f'<p class="hint">Tools called: <span class="mono">{esc(called)}</span>{link}</p>'
        f'{_judge_block(r)}{player}<div class="wrap" style="margin-top:8px"><table><tr><th class="num">Turn</th><th>Script</th><th>Agent heard (differences marked)</th>'
        f'<th class="num">WER</th><th class="num">TTFA</th><th class="num">First text</th><th class="num">Tool</th><th class="num">Replied early</th><th class="num">Agent audio</th></tr>{turn_rows}</table></div>'
        f'{_chat(r)}</div></details>')


def _baseline_section(meta: dict) -> str:
    cmp = meta.get("baseline_cmp")
    if not cmp:
        return ""
    c = cmp["counts"]

    def cell(d, k, fmt):
        return "–" if d.get(k) is None else fmt(d[k])

    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    ms = lambda v: f"{v:.0f}"  # noqa: E731
    rows = []
    for r in cmp["rows"]:
        if r["status"] == "new":
            rows.append(f'<tr><td>{esc(r["scenario"])}</td><td>{esc(r["voice"])}</td><td colspan="5"><i>not in baseline</i></td></tr>')
            continue
        cls = {"regression": "fail", "improved": "pass"}.get(r["status"], "part")
        notes = "; ".join(esc(n["text"]) for n in r["notes"]) or "no significant change"
        vs = "" if r["vs"] == cmp["augment"] else f' <small>(vs {esc(r["vs"])})</small>'
        b, cu = r["base"], r["cur"]
        player = "".join(
            f'<div><small>{lab}</small><audio controls preload="none" src="{esc(a)}" style="width:100%;height:30px"></audio></div>'
            for lab, a in [(f"caller clip {i + 1}", c) for i, c in enumerate(r.get("clips", []))]
            + [("baseline call", r.get("base_audio")), ("this run", r.get("cur_audio"))] if a)
        rows.append(
            f'<tr><td>{esc(r["scenario"])}</td><td>{esc(r["voice"])}{vs}</td>'
            f'<td><span class="pill {cls}">{r["status"]}</span></td><td>{notes}</td>'
            f'<td class="num">{cell(b, "wer", pct)} → {cell(cu, "wer", pct)}</td>'
            f'<td class="num">{cell(b, "ttfa_ms", ms)} → {cell(cu, "ttfa_ms", ms)} ms</td>'
            f'<td style="min-width:260px">{player}</td></tr>')
    return (f'<h2>Versus baseline “{esc(cmp["name"])}”</h2><p class="hint">Saved {esc(cmp["created"][:10])}. '
            f'{c["regression"]} regression(s), {c["improved"]} improved, {c["same"]} unchanged, {c["new"]} not in baseline. '
            f'Rows compare the same scenario and voice (falling back to the clean baseline for noisy runs).</p>'
            f'<div class="wrap"><table><tr><th>Scenario</th><th>Voice</th><th>Status</th><th>What changed</th>'
            f'<th class="num">WER</th><th class="num">TTFA</th><th>Recordings</th></tr>{"".join(rows)}</table></div>')


def build_html(runs: list[dict], meta: dict) -> str:
    total = {"name": "all", **metrics.aggregate([{**r, "k": "all"} for r in runs], "k")[0]} if runs else None
    by_voice = metrics.aggregate(runs, "voice")
    by_scenario = metrics.aggregate(runs, "scenario")
    scripts = _scripts()
    failed = [r for r in runs if not r["passed"]]
    cards = ""
    if total:
        cards = "".join(f'<div class="card"><b>{v}</b><span>{k}</span></div>' for k, v in [
            ("calls", total["runs"]), ("pass rate", f'{100 * total["pass_rate"]:.0f}%'),
            ("tool score", f'{total["tool_score"]:.2f}'),
            ("agent spoke before caller finished", f'{total["early_turns"]} of {total["turns"]} turns'), ("word error rate", _pct(total["wer"])),
            ("LLM judge (avg)", _judge(total.get("judge"))),
            ("time to first audio p50", _ms(total["ttfa_p50"])), ("time to first audio p95", _ms(total["ttfa_p95"])),
            ("tool call latency", _ms(total["tool_ms"]))])
    cfg = [("Run", meta.get("created", "")), ("Agent", meta.get("url", "")), ("Voices", ", ".join(meta.get("voices", []))),
           ("Repeats", meta.get("repeat", "")), ("Caller audio", meta.get("augment", "clean")), ("Extra noise SNR", f'{meta["noise_snr_db"]} dB' if meta.get("noise_snr_db") is not None else "none"),
           ("Judge", meta.get("judge_model") or "off"),
           ("Characters synthesized", ", ".join(f"{k}: {v}" for k, v in (meta.get("chars_rendered") or {}).items() if v) or "all cached")]
    fails = "".join(f'<li><a href="#run-{esc(r["run_id"])}">{esc(r["scenario"])} · {esc(r["voice"])}</a> — {esc("; ".join(verdict_reasons(r)) or "failed")}</li>'
                    for r in failed)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Agent Eval Report</title><style>{CSS}</style></head><body><main>
<h1>Voice agent evaluation</h1>
<p class="sub">{" · ".join(f"{esc(k)}: {esc(str(v))}" for k, v in cfg if v != "")}</p>
<div class="cards">{cards}</div>
{_baseline_section(meta)}
<h2>By voice</h2>{_group_table(by_voice, "Voice")}
<p class="hint">TTFA = time from the caller's last audio frame to the agent's first audio byte; it includes the agent's end-of-speech wait. WER = caller script vs the agent's own transcript of it. "Replied early" = the agent began answering (audio, text or a tool call) before the caller's last audio frame, a sign it cut the caller off at a pause; it is reported, not counted as a failure.</p>
<h2>By scenario</h2>{_group_table(by_scenario, "Scenario")}
<h2>Voice × scenario</h2>{_matrix(runs)}
<h2>Failures ({len(failed)})</h2>{f'<ul class="why" style="color:inherit">{fails}</ul>' if failed else "<p>None 🎉</p>"}
<h2>All calls ({len(runs)})</h2>{"".join(_run_block(r, scripts) for r in runs)}
</main></body></html>"""


# --------------------------------------------------------------------- main ---
def write_all(runs: list[dict], out: Path, meta: dict) -> Path:
    """Writes transcripts/, report.html and summary.json into `out` and returns the report path."""
    out.mkdir(parents=True, exist_ok=True)
    runs.sort(key=lambda r: (r["voice"], r["scenario"], r.get("run_id", "")))
    for i, r in enumerate(runs, 1):
        r.setdefault("run_id", f"{i:02d}")
    write_transcripts(runs, out)
    (out / "report.html").write_text(build_html(runs, meta), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "meta": meta, "by_voice": metrics.aggregate(runs, "voice"), "by_scenario": metrics.aggregate(runs, "scenario"),
    }, indent=2), encoding="utf-8")
    return out / "report.html"


def load_dir(folder: Path) -> tuple[list[dict], dict]:
    runs = [json.loads(line) for line in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {"created": folder.name}
    return runs, meta


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else max((HERE / "results").glob("*/runs.jsonl")).parent
    runs, meta = load_dir(target)
    for i, r in enumerate(runs, 1):
        r.setdefault("run_id", f"{i:02d}")
    path = write_all(runs, target, meta)
    # keep runs.jsonl in sync with the run ids / transcript links
    (target / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in runs), encoding="utf-8")
    print(f"report: {path}")
