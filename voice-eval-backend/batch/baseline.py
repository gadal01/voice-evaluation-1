"""Baselines: keep a known-good run (metrics + call recordings) and compare every later run against it.

    python batch/run_evals.py --save-baseline good           # run, then keep this run as baselines/good/
    python batch/run_evals.py                                # compares against baselines/default/ if present
    python batch/run_evals.py --baseline good --fail-on-regression
    python batch/baseline.py save good batch/results/<run> [<run> ...]   # build one from finished runs (merged)
    python batch/baseline.py compare good batch/results/<run>             # re-check a finished run

By default a baseline keeps NO call audio: the caller's side is the test input and lives in the clip library
(clips.py), which the baseline references by id + checksum; the agent's side is non-deterministic. Pass --with-audio
(or run_evals --baseline-audio) to also keep one stereo recording per key for listening.

A baseline is keyed by (scenario, voice, augment profile). A run is compared with the exact same key; if the baseline
has no recording for that profile, it falls back to the same scenario+voice on "clean", which shows what the noise
costs. Per key we keep: pass rate, tool score, WER, judge score, median TTFA, the tools called, the agent's reply,
and for clip voices the caller clips used. The report plays the caller clip next to the baseline/current recordings.

A key is a REGRESSION when pass rate drops, the judge score drops >= 0.5, WER rises >= 5 points, the median TTFA
rises by >= 30% and >= 500 ms, or a tool the baseline called is no longer called. Mirror changes are IMPROVED.
"""
import argparse
import difflib
import json
import os
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clips  # noqa: E402
DIR = HERE / "baselines"
JUDGE_DROP, WER_RISE, TTFA_PCT, TTFA_MS = 0.5, 0.05, 0.30, 500


def run_aug(meta: dict) -> str:
    if meta.get("augment"):
        return meta["augment"] + (f"+snr{meta['noise_snr_db']}" if meta.get("noise_snr_db") is not None else "")
    return "clean" if meta.get("noise_snr_db") is None else f"snr{meta['noise_snr_db']}"


def key_of(scenario: str, voice: str, aug: str) -> str:
    return f"{scenario}|{voice}|{aug}"


def summarize(runs: list[dict], aug: str, run_dir: Path | None = None) -> dict[str, dict]:
    """One entry per (scenario, voice, augment): the numbers used for comparison."""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(key_of(r["scenario"], r["voice"], aug), []).append(r)
    out = {}
    for k, rs in groups.items():
        words = sum(r.get("wer_words", 0) for r in rs)
        ttfa = [t["ttfa_ms"] for r in rs for t in r.get("turns", []) if t.get("ttfa_ms") is not None]
        judged = [r["judge"]["overall"] for r in rs if r.get("judge", {}).get("scores")]
        first = next((r for r in rs if r.get("audio_file")), rs[0])
        used = []
        if rs[0]["voice"].startswith("clip:"):  # which stored caller recordings this key replayed
            for c in rs[0].get("conversation", []):
                e = c["role"] == "caller" and clips.find(rs[0]["voice"][5:], c["text"])
                if e:
                    used.append({"id": e["id"], "sha256": e["sha256"], "file": e["file"]})
        out[k] = {
            "runs": len(rs), "pass_rate": sum(bool(r["passed"]) for r in rs) / len(rs),
            "tool_score": statistics.fmean(r.get("tool_score", 0.0) for r in rs),
            "wer": sum(r.get("wer_errors", 0) for r in rs) / words if words else None,
            "judge": statistics.fmean(judged) if judged else None,
            "ttfa_ms": statistics.median(ttfa) if ttfa else None,
            "clips": used, "tools_called": rs[0].get("tools_called", []), "reply": (rs[0].get("reply") or "")[:1500],
            "audio_src": str(run_dir / first["audio_file"]) if run_dir and first.get("audio_file") else None,
        }
    return out


def load_run(folder: Path) -> tuple[list[dict], dict]:
    runs = [json.loads(line) for line in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    meta_file = folder / "meta.json"
    return runs, (json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {})


# ---------------------------------------------------------------------- save ---
def save(name: str, run_dirs: list[Path], force: bool = False, with_audio: bool = False) -> Path:
    dest = DIR / name
    if dest.exists() and not force:
        raise FileExistsError(f"baseline {name!r} already exists ({dest}); pass --force to replace it")
    groups: dict[str, dict] = {}
    for folder in run_dirs:
        runs, meta = load_run(folder)
        groups.update(summarize(runs, run_aug(meta), folder))
    if dest.exists():
        shutil.rmtree(dest)  # our own baseline folder, replaced on request
    (dest / "audio").mkdir(parents=True)
    for k, g in groups.items():
        src = g.pop("audio_src")
        if with_audio and src and Path(src).exists():
            fname = "".join(c if c.isalnum() or c in "-_" else "_" for c in k) + ".wav"
            shutil.copyfile(src, dest / "audio" / fname)
            g["audio"] = f"audio/{fname}"
    (dest / "baseline.json").write_text(json.dumps({
        "name": name, "created": datetime.now().isoformat(timespec="seconds"),
        "sources": [str(p) for p in run_dirs], "groups": groups}, indent=2), encoding="utf-8")
    return dest


# ------------------------------------------------------------------- compare ---
def load(name: str) -> dict | None:
    f = DIR / name / "baseline.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _judge(cur: dict, base: dict) -> list[tuple[str, str]]:
    """(severity, note) pairs for one key; severity is 'bad', 'good' or 'info'."""
    notes = []
    if cur["pass_rate"] < base["pass_rate"]:
        notes.append(("bad", f"pass rate {100 * base['pass_rate']:.0f}% -> {100 * cur['pass_rate']:.0f}%"))
    elif cur["pass_rate"] > base["pass_rate"]:
        notes.append(("good", f"pass rate {100 * base['pass_rate']:.0f}% -> {100 * cur['pass_rate']:.0f}%"))
    if cur["judge"] is not None and base["judge"] is not None:
        d = cur["judge"] - base["judge"]
        if d <= -JUDGE_DROP:
            notes.append(("bad", f"judge {base['judge']:.1f} -> {cur['judge']:.1f}"))
        elif d >= JUDGE_DROP:
            notes.append(("good", f"judge {base['judge']:.1f} -> {cur['judge']:.1f}"))
    if cur["wer"] is not None and base["wer"] is not None:
        d = cur["wer"] - base["wer"]
        if d >= WER_RISE:
            notes.append(("bad", f"WER {100 * base['wer']:.1f}% -> {100 * cur['wer']:.1f}%"))
        elif d <= -WER_RISE:
            notes.append(("good", f"WER {100 * base['wer']:.1f}% -> {100 * cur['wer']:.1f}%"))
    if cur["ttfa_ms"] is not None and base["ttfa_ms"]:
        d = cur["ttfa_ms"] - base["ttfa_ms"]
        if d >= TTFA_MS and d / base["ttfa_ms"] >= TTFA_PCT:
            notes.append(("bad", f"TTFA {base['ttfa_ms']:.0f} -> {cur['ttfa_ms']:.0f} ms"))
        elif -d >= TTFA_MS and -d / base["ttfa_ms"] >= TTFA_PCT:
            notes.append(("good", f"TTFA {base['ttfa_ms']:.0f} -> {cur['ttfa_ms']:.0f} ms"))
    lost = [t for t in dict.fromkeys(base["tools_called"]) if t not in cur["tools_called"]]
    if lost:
        notes.append(("bad", "no longer calls " + ", ".join(lost)))
    return notes


def _clip_notes(cur: dict, base: dict) -> list[tuple[str, str]]:
    b = {c["id"]: c["sha256"] for c in base.get("clips", [])}
    changed = [c["id"] for c in cur.get("clips", []) if c["id"] in b and b[c["id"]] != c["sha256"]]
    return [("info", "caller audio differs from the baseline's (" + ", ".join(changed) + ")")] if changed else []


def compare(runs: list[dict], meta: dict, name: str, out_dir: Path, run_dir: Path | None = None) -> dict | None:
    """Compares a run with baseline `name`; returns the structure the report and console print, or None if no baseline."""
    base = load(name)
    if not base:
        return None
    aug = run_aug(meta)
    cur_groups = summarize(runs, aug, run_dir or out_dir)
    rows = []
    for k, cur in sorted(cur_groups.items()):
        scenario, voice, _ = k.split("|")
        ref_key, ref = k, base["groups"].get(k)
        if ref is None:  # no recording for this profile: fall back to the clean one
            ref_key = key_of(scenario, voice, "clean")
            ref = base["groups"].get(ref_key)
        if ref is None:
            rows.append({"key": k, "scenario": scenario, "voice": voice, "status": "new", "notes": [], "vs": None})
            continue
        notes = _judge(cur, ref) + _clip_notes(cur, ref)
        status = "regression" if any(s == "bad" for s, _ in notes) else ("improved" if any(s == "good" for s, _ in notes) else "same")
        sim = difflib.SequenceMatcher(a=ref["reply"].lower(), b=cur["reply"].lower()).ratio() if ref["reply"] and cur["reply"] else None
        base_audio = DIR / name / ref["audio"] if ref.get("audio") else None
        rows.append({
            "key": k, "scenario": scenario, "voice": voice, "status": status, "vs": ref_key.split("|")[2],
            "notes": [{"severity": s, "text": t} for s, t in notes], "reply_similarity": sim,
            "base": {x: ref[x] for x in ("pass_rate", "wer", "judge", "ttfa_ms", "tool_score")},
            "cur": {x: cur[x] for x in ("pass_rate", "wer", "judge", "ttfa_ms", "tool_score")},
            "base_audio": os.path.relpath(base_audio, out_dir).replace("\\", "/") if base_audio and base_audio.exists() else None,
            "clips": [os.path.relpath(clips.DIR / c["file"], out_dir).replace("\\", "/") for c in cur.get("clips", [])],
            "cur_audio": (cur["audio_src"] and os.path.relpath(cur["audio_src"], out_dir).replace("\\", "/")),
        })
    counts = {s: sum(r["status"] == s for r in rows) for s in ("regression", "improved", "same", "new")}
    return {"name": name, "created": base["created"], "augment": aug, "counts": counts, "rows": rows}


def print_comparison(cmp: dict) -> None:
    c = cmp["counts"]
    print(f"\nBaseline '{cmp['name']}' ({cmp['created'][:10]}): {c['regression']} regression(s), {c['improved']} improved, "
          f"{c['same']} unchanged, {c['new']} not in baseline")
    for r in cmp["rows"]:
        if r["status"] in ("regression", "improved"):
            tag = "REGRESSION" if r["status"] == "regression" else "improved  "
            vs = "" if r["vs"] == cmp["augment"] else f" (vs {r['vs']})"
            print(f"  {tag} {r['voice']:<26} {r['scenario']:<20}{vs} " + "; ".join(n["text"] for n in r["notes"]))


# ----------------------------------------------------------------------- cli ---
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("name")
    s.add_argument("runs", nargs="+", type=Path)
    s.add_argument("--force", action="store_true")
    s.add_argument("--with-audio", action="store_true", help="also keep one call recording per key")
    c = sub.add_parser("compare")
    c.add_argument("name")
    c.add_argument("run", type=Path)
    args = ap.parse_args()
    if args.cmd == "save":
        dest = save(args.name, args.runs, args.force, args.with_audio)
        print(f"saved baseline {args.name!r} -> {dest} ({len(load(args.name)['groups'])} keys)")
        return 0
    runs, meta = load_run(args.run)
    cmp = compare(runs, meta, args.name, args.run, args.run)
    if not cmp:
        print(f"no baseline named {args.name!r} in {DIR}")
        return 2
    print_comparison(cmp)
    return 1 if cmp["counts"]["regression"] else 0


if __name__ == "__main__":
    sys.exit(main())
