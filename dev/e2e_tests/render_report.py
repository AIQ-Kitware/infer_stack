#!/usr/bin/env python3
"""Emit/assemble the infer-stack leasing e2e report.

Three modes (lib.sh uses the first two; run.sh the third):

  --emit-step    read an assert-file + note-file and print ONE results.jsonl line
  --emit-skip    print one skipped results.jsonl line
  --assemble     read results.jsonl (+ run_meta.json, environment.txt) and write
                 report.md

Kept dependency-free (stdlib only) so it runs under whatever python the host
has. This is a developer harness, not shipped code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _read_asserts(path: str) -> list[dict]:
    out: list[dict] = []
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        status, _, desc = line.partition('|')
        out.append({'ok': status == 'PASS', 'desc': desc})
    return out


def _read_notes(path: str) -> list[str]:
    if not path or not Path(path).exists():
        return []
    return [ln for ln in Path(path).read_text().splitlines() if ln.strip()]


def emit_step(args) -> None:
    rec = {
        'kind': 'step',
        'section': args.section,
        'id': args.id,
        'title': args.title,
        'verdict': args.verdict,
        'duration': float(os.environ.get('E2E_DUR', '0') or 0),
        'log': os.environ.get('E2E_LOGREL', ''),
        'asserts': _read_asserts(args.asserts),
        'notes': _read_notes(args.notes),
    }
    print(json.dumps(rec))


def emit_skip(args) -> None:
    rec = {
        'kind': 'step',
        'section': args.section,
        'id': args.id,
        'title': os.environ.get('E2E_SKIP_REASON', 'skipped'),
        'verdict': 'skip',
        'duration': 0.0,
        'log': '',
        'asserts': [],
        'notes': [],
    }
    print(json.dumps(rec))


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

# Which step ids feed each "wiring" axis the user cares about. A step counts
# toward an axis if its id contains any of these substrings.
AXES = {
    'Correctness': ['acquire', 'serve', 'chat', 'models', 'coalesce',
                    'release', 'leases', 'bundle', 'ttl', 'run-', 'ollama',
                    'unknown', 'kubeai', 'missing', 'no-names', 'version'],
    'Efficiency': ['coalesce', 'reclaim', 'concurrency', 'placement', 'ttl'],
    'Ergonomics': ['paths', 'secrets', 'status', 'envfile', 'env', 'day2',
                   'docker'],
}


def _load(results: Path) -> list[dict]:
    recs: list[dict] = []
    if not results.exists():
        return recs
    for line in results.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def _badge(verdict: str) -> str:
    return {'pass': '✅ pass', 'fail': '❌ FAIL',
            'skip': '⏭️ skip'}.get(verdict, verdict)


def assemble(args) -> None:
    rdir = Path(args.results)
    recs = [r for r in _load(rdir / 'results.jsonl') if r.get('kind') == 'step']
    meta = {}
    if (rdir / 'run_meta.json').exists():
        meta = json.loads((rdir / 'run_meta.json').read_text())

    n_pass = sum(r['verdict'] == 'pass' for r in recs)
    n_fail = sum(r['verdict'] == 'fail' for r in recs)
    n_skip = sum(r['verdict'] == 'skip' for r in recs)
    total_dur = sum(r.get('duration', 0) for r in recs)
    wall = None
    wf = rdir / 'wall_seconds.txt'
    if wf.exists():
        try:
            secs = int(wf.read_text().strip())
            wall = f'{secs // 60}m {secs % 60:02d}s'
        except ValueError:
            wall = None

    L: list[str] = []
    L.append('# infer-stack leasing — e2e report')
    L.append('')
    L.append(f"- host: `{meta.get('host', '?')}`")
    L.append(f"- when: {meta.get('when', '?')}")
    L.append(f"- infer-stack: `{meta.get('version', '?')}`  "
             f"git: `{meta.get('git', '?')}`")
    L.append(f"- tiers: {meta.get('tiers', '?')}  "
             f"GPU serving: **{'on' if meta.get('gpu') else 'off'}**")
    L.append(f"- data dir: `{meta.get('data_dir', '?')}`")
    L.append('')
    verdict = '❌ FAILURES' if n_fail else '✅ all green'
    L.append(f"## Result: {verdict} — "
             f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    L.append('')
    if wall:
        L.append(f"- total wall time: **{wall}**  "
                 f"({total_dur:.0f}s in step bodies)")
    else:
        L.append(f"- time in step bodies: {total_dur:.0f}s")
    L.append('')

    # Wiring axes rollup
    L.append('### Wiring axes')
    L.append('')
    L.append('| axis | pass | fail | skip |')
    L.append('| --- | --- | --- | --- |')
    for axis, keys in AXES.items():
        sel = [r for r in recs if any(k in r['id'] for k in keys)]
        p = sum(r['verdict'] == 'pass' for r in sel)
        f = sum(r['verdict'] == 'fail' for r in sel)
        s = sum(r['verdict'] == 'skip' for r in sel)
        L.append(f'| {axis} | {p} | {f} | {s} |')
    L.append('')

    # Failures first — the thing to read
    fails = [r for r in recs if r['verdict'] == 'fail']
    if fails:
        L.append('## ❌ Failures (read these)')
        L.append('')
        for r in fails:
            L.append(f"### {r['section']} / {r['id']} — {r['title']}")
            for a in r['asserts']:
                if not a['ok']:
                    L.append(f"- **failed:** {a['desc']}")
            for nt in r.get('notes', []):
                L.append(f"- note: {nt}")
            log = rdir / r['log'] if r.get('log') else None
            if log and log.exists():
                tail = log.read_text().splitlines()[-40:]
                L.append('')
                L.append('<details><summary>log tail</summary>')
                L.append('')
                L.append('```')
                L.extend(tail)
                L.append('```')
                L.append('</details>')
            L.append('')

    # Full table grouped by section
    L.append('## All steps')
    L.append('')
    sections: dict[str, list[dict]] = {}
    for r in recs:
        sections.setdefault(r['section'], []).append(r)
    for section, items in sections.items():
        L.append(f'### {section}')
        L.append('')
        L.append('| step | verdict | dur (s) | checks | title |')
        L.append('| --- | --- | --- | --- | --- |')
        for r in items:
            n_ok = sum(a['ok'] for a in r['asserts'])
            n_all = len(r['asserts'])
            checks = f'{n_ok}/{n_all}' if n_all else '-'
            L.append(f"| `{r['id']}` | {_badge(r['verdict'])} | "
                     f"{r.get('duration', 0):.2f} | {checks} | "
                     f"{r['title']} |")
        L.append('')

    # Environment dump
    env_txt = rdir / 'environment.txt'
    if env_txt.exists():
        L.append('## Environment')
        L.append('')
        L.append('```')
        L.append(env_txt.read_text().rstrip())
        L.append('```')
        L.append('')

    L.append('## Artifacts in this results dir')
    L.append('')
    L.append('- `report.md` — this file')
    L.append('- `results.jsonl` — machine-readable per-step records')
    L.append('- `environment.txt` — host/docker/gpu capture')
    L.append('- `logs/` — full combined output of every step')
    L.append('- `infer-stack-data/leasing/compose/` — rendered compose, '
             'litellm config, secrets `.env`, sidecar state (if GPU ran)')
    L.append('')

    (rdir / 'report.md').write_text('\n'.join(L) + '\n')
    sys.stderr.write(f'wrote {rdir / "report.md"}\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--emit-step', action='store_true')
    ap.add_argument('--emit-skip', action='store_true')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--section', default='')
    ap.add_argument('--id', default='')
    ap.add_argument('--title', default='')
    ap.add_argument('--verdict', default='pass')
    ap.add_argument('--asserts', default='')
    ap.add_argument('--notes', default='')
    ap.add_argument('--results', default=os.environ.get('E2E_RESULTS', '.'))
    args = ap.parse_args()
    if args.emit_step:
        emit_step(args)
    elif args.emit_skip:
        emit_skip(args)
    elif args.assemble:
        assemble(args)
    else:
        ap.error('one of --emit-step/--emit-skip/--assemble required')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
