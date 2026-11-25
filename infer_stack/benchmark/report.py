"""Readable, dependency-free reports from local or collected JSONL ledgers."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import statistics
from pathlib import Path


def _stable(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _identity(row):
    # Never aggregate across physical devices, configured runtimes, actual
    # image IDs, different driver/BIOS revisions or workload options.
    gpus = [{k: g.get(k) for k in ('uuid', 'name', 'pci.bus_id', 'driver_version', 'vbios_version',
                                     'pcie.link.gen.max', 'pcie.link.width.max', 'power.limit')}
            | {'index_if_no_uuid': g.get('index') if not g.get('uuid') else None,
               'pci_path': [{k: p.get(k) for k in ('bdf', 'max_link_speed', 'max_link_width')}
                            for p in g.get('pci_path', [])]} for g in row.get('gpus', [])]
    host = row.get('host', {})
    return {'host': host, 'gpus': gpus, 'provenance': row.get('provenance', {}),
            'image': row.get('runtime_image', {}).get('image_id'),
            'settings': row.get('settings', {}), 'endpoint': row['endpoint'],
            'infer_stack_version': row.get('infer_stack_version'),
            'unidentified_run': row.get('run_id') if not gpus or any(not g.get('uuid') for g in gpus) else None}


def _median(rows, key):
    values = [r.get('metrics', {}).get(key) for r in rows]
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def summarize(records):
    groups = {}
    identities = {}
    for row in records:
        if row.get('kind') != 'trial':
            continue
        ident = _identity(row)
        digest = hashlib.sha256(_stable(ident).encode()).hexdigest()[:16]
        identities[digest] = ident
        groups.setdefault((digest, row.get('target_context'), row.get('phase')), []).append(row)
    result = []
    for (digest, target, phase), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2] or '')):
        ident = identities[digest]
        ok = [r for r in rows if r['status'] == 'ok']
        gpu = '; '.join(f"{g.get('name')} {g.get('pci.bus_id') or '?'} {g.get('uuid') or '?'}" for g in ident['gpus']) or 'unobserved'
        links = set()
        peak = None
        for r in ok:
            for s in r.get('telemetry', []):
                if s.get('pcie.link.gen.current') and s.get('pcie.link.width.current'):
                    links.add(f"{s['uuid']}: Gen{s['pcie.link.gen.current']} x{s['pcie.link.width.current']}")
                try:
                    memory = float(s['memory.used'])
                    peak = memory if peak is None else max(peak, memory)
                except (KeyError, ValueError, TypeError):
                    pass
        result.append({'configuration': digest, 'endpoint': ident['endpoint'], 'host': ident['host'].get('hostname'),
                       'gpu': gpu, 'target': target, 'phase': phase, 'ok': len(ok), 'total': len(rows),
                       'prompt_tokens': _median(ok, 'prompt_tokens'), 'completion_tokens': _median(ok, 'completion_tokens'),
                       'ttft_s': _median(ok, 'ttft_seconds'), 'stream_tps': _median(ok, 'stream_tps_estimate'),
                       'e2e_tps': _median(ok, 'e2e_completion_tps'), 'prompt_per_ttft': _median(ok, 'prompt_tokens_per_ttft'),
                       'cached_tokens': _median(ok, 'cached_prompt_tokens'), 'peak_gpu_mib': peak,
                       'observed_links': '; '.join(sorted(links)),
                       'reason': '; '.join(sorted({str(r['reason']) for r in rows if r.get('reason')}))})
    return result, identities


def _fmt(value):
    if value is None:
        return '-'
    return f'{value:.3g}' if isinstance(value, float) else str(value)


COLUMNS = ('endpoint', 'host', 'configuration', 'target', 'phase', 'ok', 'total', 'prompt_tokens',
           'completion_tokens', 'ttft_s', 'stream_tps', 'e2e_tps', 'prompt_per_ttft', 'cached_tokens',
           'peak_gpu_mib', 'observed_links', 'reason')
NOTES = ('Medians across matching observations. Target context is approximate; actual tokens are API-reported. '
         'Stream t/s estimates generation from first to last observed output and can include reasoning tokens '
         'or transport batching; it is not an engine-internal decode counter. Prompt tokens/TTFT includes '
         'network, queuing and processing, not pure prefill. Prefix reuse is attempted, not guaranteed. '
         'GPU peak is sampled, per-device, not an allocator high-water mark. Missing measurements are shown as -. '
         'Link observations can include idle intervals; see raw utilization and bridge samples. '
         'Serving provenance is configured identity, not weight attestation.')


def write_report(records, directory):
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    rows, identities = summarize(records)
    issues = [r for r in records if r.get('kind') != 'trial' and r.get('status') not in ('ok', 'ready')]
    ready = [r for r in records if r.get('kind') == 'endpoint' and r.get('status') == 'ready']
    columns = (*COLUMNS, 'gpu')
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (directory / 'configurations.json').write_text(json.dumps(identities, indent=2, sort_keys=True) + '\n')
    md = ['# Infer-stack benchmark report', '', NOTES, '',
          f'{len(records)} records; {len(ready)} endpoint acquisitions ready; {len(issues)} endpoint/release issues.', '']
    compact = ('endpoint', 'host', 'configuration', 'target', 'phase', 'ok', 'total', 'prompt_tokens', 'completion_tokens', 'ttft_s', 'stream_tps', 'e2e_tps')
    def md_cell(x):
        return _fmt(x).replace('|', '\\|').replace('\n', ' ')
    md.extend(['| ' + ' | '.join(compact) + ' |', '| ' + ' | '.join(['---'] * len(compact)) + ' |'])
    md.extend('| ' + ' | '.join(md_cell(r[k]) for k in compact) + ' |' for r in rows)
    md.extend(['', '## Acquisitions and failures', ''])
    md.extend(f"- {r['endpoint']} on {r.get('host', {}).get('hostname')}: ready in {_fmt(r.get('acquire_seconds'))} s" for r in ready)
    md.extend(f"- {r['endpoint']}: {r['status']} ({r.get('reason', r.get('kind'))})" for r in issues)
    md.extend(['', '## Configuration identities', ''])
    for key, val in identities.items():
        md.extend([f'### {key} — {val["endpoint"]}', '', '```json', json.dumps(val, indent=2, sort_keys=True), '```', ''])
    (directory / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    def esc(value):
        return html.escape(_fmt(value))
    table = '<table><thead><tr>' + ''.join(f'<th>{esc(k)}</th>' for k in columns) + '</tr></thead><tbody>'
    for r in rows:
        table += '<tr>' + ''.join(f'<td>{esc(r[k])}</td>' for k in columns) + '</tr>'
    table += '</tbody></table>'
    details = ''.join(f'<details><summary>{esc(k)} — {esc(v["endpoint"])}</summary><pre>{esc(json.dumps(v, indent=2, sort_keys=True))}</pre></details>' for k, v in identities.items())
    failures = ''.join(f'<li>{esc(r["endpoint"])}: {esc(r["status"])} — {esc(r.get("reason", r.get("kind")))}</li>' for r in issues)
    startups = ''.join(f'<li>{esc(r["endpoint"])} on {esc(r.get("host", {}).get("hostname"))}: {esc(r.get("acquire_seconds"))} s</li>' for r in ready)
    (directory / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Infer-stack benchmark</title>'
        '<style>body{font:15px system-ui;margin:2rem;color:#17212b}table{border-collapse:collapse;font-size:13px}th,td{padding:.5em;border:1px solid #d8dce0;text-align:left}th{background:#eef2f5}tr:nth-child(even){background:#f6f8fa}main{overflow-x:auto}pre{white-space:pre-wrap}details{margin:1em 0}</style>'
        f'<h1>Infer-stack benchmark report</h1><p>{esc(NOTES)}</p><p>{len(records)} records; {len(ready)} ready acquisitions; {len(issues)} issues.</p><main>{table}</main>'
        f'<h2>Acquisitions</h2><ul>{startups}</ul><h2>Endpoint and release issues</h2><ul>{failures}</ul><h2>Configuration identities</h2>{details}</html>', encoding='utf-8')
    return directory / 'report.html'
