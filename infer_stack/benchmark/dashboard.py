"""Dependency-free, self-contained comparison views for benchmark reports.

Consumes the existing summary rows and configuration identities, so old ledgers
can be re-rendered without measuring again. Configuration grouping remains the
report's responsibility; comparisons never combine independent identities.
"""
from __future__ import annotations

import html
import json
import math
import statistics
from collections import defaultdict


COLORS = ('#1769aa', '#b34d19', '#147d64', '#7954ad', '#b82e5b', '#82600f',
          '#357887', '#655b51', '#5269cd', '#ad641c')
DASHES = ('', '7 4', '2 3', '10 3 2 3')
PHASES = ('fresh', 'exact-prefix', 'extended-prefix')


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=1):
    if not number(value):
        return '—'
    return f'{value:,.{digits}f}'


def seconds(value):
    return fmt(value, 2 if number(value) and value < 10 else 1)


def context(value):
    if not number(value):
        return '—'
    if value >= 1024 and value % 1024 == 0:
        return f'{value // 1024:,} Ki'
    return f'{value:,}'


def cache_percent(row):
    """Ratio of reported median counts; unavailable is distinct from zero."""
    p, c = row.get('prompt_tokens'), row.get('cached_tokens')
    if number(p) and p > 0 and number(c) and 0 <= c <= p:
        return 100 * c / p
    return None


def successful(row):
    return bool(row and row.get('ok', 0) > 0)


def metric(row, key):
    if not successful(row):
        return None
    return cache_percent(row) if key == 'cache_percent' else row.get(key)


def cohort_key(identity):
    # Compare endpoints only within the same observed host/GPU topology and
    # workload settings. Keep differing driver, firmware, power limits, unknown
    # device identities and infer-stack versions separate, as summarize does.
    keys = ('host', 'gpus', 'settings', 'infer_stack_version', 'unidentified_run')
    return json.dumps({k: identity.get(k) for k in keys}, sort_keys=True, separators=(',', ':'))


def comparison_groups(rows, identities):
    cohorts = defaultdict(list)
    for key, ident in identities.items():
        cohorts[cohort_key(ident)].append(key)
    rows_by_id = defaultdict(list)
    for row in rows:
        rows_by_id[row['configuration']].append(row)
    result = []
    for keys in sorted(cohorts.values(), key=lambda ks: (identities[ks[0]].get('host', {}).get('hostname') or '', min(ks))):
        keys.sort(key=lambda k: (identities[k]['endpoint'], k))
        records = [r for k in keys for r in rows_by_id[k]]
        targets = sorted({r['target'] for r in records if number(r.get('target')) and r['target'] > 0})
        result.append((keys, records, targets))
    return result


def lookup(rows):
    return {(r['configuration'], r['target'], r['phase']): r for r in rows}


def style(index):
    return COLORS[index % len(COLORS)], DASHES[(index // len(COLORS) + index % 3) % len(DASHES)]


def label(index):
    return str(index + 1)


def endpoint_label(key, keys, identities):
    endpoint = identities[key]['endpoint']
    if sum(identities[k]['endpoint'] == endpoint for k in keys) > 1:
        return f'{endpoint} [{key[:8]}]'
    return endpoint


def chart(rows, keys, targets, identities, *, title, subtitle, phase, field, unit,
          log=False, percent=False):
    """SVG line chart. Missing/failed points break lines rather than becoming zero."""
    index = lookup(rows)
    def row_at(k, t):
        return index.get((k, t, phase)) or {}
    def valid(v):
        return number(v) and (v > 0 if log else v >= 0)
    values = [metric(row_at(k, t), field) for k in keys for t in targets]
    values = [v for v in values if valid(v)]
    head = f'<figure class="chart"><h3>{esc(title)}</h3><p>{esc(subtitle)}</p>'
    if not values or not targets:
        return head + '<p class="empty">No successful measurements available.</p></figure>'
    # Equal spacing makes all requested context buckets readable; the axis is
    # explicitly categorical rather than implying a linear token-count scale.
    w, h, left, right, top, bottom = 650, 290, 70, 20, 16, 61
    pw, ph = w - left - right, h - top - bottom
    if log:
        lo = math.floor(math.log10(min(values)))
        hi = max(lo + 1, math.ceil(math.log10(max(values))))
        ticks = range(lo, hi + 1)
        if hi - lo > 7:
            ticks = range(lo, hi + 1, math.ceil((hi - lo) / 7))
        ticks = [(10.0 ** n, str(f'{10.0 ** n:g}')) for n in ticks]
        def y(v):
            return top + ph * (hi - math.log10(v)) / (hi - lo)
    else:
        maximum = max(values)
        if percent:
            upper = 100
        elif maximum <= 0:
            upper = 1
        else:
            magnitude = 10 ** math.floor(math.log10(maximum / 4))
            step = next(s * magnitude for s in (1, 2, 2.5, 5, 10) if s * magnitude >= maximum / 4)
            upper = step * 4
        ticks = [(upper * i / 4, f'{upper * i / 4:,.3g}') for i in range(5)]
        def y(v):
            return top + ph * (1 - v / upper)
    def x(j):
        return left + pw * (j / (len(targets) - 1) if len(targets) > 1 else .5)
    out = [f'<svg role="img" aria-label="{esc(title)}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for tick, tick_label in ticks:
        yy = y(tick)
        out += [f'<line x1="{left}" y1="{yy:.2f}" x2="{w-right}" y2="{yy:.2f}" class="gridline"/>',
                f'<text x="{left-9}" y="{yy+4:.2f}" text-anchor="end">{esc(tick_label)}</text>']
    for j, t in enumerate(targets):
        xx = x(j)
        if len(targets) <= 9 or j in (0, len(targets) - 1) or j % math.ceil(len(targets) / 8) == 0:
            out += [f'<text x="{xx:.2f}" y="{h-bottom+21}" text-anchor="middle">{esc(context(t))}</text>']
    out += [f'<text x="{(left+w-right)/2}" y="{h-10}" text-anchor="middle">Target context (categorical)</text>',
            f'<text transform="translate(15 {top+ph/2}) rotate(-90)" text-anchor="middle">{esc(unit)}{" · log scale" if log else ""}</text>']
    for i, k in enumerate(keys):
        color, dash = style(i)
        points, segments = [], []
        for j, t in enumerate(targets):
            row = row_at(k, t)
            val = metric(row, field)
            if not valid(val):
                if points:
                    segments.append(points)
                    points = []
                continue
            points.append((x(j), y(val), row, val))
        if points:
            segments.append(points)
        for segment in segments:
            if len(segment) > 1:
                coords = ' '.join(f'{xx:.2f},{yy:.2f}' for xx, yy, _, _ in segment)
                out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
            for xx, yy, row, val in segment:
                title_text = (f'{label(i)} · {endpoint_label(k, keys, identities)}\n'
                              f'{context(row.get("target"))} target; {fmt(row.get("prompt_tokens"), 0)} reported prompt tokens\n'
                              f'{fmt(val, 2)} {unit}; {row["ok"]}/{row["total"]} successful observations')
                out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1"><title>{esc(title_text)}</title></circle>')
    return head + ''.join(out) + '</svg></figure>'


def table(headers, body, *, klass=''):
    return (f'<div class="table-wrap"><table class="{klass}"><thead><tr>' +
            ''.join(f'<th scope="col">{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' +
            ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in body) +
            '</tbody></table></div>')


def marker(i, name, key):
    color, _ = style(i)
    return f'<span class="series-id" style="--series:{color}">{label(i)}</span> <span title="{esc(key)}">{esc(name)}</span>'


def hardware_description(keys, rows, identities):
    ident = identities[keys[0]]
    host = ident.get('host', {})
    parts = [esc(host.get('hostname') or 'Unknown host')]
    if host.get('cpu'):
        parts.append(esc(host['cpu']))
    if host.get('board_name'):
        parts.append(esc(host['board_name']))
    links = sorted({link.strip() for row in rows for link in (row.get('observed_links') or '').split(';') if link.strip()})
    gpu_items = []
    for gpu in ident.get('gpus', []):
        uuid = gpu.get('uuid')
        gpu_links = [s.split(': ', 1)[-1] for s in links if uuid and s.startswith(uuid + ': ')]
        path = []
        for bridge in gpu.get('pci_path', []):
            if bridge.get('max_link_width') or bridge.get('max_link_speed'):
                path.append(f"{bridge.get('bdf') or '?'}: {bridge.get('max_link_speed') or '?'} ×{bridge.get('max_link_width') or '?'}")
        info = [f"{gpu.get('name') or 'Unknown GPU'} · {gpu.get('pci.bus_id') or 'unknown bus'}",
                f"GPU ID: {uuid or 'unobserved'}",
                'Observed link: ' + (', '.join(gpu_links) or 'not sampled')]
        if path:
            info.append('PCI path maxima: ' + ' → '.join(path))
        gpu_items.append('<div class="gpu"><strong>' + esc(info[0]) + '</strong>' + ''.join(f'<div>{esc(s)}</div>' for s in info[1:]) + '</div>')
    settings = ident.get('settings', {})
    work = [f"Workload: {settings.get('workload') or 'unknown'}",
            f"Output budget: {settings.get('generation_tokens', 'unknown')} tokens",
            f"Effort: {settings.get('reasoning_effort') or 'endpoint default'}",
            f"Temperature: {settings.get('temperature', 'unknown')}",
            f"Repeats: {settings.get('repeats', 'unknown')}"]
    return ('<div class="hardware"><p><strong>' + ' · '.join(parts) + '</strong></p>' +
            (''.join(gpu_items) or '<p>GPU identity unavailable; this run is kept separate.</p>') +
            '<p class="muted">' + esc(' · '.join(work)) + '</p>' +
            '<p class="muted">Observed links may include idle intervals. Bridge maxima are capability limits, not measured transfer bandwidth.</p></div>')


def coverage(keys, rows, identities, acquisitions):
    body = []
    for i, key in enumerate(keys):
        subset = [r for r in rows if r['configuration'] == key]
        good = [r for r in subset if successful(r) and r['phase'] == 'fresh']
        measured = [r['target'] for r in good if number(r['target'])]
        peak = [r['peak_gpu_mib'] for r in subset if number(r.get('peak_gpu_mib'))]
        acquired = acquisitions.get(key, [])
        times = [r['acquire_seconds'] for r in acquired if number(r.get('acquire_seconds'))]
        limits = sorted({r['configured_context'] for r in acquired if number(r.get('configured_context'))})
        name = marker(i, endpoint_label(key, keys, identities), key)
        body.append([name, esc(', '.join(context(x) for x in limits) or '—'),
                     esc(context(max(measured)) if measured else '—'),
                     esc(f"{sum(r['ok'] for r in subset)}/{sum(r['total'] for r in subset)}"),
                     esc(seconds(statistics.median(times)) if times else '—'),
                     esc(fmt(max(peak) / 1024, 2) if peak else '—')])
    return table(('Configuration', 'Configured window', 'Largest successful target', 'Successful / recorded trials', 'Ready time (s)', 'Peak GPU (GiB)'), body, klass='scorecard')


def scorecard(keys, rows, identities, target):
    idx = lookup(rows)
    body = []
    specs = [('fresh', 'stream_tps', fmt), ('fresh', 'ttft_s', seconds),
             ('exact-prefix', 'ttft_s', seconds), ('exact-prefix', 'e2e_tps', fmt),
             ('extended-prefix', 'ttft_s', seconds), ('extended-prefix', 'stream_tps', fmt),
             ('extended-prefix', 'e2e_tps', fmt), ('extended-prefix', 'cache_percent', fmt)]
    for i, key in enumerate(keys):
        cells = [marker(i, endpoint_label(key, keys, identities), key)]
        for phase, field, formatter in specs:
            row = idx.get((key, target, phase))
            value = metric(row, field) if row else None
            if not row:
                cell = '<span class="missing" title="No observation for this phase">—</span>'
            elif not successful(row):
                cell = f'<span class="missing" title="{esc(row.get("reason") or "No successful observation")}">—</span>'
            else:
                title = (f"{row['ok']}/{row['total']} successful; prompt {fmt(row.get('prompt_tokens'), 0)}; "
                         f"output {fmt(row.get('completion_tokens'), 0)}")
                cell = f'<span title="{esc(title)}">{esc(formatter(value))}</span>'
            cells.append(cell)
        body.append(cells)
    return (f'<h3>{esc(context(target))} target tokens</h3>' + table(
        ('Configuration', 'Fresh stream (t/s)', 'Fresh TTFT (s)', 'Exact TTFT (s)', 'Exact e2e (t/s)',
         'Extended TTFT (s)', 'Extended stream (t/s)', 'Extended e2e (t/s)', 'Extended reported cache (%)'), body, klass='scorecard'))


def highlights(keys, rows, identities, target):
    """Metric-specific leaders, never an overall winner or a quality ranking."""
    idx = lookup(rows)
    specs = [('Fresh stream', 'fresh', 'stream_tps', max, 't/s', fmt),
             ('Extended-prefix end-to-end', 'extended-prefix', 'e2e_tps', max, 't/s', fmt),
             ('Extended-prefix first output', 'extended-prefix', 'ttft_s', min, 's', seconds)]
    items = []
    for title, phase, field, choose, unit, formatter in specs:
        values = [(k, metric(idx.get((k, target, phase)), field)) for k in keys]
        values = [(k, v) for k, v in values if number(v) and v >= 0]
        if not values:
            continue
        best = choose(v for _, v in values)
        # Treat values that round to the same displayed precision as ties.
        winners = [k for k, v in values if formatter(v) == formatter(best)]
        names = '; '.join(endpoint_label(k, keys, identities) for k in winners)
        items.append(f'<div class="highlight"><span>{esc(title)}</span><strong>{esc(formatter(best))} {esc(unit)}</strong><div>{esc(names)}</div><small>{len(values)} of {len(keys)} configurations reported</small></div>')
    if not items:
        return ''
    return (f'<h3>Observed leaders at {esc(context(target))} target tokens</h3>'
            '<p class="muted">Metric-specific observations, not an overall ranking. A single repeat is not enough to establish a stable performance difference.</p>'
            '<div class="highlights">' + ''.join(items) + '</div>')


def dashboard_html(rows, identities, acquisitions):
    groups = comparison_groups(rows, identities)
    if not groups:
        return '<section><h2>Performance comparison</h2><p class="empty">Awaiting trial observations. The report updates as trials complete.</p></section>'
    out = []
    for group_index, (keys, subset, targets) in enumerate(groups, 1):
        host = identities[keys[0]].get('host', {}).get('hostname') or 'Unknown host'
        out.append(f'<section class="cohort"><h2>{esc(host)} · Comparison {group_index}</h2>')
        out.append(hardware_description(keys, subset, identities))
        if targets:
            out.append(highlights(keys, subset, identities, targets[-1]))
        out.append('<h3>Configuration coverage</h3>' + coverage(keys, subset, identities, acquisitions))
        out.append('<h3 class="section-label">Performance at the largest target contexts</h3><p class="muted">Compare the same target and workload phase. Missing or unsuccessful measurements are not ranked or replaced with zero. Hover over cells for actual prompt/output counts.</p>')
        for target in targets[-2:]:
            out.append(scorecard(keys, subset, identities, target))
        out.append('<h3 class="section-label">Performance across context lengths</h3><div class="legend">')
        for i, key in enumerate(keys):
            color, dash = style(i)
            out.append(f'<div><svg width="30" height="12" aria-hidden="true"><line x1="1" y1="6" x2="29" y2="6" stroke="{color}" stroke-width="3" stroke-dasharray="{dash}"/></svg> {marker(i, endpoint_label(key, keys, identities), key)}</div>')
        out.append('</div><div class="charts">')
        plots = [
            dict(title='Fresh generation rate', subtitle='Estimated streamed output after first output arrives; higher is better.', phase='fresh', field='stream_tps', unit='tokens/s'),
            dict(title='Fresh time to first output', subtitle='Initial ingestion and request overhead; lower is better.', phase='fresh', field='ttft_s', unit='seconds', log=True),
            dict(title='Extended-prefix end-to-end throughput', subtitle='Agent-like follow-up: includes prefix processing and generation; higher is better.', phase='extended-prefix', field='e2e_tps', unit='tokens/s'),
            dict(title='Extended-prefix time to first output', subtitle='Cost of appending to an earlier prompt; lower is better.', phase='extended-prefix', field='ttft_s', unit='seconds', log=True),
            dict(title='Exact-prefix time to first output', subtitle='Repeated identical prompt; lower is better.', phase='exact-prefix', field='ttft_s', unit='seconds', log=True),
            dict(title='Extended-prefix reported cache share', subtitle='API-reported cached tokens / prompt tokens; zero is not proof of an internal cache miss.', phase='extended-prefix', field='cache_percent', unit='percent', percent=True),
        ]
        out.extend(chart(subset, keys, targets, identities, **spec) for spec in plots)
        out.append('</div></section>')
    return ''.join(out)


STYLE = '''
:root{color-scheme:light;--ink:#172b3b;--sub:#516474;--border:#d6e0e8;--paper:#fff;--bg:#f4f7fa;--accent:#1769aa}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--ink)}
.page{max-width:1540px;margin:0 auto;padding:30px clamp(14px,3vw,48px) 60px}h1,h2,h3,p{margin:0}h1{font-size:clamp(25px,3vw,36px);letter-spacing:-.04em;line-height:1.15}h2{font-size:22px;letter-spacing:-.02em;margin-bottom:15px}h3{font-size:16px;margin:18px 0 8px}p{margin:8px 0}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:var(--accent)}.muted,.chart p{font-size:13px;color:var(--sub)}
.header{margin-bottom:22px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(142px,1fr));gap:12px;margin:20px 0}.metric{border:1px solid var(--border);background:white;padding:14px 16px;border-radius:10px}.metric strong{font-size:28px;line-height:1.25;display:block;font-variant-numeric:tabular-nums}.metric span{font-size:12px;color:var(--sub)}
section{margin:18px 0;padding:22px;background:var(--paper);border:1px solid var(--border);border-radius:12px}.hardware{border-left:3px solid #6c93b2;background:#f5f8fb;padding:8px 15px;margin-bottom:20px}.gpu{font-size:13px;overflow-wrap:anywhere;margin:8px 0}.gpu strong{font-size:14px}.highlights{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:12px;margin:10px 0 20px}.highlight{padding:12px 14px;border:1px solid var(--border);border-radius:8px;overflow-wrap:anywhere}.highlight span,.highlight small{display:block;font-size:12px;color:var(--sub)}.highlight strong{display:block;font-size:24px;font-variant-numeric:tabular-nums;margin:3px 0}.highlight div{font-weight:600;font-size:13px}.section-label{margin-top:30px}.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:8px;margin:10px 0 16px}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:9px 11px;border-bottom:1px solid #e4ebf0;vertical-align:middle}th{background:#eaf1f6;color:#34495a;font-weight:650;line-height:1.25}td:first-child,th:first-child{text-align:left}td:first-child{min-width:200px;overflow-wrap:anywhere}tr:last-child td{border-bottom:none}tbody tr:nth-child(even){background:#f8fafc}.scorecard td:not(:first-child){white-space:nowrap}.scorecard th{min-width:85px}.series-id{display:inline-flex;width:23px;height:23px;align-items:center;justify-content:center;border-radius:5px;background:var(--series);color:white;font-weight:700;font-size:12px;vertical-align:middle;flex-shrink:0}.missing{color:#8492a0}.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px 14px;margin:8px 0 15px;font-size:13px}.legend>div{display:flex;align-items:center;gap:6px;overflow-wrap:anywhere}.legend svg{flex-shrink:0}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:14px}.chart{min-width:0;margin:0;border:1px solid var(--border);padding:15px;border-radius:9px;background:#fff}.chart h3{margin:0}.chart p{min-height:2.6em;margin:4px 0}.chart svg{display:block;width:100%;height:auto;overflow:visible}.chart svg text{font:11px system-ui;fill:#4e6070}.gridline{stroke:#dce5ed;stroke-width:1}.empty{padding:20px;color:var(--sub)}details{margin:14px 0}summary{cursor:pointer;font-weight:650}details .table-wrap{margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;border-radius:8px;padding:12px;font-size:12px}a{color:var(--accent)}.raw td,.raw th{text-align:left;white-space:nowrap}.raw td:first-child{min-width:0}.raw td{max-width:400px;white-space:normal;overflow-wrap:anywhere}.footer{font-size:12px;color:var(--sub)}
@media(max-width:640px){section{padding:14px}.page{padding:18px 10px 40px}.metric strong{font-size:23px}.scorecard td:first-child{min-width:170px}.chart{padding:9px}}
@media print{body{background:white}.page{padding:0}section,.chart{break-inside:avoid}.table-wrap{overflow:visible}.charts{grid-template-columns:1fr 1fr}}
'''
