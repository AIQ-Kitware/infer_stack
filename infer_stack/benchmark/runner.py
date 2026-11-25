"""Serial, lease-scoped benchmarking of local Compose endpoints."""
from __future__ import annotations

import datetime
import hashlib
import os
import time
import uuid
from functools import partial

from .. import __version__
from . import hardware, ledger, workload

DEFAULT_CONTEXTS = (1024, 8192, 32768, 65536, 131072, 163840)


def record_base(run_id, endpoint, host, settings):
    return {'schema': ledger.SCHEMA, 'run_id': run_id, 'endpoint': endpoint,
            'host': host, 'settings': settings, 'infer_stack_version': __version__}


def emit(path, base, *, on_record=None, **fields):
    row = {**base, **fields, 'id': uuid.uuid4().hex,
           'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    ledger.append(path, row)
    if row.get('kind') == 'trial':
        print(format_trial(row), flush=True)
    if on_record is not None:
        on_record(row)
    return row


def format_trial(row):
    """Report observed counts, not the approximate requested context."""
    if row['status'] != 'ok':
        return (f"    {row['target_context']:,} target / {row['phase']}: "
                f"{row['status']} ({row.get('reason') or 'no reason recorded'})")
    metrics = row.get('metrics', {})

    def value(key, fmt):
        v = metrics.get(key)
        if not isinstance(v, (int, float)):
            return '-'
        return format(v, fmt)

    return (f"    prompt {value('prompt_tokens', ',')} tok | "
            f"output {value('completion_tokens', ',')} tok | "
            f"TTFT {value('ttft_seconds', '.2f')} s | "
            f"stream {value('stream_tps_estimate', '.1f')} tok/s | "
            f"e2e {value('e2e_completion_tps', '.1f')} tok/s | "
            f"cached {value('cached_prompt_tokens', ',')} tok | "
            f"elapsed {value('elapsed_seconds', '.2f')} s")


def _reason(exc):
    # No arbitrary upstream response bodies, command lines, or credentials in
    # a shareable ledger. Workload.request only raises sanitized RuntimeErrors.
    if isinstance(exc, RuntimeError) and str(exc).startswith(('HTTP ', 'stream reported', 'incomplete SSE')):
        return str(exc)
    return type(exc).__name__


def benchmark_endpoint(controller, endpoint, config, path, run_id, host, *, descriptor_for,
                       contexts=DEFAULT_CONTEXTS, repeats=1, generation_tokens=512,
                       request_timeout=180, effort=None, request_fn=workload.request, on_record=None):
    """Always release our own lease. A release failure aborts the outer run.

    A normal release may retain a keep-warm model, so atomically evict only
    deployments newly idled by this lease. Never release or evict another lease.
    """
    from ..leasing.launch import effective_max_model_len

    record = partial(emit, path, on_record=on_record)
    settings = {'workload': workload.WORKLOAD, 'generation_tokens': generation_tokens,
                'temperature': 0.0, 'reasoning_effort': effort, 'request_timeout': request_timeout,
                'repeats': repeats, 'append_lines': 16, 'transport': 'openai-sse'}
    base = record_base(run_id, endpoint.alias, host, settings)
    started = time.monotonic()
    outcome = None
    failures = 0
    try:
        if not endpoint.managed or endpoint.protocol not in ('chat', 'completions'):
            record(base, kind='endpoint', status='skipped', reason='requires a managed chat/completions endpoint')
            return 1
        # A bounded lease backstop exceeds the sum of configured waits and
        # request deadlines even when every benchmark request is slow.
        ttl = float(config.timeout) * 2 + (len(contexts) * repeats * 3 + 3) * request_timeout + 600
        print(f'Benchmarking {endpoint.alias}: acquiring', flush=True)
        endpoint_request = endpoint.to_request()
        outcome = controller.acquire('benchmark-' + run_id[:12], [endpoint_request],
                                     ttl_seconds=ttl, wait=True, timeout=float(config.timeout),
                                     interval=2.0, wait_for_placement=False)
        acquire_seconds = time.monotonic() - started
        if outcome.released_on_timeout or (outcome.wait is not None and not outcome.wait.ready):
            record(base, kind='endpoint', status='failed', reason='endpoint not ready', acquire_seconds=acquire_seconds)
            return 1
        if outcome.reconcile.publication_pending:
            record(base, kind='endpoint', status='failed', reason='publication pending', acquire_seconds=acquire_seconds)
            return 1
        dep = next(d for d in outcome.deployments if endpoint.alias in d.served)
        assigned = dep.assigned_gpus
        if assigned is None:
            assigned = outcome.reconcile.assignments.get(dep.id, [])
        gpus = hardware.gpu_facts(assigned, getattr(controller.backend, 'inventory', None))
        descriptor = descriptor_for(controller, outcome.lease, outcome.deployments, config)
        provenance = descriptor['serving_provenance'][endpoint.alias]
        base = {**base, 'deployment_id': dep.id, 'gpus': gpus,
                'provenance': provenance, 'runtime_image': hardware.image_facts(dep.id),
                'assigned_gpu_indices': assigned, 'acquire_seconds': acquire_seconds,
                'sharing_demand_at_acquire': dep.demand}
        # A shared live deployment can receive unrelated traffic. Avoid adding
        # benchmark load to a deployment held by another client at acquisition.
        _, live_deployments = controller.ledger.status()
        competing = any(d.id != dep.id and d.demand > 0 and endpoint.alias in d.served
                        for d in live_deployments)
        demand = next((d.demand for d in live_deployments if d.id == dep.id), dep.demand)
        if demand > 1 or competing:
            record(base, kind='endpoint', status='skipped', reason='deployment already leased by another client')
            return 1
        base_url = descriptor.get('base_url')
        if not base_url or not controller.backend.connection_info():
            record(base, kind='endpoint', status='failed', reason='no OpenAI connection')
            return 1
        key = descriptor.get('api_key') or os.environ.get(descriptor.get('api_key_env') or '')
        model = descriptor['endpoints'][endpoint.alias]
        options = {'protocol': endpoint.protocol, 'timeout': request_timeout, 'effort': effort}
        # Two inexpensive probes estimate per-line tokenizer costs and the
        # template overhead. Cache tests deliberately use different seeds.
        counts = []
        for n in (16, 64):
            result = request_fn(base_url, key, model, workload.prompt(n, n), max_tokens=1, **options)
            counts.append(result.get('prompt_tokens'))
        if all(isinstance(n, int) and n > 0 for n in counts) and counts[1] > counts[0]:
            slope = (counts[1] - counts[0]) / 48
            overhead = counts[0] - slope * 16
            calibration = 'server-reported'
        else:
            # Some SSE implementations omit usage. Preserve timing only and
            # mark the prompt construction as estimated, never invent counts.
            slope, overhead = 25, 64
            calibration = 'heuristic-no-usage'
        max_context = min(effective_max_model_len(dep.spec.get('runtime')),
                          effective_max_model_len(endpoint_request.spec.get('runtime'))) if dep.engine == 'vllm' else None
        record(base, kind='endpoint', status='ready', calibration=calibration,
             calibration_counts=counts, configured_context=max_context)
        for target in contexts:
            for repeat in range(repeats):
                # Leave room for tokenizer variation, the appended suffix and
                # the generation allowance; don't label an impossible prompt.
                if max_context and target + generation_tokens + 512 > max_context:
                    record(base, kind='trial', status='skipped', reason='target plus generation exceeds configured context headroom',
                         target_context=target, repeat=repeat, phase='fresh')
                    continue
                nlines = max(1, int((target - overhead) / slope))
                seed = int.from_bytes(hashlib.sha256(f'{run_id}:{target}:{repeat}'.encode()).digest()[:16], 'big')
                text = workload.prompt(nlines, seed)
                # Keep the exact prefix in all three requests. Whether the
                # implementation actually reuses it is an observation, not a promise.
                phases = [('fresh', text), ('exact-prefix', text),
                          ('extended-prefix', workload.prompt(nlines + 16, seed))]
                for phase, content in phases:
                    print(f'  {target:,} target tokens / {phase} ({repeat + 1}/{repeats})', flush=True)
                    with hardware.Sampler(gpus) as sampler:
                        try:
                            metrics = request_fn(base_url, key, model, content, max_tokens=generation_tokens, **options)
                            status, reason = 'ok', None
                        except Exception as exc:
                            metrics, status, reason = {}, 'failed', _reason(exc)
                    record(base, kind='trial', status=status, reason=reason, phase=phase,
                         target_context=target, repeat=repeat, seed=str(seed), prompt_lines=nlines + (16 if phase == 'extended-prefix' else 0),
                         prompt_bytes=len(content.encode()), calibration=calibration, metrics=metrics,
                         telemetry=sampler.samples)
                    if status != 'ok':
                        failures += 1
                        # A failed fresh/identical request cannot establish a
                        # prefix cache for subsequent phases in this trial.
                        break
        return failures
    except Exception as exc:
        record(base, kind='endpoint', status='failed', reason=_reason(exc),
             acquire_seconds=time.monotonic() - started)
        print(f'  failed: {_reason(exc)}', flush=True)
        return failures + 1
    finally:
        if outcome is not None and not outcome.released_on_timeout:
            print(f'  releasing {outcome.lease.id}', flush=True)
            try:
                released = controller.release_leases([outcome.lease.id], evict=True)
                if released.reconcile and released.reconcile.publication_pending:
                    raise RuntimeError('release publication pending')
            except BaseException:
                record(base, kind='release', status='failed', lease_id=outcome.lease.id)
                raise
            record(base, kind='release', status='ok', lease_id=outcome.lease.id)
