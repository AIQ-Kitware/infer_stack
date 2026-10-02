"""Small, read-only live snapshots for interactive cluster monitoring.

Three list requests per refresh, independent errors, no per-node/per-pod probes,
Helm inspection, host metrics, or API generation. The caller owns the cadence.
"""
from __future__ import annotations

import time

from ..backends.kubeai import model_replica_counts
from .manage import KubeManager


def gpu_request(pod: dict) -> int:
    """Scheduler GPU request including init containers and restartable sidecars."""
    spec = pod.get('spec') or {}

    def request(container):
        resources = container.get('resources') or {}
        requests = resources.get('requests') or {}
        limits = resources.get('limits') or {}
        return int(requests.get('nvidia.com/gpu', limits.get('nvidia.com/gpu', 0)))

    steady = sum(request(c) for c in spec.get('containers') or [])
    sidecars = peak = 0
    for container in spec.get('initContainers') or []:
        value = request(container)
        if container.get('restartPolicy') == 'Always':
            sidecars += value
            peak = max(peak, sidecars)
        else:
            peak = max(peak, sidecars + value)
    return max(steady + sidecars, peak) + int((spec.get('overhead') or {}).get('nvidia.com/gpu', 0))


def snapshot(manager: KubeManager, *, namespace: str, resources=('nodes', 'pods', 'models')) -> dict:
    report = {'nodes': None, 'pods': None, 'models': None, 'errors': {},
              'sampled_at': time.monotonic(), 'namespace': namespace}
    for key, args in (
        ('nodes', ['get', 'nodes', '-o', 'json']),
        ('pods', ['get', 'pods', '-A', '-o', 'json']),
        ('models', ['-n', namespace, 'get', 'models.kubeai.org',
                    '-l', 'infer-stack/managed=true', '-o', 'json']),
    ):
        if key not in resources:
            continue
        try:
            result = manager.kubectl_json(args)
            if not isinstance(result.get('items'), list):
                raise ValueError(f'{key} response has no items list')
            report[key] = result['items']
        except Exception as ex:  # independent diagnostic boundary
            report['errors'][key] = str(ex)
    return report


def node_rows(report: dict) -> list[tuple]:
    requested: dict[str, int] = {}
    for pod in report['pods'] or []:
        if pod.get('status', {}).get('phase') in {'Succeeded', 'Failed'}:
            continue
        node = pod.get('spec', {}).get('nodeName', '')
        requested[node] = requested.get(node, 0) + gpu_request(pod)
    runtime_nodes = {p.get('spec', {}).get('nodeName') for p in report['pods'] or []
                     if p.get('spec', {}).get('runtimeClassName') == 'nvidia' and
                     any(c.get('state', {}).get('running') or c.get('state', {}).get('terminated')
                         for c in p.get('status', {}).get('containerStatuses') or [])}
    return sorted([
        (r['name'], 'yes' if r['ready'] else 'NO',
         'detached' if r['detached_for_compose'] else 'yes' if r['schedulable'] else 'cordoned',
         str(r['gpu_count']), str(requested.get(r['name'], 0)) if report['pods'] is not None else '?',
         r['gpu_product'] or '-', str(r['gpu_memory_gib'] or '-'),
         'yes' if r['gpu_product'] and r['gpu_memory_gib'] else 'missing',
         'pod started' if r['name'] in runtime_nodes else 'unknown')
        for r in KubeManager.node_rows(report['nodes'] or [])
    ])


def pod_rows(report: dict) -> list[tuple]:
    rows = []
    for pod in report['pods'] or []:
        meta, status = pod.get('metadata', {}), pod.get('status', {})
        containers_spec = pod.get('spec', {}).get('containers') or []
        nvidia = any(any(part in c.get('image', '') for part in ('k8s-device-plugin', 'gpu-feature-discovery'))
                     for c in containers_spec)
        if meta.get('namespace') != report['namespace'] and not nvidia:
            continue
        containers = status.get('containerStatuses') or []
        reasons = [c.get('state', {}).get('waiting', {}).get('reason') for c in containers]
        reasons = [r for r in reasons if r]
        if not reasons:
            reasons = [c.get('reason') for c in status.get('conditions') or []
                       if c.get('status') == 'False' and c.get('reason')]
        phase = 'deleting' if meta.get('deletionTimestamp') else status.get('phase', '?')
        rows.append((KubeManager._pod_name(pod), pod.get('spec', {}).get('nodeName') or '(pending)',
                     ', '.join(reasons) or phase,
                     f"{sum(bool(c.get('ready')) for c in containers)}/{len(pod.get('spec', {}).get('containers') or [])}",
                     str(sum(c.get('restartCount', 0) for c in containers)), str(gpu_request(pod))))
    return sorted(rows)


def model_states(report: dict) -> dict[str, dict]:
    """Lifecycle evidence from one sample; declaring a CR proves no readiness."""
    states = {}
    for model in report['models'] or []:
        meta = model.get('metadata') or {}
        gid = (meta.get('labels') or {}).get('infer-stack/deployment')
        if not gid:
            continue
        pods = [p for p in report['pods'] or []
                if p.get('metadata', {}).get('namespace') == report['namespace']
                and p.get('metadata', {}).get('labels', {}).get('infer-stack/deployment') == gid
                and not p.get('metadata', {}).get('deletionTimestamp')
                and p.get('status', {}).get('phase') not in {'Succeeded', 'Failed'}]
        scheduled = any(p.get('spec', {}).get('nodeName') for p in pods)
        running = any(c.get('state', {}).get('running')
                      for p in pods for c in p.get('status', {}).get('containerStatuses') or [])
        ready_pods = any(any(c.get('type') == 'Ready' and c.get('status') == 'True'
                            for c in p.get('status', {}).get('conditions') or []) for p in pods)
        total, ready = model_replica_counts(model)
        replica_ready = bool(ready and (report['pods'] is None or ready_pods))
        stage = 'replica ready' if replica_ready else 'pod running' if running else 'scheduled' if scheduled else 'declared'
        if meta.get('deletionTimestamp'):
            stage, replica_ready = 'deleting', False
        # Proof is scoped to this incarnation; replacement/restarts invalidate
        # an earlier successful API request. A Model-only sample has no pod proof.
        identity = (meta.get('uid'), meta.get('generation'), tuple(sorted(
            (p.get('metadata', {}).get('uid', ''), tuple(
                (c.get('containerID', ''), c.get('restartCount', 0))
                for c in p.get('status', {}).get('containerStatuses') or [])) for p in pods))
                    if report['pods'] is not None else None)
        states[gid] = {'name': meta.get('name'), 'stage': stage,
                       'replica_ready': replica_ready, 'all_replicas': total,
                       'ready_replicas': ready, 'identity': identity}
    return states
