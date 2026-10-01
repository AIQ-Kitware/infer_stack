"""Best-effort Kubernetes facts and dependency-ordered readiness policy.

Each probe owns its error. In particular, chart configuration is never a
prerequisite for node discovery or proposed resource profiles.
"""
from __future__ import annotations

import csv
import io
from typing import Any

import requests
import yaml

from ..backends.kubeai import node_gpus
from .manage import KubeManager


def api_probe(base_url: str, http: Any = requests) -> dict:
    try:
        code = http.get(base_url.rstrip('/') + '/models', timeout=5).status_code
        return {'reachable': code == 200, 'status_code': code, 'base_url': base_url}
    except Exception as ex:  # diagnostic boundary
        return {'reachable': False, 'error': str(ex), 'base_url': base_url}


def inventory(manager: KubeManager, *, namespace='kubeai', release='kubeai',
              base_url='http://127.0.0.1:8000/openai/v1', http=requests) -> dict:
    report: dict[str, Any] = {
        'tools': {name: manager.command_exists(name) for name in ('kubectl', 'helm')},
        'cluster': {'context': '', 'reachable': False, 'distribution': None},
        'host_gpus': None, 'gpu_count': 0, 'nodes': [], 'runtime_classes': None, 'device_plugin': None,
        'kubeai': {'namespace': namespace, 'release_name': release, 'crd': None,
                   'namespace_exists': None, 'release': None, 'pods': None,
                   'services': None, 'models': None,
                   'api': {'base_url': base_url, 'reachable': False, 'blocked': 'Cluster unavailable'}},
        'resource_profiles': {'proposed': {}, 'installed': {}}, 'errors': {},
    }

    def probe(key, func, default=None):
        try:
            return func()
        except Exception as ex:  # independent, best-effort diagnostic boundary
            report['errors'][key] = str(ex)
            return default

    if manager.command_exists('nvidia-smi'):
        def host_gpus():
            text = manager.run(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                '--format=csv,noheader,nounits'])
            return [{'index': int(row[0]), 'product': row[1].strip(),
                     'memory_mib': int(row[2])}
                    for row in csv.reader(io.StringIO(text)) if row]
        report['host_gpus'] = probe('host_gpus', host_gpus)
    if not report['tools']['kubectl']:
        report['errors']['cluster'] = 'kubectl unavailable'
        return report
    report['cluster']['context'] = manager.current_context()
    reachable, detail = manager.cluster_reachable()
    report['cluster']['reachable'] = reachable
    if not reachable:
        report['errors']['cluster'] = detail
        return report
    nodes = probe('nodes', manager.nodes, [])
    report['nodes'] = manager.node_rows(nodes)
    facts = node_gpus(nodes)
    for row in report['nodes']:
        row['gfd_labels'] = bool(row['gpu_product'] and row['gpu_memory_gib'])
        row['kubelet_version'] = next(
            ((n.get('status', {}).get('nodeInfo') or {}).get('kubeletVersion', '')
             for n in nodes if n.get('metadata', {}).get('name') == row['name']), '')
    versions = [r['kubelet_version'] for r in report['nodes']]
    report['cluster']['distribution'] = ('k3s' if any('k3s' in v for v in versions)
                                         else 'unknown')
    runtime = probe('runtime_classes', lambda: manager.kubectl_json(
        ['get', 'runtimeclasses', '-o', 'json']))
    if runtime is not None:
        report['runtime_classes'] = [i.get('metadata', {}).get('name')
                                     for i in runtime.get('items', [])]
    runtime_ok = 'nvidia' in (report['runtime_classes'] or [])
    for row in report['nodes']:
        row['nvidia_runtime_class'] = runtime_ok
    ds = probe('device_plugin', lambda: manager.kubectl_json(
        ['get', 'daemonsets', '-A', '-o', 'json']))
    if ds is not None:
        plugins = []
        for item in ds.get('items', []):
            containers = item.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
            if any('k8s-device-plugin' in c.get('image', '') for c in containers):
                status = item.get('status', {})
                desired = status.get('desiredNumberScheduled', 0)
                plugins.append({'name': manager._pod_name(item), 'desired': desired,
                                'ready': status.get('numberReady', 0),
                                'healthy': desired > 0 and status.get('numberReady', 0) == desired})
        report['device_plugin'] = plugins
    report['resource_profiles']['proposed'] = manager.resource_profiles(
        nodes, runtime_class_name='nvidia' if runtime_ok else None)
    kubeai = report['kubeai']
    # --ignore-not-found distinguishes absence from permission/transport errors.
    kubeai['crd'] = probe('crd', lambda: bool(manager.run([
        'kubectl', 'get', 'crd', 'models.kubeai.org', '--ignore-not-found', '-o', 'name']).strip()))
    kubeai['namespace_exists'] = probe('namespace', lambda: bool(manager.run([
        'kubectl', 'get', 'namespace', namespace, '--ignore-not-found', '-o', 'name']).strip()))
    kubeai['release'] = None
    if report['tools']['helm']:
        releases = probe('helm_releases', manager.helm_releases, [])
        kubeai['release'] = next((r for r in releases if r.get('name') == release
                                  and r.get('namespace') == namespace), None)
    if kubeai['namespace_exists']:
        for resource in ('pods', 'services'):
            result = probe(resource, lambda resource=resource: manager.kubectl_json(
                ['-n', namespace, 'get', resource, '-o', 'json']))
            kubeai[resource] = result.get('items', []) if result is not None else None
        if kubeai['crd']:
            result = probe('models', lambda: manager.kubectl_json([
                '-n', namespace, 'get', 'models.kubeai.org',
                '-l', 'infer-stack/managed=true', '-o', 'json']))
            kubeai['models'] = result.get('items', []) if result is not None else None
        config = probe('installed_profiles', lambda: manager.kubectl_json([
            '-n', namespace, 'get', 'configmap', 'kubeai-config',
            '--ignore-not-found', '-o', 'json']), {})
        def installed_profiles():
            system = yaml.safe_load((config or {}).get('data', {}).get('system.yaml', '')) or {}
            if not isinstance(system, dict):
                raise ValueError('kubeai-config system.yaml must contain a mapping')
            profiles = system.get('resourceProfiles') or {}
            if not isinstance(profiles, dict):
                raise ValueError('Installed resourceProfiles must contain a mapping')
            return profiles
        report['resource_profiles']['installed'] = probe('profile_yaml', installed_profiles, {})
    kubeai['api'] = api_probe(base_url, http) if kubeai['namespace_exists'] else {
        'base_url': base_url, 'reachable': False, 'blocked': 'KubeAI namespace missing'}
    report['gpu_count'] = sum(f['count'] for f in facts.values())
    return report


def readiness(report: dict, *, installation=True, gpu=True) -> list[dict]:
    """Policy over inventory; stop dependent chains at their first blocker."""
    checks = []

    def check(name, ok, fix='', detail='', blocked=False):
        checks.append({'name': name, 'ok': bool(ok), 'status': 'blocked' if blocked else
                       ('ok' if ok else 'fail'), 'detail': detail, 'fix': fix if not ok else ''})
        return ok

    if not check('kubectl available', report['tools']['kubectl'],
                 'infer-stack kube bootstrap --provider=k3s'):
        return checks
    check('Helm available', report['tools']['helm'], 'infer-stack kube bootstrap')
    if not check('cluster reachable', report['cluster']['reachable'],
                 'Select a reachable kubeconfig/context or run infer-stack kube bootstrap',
                 report['errors'].get('cluster', '')):
        return checks
    rows = report['nodes']
    check('nodes Ready', bool(rows) and all(r['ready'] for r in rows),
          'infer-stack kube bootstrap', report['errors'].get('nodes', ''))
    if gpu:
        gpu_rows = [r for r in rows if r['gpu_count']]
        runtime_ok = 'nvidia' in (report['runtime_classes'] or [])
        # Working external integrations need not use a named NVIDIA RuntimeClass.
        check('NVIDIA runtime', runtime_ok or bool(gpu_rows),
              'Install NVIDIA driver/container toolkit on GPU nodes; then infer-stack kube bootstrap',
              'RuntimeClass nvidia' if runtime_ok else 'No nvidia RuntimeClass; GPU allocation proves an external integration' if gpu_rows else '')
        plugin = report['device_plugin']
        check('NVIDIA device plugin', bool(plugin) and all(p['healthy'] for p in plugin),
              'infer-stack kube bootstrap', report['errors'].get('device_plugin', ''))
        if check('nvidia.com/gpu allocatable', bool(gpu_rows), 'infer-stack kube bootstrap'):
            check('GFD product + memory labels', all(r['gfd_labels'] for r in gpu_rows),
                  'infer-stack kube bootstrap')
        else:
            check('GFD product + memory labels', False, blocked=True,
                  detail='Waiting for allocatable GPUs')
    if not installation:
        return checks
    kubeai = report['kubeai']
    fix = f"infer-stack kube install --namespace={kubeai['namespace']} --release={kubeai['release_name']}"
    check('KubeAI Model CRD', kubeai.get('crd'), fix, report['errors'].get('crd', ''))
    ns = check('KubeAI namespace', kubeai.get('namespace_exists'), fix,
               report['errors'].get('namespace', ''))
    release = kubeai.get('release')
    check('KubeAI Helm release', bool(release) and release.get('status') == 'deployed',
          fix, report['errors'].get('helm_releases', ''))
    api = kubeai.get('api', {})
    check('KubeAI API', api.get('reachable'),
          'infer-stack config set kubeai_base_url <reachable OpenAI URL>; inspect infer-stack kube status',
          api.get('error', api.get('blocked', api.get('base_url', ''))), blocked=not ns)
    return checks


def failed(checks: list[dict]) -> bool:
    return any(c['status'] == 'fail' for c in checks)
