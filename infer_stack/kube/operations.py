"""Converge prerequisites and KubeAI using the shared inventory policy."""
from __future__ import annotations

import time

from . import k3s
from .inspect import failed, inventory, readiness
from .manage import KubeManager


class K3sProvider:
    """Host provisioning adapter; generic inventory never assumes K3s."""

    def plan(self, manager: KubeManager, report: dict) -> list[str]:
        actions = []
        actions.append('Provision the local K3s server; preserve unrelated selected context; use private local kubeconfig for all reconciliation')
        if not report['tools']['helm']:
            actions.append('Install Helm')
        if not report.get('gpu_count'):
            actions.append('Verify installed NVIDIA container runtime; restart local K3s only if runtime discovery needs it')
        if failed(readiness(report, installation=False)):
            actions.append('Reconcile NVIDIA device plugin 0.17.1 + GFD when runtime is available')
        actions.append('Wait for Ready nodes, allocatable GPUs and GFD labels; run prerequisite doctor')
        return actions

    def apply(self, manager: KubeManager, report: dict, *, version=None, timeout=180) -> dict:
        # provider=k3s always targets this host's K3s, never the selected EKS/etc.
        k3s.bootstrap(version=version, run=manager.run)
        manager = self._local_manager(manager)
        fresh = inventory(manager)
        if not fresh.get('gpu_count') and (manager.command_exists('k3s')
                or 'nvidia' not in (fresh['runtime_classes'] or [])):
            if not k3s._active(manager.run, 'k3s'):
                raise RuntimeError('Configure NVIDIA runtime on this cluster provider; local K3s is not active')
            if not manager.command_exists('nvidia-container-runtime'):
                raise RuntimeError('Install NVIDIA driver and nvidia-container-toolkit, then retry infer-stack kube bootstrap')
            selected_uid = manager.run(['kubectl', 'get', 'namespace', 'kube-system',
                                        '-o', 'jsonpath={.metadata.uid}']).strip()
            local_uid = manager.run(['sudo', '-n', 'k3s', 'kubectl', 'get',
                                     'namespace', 'kube-system', '-o',
                                     'jsonpath={.metadata.uid}']).strip()
            if not selected_uid or selected_uid != local_uid:
                raise RuntimeError('Selected cluster differs from local K3s; configure runtime on its GPU nodes')
            config_path = '/var/lib/rancher/k3s/agent/etc/containerd/config.toml'
            config = manager.run(['sudo', '-n', 'cat', config_path])
            if 'nvidia-container-runtime' not in config:
                manager.run(['sudo', '-n', 'systemctl', 'restart', 'k3s'])
            deadline = time.monotonic() + timeout
            while True:
                fresh = inventory(manager)
                if fresh['cluster']['reachable'] and 'nvidia' in (fresh['runtime_classes'] or []):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('K3s runtime discovery timed out; ensure nvidia-container-runtime is in the service PATH')
                time.sleep(min(2, max(0, deadline - time.monotonic())))
            config = manager.run(['sudo', '-n', 'cat',
                                  '/var/lib/rancher/k3s/agent/etc/containerd/config.toml'])
            if 'nvidia-container-runtime' not in config or 'nvidia' not in (fresh['runtime_classes'] or []):
                raise RuntimeError('K3s did not detect NVIDIA runtime; ensure nvidia-container-runtime is in the K3s service PATH')
        manager.run(['kubectl', 'wait', '--for=condition=Ready', 'node', '--all', f'--timeout={timeout}s'])
        plugin = fresh.get('device_plugin') or []
        if (not plugin or not any(p['desired'] > 0 for p in plugin)
                or not all(p['healthy'] for p in plugin if p['desired'] > 0)
                or not fresh.get('gpu_count')
                or not all(r['gfd_labels'] for r in fresh['nodes'] if r['gpu_count'])):
            if 'nvidia' not in (fresh['runtime_classes'] or []):
                raise RuntimeError('NVIDIA chart remediation requires RuntimeClass nvidia; configure node runtimes first')
            manager.install_nvidia_device_plugin()
        deadline = time.monotonic() + timeout
        verified_nodes = set()
        while True:
            fresh = inventory(manager)
            if fresh.get('gpu_count') and 'nvidia' in (fresh['runtime_classes'] or []):
                # Fresh per apply, once per node during this readiness loop.
                pending = [n for n in fresh['nodes'] if n['gpu_count'] and n['name'] not in verified_nodes]
                verify_node_runtimes(manager, {'nodes': pending}, timeout=timeout)
                verified_nodes.update(n['name'] for n in pending)
                fresh = inventory(manager)
            if not failed(readiness(fresh, installation=False)):
                return fresh
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU readiness timed out; run infer-stack kube doctor')
            time.sleep(min(2, max(0, deadline - time.monotonic())))

    @staticmethod
    def _local_manager(manager):
        def run(args, **kwargs):
            if args[0] in {'kubectl', 'helm'}:
                args = [args[0], '--kubeconfig', str(k3s.user_kubeconfig()), *args[1:]]
            return manager.run(args, **kwargs)
        local = KubeManager(run=run)
        local.command_exists = manager.command_exists
        return local


def verify_node_runtimes(manager, report, *, timeout=180):
    """Explicit bootstrap canaries test each GPU node's nvidia handler.

    No GPU reservation: allocatable resources are independently checked. A
    successful retained Pod supplies historical evidence for later inventory.
    Always replace our own canary: prior startup cannot verify today's handler.
    No hostRuntime claim is inferred from the RuntimeClass object alone.
    """
    import hashlib
    import json
    for node in report['nodes']:
        if not node['gpu_count']:
            continue
        name = 'infer-stack-runtime-' + hashlib.sha256(node['name'].encode()).hexdigest()[:12]
        existing = manager.kubectl_json(['-n', 'kube-system', 'get', 'pod', name, '--ignore-not-found', '-o', 'json'])
        if existing and existing.get('metadata', {}).get('labels', {}).get('infer-stack/runtime-canary') != 'true':
            raise RuntimeError(f'Refusing to replace unrelated pod kube-system/{name}')
        manager.run(['kubectl', '-n', 'kube-system', 'delete', 'pod', name, '--ignore-not-found', '--wait=true'])
        doc = {'apiVersion': 'v1', 'kind': 'Pod',
               'metadata': {'name': name, 'namespace': 'kube-system',
                            'labels': {'infer-stack/runtime-canary': 'true'}},
               'spec': {'nodeName': node['name'], 'runtimeClassName': 'nvidia',
                        'restartPolicy': 'Never', 'automountServiceAccountToken': False,
                        'containers': [{'name': 'runtime', 'image': 'busybox:1.37.0', 'command': ['/bin/true']}],
                        'tolerations': [{'operator': 'Exists'}]}}
        manager.run(['kubectl', 'apply', '-f', '-'], input_text=json.dumps(doc))
        manager.run(['kubectl', '-n', 'kube-system', 'wait', '--for=jsonpath={.status.phase}=Succeeded',
                     f'pod/{name}', f'--timeout={timeout}s'])



def install_plan(manager: KubeManager, report: dict, *, operator_values=None) -> dict:
    kubeai = report['kubeai']
    values = manager._merge_profiles(manager._deep_merge(
        manager._existing_kubeai_values(kubeai['release_name'], kubeai['namespace']),
        operator_values or {}), report['resource_profiles']['proposed'])
    public, _ = manager._split_secret_values(values)
    return public


def install(manager: KubeManager, report: dict, *, operator_values=None,
            chart='kubeai/kubeai', version=None, gpu=True) -> None:
    checks = readiness(report, installation=False, gpu=gpu)
    if report['errors'].get('helm_releases'):
        raise RuntimeError('Cannot inspect existing Helm releases; repair Helm access before installation')
    blockers = [c for c in checks if c['status'] == 'fail' and not c['name'].endswith('NVIDIA runtime evidence')]
    if blockers:
        raise RuntimeError('KubeAI prerequisites failed: ' + ', '.join(c['name'] for c in blockers)
                           + '. Run infer-stack kube doctor / kube bootstrap')
    if gpu and 'nvidia' in (report['runtime_classes'] or []):
        verify_node_runtimes(manager, report)
        report = inventory(manager, namespace=report['kubeai']['namespace'], release=report['kubeai']['release_name'], base_url=report['kubeai']['api']['base_url'])
    checks = readiness(report, installation=False, gpu=gpu)
    if failed(checks):
        raise RuntimeError('Node runtime verification incomplete; run infer-stack kube doctor')
    kubeai = report['kubeai']
    manager.install_kubeai(namespace=kubeai['namespace'], release=kubeai['release_name'],
                          resource_profiles=report['resource_profiles']['proposed'] if gpu else {},
                          operator_values=operator_values, chart=chart,
                          version=version or manager._release_chart_version(kubeai.get('release')))
