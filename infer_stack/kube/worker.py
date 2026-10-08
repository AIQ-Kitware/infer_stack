"""Named-worker acceptance, separate from the normal lease authority.

Tests reserve the target node's GPUs briefly, then create one isolated KubeAI
Model. They never converge/prune the application's managed Model set or ledger.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
import selectors
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from ..backends.kubeai import (
    GPU_MEMORY_LABEL,
    GPU_PRODUCT_LABEL,
    model_replica_counts,
    render_models,
)
from ..leasing.models import Deployment
from .manage import KubeManager

RUN_LABEL = 'infer-stack/acceptance-run'
NODE_LABEL = 'infer-stack/acceptance-node'
DEFAULT_MODEL = 'HuggingFaceTB/SmolLM2-135M-Instruct'
GPU_PROBE_IMAGE = 'nvidia/cuda:12.4.1-base-ubuntu22.04'


def scoped_manager(manager: KubeManager, kubeconfig: str | Path | None) -> KubeManager:
    """Scope both Helm and kubectl without changing the operator's environment."""
    if not kubeconfig:
        return manager
    path = str(Path(kubeconfig).expanduser().resolve())
    def run(args, **kwargs):
        if args[0] in {'kubectl', 'helm'}:
            args = [args[0], '--kubeconfig', path, *args[1:]]
        return manager.run(args, **kwargs)
    class ScopedManager(KubeManager):
        def command_exists(self, name: str) -> bool:
            return manager.command_exists(name)
    return ScopedManager(run=run)


def devices(text: str) -> list[dict]:
    """Preserve each device's identity/product/memory, including mixed nodes."""
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if row:
            uuid, product, memory = (value.strip() for value in row)
            rows.append({'uuid': uuid, 'product': product, 'memory_mib': int(float(memory))})
    if not rows or len({row['uuid'] for row in rows}) != len(rows):
        raise RuntimeError('GPU query returned no devices or duplicate device identities')
    return sorted(rows, key=lambda row: row['uuid'])


def local_devices(manager: KubeManager) -> list[dict]:
    return devices(manager.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                '--format=csv,noheader,nounits']))


def worker_facts(manager: KubeManager, node: str, expected_gpus: int) -> dict:
    if expected_gpus < 1:
        raise RuntimeError('expected GPUs must be positive')
    raw = manager.node(node)
    labels = raw.get('metadata', {}).get('labels') or {}
    count = int(raw.get('status', {}).get('allocatable', {}).get('nvidia.com/gpu', 0))
    facts = {'node': node, 'ready': manager._node_ready(raw), 'gpu_count': count,
             'product': labels.get(GPU_PRODUCT_LABEL), 'memory_mib': labels.get(GPU_MEMORY_LABEL),
             'hostname': labels.get('kubernetes.io/hostname'),
             'schedulable': not raw.get('spec', {}).get('unschedulable', False)}
    facts['errors'] = []
    if not facts['ready']:
        facts['errors'].append(f'node {node} is not Ready')
    if not facts['schedulable']:
        facts['errors'].append(f'node {node} is cordoned; use infer-stack kube node attach {node} when appropriate')
    if count != expected_gpus:
        facts['errors'].append(f'node {node} exposes {count} GPUs, expected {expected_gpus}; check its NVIDIA runtime/device-plugin')
    if not facts['product'] or not facts['memory_mib']:
        facts['errors'].append(f'node {node} is missing GFD product/memory labels')
    if not facts['hostname']:
        facts['errors'].append(f'node {node} has no kubernetes.io/hostname label')
    return facts


def worker_profile(manager: KubeManager, facts: dict, *, namespace: str,
                   resource_profile: str | None = None) -> tuple[str, dict, bool]:
    """Add a stable, node-constrained profile without rewriting operator profiles."""
    config = manager.kubectl_json(['-n', namespace, 'get', 'configmap', 'kubeai-config', '-o', 'json'])
    import yaml
    system = yaml.safe_load(config.get('data', {}).get('system.yaml') or '{}')
    installed = (system or {}).get('resourceProfiles') or {}
    if resource_profile:
        if ':' in resource_profile or resource_profile not in installed:
            raise RuntimeError('resource-profile must name an installed profile (without a :count suffix)')
        profile = copy.deepcopy(installed[resource_profile])
    else:
        profile: dict[str, Any] = {'imageName': 'nvidia-gpu', 'runtimeClassName': 'nvidia',
                   'requests': {'nvidia.com/gpu': '1'}, 'limits': {'nvidia.com/gpu': '1'}}
    for field in ('requests', 'limits'):
        if str(profile.get(field, {}).get('nvidia.com/gpu')) != '1':
            raise RuntimeError('Worker acceptance requires a base profile requesting/limiting one NVIDIA GPU')
    if profile.get('runtimeClassName') != 'nvidia':
        raise RuntimeError('Worker acceptance requires runtimeClassName nvidia')
    selector = profile.setdefault('nodeSelector', {})
    labels = manager.node(facts['node']).get('metadata', {}).get('labels') or {}
    if any(labels.get(k) != v for k, v in selector.items()):
        raise RuntimeError('Base resource profile selectors do not match the requested node')
    selector['kubernetes.io/hostname'] = facts['hostname']
    digest = hashlib.sha256(json.dumps([facts['node'], resource_profile], sort_keys=True).encode()).hexdigest()[:12]
    name = 'infer-stack-node-' + digest
    if name in installed and installed[name] != profile:
        raise RuntimeError(f'Existing profile {name} differs; refusing to overwrite it')
    return name, profile, name not in installed


def _names(node: str, run_id: str) -> tuple[str, str]:
    if not re.fullmatch(r'[a-z0-9]{8,20}', run_id):
        raise RuntimeError('run-id must contain 8-20 lowercase letters/digits')
    if not re.fullmatch(r'[a-z0-9](?:[-.a-z0-9]*[a-z0-9])?', node) or len(node) > 63:
        raise RuntimeError('Node name must fit a Kubernetes label value (at most 63 characters)')
    return f'infer-stack-gpu-{run_id}', f'infer-stack-e2e-{run_id}'


def _wait_for_acceptance_pods_deleted(manager: KubeManager, *, namespace: str,
                                      selector: str, timeout: int = 180) -> None:
    """Wait for acceptance pods, tolerating the kubectl timeout/delete race.

    ``kubectl wait --for=delete`` can return its timeout just as the final pod
    disappears. Re-read the selector before calling that a cleanup failure, and
    include the remaining pod state when deletion really is still incomplete.
    """
    try:
        manager.run(['kubectl', '-n', namespace, 'wait', '--for=delete', 'pods',
                     '-l', selector, f'--timeout={timeout}s'])
        return
    except Exception as ex:
        remaining = manager.kubectl_json([
            '-n', namespace, 'get', 'pods', '-l', selector, '-o', 'json',
        ]).get('items') or []
        if not remaining:
            return
        states = []
        for pod in remaining:
            meta = pod.get('metadata') or {}
            status = pod.get('status') or {}
            state = status.get('phase') or 'Unknown'
            if meta.get('deletionTimestamp'):
                state += f' deleting-since={meta["deletionTimestamp"]}'
            states.append(f'{meta.get("name", "?")}({state})')
        raise RuntimeError(
            f'Acceptance cleanup timed out waiting for pods matching {selector!r}: '
            + ', '.join(states)
        ) from ex


def cleanup(manager: KubeManager, *, node: str, namespace: str, run_id: str) -> None:
    """Explicit retry/cleanup deletes only exact, matching acceptance resources."""
    pod_name, model_name = _names(node, run_id)
    for kind, name in [('models.kubeai.org', model_name), ('pod', pod_name)]:
        existing = manager.kubectl_json(['-n', namespace, 'get', kind, name,
                                         '--ignore-not-found', '-o', 'json'])
        if not existing:
            continue
        labels = existing.get('metadata', {}).get('labels') or {}
        if labels.get(RUN_LABEL) != run_id or labels.get(NODE_LABEL) != node:
            raise RuntimeError(f'Refusing to delete unrelated {kind}/{name}')
        manager.run(['kubectl', '-n', namespace, 'delete', kind, name, '--wait=true', '--timeout=180s'])
    # Model deletion is asynchronous; wait for its backing Pods to release GPUs.
    # A timed-out kubectl wait is re-checked because the final deletion can race
    # the client's deadline by a fraction of a second.
    _wait_for_acceptance_pods_deleted(
        manager, namespace=namespace, selector=f'{RUN_LABEL}={run_id},{NODE_LABEL}={node}')
    _wait_for_acceptance_pods_deleted(
        manager, namespace=namespace, selector=f'model={model_name}')


def plan(manager: KubeManager, *, node: str, namespace: str, expected_gpus: int,
         resource_profile: str | None = None) -> dict:
    facts = worker_facts(manager, node, expected_gpus)
    result = {'facts': facts, 'actions': [
        f'Fresh NVIDIA runtime/device query reserving all {expected_gpus} GPUs on {node} temporarily',
        f'Create one temporary KubeAI Model on {node}; verify GPU requests, replica readiness and real generation',
        'Delete only this run\'s Model/Pod; retain its reusable node profile; leave catalog/ledger/gateway untouched'],
        'context': manager.current_context(), 'namespace': namespace, 'profile_name': None, 'profile': None, 'needs_profile': False}
    if facts['errors']:
        return result
    name, profile, missing = worker_profile(manager, facts, namespace=namespace, resource_profile=resource_profile)
    result.update(profile_name=name, profile=profile, needs_profile=missing)
    if missing:
        result['actions'].insert(0, f'Add node profile {name} through the existing KubeAI Helm installer (preserve installed chart version/values)')
    result['profile'] = profile
    return result


def wait_gpu_probe(manager: KubeManager, *, node: str, namespace: str, name: str,
                   timeout: int, progress=print, sleep=time.sleep, clock=time.monotonic):
    """Bounded fresh launch with changing startup details, not a silent long wait."""
    from .monitor import pod_rows

    deadline, last = clock() + timeout, None
    while True:
        pod = manager.kubectl_json(['-n', namespace, 'get', 'pod', name, '-o', 'json'])
        phase = pod.get('status', {}).get('phase')
        rows = pod_rows({'namespace': namespace, 'pods': [pod]})
        detail = rows[0][2] if rows else str(phase or 'pending')
        if pod.get('spec', {}).get('nodeName') not in {None, node}:
            raise RuntimeError('GPU probe landed on another node')
        if phase == 'Succeeded':
            return
        if phase == 'Failed':
            raise RuntimeError(f'Fresh GPU runtime probe failed: {detail}')
        if detail != last:
            progress(f'{node}: GPU runtime probe {detail}')
            last = detail
        if clock() >= deadline:
            raise RuntimeError(f'Fresh GPU runtime probe timed out: {detail}')
        sleep(min(5, max(0, deadline - clock())))


def acceptance(manager: KubeManager, *, node: str, namespace: str, release: str,
               expected_gpus: int, base_url: str | None, run_id: str, timeout=900,
               kubeconfig: str | None = None,
               resource_profile: str | None = None, model=DEFAULT_MODEL,
               expected_devices: list[dict] | None = None, http=requests,
               progress=print, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Fresh GPU launch + node-specific KubeAI generation with bounded cleanup."""
    pod_name, model_name = _names(node, run_id)
    scope = f' --kubeconfig={shlex.quote(str(kubeconfig))}' if kubeconfig else ''
    progress(f'Acceptance run {run_id} on {node}; cleanup: infer-stack kube node test {node} --run-id={run_id} --cleanup --apply --namespace={namespace}{scope}')
    prepared = plan(manager, node=node, namespace=namespace, expected_gpus=expected_gpus, resource_profile=resource_profile)
    if prepared['facts']['errors']:
        raise RuntimeError('; '.join(prepared['facts']['errors']))
    for kind, name in [('pod', pod_name), ('models.kubeai.org', model_name)]:
        if manager.kubectl_json(['-n', namespace, 'get', kind, name, '--ignore-not-found', '-o', 'json']):
            raise RuntimeError(f'Acceptance resource {kind}/{name} already exists; use explicit --cleanup --run-id={run_id} before retrying')
    if prepared['needs_profile']:
        found = manager.kubeai_release(release)
        if not found or found.get('namespace') != namespace:
            raise RuntimeError('Cannot add node profile to an externally managed/mismatched chart; use the configured Helm release')
        manager.install_kubeai(namespace=namespace, release=release,
                               resource_profiles={prepared['profile_name']: prepared['profile']},
                               version=manager._release_chart_version(found))
        # The controller config must reflect the Helm rollout before testing.
        name, _, missing = worker_profile(manager, prepared['facts'], namespace=namespace, resource_profile=resource_profile)
        if missing or name != prepared['profile_name']:
            raise RuntimeError('KubeAI did not load the node resource profile after installation')
    labels = {RUN_LABEL: run_id, NODE_LABEL: node}
    pod = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': pod_name, 'namespace': namespace, 'labels': labels},
           'spec': {'nodeSelector': {'kubernetes.io/hostname': prepared['facts']['hostname']},
                    'runtimeClassName': 'nvidia', 'restartPolicy': 'Never', 'automountServiceAccountToken': False,
                    'containers': [{'name': 'gpu', 'image': GPU_PROBE_IMAGE,
                        'command': ['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'],
                        'resources': {'requests': {'nvidia.com/gpu': str(expected_gpus)},
                                      'limits': {'nvidia.com/gpu': str(expected_gpus)}}}]}}
    deployment = Deployment(id=f'test-{run_id}', compat_key='acceptance', engine='vllm',
                            sharing='dedicated', capacity={}, state='live', created_at=0, updated_at=0,
                            spec={'engine': 'vllm', 'hf_model_id': model,
                                  'runtime': {'resource_profile': prepared['profile_name'], 'max_model_len': 2048,
                                              'gpu_memory_utilization': 0.5},
                                  'topology': {'tensor_parallel_size': 1}},
                            served={model_name: {'served_model_name': model_name, 'protocol': 'chat'}})
    rendered = render_models([deployment], namespace=namespace, default_resource_profile=None)
    if rendered.errors:
        raise RuntimeError('; '.join(rendered.errors))
    doc = rendered.docs[0]
    # Deliberately outside the normal lease authority's managed selector.
    doc['metadata']['labels'] = labels
    created = False
    primary_error: BaseException | None = None
    result = {'context': prepared['context'], 'namespace': namespace, 'run_id': run_id, 'node': node, 'profile': prepared['profile_name'], 'model': model_name}
    try:
        # Atomic create refuses occupied names, even an interrupted test.
        manager.run(['kubectl', 'create', '-f', '-'], input_text=json.dumps(pod))
        created = True
        progress(f'{node}: launching fresh runtime/GPU probe ({expected_gpus} GPUs)')
        wait_gpu_probe(manager, node=node, namespace=namespace, name=pod_name, timeout=timeout,
                       progress=progress, sleep=sleep, clock=clock)
        actual = manager.kubectl_json(['-n', namespace, 'get', 'pod', pod_name, '-o', 'json'])
        if actual.get('spec', {}).get('nodeName') != node:
            raise RuntimeError('GPU probe landed on another node; refusing an aggregate-cluster success')
        observed = devices(manager.run(['kubectl', '-n', namespace, 'logs', pod_name]))
        if len(observed) != expected_gpus or (expected_devices is not None and observed != sorted(expected_devices, key=lambda row: row['uuid'])):
            raise RuntimeError(f'Runtime GPU identities differ from expected hardware: {observed}')
        product_names = {d['product'] for d in observed}
        if len(product_names) == 1:
            def normalize(value):
                return re.sub('[^a-z0-9]', '', str(value).lower())
            if normalize(prepared['facts']['product']) != normalize(observed[0]['product']):
                raise RuntimeError('GFD product label disagrees with freshly observed GPU product')
            if any(abs(int(prepared['facts']['memory_mib']) - d['memory_mib']) > 512 for d in observed):
                raise RuntimeError('GFD memory label disagrees with freshly observed GPU memory')
        else:
            result['limitations'] = ['Heterogeneous node: GFD labels are node-level; this test verifies every device, but generation uses one Kubernetes-allocated GPU, not every product.']
        result['devices'] = observed
        result['gfd'] = {'product': prepared['facts']['product'], 'memory_mib': prepared['facts']['memory_mib']}
        manager.run(['kubectl', '-n', namespace, 'delete', 'pod', pod_name, '--wait=true', '--timeout=180s'])
        manager.run(['kubectl', 'create', '-f', '-'], input_text=json.dumps(doc))
        deadline, last = clock() + timeout, None
        while True:
            cr = manager.kubectl_json(['-n', namespace, 'get', 'models.kubeai.org', model_name, '-o', 'json'])
            pods = manager.kubectl_json(['-n', namespace, 'get', 'pods', '-l', f'model={model_name}', '-o', 'json']).get('items') or []
            failed = [p for p in pods if not p.get('metadata', {}).get('deletionTimestamp')
                      and p.get('status', {}).get('phase') == 'Failed']
            if failed:
                pod = failed[0]
                status = pod.get('status') or {}
                meta = pod.get('metadata') or {}
                reason = status.get('reason') or 'Failed'
                detail = status.get('message') or ''
                terminated = []
                for container in status.get('containerStatuses') or []:
                    term = (container.get('state') or {}).get('terminated') or {}
                    if term:
                        terminated.append(
                            f'{container.get("name", "container")}: '
                            f'{term.get("reason") or "terminated"} exit={term.get("exitCode", "?")}'
                        )
                suffix = '; '.join(part for part in [detail, *terminated] if part)
                where = (pod.get('spec') or {}).get('nodeName') or 'unscheduled'
                raise RuntimeError(
                    f'Model pod {meta.get("name", "?")} failed on {where}: {reason}'
                    + (f': {suffix}' if suffix else '')
                )
            active = [p for p in pods if not p.get('metadata', {}).get('deletionTimestamp')
                      and p.get('status', {}).get('phase') not in {'Succeeded', 'Failed'}]
            state = 'waiting for scheduler'
            for p in active:
                spec = p.get('spec') or {}
                scheduled = spec.get('nodeName')
                if scheduled and scheduled != node:
                    raise RuntimeError(f'Model scheduled on {scheduled}, expected {node}')
                containers = spec.get('containers') or []
                requested = sum(int(c.get('resources', {}).get('requests', {}).get('nvidia.com/gpu', 0)) for c in containers)
                limited = sum(int(c.get('resources', {}).get('limits', {}).get('nvidia.com/gpu', 0)) for c in containers)
                if scheduled and (requested != 1 or limited != 1 or spec.get('runtimeClassName') != 'nvidia'):
                    raise RuntimeError('Model pod does not request/limit exactly one GPU with NVIDIA runtime')
                state = f'scheduled on {scheduled}' if scheduled else state
                for c in p.get('status', {}).get('containerStatuses') or []:
                    waiting = c.get('state', {}).get('waiting') or {}
                    if waiting:
                        state += ': ' + str(waiting.get('reason', 'starting')) + ' ' + str(waiting.get('message', ''))
                    if c.get('state', {}).get('running'):
                        state += ': container running; model loading'
            ready_pods = [p for p in active if p.get('spec', {}).get('nodeName') == node
                          and any(c.get('type') == 'Ready' and c.get('status') == 'True'
                                  for c in p.get('status', {}).get('conditions') or [])]
            _, ready = model_replica_counts(cr)
            if ready and ready_pods:
                state = f'{node}: replica ready; verifying generation'
                if state != last:
                    progress(state)
                    last = state
                serving_devices = devices(manager.run(['kubectl', '-n', namespace, 'exec',
                                                          ready_pods[0]['metadata']['name'], '--',
                                                          'nvidia-smi', '--query-gpu=uuid,name,memory.total',
                                                          '--format=csv,noheader,nounits']))
                if len(serving_devices) != 1 or serving_devices[0] not in observed:
                    raise RuntimeError("Serving replica must expose exactly one of this worker's verified GPUs")
                result['serving_devices'] = serving_devices
                with api_access(namespace=namespace, kubeconfig=kubeconfig, base_url=base_url) as base:
                    try:
                        response = http.post(base.rstrip('/') + '/chat/completions',
                                             json={'model': model_name, 'messages': [{'role': 'user', 'content': 'Reply with ready.'}], 'max_tokens': 8}, timeout=45)
                        response.raise_for_status()
                        if not response.json().get('choices'):
                            raise RuntimeError('Generation response contains no choices')
                    except requests.RequestException as ex:
                        raise RuntimeError(f'Worker generation failed: {ex}') from ex
                result['generation_verified'] = True
                progress(f'{node}: real generation verified')
                return result
            if state != last:
                progress(f'{model_name}: {state}')
                last = state
            if clock() >= deadline:
                raise RuntimeError(f'Worker acceptance timed out: {state}')
            sleep(min(5, max(0, deadline - clock())))
    except BaseException as ex:
        primary_error = ex
        raise
    finally:
        if created:
            try:
                cleanup(manager, node=node, namespace=namespace, run_id=run_id)
            except Exception as cleanup_ex:
                if primary_error is None:
                    raise
                progress(
                    f'{node}: acceptance cleanup incomplete after the primary failure: '
                    f'{cleanup_ex}; retry the printed --cleanup command'
                )


@contextmanager
def api_access(*, namespace: str, kubeconfig: str | None = None, base_url: str | None = None):
    """Temporary loopback-only service forwarding; no gateway/config changes."""
    if base_url:
        yield base_url
        return
    args = ['kubectl']
    if kubeconfig:
        args += ['--kubeconfig', str(Path(kubeconfig).expanduser().resolve())]
    args += ['-n', namespace, 'port-forward', '--address=127.0.0.1', 'service/kubeai', ':80']
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        assert proc.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            deadline, lines = time.monotonic() + 30, []
            while time.monotonic() < deadline:
                if not selector.select(timeout=1):
                    continue
                line = proc.stdout.readline()
                lines.append(line.strip())
                match = re.search(r'Forwarding from 127\.0\.0\.1:(\d+)', line)
                if match:
                    yield f'http://127.0.0.1:{match[1]}/openai/v1'
                    return
                if not line:
                    break
            raise RuntimeError('Could not forward KubeAI API: ' + '; '.join(lines))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()


def private_file(path: str | Path, purpose: str) -> Path:
    """Onboarding credentials must be user-private; never print their contents."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise RuntimeError(f'{purpose} file is missing: {source}')
    if source.stat().st_mode & 0o077:
        raise RuntimeError(f'{purpose} file must be private (chmod 600): {source}')
    return source


def onboarding_plan(local: KubeManager, cluster: KubeManager, *, server: str,
                    node: str, version: str | None = None) -> dict:
    """Verify host prerequisites and the explicit administrative cluster target."""
    _names(node, 'onboardcheck')
    if not server.startswith('https://'):
        raise RuntimeError('K3s server must use an https:// URL')
    for tool in ('nvidia-smi', 'kubectl', 'helm'):
        if not local.command_exists(tool):
            raise RuntimeError(f'{tool} is required before onboarding; prepare host driver and admin tooling first')
    hardware = local_devices(local)
    if not local.command_exists('nvidia-container-runtime'):
        raise RuntimeError('Install NVIDIA container toolkit on this GPU worker first; nvidia-container-runtime is missing. No agent changes made.')
    selected = cluster.kubectl_json(['config', 'view', '--minify', '-o', 'json'])
    servers = [c.get('cluster', {}).get('server', '').rstrip('/') for c in selected.get('clusters') or []]
    if servers != [server.rstrip('/')]:
        raise RuntimeError(f'Explicit admin kubeconfig server {servers} differs from requested join server {server!r}')
    if not cluster.cluster_reachable()[0]:
        raise RuntimeError('Explicit admin kubeconfig cannot reach the requested cluster')
    nodes = cluster.nodes()
    versions = {n.get('status', {}).get('nodeInfo', {}).get('kubeletVersion')
                for n in nodes if cluster._node_control_plane(n)}
    if not version:
        if len(versions) != 1 or not next(iter(versions), '') or 'k3s' not in str(next(iter(versions))):
            raise RuntimeError('Cannot infer one K3s server version; pass --version explicitly')
        version = next(iter(versions))
    if versions and version not in versions:
        raise RuntimeError(f'Requested K3s version {version} differs from control-plane versions {sorted(str(v) for v in versions)}')
    return {'node': node, 'server': server, 'version': version, 'devices': hardware,
            'expected_gpus': len(hardware), 'actions': [
                'Verify existing agent membership or install/start a K3s agent using the private token file',
                'Verify local containerd NVIDIA handler; restart this agent only if discovery needs it',
                'Ensure cluster NVIDIA device plugin/GFD via existing installer if absent',
                f'Wait specifically for {node}: Ready, {len(hardware)} GPUs, product/memory labels',
                'Ensure KubeAI via the shared installer if absent; run node-specific GPU/generation acceptance']}


def onboard(local: KubeManager, cluster: KubeManager, prepared: dict, *, token_file: str,
            namespace: str, release: str, base_url: str | None, run_id: str,
            kubeconfig: str | None = None,
            timeout=900, model=DEFAULT_MODEL, resource_profile=None,
            progress=print, sleep=time.sleep, clock=time.monotonic, http=requests) -> dict:
    """Join, wait for the exact worker, then use the same targeted acceptance."""
    from . import k3s
    from .inspect import inventory
    from .operations import install

    node = prepared['node']
    k3s.join(server=prepared['server'], token_file=token_file, node_name=node,
             version=prepared['version'], run=local.run)
    config_path = '/var/lib/rancher/k3s/agent/etc/containerd/config.toml'
    runtime_config = local.run(['sudo', '-n', 'cat', config_path])
    if 'nvidia-container-runtime' not in runtime_config:
        progress(f'{node}: restarting local agent for NVIDIA runtime discovery')
        local.run(['sudo', '-n', 'systemctl', 'restart', 'k3s-agent'])
        discovery_deadline = clock() + min(timeout, 60)
        while True:
            runtime_config = local.run(['sudo', '-n', 'cat', config_path])
            if 'nvidia-container-runtime' in runtime_config:
                break
            if clock() >= discovery_deadline:
                raise RuntimeError('Local K3s agent has no NVIDIA handler; ensure nvidia-container-runtime is in its service PATH')
            sleep(2)
    report = inventory(cluster, namespace=namespace, release=release, base_url=base_url or 'http://127.0.0.1:8000/openai/v1')
    if report.get('errors', {}).get('device_plugin'):
        raise RuntimeError('Cannot inspect NVIDIA components; repair admin access before reconciling the plugin')
    if not report['device_plugin']:
        if 'nvidia' not in (report['runtime_classes'] or []):
            raise RuntimeError('Cluster requires RuntimeClass nvidia before GPU plugin setup; use infer-stack kube bootstrap on the server')
        cluster.install_nvidia_device_plugin()
    deadline, last = clock() + timeout, None
    while True:
        try:
            facts = worker_facts(cluster, node, prepared['expected_gpus'])
            detail = '; '.join(facts['errors'])
        except RuntimeError as ex:
            detail = str(ex)
        if not detail:
            break
        if detail != last:
            progress(f'{node}: {detail}')
            last = detail
        if clock() >= deadline:
            raise RuntimeError(f'{node}: worker GPU readiness timed out: {detail}. Other nodes cannot satisfy this test.')
        sleep(min(5, max(0, deadline - clock())))
    report = inventory(cluster, namespace=namespace, release=release, base_url=base_url or 'http://127.0.0.1:8000/openai/v1')
    if not report['kubeai']['crd'] or not report['kubeai']['namespace_exists']:
        install(cluster, report)
    return acceptance(cluster, node=node, namespace=namespace, release=release,
                      expected_gpus=prepared['expected_gpus'], expected_devices=prepared['devices'],
                      base_url=base_url, kubeconfig=kubeconfig, run_id=run_id, timeout=timeout, model=model,
                      resource_profile=resource_profile, progress=progress, sleep=sleep, clock=clock, http=http)
