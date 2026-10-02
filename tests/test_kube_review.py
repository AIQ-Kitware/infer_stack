"""Real-host bootstrap review regressions; no privileged host mutations."""
import json
from pathlib import Path

import pytest
from test_kube_readiness import HTTP, Cluster, manager, node

from infer_stack.kube import k3s
from infer_stack.kube.inspect import failed, inventory, readiness
from infer_stack.kube.manage import KubeManager
from infer_stack.kube.operations import K3sProvider, verify_node_runtimes


def test_aiq_optional_mps_zero_zero_is_not_failure(monkeypatch, capsys):
    cluster = Cluster(installed=True)
    def run(args, **kw):
        if args[:3] == ['kubectl', 'get', 'daemonsets']:
            return json.dumps({'items': [
                {'metadata': {'name': name}, 'spec': {'template': {'spec': {
                    'containers': [{'image': 'nvcr.io/nvidia/k8s-device-plugin:v0.17.1'}]}}},
                 'status': {'desiredNumberScheduled': count, 'numberReady': count}}
                for name, count in [('nvdp-nvidia-device-plugin', 1),
                    ('nvdp-nvidia-device-plugin-gpu-feature-discovery', 1),
                    ('nvdp-nvidia-device-plugin-mps-control-daemon', 0)]]})
        return cluster(args, **kw)
    m = manager(cluster)
    m.run = run
    report = inventory(m, http=HTTP)
    assert not failed(readiness(report))
    assert report['device_plugin'][2]['applicable'] is False
    assert report['device_plugin'][2]['healthy'] is None
    from infer_stack.cli.commands_kube import _print_inventory
    _print_inventory(report)
    assert 'mps-control-daemon (N/A: zero scheduled)' in capsys.readouterr().out
    monkeypatch.setattr(k3s, 'bootstrap', lambda **kw: True)
    monkeypatch.setattr(K3sProvider, '_local_manager', staticmethod(lambda m: m))
    K3sProvider().apply(m, report, timeout=0)
    assert not any('upgrade' in args for args in cluster.calls)


def test_runtime_class_is_not_a_node_handler():
    cluster = Cluster(nodes=[node('aiq-gpu'), node('namek')])
    m = manager(cluster)
    def run(args, **kw):
        if args[:3] == ['kubectl', 'get', 'pods']:
            return json.dumps({'items': [{'metadata': {'name': 'plugin'},
                'spec': {'nodeName': 'aiq-gpu', 'runtimeClassName': 'nvidia'},
                'status': {'containerStatuses': [{'state': {'running': { 'startedAt': 'now'}}}]}}]})
        return cluster(args, **kw)
    m.run = run
    report = inventory(m, http=HTTP)
    assert report['runtime_classes'] == ['nvidia']
    assert report['nodes'][0]['nvidia_runtime_verified'] is True
    assert report['nodes'][1]['nvidia_runtime_verified'] is None
    assert failed(readiness(report, installation=False))
    assert all('nvidia_runtime_class' not in row for row in report['nodes'])


def test_private_user_config_keeps_root_and_default_authorities(tmp_path, monkeypatch):
    monkeypatch.setattr(k3s.Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('KUBECONFIG', raising=False)
    default = tmp_path / '.kube/config'
    default.parent.mkdir()
    default.write_text('unrelated EKS credentials')
    secret = 'cluster-admin-secret'
    k3s._provision_user_kubeconfig(lambda args: secret)
    copy = k3s.user_kubeconfig()
    assert copy.read_text() == secret
    assert copy.stat().st_mode & 0o777 == 0o600
    assert not k3s._ensure_default_kubeconfig_link()
    assert default.read_text() == 'unrelated EKS credentials'
    k3s._provision_user_kubeconfig(lambda args: secret + '-rotated')
    assert copy.stat().st_mode & 0o777 == 0o600
    assert copy.read_text().endswith('-rotated')


def test_bootstrap_enforces_root_0600_without_restart(monkeypatch):
    calls, fragments = [], []
    def run(args, **kw):
        calls.append(args)
        if args[:2] == ['systemctl', 'is-active'] and args[-1] == 'k3s-agent':
            raise RuntimeError('inactive')
        if args[:4] == ['sudo', '-n', 'install', '-m']:
            fragments.append(Path(args[5]).read_text())
        return ''
    monkeypatch.setattr(k3s, '_provision_user_kubeconfig', lambda run: None)
    monkeypatch.setattr(k3s, '_ensure_default_kubeconfig_link', lambda: True)
    monkeypatch.setattr(k3s, '_ensure_helm', lambda run: None)
    monkeypatch.setattr(k3s, '_require_local_tool', lambda name: None)
    k3s.bootstrap(run=run)
    assert fragments == ['write-kubeconfig-mode: "0600"\n']
    assert ['sudo', '-n', 'chmod', '0600', str(k3s.K3S_KUBECONFIG)] in calls
    assert not any('restart' in c or ('chmod' in c and '0644' in c) for c in calls)


def agent_runner(server='https://old:6443', name='namek', config=None):
    calls = []
    def run(args, **kw):
        calls.append(args)
        if args[:2] == ['systemctl', 'is-active']:
            if args[-1] == 'k3s':
                raise RuntimeError('inactive server')
            return ''
        if args[:2] == ['systemctl', 'show']:
            return '321'
        if args == ['k3s', '--version']:
            return 'k3s version v1.34.3+k3s1 (fake)'
        if args[-1] == '/proc/321/cmdline':
            return '/usr/local/bin/k3s\0agent\0'
        if args[-1] == '/proc/321/environ':
            return f'K3S_URL={server}\0K3S_NODE_NAME={name}\0K3S_TOKEN=NEVER-PRINT-ME\0' if config is None else ''
        if args[:3] == ['sudo', '-n', 'python3']:
            return json.dumps(config or [])
        raise AssertionError(args)
    return run, calls


@pytest.mark.parametrize('server,name,version,match', [
    ('https://new:6443', 'namek', None, 'differs from requested'),
    ('https://old:6443', 'new-name', None, 'node name'),
    ('https://old:6443', 'namek', 'v1.34.3+k3s10', 'implicit agent upgrade'),
])
def test_active_agent_join_refuses_incompatible_state(monkeypatch, server, name, version, match):
    run, calls = agent_runner()
    monkeypatch.setattr(k3s, '_require_local_tool', lambda name: None)
    with pytest.raises(RuntimeError, match=match):
        k3s.join(server=server, node_name=name, token_file='unused', version=version, run=run)
    assert not any(c[0] == 'curl' for c in calls)


def test_active_agent_correct_state_is_idempotent_and_secrets_hidden(monkeypatch):
    run, calls = agent_runner()
    monkeypatch.setattr(k3s, '_require_local_tool', lambda name: None)
    k3s.join(server='https://old:6443/', node_name='namek', token_file='unused', version='v1.34.3+k3s1', run=run)
    report = k3s.local_status(run=run)
    assert report['server'] == 'https://old:6443'
    assert report['node_name'] == 'namek'
    assert 'NEVER-PRINT-ME' not in json.dumps(report)
    assert not report['errors']


def test_agent_config_dropins_inspected():
    run, _ = agent_runner(config=['server: https://first:6443\nnode-name: a',
                                 'server: https://second:6443\nnode-name: namek'])
    report = k3s.local_status(run=run)
    assert report['server'] == 'https://second:6443'
    assert report['node_name'] == 'namek'


def test_explicit_provider_scopes_every_cluster_mutation_to_local_copy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv('KUBECONFIG', str(tmp_path / 'eks.yaml'))
    monkeypatch.setattr(k3s, 'user_kubeconfig', lambda: tmp_path / 'local.yaml')
    scoped = K3sProvider._local_manager(KubeManager(run=lambda args, **kw: calls.append(args) or ''))
    scoped.run(['kubectl', 'wait', 'node'])
    scoped.run(['helm', 'upgrade', '--install', 'nvdp'])
    assert all(c[1:3] == ['--kubeconfig', str(tmp_path / 'local.yaml')] for c in calls)
    assert 'eks' not in str(calls)


def test_canary_is_node_scoped_and_does_not_reserve_gpu():
    calls, documents = [], []
    def run(args, **kw):
        calls.append(args)
        if 'get' in args:
            return '{}'
        if 'apply' in args:
            documents.append(json.loads(kw['input_text']))
        return ''
    report = {'nodes': [{'name': 'namek', 'gpu_count': 1, 'nvidia_runtime_verified': None}]}
    verify_node_runtimes(KubeManager(run=run), report)
    assert documents[0]['spec']['nodeName'] == 'namek'
    assert documents[0]['spec']['runtimeClassName'] == 'nvidia'
    assert 'resources' not in documents[0]['spec']['containers'][0]
    assert documents[0]['spec']['automountServiceAccountToken'] is False
    assert any('wait' in c and '--for=jsonpath={.status.phase}=Succeeded' in c for c in calls)


def test_profile_vendor_prefix_is_not_duplicated():
    report = inventory(manager(Cluster(nodes=[node(product='NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition')])), http=HTTP)
    assert list(report['resource_profiles']['proposed']) == ['nvidia-rtx-pro-6000-blackwell-max-q-workstation-edition']


def test_setup_cli_is_deprecated_alias_of_install(monkeypatch):
    from infer_stack.cli.commands_kube import KubeInstallCLI, KubeSetupCLI
    calls = []
    monkeypatch.setattr(KubeInstallCLI, 'main', lambda **kw: calls.append(kw) or 0)
    assert KubeSetupCLI.main(argv=False, apply=True, namespace='default', gpu='none') == 0
    assert calls == [dict(argv=False, apply=True, namespace='default', release='kubeai', gpu='none', values=None, version=None)]


def test_controller_reports_only_readiness_transitions():
    from infer_stack.leasing import (
        Catalog,
        Controller,
        Ledger,
        NullBackend,
        SqliteStore,
    )
    from infer_stack.leasing.backend import Readiness
    catalog = Catalog.from_dict({'models': {'q': {'source': 'hf://org/model'}},
                                 'endpoints': {'qwen-coder': {'engine': 'vllm', 'model': 'q'}}})
    controller = Controller(Ledger(SqliteStore(':memory:')), NullBackend())
    deployment = controller.acquire('test', catalog.resolve_names(['qwen-coder'])).deployments[0]
    transitions = iter([Readiness(False, 'pending'), Readiness(False, 'pending'),
                        Readiness(False, 'scheduled on namek; pulling image'),
                        Readiness(False, 'scheduled on namek; pulling image'), Readiness(True, 'ok')])
    controller.backend.probe_ready = lambda *a: next(transitions)
    messages = []
    controller.backend.progress = messages.append
    controller.sleep = lambda seconds: None
    assert controller.wait_ready([deployment], timeout=10).ready
    assert len(messages) == 3
    assert messages[-1].endswith('generation verified')


def test_startup_details_expose_node_image_and_replica_state():
    from infer_stack.backends.kubeai import model_startup_progress
    pod = {'spec': {'containers': [{'image': 'vllm/vllm-openai:v0.11.2'}]}, 'status': {}}
    assert 'waiting for scheduler' in model_startup_progress([pod])
    pod['spec']['nodeName'] = 'namek'
    assert 'scheduled on namek' in model_startup_progress([pod])
    assert 'vllm/vllm-openai:v0.11.2' in model_startup_progress([pod])
    pod['status']['containerStatuses'] = [{'ready': False, 'state': {'waiting': {'reason': 'ImagePullBackOff'}}}]
    assert 'ImagePullBackOff' in model_startup_progress([pod])
    pod['status']['containerStatuses'] = [{'ready': False, 'state': {'running': {'startedAt': 'now'}}}]
    assert 'container started; model loading' in model_startup_progress([pod])
    pod['status']['containerStatuses'][0]['ready'] = True
    pod['status']['conditions'] = [{'type': 'Ready', 'status': 'True'}]
    assert '1/1 replicas ready' in model_startup_progress([pod])
    assert 'verifying generation' in model_startup_progress([pod])
