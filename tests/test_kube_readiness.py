"""Kubernetes setup contracts without a cluster or privileged host changes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infer_stack.backends.kubeai import KubeaiBackend
from infer_stack.cli import ManageCLI
from infer_stack.cli.commands_catalog import CatalogSuggestCLI
from infer_stack.cli.commands_kube import (
    KubeBootstrapCLI,
    KubeDoctorCLI,
    KubeInstallCLI,
    KubeInventoryCLI,
)
from infer_stack.kube.inspect import failed, inventory, readiness
from infer_stack.kube.manage import KubeManager
from infer_stack.kube.operations import K3sProvider, install


def node(name='gpu-a', product='RTX-4090', count=4, memory='24576'):
    return {'metadata': {'name': name, 'labels': {
        'nvidia.com/gpu.product': product, 'nvidia.com/gpu.memory': memory} if product else {}},
        'status': {'nodeInfo': {'kubeletVersion': 'v1.34.3+k3s1'},
                   'conditions': [{'type': 'Ready', 'status': 'True'}],
                   'allocatable': {'nvidia.com/gpu': str(count)}}}


class Cluster:
    def __init__(self, *, nodes=None, installed=False, crd=False, plugin=True, reachable=True):
        self.nodes = [node()] if nodes is None else nodes
        self.installed = installed
        self.crd = installed or crd
        self.plugin = plugin
        self.reachable = reachable
        self.calls = []
        self.documents = []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        if args[:3] == ['kubectl', 'config', 'current-context']:
            return 'test-context'
        if args[:2] == ['kubectl', 'version']:
            if not self.reachable:
                raise RuntimeError('connection refused')
            return '{}'
        if args[0] == 'kubectl' and 'nodes' in args and 'get' in args:
            return json.dumps({'items': self.nodes})
        if args[:3] == ['kubectl', 'get', 'runtimeclasses']:
            return json.dumps({'items': [{'metadata': {'name': 'nvidia'}}]})
        if args[:3] == ['kubectl', 'get', 'daemonsets']:
            return json.dumps({'items': [{
                'metadata': {'name': 'nvdp', 'namespace': 'nvidia-device-plugin'},
                'spec': {'template': {'spec': {'containers': [{'image': 'nvcr.io/nvidia/k8s-device-plugin:v0.17.1'}]}}},
                'status': {'desiredNumberScheduled': 1, 'numberReady': 1}}] if self.plugin else []})
        if args[:3] == ['kubectl', 'get', 'crd']:
            return 'models.kubeai.org' if self.crd else ''
        if args[:3] == ['kubectl', 'get', 'namespace']:
            return 'namespace/kubeai' if self.installed else ''
        if args[0] == 'kubectl' and 'configmap' in args:
            if not self.installed:
                raise RuntimeError('NotFound: kubeai-config')
            return json.dumps({'data': {'system.yaml': 'resourceProfiles: {custom: {}}'}})
        if args[0] == 'kubectl' and any(r in args for r in ('pods', 'services', 'models.kubeai.org')):
            return '{"items": []}'
        if args[:2] == ['helm', 'list']:
            return json.dumps([{'name': 'kubeai', 'namespace': 'kubeai',
                                'status': 'deployed', 'chart': 'kubeai-0.22.0'}] if self.installed else [])
        if args[:3] == ['helm', 'get', 'values']:
            return 'resourceProfiles: {}'
        if args[:3] == ['helm', 'upgrade', '--install']:
            self.documents.append([yaml.safe_load(Path(args[i + 1]).read_text())
                                   for i, a in enumerate(args) if a == '-f'])
            if args[3] == 'nvdp':
                self.plugin = True
                self.nodes = [node()]
            else:
                self.installed = self.crd = True
            return ''
        if args[:2] == ['helm', 'repo'] or (args[0] == 'kubectl' and 'wait' in args):
            return ''
        raise AssertionError(f'Unexpected command: {args}')


def manager(cluster, tools=('kubectl', 'helm')):
    result = KubeManager(run=cluster)
    result.command_exists = lambda name: name in tools
    return result


HTTP = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(status_code=200))


@pytest.mark.parametrize('nodes,plugin,installed,crd', [
    ([node(count=0, product=None)], False, False, False),
    ([node(count=1)], True, False, False),
    ([node(count=4)], True, False, False),
    ([node(), node('gpu-b', 'A100', 8, '81920')], True, False, False),
    ([node(product=None)], True, False, False),
    ([node()], False, False, False),
    ([node()], True, False, True),
    ([node()], True, True, True),
])
def test_inventory_partial_states(nodes, plugin, installed, crd):
    cluster = Cluster(nodes=nodes, plugin=plugin, installed=installed, crd=crd)
    report = inventory(manager(cluster), http=HTTP)
    assert report['cluster']['reachable']
    assert report['gpu_count'] == sum(int(n['status']['allocatable']['nvidia.com/gpu']) for n in nodes)
    assert bool(report['device_plugin']) == plugin
    assert report['kubeai']['crd'] == crd
    assert report['kubeai']['namespace_exists'] == installed
    assert bool(report['kubeai']['release']) == installed
    assert len(report['resource_profiles']['proposed']) == len({
        n['metadata']['labels']['nvidia.com/gpu.product'] for n in nodes
        if n['status']['allocatable']['nvidia.com/gpu'] != '0' and n['metadata']['labels']})


def test_missing_kubectl_and_unreachable_do_not_cascade():
    cluster = Cluster(reachable=False)
    report = inventory(manager(cluster, tools=()))
    assert len(readiness(report)) == 1
    assert not cluster.calls
    report = inventory(manager(cluster))
    assert failed(readiness(report))
    assert [c['name'] for c in readiness(report)] == ['kubectl available', 'Helm available', 'cluster reachable']
    assert not any('daemonsets' in c for c in cluster.calls)


def test_probe_error_keeps_gpu_facts():
    cluster = Cluster(installed=True)
    def run(args, **kwargs):
        if 'daemonsets' in args:
            raise RuntimeError('Forbidden')
        return cluster(args, **kwargs)
    m = manager(cluster)
    m.run = run
    report = inventory(m, http=HTTP)
    assert report['gpu_count'] == 4
    assert report['device_plugin'] is None
    assert 'Forbidden' in report['errors']['device_plugin']
    assert report['resource_profiles']['proposed']


def test_preinstall_configmap_notfound_inventory_and_catalog(tmp_path, monkeypatch, capsys):
    cluster = Cluster(crd=True)
    backend = KubeaiBackend(state_dir=tmp_path, run=cluster)
    facts, sizes = backend.gpu_facts()
    assert facts['gpu-a']['count'] == 4
    assert sizes == {}
    inv, profiles = backend.suggestion_inventory()
    assert inv['gpu_count'] == 4
    assert profiles
    monkeypatch.setattr('infer_stack.cli.commands_leasing._make_backend', lambda *a: backend)
    assert CatalogSuggestCLI.main(argv=False, backend='kubeai') == 0
    out = capsys.readouterr()
    assert 'resourceProfiles' in out.err
    assert 'infer-stack kube install' in out.err
    assert 'models:' in out.out
    monkeypatch.setattr('infer_stack.cli.commands_kube.KubeManager', lambda: manager(cluster))
    assert KubeInventoryCLI.main(argv=False, json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['gpu_count'] == 4
    assert not report['kubeai']['namespace_exists']
    assert report['resource_profiles']['proposed'] == profiles


def test_doctor_green_and_fail_exit(monkeypatch, capsys):
    cluster = Cluster(crd=True)
    monkeypatch.setattr('infer_stack.cli.commands_kube.KubeManager', lambda: manager(cluster))
    monkeypatch.setattr('infer_stack.kube.inspect.requests.get', HTTP.get)
    assert KubeDoctorCLI.main(argv=False) == 1
    out = capsys.readouterr().out
    assert 'fix: infer-stack kube install' in out
    assert '[SKIP] KubeAI API' in out
    cluster.installed = True
    assert KubeDoctorCLI.main(argv=['--json']) == 0
    assert not failed(json.loads(capsys.readouterr().out)['checks'])


def test_install_values_token_upgrade_dryrun(tmp_path, monkeypatch, capsys):
    cluster = Cluster(nodes=[node(), node('gpu-b', 'A100', 8, '81920')])
    m = manager(cluster)
    monkeypatch.setattr('infer_stack.cli.commands_kube.KubeManager', lambda: m)
    monkeypatch.setattr('infer_stack.kube.inspect.requests.get', HTTP.get)
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('HF_TOKEN', 'sensitive-token')
    assert KubeInstallCLI.main(argv=['--dry-run', '--json']) == 0
    preview = json.loads(capsys.readouterr().out)
    assert len(preview['values']['resourceProfiles']) == 2
    assert not any('upgrade' in c or 'add' in c for c in cluster.calls)
    assert not (tmp_path / 'generated').exists()
    assert KubeInstallCLI.main(argv=False, apply=True) == 0
    assert len(cluster.documents[-1][0]['resourceProfiles']) == 2
    assert cluster.documents[-1][1]['secrets']['huggingface']['token'] == 'sensitive-token'
    cmd = next(c for c in cluster.calls if c[:3] == ['helm', 'upgrade', '--install'])
    assert 'sensitive-token' not in ' '.join(cmd)
    assert not Path(cmd[-1]).exists()  # temporary secret values cleaned up
    assert 'sensitive-token' not in (tmp_path / 'generated/kube/kubeai-values.yaml').read_text()
    assert KubeInstallCLI.main(argv=False, apply=True) == 0
    upgrades = [c for c in cluster.calls if c[:3] == ['helm', 'upgrade', '--install']]
    assert len(upgrades) == 2
    assert '--version' in upgrades[-1]


def test_install_custom_settings_and_profiles(tmp_path, monkeypatch):
    cluster = Cluster()
    m = manager(cluster)
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    report = inventory(m, namespace='custom-ns', release='custom-release', http=HTTP)
    profiles = report['resource_profiles']['proposed']
    name = next(iter(profiles))
    install(m, report, operator_values={'resourceProfiles': {name: {'custom': True}}}, chart='custom/chart', version='1.2')
    assert cluster.documents[-1][0]['resourceProfiles'][name] == {'custom': True}
    cmd = next(c for c in cluster.calls if 'upgrade' in c)
    assert cmd[3:5] == ['custom-release', 'custom/chart']
    assert cmd[cmd.index('-n') + 1] == 'custom-ns'


def test_install_requires_prerequisites():
    cluster = Cluster(nodes=[node(product=None)])
    m = manager(cluster)
    with pytest.raises(RuntimeError, match='GFD'):
        install(m, inventory(m))
    assert not any('upgrade' in c for c in cluster.calls)


def test_bootstrap_working_cluster_idempotent():
    cluster = Cluster()
    m = manager(cluster)
    provider = K3sProvider()
    for _ in range(2):
        provider.apply(m, inventory(m), timeout=0)
    assert not any(c[0] in ('sudo', 'curl', 'systemctl') or 'upgrade' in c for c in cluster.calls)


def test_bootstrap_partial_state_resumes():
    cluster = Cluster(nodes=[node(count=0, product=None)], plugin=False)
    m = manager(cluster)
    provider = K3sProvider()
    provider.apply(m, inventory(m), timeout=0)
    provider.apply(m, inventory(m), timeout=0)
    upgrades = [c for c in cluster.calls if 'upgrade' in c]
    assert len(upgrades) == 1
    assert '--version' in upgrades[0] and '0.17.1' in upgrades[0]
    assert 'gfd.enabled=true' in upgrades[0] and 'runtimeClassName=nvidia' in upgrades[0]


def test_fresh_bootstrap_plan_no_mutation(monkeypatch, capsys):
    cluster = Cluster()
    monkeypatch.setattr('infer_stack.cli.commands_kube.KubeManager', lambda: manager(cluster, tools=()))
    assert KubeBootstrapCLI.main(argv=['--provider=k3s', '--plan', '--json']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert any('K3s' in a for a in plan['actions'])
    assert any('Helm' in a for a in plan['actions'])
    assert not cluster.calls


@pytest.mark.parametrize('leaf', ['inventory', 'doctor', 'bootstrap', 'install', 'status'])
def test_real_modal_help_registration(leaf, tmp_path):
    env = os.environ | {'INFER_STACK_CONFIG_DIR': str(tmp_path), 'INFER_STACK_DATA_DIR': str(tmp_path)}
    proc = subprocess.run([sys.executable, '-m', 'infer_stack', 'kube', leaf, '--help'],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert leaf in proc.stdout
    assert '--config_dir' in proc.stdout


def test_modal_doctor_and_tree(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr('infer_stack.cli.commands_kube.KubeManager', lambda: manager(Cluster(), tools=()))
    assert ManageCLI.main(argv=['kube', 'doctor']) == 1
    capsys.readouterr()
    proc = subprocess.run([sys.executable, '-m', 'infer_stack', 'help', 'tree'],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    for leaf in ('inventory', 'doctor', 'bootstrap', 'install', 'status'):
        assert leaf in proc.stdout


def test_host_gpu_facts_survive_missing_kubectl():
    def run(args, **kwargs):
        assert args[0] == 'nvidia-smi'
        return '0, RTX 4090, 24576\n1, RTX 4090, 24576\n'
    m = KubeManager(run=run)
    m.command_exists = lambda name: name == 'nvidia-smi'
    report = inventory(m)
    assert len(report['host_gpus']) == 2
    assert report['gpu_count'] == 0
    assert not report['cluster']['reachable']


def test_existing_values_probe_failure_prevents_replacement(tmp_path, monkeypatch):
    cluster = Cluster(installed=True)
    m = manager(cluster)
    def run(args, **kwargs):
        if args[:3] == ['helm', 'get', 'values']:
            raise RuntimeError('Forbidden')
        return cluster(args, **kwargs)
    m.run = run
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='Forbidden'):
        install(m, inventory(m, http=HTTP))
    assert not any('upgrade' in c for c in cluster.calls)


def test_bootstrap_installs_helm_only_if_missing(monkeypatch):
    cluster = Cluster()
    m = manager(cluster, tools=('kubectl',))
    def ensure(run):
        assert run == cluster
        m.command_exists = lambda name: name in ('kubectl', 'helm')
    monkeypatch.setattr('infer_stack.kube.k3s._ensure_helm', ensure)
    result = K3sProvider().apply(m, inventory(m), timeout=0)
    assert result['tools']['helm']


def test_k3s_fresh_then_interrupted_start_never_reinstalls(monkeypatch):
    from infer_stack.kube import k3s

    calls = []
    installed = False
    def run(args, **kwargs):
        nonlocal installed
        calls.append(args)
        if args[:2] == ['systemctl', 'is-active']:
            raise RuntimeError('inactive')
        if args == ['curl', '-sfL', k3s.K3S_INSTALL_URL]:
            return '# installer'
        if args[-2:] == ['sh', '-']:
            installed = True
        return ''
    monkeypatch.setattr(k3s, '_require_local_tool', lambda name: None)
    monkeypatch.setattr(k3s.shutil, 'which', lambda name: '/usr/local/bin/k3s' if name == 'k3s' and installed else None)
    monkeypatch.setattr(k3s, '_ensure_default_kubeconfig_link', lambda: True)
    monkeypatch.setattr(k3s, '_ensure_helm', lambda run: None)
    k3s.bootstrap(run=run)
    k3s.bootstrap(run=run)
    assert sum(c[0] == 'curl' for c in calls) == 1
    assert ['sudo', '-n', 'systemctl', 'start', 'k3s'] in calls
    assert not any('restart' in c for c in calls)


def test_kubeconfig_preserved(tmp_path, monkeypatch):
    from infer_stack.kube import k3s

    monkeypatch.delenv('KUBECONFIG', raising=False)
    monkeypatch.setattr(k3s.Path, 'home', lambda: tmp_path)
    config = tmp_path / '.kube/config'
    config.parent.mkdir()
    config.write_text('operator config')
    assert not k3s._ensure_default_kubeconfig_link()
    assert config.read_text() == 'operator config'


def test_runtime_repair_refuses_unrelated_local_cluster(monkeypatch):
    cluster = Cluster(nodes=[node(count=0)], plugin=False)
    m = manager(cluster, tools=('kubectl', 'helm', 'nvidia-container-runtime'))
    original = m.run
    def run(args, **kwargs):
        if args[:3] == ['kubectl', 'get', 'runtimeclasses']:
            return '{"items": []}'
        if 'jsonpath={.metadata.uid}' in args:
            return 'local' if args[0] == 'sudo' else 'remote'
        return original(args, **kwargs)
    m.run = run
    monkeypatch.setattr('infer_stack.kube.k3s._active', lambda *args: True)
    with pytest.raises(RuntimeError, match='differs from local K3s'):
        K3sProvider().apply(m, inventory(m), timeout=0)
    assert not any('restart' in c for c in cluster.calls)


def test_k3s_runtimeclass_alone_does_not_prove_runtime_configured(monkeypatch):
    cluster = Cluster(nodes=[node(count=0)], plugin=False)
    m = manager(cluster, tools=('kubectl', 'helm', 'k3s'))
    monkeypatch.setattr('infer_stack.kube.k3s._active', lambda *args: True)
    with pytest.raises(RuntimeError, match='nvidia-container-toolkit'):
        K3sProvider().apply(m, inventory(m), timeout=0)
    assert not any('upgrade' in c for c in cluster.calls)


def test_k3s_configured_runtime_does_not_restart(monkeypatch):
    cluster = Cluster(nodes=[node(count=0, product=None)], plugin=False)
    m = manager(cluster, tools=('kubectl', 'helm', 'k3s', 'nvidia-container-runtime'))
    original = m.run
    host_calls = []
    def run(args, **kwargs):
        host_calls.append(args)
        if 'jsonpath={.metadata.uid}' in args:
            return 'same-cluster'
        if args[:3] == ['sudo', '-n', 'cat']:
            return 'BinaryName = "/usr/bin/nvidia-container-runtime"'
        return original(args, **kwargs)
    m.run = run
    monkeypatch.setattr('infer_stack.kube.k3s._active', lambda *args: True)
    result = K3sProvider().apply(m, inventory(m), timeout=0)
    assert result['gpu_count'] == 4
    assert not any('restart' in c for c in host_calls)


def test_explicit_cpu_install_escape_hatch(tmp_path, monkeypatch, capsys):
    cluster = Cluster(nodes=[node(count=0, product=None)], plugin=False)
    m = manager(cluster)
    monkeypatch.setattr('infer_stack.cli.commands_kube.KubeManager', lambda: m)
    monkeypatch.setattr('infer_stack.kube.inspect.requests.get', HTTP.get)
    values = tmp_path / 'cpu.yaml'
    values.write_text('resourceProfiles: {cpu: {imageName: cpu, requests: {cpu: 8}}}')
    assert KubeInstallCLI.main(argv=False, gpu='none', values=str(values), apply=True) == 0
    assert cluster.documents[-1][0]['resourceProfiles']['cpu']['imageName'] == 'cpu'
    assert 'NVIDIA' not in capsys.readouterr().out


def test_malformed_chart_config_does_not_erase_inventory():
    cluster = Cluster(installed=True)
    m = manager(cluster)
    original = m.run
    def run(args, **kwargs):
        if 'configmap' in args:
            return json.dumps({'data': {'system.yaml': '- invalid\n- mapping'}})
        return original(args, **kwargs)
    m.run = run
    report = inventory(m, http=HTTP)
    assert report['gpu_count'] == 4
    assert report['resource_profiles']['proposed']
    assert report['resource_profiles']['installed'] == {}
    assert 'mapping' in report['errors']['profile_yaml']
