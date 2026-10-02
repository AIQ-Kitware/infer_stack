"""Named GPU workers: deterministic plan, join, device/placement/generation tests."""
import copy
import json
from types import SimpleNamespace

import pytest
import yaml
from test_kube_readiness import node

from infer_stack.cli.commands_kube_worker import K3sOnboardCLI, KubeNodeTestCLI
from infer_stack.kube import k3s
from infer_stack.kube.manage import KubeManager
from infer_stack.kube.worker import (
    DEFAULT_MODEL,
    NODE_LABEL,
    RUN_LABEL,
    acceptance,
    cleanup,
    devices,
    onboard,
    onboarding_plan,
    plan,
    scoped_manager,
)


class Worker:
    """Commands traverse real rendering and operations; only the cluster is fake."""
    def __init__(self, name='namek', count=1, products=None):
        self.name, self.count = name, count
        self.devices = [{'uuid': f'GPU-{i}', 'product': p, 'memory_mib': 24576 - (i * 8192 if products and len(set(products)) > 1 else 0)}
                        for i, p in enumerate(products or ['NVIDIA RTX 3090'] * count)]
        self.nodes = [node('aiq-gpu', count=4), node(name, count=count)]
        self.nodes[1]['metadata']['labels']['nvidia.com/gpu.product'] = self.devices[0]['product'].replace(' ', '-') if self.devices else 'NVIDIA-RTX-3090'
        for n in self.nodes:
            n['metadata']['labels']['kubernetes.io/hostname'] = n['metadata']['name']
        self.nodes[0]['metadata']['labels']['node-role.kubernetes.io/control-plane'] = 'true'
        self.profiles, self.resources, self.calls, self.upgrades = {}, {}, [], []
        self.landing, self.model_gpu_count, self.model_ready = name, 1, 1
        self.handler = True
        self.generation_calls = []
        self.manager = KubeManager(run=self.run)
        self.manager.command_exists = lambda name: True
        self.manager.install_kubeai = self.install
        self.http = SimpleNamespace(post=self.post)

    def install(self, **kw):
        self.upgrades.append(kw)
        self.profiles.update(copy.deepcopy(kw['resource_profiles']))

    def post(self, url, **kw):
        self.generation_calls.append((url, kw))
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {'choices': [{'message': {'content': 'ready'}}]})

    def run(self, args, **kw):
        self.calls.append(args)
        if args[0] == 'nvidia-smi':
            return '\n'.join(f"{d['uuid']}, {d['product']}, {d['memory_mib']}" for d in self.devices)
        if args[0] == 'sudo' and 'cat' in args:
            return 'nvidia-container-runtime' if self.handler else 'no handler'
        if args[0] == 'sudo' and 'restart' in args:
            self.handler = True
            return ''
        if args == ['kubectl', 'config', 'current-context']:
            return 'aiq-gpu'
        if args[:3] == ['kubectl', 'config', 'view']:
            return json.dumps({'clusters': [{'cluster': {'server': 'https://aiq-gpu:6443'}}]})
        if args[:2] == ['kubectl', 'version']:
            return '{}'
        if args == ['kubectl', 'get', 'nodes', '-o', 'json']:
            return json.dumps({'items': self.nodes})
        if args[:3] == ['kubectl', 'get', 'node']:
            return json.dumps(next(n for n in self.nodes if n['metadata']['name'] == args[3]))
        if args[0] == 'kubectl' and 'configmap' in args:
            return json.dumps({'data': {'system.yaml': yaml.safe_dump({'resourceProfiles': self.profiles})}})
        if args[:2] == ['helm', 'list']:
            return json.dumps([{'name': 'kubeai', 'namespace': 'default', 'chart': 'kubeai-0.23.2'}])
        if args[:2] == ['kubectl', 'create']:
            doc = json.loads(kw['input_text'])
            key = (doc['kind'], doc['metadata']['name'])
            if key in self.resources:
                raise RuntimeError('AlreadyExists')
            self.resources[key] = doc
            return ''
        if args[:3] == ['kubectl', '-n', 'default']:
            verb = args[3]
            if verb == 'wait':
                return ''
            if verb == 'exec':
                return f"{self.devices[0]['uuid']}, {self.devices[0]['product']}, {self.devices[0]['memory_mib']}"
            if verb == 'logs':
                return self.run(['nvidia-smi'])
            if verb in {'get', 'delete'}:
                kind, name = args[4:6]
                kind = {'models.kubeai.org': 'Model', 'pod': 'Pod'}.get(kind, kind)
                if kind == 'pods':
                    model_name = args[args.index('-l') + 1].split('=', 1)[1]
                    cr = self.resources[('Model', model_name)]
                    return json.dumps({'items': [{'metadata': {'name': 'model-pod', 'namespace': 'default',
                        'labels': dict(cr['metadata']['labels'], model=cr['metadata']['name'])},
                        'spec': {'nodeName': self.landing, 'runtimeClassName': 'nvidia',
                                 'containers': [{'resources': {'requests': {'nvidia.com/gpu': str(self.model_gpu_count)},
                                                              'limits': {'nvidia.com/gpu': str(self.model_gpu_count)}}}]},
                        'status': {'phase': 'Running', 'conditions': [{'type': 'Ready', 'status': 'True'}]}}]})
                if verb == 'delete':
                    self.resources.pop((kind, name), None)
                    return ''
                doc = copy.deepcopy(self.resources.get((kind, name), {}))
                if doc and kind == 'Pod':
                    doc['spec']['nodeName'] = self.name
                    doc['status'] = {'phase': 'Succeeded'}
                if doc and kind == 'Model':
                    doc['status'] = {'replicas': {'all': 1, 'ready': self.model_ready}}
                return json.dumps(doc)
        raise AssertionError(args)


def run_test(worker, **kwargs):
    return acceptance(worker.manager, node=worker.name, namespace='default', release='kubeai',
                      expected_gpus=worker.count, base_url='http://kubeai/openai/v1',
                      run_id=kwargs.pop('run_id', 'abcdef123456'), http=worker.http,
                      expected_devices=worker.devices, **kwargs)


@pytest.mark.parametrize('name,products', [
    ('namek', ['NVIDIA RTX 3090']),
    ('yardrat', ['NVIDIA RTX A6000', 'NVIDIA RTX 4080']),
    ('aiq-gpu2', ['NVIDIA RTX 4070'] * 4),
])
def test_named_worker_real_rendering_device_inventory_and_cleanup(name, products):
    worker = Worker(name, len(products), products)
    out = run_test(worker)
    assert out['node'] == name and out['generation_verified']
    assert out['devices'] == worker.devices
    assert out['serving_devices'] == worker.devices[:1]
    assert out['profile'] in worker.profiles  # deliberate reusable profile retained
    assert worker.resources == {}
    assert worker.upgrades[0]['version'] == '0.23.2'
    profile = worker.profiles[out['profile']]
    assert profile['nodeSelector'] == {'kubernetes.io/hostname': name}
    assert profile['limits']['nvidia.com/gpu'] == '1'
    assert worker.generation_calls[0][1]['json']['model'] == out['model']
    assert not any('apply' in args or 'prune' in args for args in worker.calls)
    run_test(worker, run_id='123456abcdef')
    assert len(worker.upgrades) == 1  # repeats preserve profile/version


def test_plan_is_read_only_and_aggregate_does_not_mask_broken_worker():
    w = Worker(count=0)
    result = plan(w.manager, node='namek', namespace='default', expected_gpus=1)
    assert 'exposes 0 GPUs' in '; '.join(result['facts']['errors'])
    assert w.nodes[0]['status']['allocatable']['nvidia.com/gpu'] == '4'
    assert not w.resources and not w.upgrades
    with pytest.raises(RuntimeError, match='expected 1'):
        acceptance(w.manager, node='namek', namespace='default', release='kubeai', expected_gpus=1,
                   base_url='unused', run_id='abcdefgh')
    assert not any('create' in c for c in w.calls)


@pytest.mark.parametrize('change,match', [('landing', 'expected namek'), ('model_gpu_count', 'exactly one GPU')])
def test_wrong_placement_or_gpu_request_fails_and_cleans(change, match):
    w = Worker()
    setattr(w, change, 'aiq-gpu' if change == 'landing' else 2)
    with pytest.raises(RuntimeError, match=match):
        run_test(w)
    assert not w.resources and not w.generation_calls


def test_generation_failure_and_device_mismatch_cleanup():
    w = Worker()
    w.http.post = lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    with pytest.raises(RuntimeError, match='no choices'):
        run_test(w)
    assert not w.resources
    w.devices.append(dict(w.devices[0], uuid='GPU-extra'))
    with pytest.raises(RuntimeError, match='identities differ'):
        run_test(w)
    assert not w.resources


def test_pending_model_timeout_cleanup_reports_wait_state():
    w = Worker()
    w.model_ready = 0
    with pytest.raises(RuntimeError, match='timed out'):
        run_test(w, timeout=0)
    assert not w.resources


def test_explicit_interrupted_run_cleanup_refuses_unrelated_and_retries_safely():
    w = Worker()
    name = 'infer-stack-e2e-abcdefgh'
    w.resources[('Model', name)] = {'metadata': {'labels': {}}}
    with pytest.raises(RuntimeError, match='unrelated'):
        cleanup(w.manager, node='namek', namespace='default', run_id='abcdefgh')
    assert ('Model', name) in w.resources
    w.resources[('Model', name)]['metadata']['labels'] = {RUN_LABEL: 'abcdefgh', NODE_LABEL: 'namek'}
    cleanup(w.manager, node='namek', namespace='default', run_id='abcdefgh')
    cleanup(w.manager, node='namek', namespace='default', run_id='abcdefgh')
    assert not w.resources
    run_test(w, run_id='abcdefgh')


def test_profile_constraints_preserve_base_and_refuse_operator_collision():
    w = Worker()
    w.profiles['custom'] = {'imageName': 'custom-vllm', 'runtimeClassName': 'nvidia',
                            'requests': {'nvidia.com/gpu': '1'}, 'limits': {'nvidia.com/gpu': '1'},
                            'nodeSelector': {'nvidia.com/gpu.product': 'NVIDIA-RTX-3090'}}
    before = copy.deepcopy(w.profiles)
    p = plan(w.manager, node='namek', namespace='default', expected_gpus=1, resource_profile='custom')
    assert w.profiles == before
    assert p['profile']['imageName'] == 'custom-vllm'
    assert p['profile']['nodeSelector']['kubernetes.io/hostname'] == 'namek'
    w.profiles[p['profile_name']] = {}
    with pytest.raises(RuntimeError, match='overwrite'):
        plan(w.manager, node='namek', namespace='default', expected_gpus=1, resource_profile='custom')


def test_scoping_never_replaces_stale_context(tmp_path, monkeypatch):
    monkeypatch.setenv('KUBECONFIG', '/stale/eks')
    w = Worker()
    m = scoped_manager(w.manager, tmp_path / 'admin')
    for args in [['kubectl', 'version'], ['helm', 'list', '-A', '-o', 'json'], ['nvidia-smi']]:
        # Wrapper construction only: real fake expects commands without prefix.
        w.manager.run = lambda args, **kw: w.calls.append(args) or '{}'
        m.run(args)
    assert w.calls[0][1:3] == ['--kubeconfig', str(tmp_path / 'admin')]
    assert w.calls[1][1:3] == ['--kubeconfig', str(tmp_path / 'admin')]
    assert w.calls[2] == ['nvidia-smi']


def test_onboarding_plan_infers_version_and_preserves_heterogeneous_hardware():
    w = Worker('yardrat', 2, ['RTX A6000', 'RTX 4080'])
    prepared = onboarding_plan(w.manager, w.manager, server='https://aiq-gpu:6443', node='yardrat')
    assert prepared['version'] == 'v1.34.3+k3s1'
    assert prepared['expected_gpus'] == 2
    assert prepared['devices'] == w.devices
    assert not w.upgrades and not w.resources
    with pytest.raises(RuntimeError, match='differs'):
        onboarding_plan(w.manager, w.manager, server='https://wrong:6443', node='yardrat')
    w.manager.command_exists = lambda name: name != 'nvidia-container-runtime'
    with pytest.raises(RuntimeError, match='container toolkit'):
        onboarding_plan(w.manager, w.manager, server='https://aiq-gpu:6443', node='yardrat')


def test_onboard_integration_join_runtime_discovery_then_exact_worker_test(monkeypatch):
    w = Worker()
    prepared = onboarding_plan(w.manager, w.manager, server='https://aiq-gpu:6443', node='namek')
    joins = []
    monkeypatch.setattr(k3s, 'join', lambda **kw: joins.append(kw))
    from infer_stack.kube import inspect
    monkeypatch.setattr(inspect, 'inventory', lambda *a, **kw: {
        'device_plugin': [{'name': 'nvdp'}], 'kubeai': {'crd': True, 'namespace_exists': True}})
    w.handler = False
    out = onboard(w.manager, w.manager, prepared, token_file='private-file', namespace='default',
                  release='kubeai', base_url='http://kubeai/openai/v1', run_id='onboard12345', http=w.http)
    assert joins[0]['node_name'] == 'namek' and joins[0]['version'] == prepared['version']
    assert ['sudo', '-n', 'systemctl', 'restart', 'k3s-agent'] in w.calls
    assert out['generation_verified'] and not w.resources


def test_cli_registration_plan_apply_and_cleanup(tmp_path, monkeypatch, capsys):
    from infer_stack.cli import ManageCLI
    from infer_stack.cli import commands_kube_worker as cli
    w = Worker()
    monkeypatch.setattr(cli, 'KubeManager', lambda: w.manager)
    assert KubeNodeTestCLI.main(argv=['namek', '--expected-gpus=1', '--namespace=default', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['profile']['nodeSelector']['kubernetes.io/hostname'] == 'namek'
    assert not w.upgrades and not w.resources
    import requests
    monkeypatch.setattr(requests, 'post', w.post)
    assert KubeNodeTestCLI.main(argv=False, node='namek', expected_gpus=1, namespace='default', apply=True, base_url='http://kubeai/openai/v1') == 0
    assert ManageCLI.main(argv=['kube', 'node', 'test', 'namek', '--expected-gpus=1', '--namespace=default', '--apply', '--base-url=http://kubeai/openai/v1']) in (None, 0)
    assert not w.resources
    with pytest.raises(SystemExit, match='run-id'):
        KubeNodeTestCLI.main(argv=False, node='namek', cleanup=True)
    ManageCLI.main(argv=['help', 'tree'])
    tree = capsys.readouterr().out
    assert 'onboard' in tree and 'test' in tree
    for cls in (K3sOnboardCLI, KubeNodeTestCLI):
        with pytest.raises(SystemExit) as exc:
            cls.main(argv=['--help'])
        assert exc.value.code == 0
    assert DEFAULT_MODEL == 'HuggingFaceTB/SmolLM2-135M-Instruct'


def test_device_parser_rejects_missing_or_duplicate_identity():
    with pytest.raises(RuntimeError):
        devices('')
    with pytest.raises(RuntimeError):
        devices('GPU-a,3090,24576\nGPU-a,4080,16384')


def test_onboard_does_not_accept_existing_other_node_gpus(monkeypatch):
    from infer_stack.kube import inspect
    w = Worker()
    prepared = onboarding_plan(w.manager, w.manager, server='https://aiq-gpu:6443', node='namek')
    w.nodes[1]['status']['allocatable']['nvidia.com/gpu'] = '0'
    monkeypatch.setattr(k3s, 'join', lambda **kw: None)
    monkeypatch.setattr(inspect, 'inventory', lambda *a, **kw: {'device_plugin': [{'name': 'nvdp'}]})
    with pytest.raises(RuntimeError, match='Other nodes cannot satisfy'):
        onboard(w.manager, w.manager, prepared, token_file='private', namespace='default',
                release='kubeai', base_url='http://kubeai/openai/v1', run_id='badworker123', timeout=0, http=w.http)
    assert not w.resources and not w.generation_calls


def test_onboard_uses_shared_plugin_and_kubeai_installers(monkeypatch):
    from infer_stack.kube import inspect, operations
    w = Worker()
    prepared = onboarding_plan(w.manager, w.manager, server='https://aiq-gpu:6443', node='namek')
    plugin_calls, chart_calls = [], []
    monkeypatch.setattr(k3s, 'join', lambda **kw: None)
    monkeypatch.setattr(inspect, 'inventory', lambda *a, **kw: {
        'device_plugin': [], 'runtime_classes': ['nvidia'],
        'kubeai': {'crd': False, 'namespace_exists': False}})
    monkeypatch.setattr(w.manager, 'install_nvidia_device_plugin', lambda: plugin_calls.append(True))
    monkeypatch.setattr(operations, 'install', lambda *a, **kw: chart_calls.append(a))
    result = onboard(w.manager, w.manager, prepared, token_file='private', namespace='default',
                     release='kubeai', base_url='http://kubeai/openai/v1', run_id='setupworker1', http=w.http)
    assert plugin_calls == [True] and len(chart_calls) == 1
    assert result['generation_verified'] and not w.resources


def test_onboard_kwconf_plan_and_apply_from_root(tmp_path, monkeypatch, capsys):
    import requests

    from infer_stack.cli import ManageCLI
    from infer_stack.cli import commands_kube_worker as cli
    from infer_stack.kube import inspect
    w = Worker()
    token, admin = tmp_path / 'token', tmp_path / 'admin'
    for file in (token, admin):
        file.write_text('private credentials')
        file.chmod(0o600)
    monkeypatch.setattr(cli, 'KubeManager', lambda: w.manager)
    monkeypatch.setattr(cli, 'scoped_manager', lambda manager, path: manager)
    monkeypatch.setattr(requests, 'post', w.post)
    joins = []
    monkeypatch.setattr(k3s, 'join', lambda **kw: joins.append(kw))
    monkeypatch.setattr(inspect, 'inventory', lambda *a, **kw: {
        'device_plugin': [{'name': 'nvdp'}], 'kubeai': {'crd': True, 'namespace_exists': True}})
    args = ['kube', 'k3s', 'onboard', 'namek', '--server=https://aiq-gpu:6443',
            f'--kubeconfig={admin}', f'--token-file={token}', '--namespace=default',
            '--base-url=http://kubeai/openai/v1', '--json']
    ManageCLI.main(argv=args)
    assert json.loads(capsys.readouterr().out)['expected_gpus'] == 1
    assert not joins and not w.upgrades and not w.resources
    ManageCLI.main(argv=args + ['--apply'])
    out = capsys.readouterr()
    assert json.loads(out.out)['generation_verified']
    assert 'real generation verified' in out.err
    assert len(joins) == 1 and not w.resources
    token.chmod(0o644)
    with pytest.raises(SystemExit, match='private'):
        K3sOnboardCLI.main(argv=False, node='namek', server='https://aiq-gpu:6443',
                          kubeconfig=str(admin), token_file=str(token))


def test_real_helm_operation_preserves_values_and_chart_version(tmp_path, monkeypatch):
    w = Worker()
    from infer_stack.kube import manage
    monkeypatch.setattr(manage, 'data_root', lambda: tmp_path)
    del w.manager.install_kubeai  # use the actual shared operation, not our fixture shortcut
    original = w.manager.run
    values = {'resourceProfiles': {}, 'modelServers': {'vllm': {'images': {'nvidia-gpu': 'my-image'}}}}
    commands = []
    def run(args, **kw):
        commands.append(args)
        if args[:3] == ['helm', 'get', 'values']:
            return yaml.safe_dump(values)
        if args[:2] == ['helm', 'repo']:
            return ''
        if args[:3] == ['helm', 'upgrade', '--install']:
            from pathlib import Path
            document = yaml.safe_load(Path(args[args.index('-f') + 1]).read_text())
            assert document['modelServers'] == values['modelServers']
            w.profiles.update(document['resourceProfiles'])
            return ''
        return original(args, **kw)
    w.manager.run = run
    result = run_test(w)
    cmd = next(c for c in commands if c[:3] == ['helm', 'upgrade', '--install'])
    assert cmd[3:5] == ['kubeai', 'kubeai/kubeai']
    assert cmd[cmd.index('--version') + 1] == '0.23.2'
    assert cmd[cmd.index('-n') + 1] == 'default'
    assert result['generation_verified']


def test_targeted_acceptance_preserves_unrelated_managed_models():
    w = Worker()
    old = {'apiVersion': 'kubeai.org/v1', 'kind': 'Model',
           'metadata': {'name': 'important-production-model', 'labels': {'infer-stack/managed': 'true'}},
           'spec': {'resourceProfile': 'custom:1'}}
    w.resources[('Model', old['metadata']['name'])] = copy.deepcopy(old)
    run_test(w)
    assert w.resources == {('Model', old['metadata']['name']): old}


def test_gpu_probe_failure_is_fresh_and_cleans_own_resource():
    w = Worker()
    original = w.manager.run
    def run(args, **kw):
        data = original(args, **kw)
        if args[:5] == ['kubectl', '-n', 'default', 'get', 'pod']:
            doc = json.loads(data)
            if doc:
                doc['status'] = {'phase': 'Failed'}
            return json.dumps(doc)
        return data
    w.manager.run = run
    with pytest.raises(RuntimeError, match='Fresh GPU runtime probe failed'):
        run_test(w)
    assert not w.resources and not w.generation_calls


def test_join_bundle_private_refresh_and_interrupted_retry(tmp_path):
    original = {'current-context': 'default', 'contexts': [{'name': 'default', 'context': {'cluster': 'default'}}],
                'clusters': [{'name': 'default', 'cluster': {'server': 'https://127.0.0.1:6443',
                                                           'certificate-authority-data': 'public-CA'}}],
                'users': [{'name': 'default', 'user': {'client-key-data': 'ADMIN-PRIVATE-KEY'}}]}
    calls = []
    def run(args):
        calls.append(args)
        if 'is-active' in args:
            return ''
        return yaml.safe_dump(original) if args[-1] == str(k3s.K3S_KUBECONFIG) else 'SECRET-TOKEN'
    path = tmp_path / 'bundle'
    result = k3s.export_join_info(server='https://aiq-gpu:6443', directory=path, run=run)
    assert path.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in path.iterdir())
    assert (path / 'token').read_text().strip() == 'SECRET-TOKEN'
    assert yaml.safe_load((path / 'kubeconfig.yaml').read_text())['clusters'][0]['cluster']['server'] == result['server']
    assert original['clusters'][0]['cluster']['server'] == 'https://127.0.0.1:6443'
    assert 'SECRET-TOKEN' not in json.dumps(result) and 'ADMIN-PRIVATE-KEY' not in json.dumps(result)
    (path / 'token').unlink()  # interrupted write resumes from owned manifest
    k3s.export_join_info(server='https://aiq-gpu:6443', directory=path, run=run)
    assert (path / 'token').exists()
    original['clusters'][0]['cluster']['certificate-authority-data'] = 'different-CA'
    with pytest.raises(RuntimeError, match='different cluster'):
        k3s.export_join_info(server='https://aiq-gpu:6443', directory=path, run=run)


def test_join_bundle_refuses_unrelated_directory_and_agent(tmp_path):
    path = tmp_path / 'bundle'
    path.mkdir(mode=0o700)
    (path / 'important').write_text('preserve me')
    def run(args):
        if 'is-active' in args:
            return ''
        if args[-1] == str(k3s.K3S_NODE_TOKEN):
            return 'token'
        return yaml.safe_dump({'current-context': 'default', 'contexts': [{'name': 'default', 'context': {'cluster': 'default'}}],
                              'clusters': [{'name': 'default', 'cluster': {'certificate-authority-data': 'CA'}}]})
    with pytest.raises(RuntimeError, match='nonempty'):
        k3s.export_join_info(server='https://aiq-gpu:6443', directory=path, run=run)
    assert (path / 'important').read_text() == 'preserve me'
    def inactive(args):
        raise RuntimeError('inactive')
    with pytest.raises(RuntimeError, match='local K3s server'):
        k3s.export_join_info(server='https://aiq-gpu:6443', directory=path, run=inactive)


def test_join_bundle_kwconf_plan_reads_no_credentials(tmp_path, monkeypatch, capsys):
    from infer_stack.cli import ManageCLI
    monkeypatch.setattr(k3s, 'export_join_info', lambda **kw: pytest.fail('plan read credentials'))
    ManageCLI.main(argv=['kube', 'k3s', 'export', '--server=https://aiq-gpu:6443', f'--directory={tmp_path}', '--plan', '--apply'])
    assert 'No credentials read or written' in capsys.readouterr().out
