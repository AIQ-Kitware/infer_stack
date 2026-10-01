from __future__ import annotations

import json
from pathlib import Path

import yaml

from infer_stack.kube.manage import KubeManager


GPU_PRODUCT = 'nvidia.com/gpu.product'
GPU_MEMORY = 'nvidia.com/gpu.memory'
GPU_RESOURCE = 'nvidia.com/gpu'


def node(
    name='gpu-a', *, product='RTX-TEST', memory='24576', count='1',
    schedulable=True, detached=False, control_plane=False,
):
    labels = {}
    if product is not None:
        labels[GPU_PRODUCT] = product
    if memory is not None:
        labels[GPU_MEMORY] = memory
    if control_plane:
        labels['node-role.kubernetes.io/control-plane'] = 'true'
    allocatable = {}
    if count is not None:
        allocatable[GPU_RESOURCE] = count
    annotations = {}
    if detached:
        annotations['infer-stack.io/compose-detached'] = 'true'
    return {
        'metadata': {'name': name, 'labels': labels, 'annotations': annotations},
        'spec': {'unschedulable': not schedulable},
        'status': {
            'conditions': [{'type': 'Ready', 'status': 'True'}],
            'allocatable': allocatable,
        },
    }


def pod(name, *, namespace='default', owner='ReplicaSet', mirror=False):
    meta = {'name': name, 'namespace': namespace, 'annotations': {}}
    if owner is not None:
        meta['ownerReferences'] = [{'kind': owner, 'name': f'{name}-owner'}]
    if mirror:
        meta['annotations']['kubernetes.io/config.mirror'] = 'mirror-hash'
    return {'metadata': meta}


class FakeCluster:
    def __init__(
        self,
        *,
        nodes=None,
        runtime_class=True,
        crd=True,
        namespace=True,
        service=True,
        releases=None,
        values=None,
        pods=None,
    ):
        self.nodes = nodes if nodes is not None else [node()]
        self.pods = pods if pods is not None else {}
        self.runtime_class = runtime_class
        self.crd = crd
        self.namespace = namespace
        self.service = service
        self.releases = releases if releases is not None else []
        self.values = values or {}
        self.applied_values_documents = []
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if args == ['kubectl', 'config', 'current-context']:
            return 'test-cluster\n'
        if args[:4] == ['kubectl', 'version', '--client=false', '-o']:
            return '{}'
        if args == ['kubectl', 'get', 'nodes', '-o', 'json']:
            return json.dumps({'items': self.nodes})
        if len(args) == 6 and args[:3] == ['kubectl', 'get', 'node'] and args[4:] == ['-o', 'json']:
            wanted = args[3]
            for item in self.nodes:
                if item.get('metadata', {}).get('name') == wanted:
                    return json.dumps(item)
            raise RuntimeError('not found')
        if (
            len(args) == 8
            and args[:4] == ['kubectl', 'get', 'pods', '-A']
            and args[4] == '--field-selector'
            and args[6:] == ['-o', 'json']
        ):
            wanted = args[5].split('=', 1)[1]
            return json.dumps({'items': self.pods.get(wanted, [])})
        if args[:3] == ['kubectl', 'annotate', 'node']:
            wanted = args[3]
            annotation = args[4]
            for item in self.nodes:
                if item.get('metadata', {}).get('name') != wanted:
                    continue
                annotations = item.setdefault('metadata', {}).setdefault('annotations', {})
                if annotation.endswith('-'):
                    annotations.pop(annotation[:-1], None)
                else:
                    key, value = annotation.split('=', 1)
                    annotations[key] = value
                return f'node/{wanted} annotated\n'
            raise RuntimeError('not found')
        if args[:2] == ['kubectl', 'cordon']:
            wanted = args[2]
            for item in self.nodes:
                if item.get('metadata', {}).get('name') == wanted:
                    item.setdefault('spec', {})['unschedulable'] = True
                    return f'node/{wanted} cordoned\n'
            raise RuntimeError('not found')
        if args[:2] == ['kubectl', 'drain']:
            wanted = args[2]
            self.pods[wanted] = [
                item for item in self.pods.get(wanted, [])
                if KubeManager._pod_retained_by_drain(item)
            ]
            return f'node/{wanted} drained\n'
        if args[:3] == ['kubectl', 'wait', '--for=condition=Ready']:
            return 'condition met\n'
        if args[:2] == ['kubectl', 'uncordon']:
            wanted = args[2]
            for item in self.nodes:
                if item.get('metadata', {}).get('name') == wanted:
                    item.setdefault('spec', {})['unschedulable'] = False
                    return f'node/{wanted} uncordoned\n'
            raise RuntimeError('not found')
        if args[:4] == ['kubectl', 'get', 'runtimeclass', 'nvidia']:
            if not self.runtime_class:
                raise RuntimeError('not found')
            return 'runtimeclass.node.k8s.io/nvidia\n'
        if args[:4] == ['kubectl', 'get', 'crd', 'models.kubeai.org']:
            if not self.crd:
                raise RuntimeError('not found')
            return 'customresourcedefinition.apiextensions.k8s.io/models.kubeai.org\n'
        if args[:3] == ['kubectl', 'get', 'namespace']:
            if not self.namespace:
                raise RuntimeError('not found')
            return f'namespace/{args[3]}\n'
        if len(args) >= 7 and args[:3] == ['kubectl', '-n', 'kubeai'] and args[3:6] == ['get', 'service', 'kubeai']:
            if not self.service:
                raise RuntimeError('not found')
            return 'service/kubeai\n'
        if args == ['helm', 'list', '-A', '-o', 'json']:
            return json.dumps(self.releases)
        if args[:4] == ['helm', 'get', 'values', 'kubeai']:
            return yaml.safe_dump(self.values, sort_keys=False)
        if args[:3] == ['helm', 'repo', 'add']:
            return ''
        if args == ['helm', 'repo', 'update']:
            return ''
        if args[:3] == ['helm', 'upgrade', '--install']:
            # Reflect successful KubeAI/NVIDIA installs into the fake cluster.
            if len(args) > 3 and args[3] == 'nvdp':
                self.runtime_class = True
                for item in self.nodes:
                    alloc = item.setdefault('status', {}).setdefault('allocatable', {})
                    alloc.setdefault(GPU_RESOURCE, '1')
                    labels = item.setdefault('metadata', {}).setdefault('labels', {})
                    labels.setdefault(GPU_PRODUCT, 'RTX-TEST')
                    labels.setdefault(GPU_MEMORY, '24576')
            if len(args) > 3 and args[3] == 'kubeai':
                self.crd = self.namespace = self.service = True
                chart_version = '0.22.0'
                if '--version' in args:
                    chart_version = args[args.index('--version') + 1]
                self.releases = [{
                    'name': 'kubeai', 'namespace': 'kubeai',
                    'chart': f'kubeai-{chart_version}',
                }]
                value_docs = []
                for idx, arg in enumerate(args):
                    if arg == '-f':
                        value_docs.append(
                            yaml.safe_load(Path(args[idx + 1]).read_text()) or {}
                        )
                self.applied_values_documents = value_docs
                if value_docs:
                    merged = {}
                    for doc in value_docs:
                        merged = KubeManager._deep_merge(merged, doc)
                    self.values = merged
            return ''
        raise AssertionError(f'unhandled fake command: {args!r}')


def manager_for(fake):
    manager = KubeManager(run=fake)
    manager.command_exists = lambda name: name in {'kubectl', 'helm'}
    return manager


def test_setup_accepts_external_gpu_and_kubeai_without_runtimeclass():
    fake = FakeCluster(runtime_class=False, releases=[])
    manager = manager_for(fake)
    plan = manager.plan_setup()
    assert not plan.failed
    assert plan.context == 'test-cluster'
    assert not any('install KubeAI' in action for action in plan.actions)
    assert not any('NVIDIA device plugin' in action for action in plan.actions)
    (profile,) = plan.resource_profiles.values()
    assert 'runtimeClassName' not in profile
    assert any('externally managed' in check.detail for check in plan.checks)


def test_setup_turns_missing_gpu_plugin_and_kubeai_into_actions():
    fake = FakeCluster(
        nodes=[node(product=None, memory=None, count=None)],
        runtime_class=True,
        crd=False,
        namespace=False,
        service=False,
        releases=[],
    )
    manager = manager_for(fake)
    plan = manager.plan_setup()
    assert not plan.failed
    assert any('NVIDIA device plugin' in action for action in plan.actions)
    assert any('install KubeAI release' in action for action in plan.actions)
    needs = [check for check in plan.checks if not check.ok]
    assert needs and all(check.level == 'action' for check in needs)


def test_setup_is_noop_when_managed_profiles_are_already_present():
    profile = {
        'requests': {GPU_RESOURCE: '1'},
        'limits': {GPU_RESOURCE: '1'},
        'nodeSelector': {GPU_PRODUCT: 'RTX-TEST'},
    }
    fake = FakeCluster(
        releases=[{'name': 'kubeai', 'namespace': 'kubeai', 'chart': 'kubeai-0.22.0'}],
        values={'resourceProfiles': {'nvidia-rtx-test': profile}},
    )
    manager = manager_for(fake)
    plan = manager.plan_setup()
    assert not plan.failed
    assert plan.actions == []


def test_apply_preserves_custom_profile_and_existing_chart_version(tmp_path, monkeypatch):
    custom = {
        'runtimeClassName': 'custom-runtime',
        'requests': {GPU_RESOURCE: '1'},
        'limits': {GPU_RESOURCE: '1'},
        'nodeSelector': {GPU_PRODUCT: 'RTX-A'},
    }
    fake = FakeCluster(
        nodes=[
            node('a', product='RTX-A', memory='24576'),
            node('b', product='RTX-B', memory='49152'),
        ],
        releases=[{'name': 'kubeai', 'namespace': 'kubeai', 'chart': 'kubeai-0.22.0'}],
        values={'resourceProfiles': {'nvidia-rtx-a': custom}},
    )
    manager = manager_for(fake)
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    fresh, values_path = manager.apply_setup()
    assert not fresh.failed
    assert values_path == tmp_path / 'generated' / 'kube' / 'kubeai-values.yaml'
    values = yaml.safe_load(values_path.read_text())
    assert values['resourceProfiles']['nvidia-rtx-a'] == custom
    assert 'nvidia-rtx-b' in values['resourceProfiles']
    helm_upgrade = [
        args for args, _ in fake.calls
        if args[:4] == ['helm', 'upgrade', '--install', 'kubeai']
    ][-1]
    assert helm_upgrade[helm_upgrade.index('--version') + 1] == '0.22.0'


def test_k3s_join_keeps_token_out_of_argv(tmp_path, monkeypatch):
    from infer_stack.kube import k3s

    token = 'K10-secret-token::server:abc123'
    token_file = tmp_path / 'token'
    token_file.write_text(token)
    calls = []
    active_checks = 0

    def fake_run(args, **kwargs):
        nonlocal active_checks
        calls.append((list(args), dict(kwargs)))
        if args[:2] == ['curl', '-sfL']:
            return '# installer\n'
        if args[:3] == ['systemctl', 'is-active', '--quiet']:
            active_checks += 1
            if active_checks == 1:
                raise RuntimeError('inactive')
            return 'active\n'
        return ''

    monkeypatch.setattr(k3s, '_require_local_tool', lambda name: None)
    k3s.join(
        server='https://server:6443',
        token_file=token_file,
        node_name='worker-b',
        version='v1.34.3+k3s1',
        run=fake_run,
    )
    assert all(token not in ' '.join(args) for args, _ in calls)
    install = [item for item in calls if item[0] and item[0][0] == 'sudo'][0]
    assert install[1]['env']['K3S_TOKEN'] == token
    assert 'K3S_TOKEN' in install[0][1]


def test_missing_helm_is_required_when_setup_has_managed_actions():
    fake = FakeCluster(
        nodes=[node(product=None, memory=None, count=None)],
        runtime_class=True,
        # KubeAI itself is external/healthy, but NVIDIA reconciliation needs Helm.
        crd=True,
        namespace=True,
        service=True,
        releases=[],
    )
    manager = KubeManager(run=fake)
    manager.command_exists = lambda name: name == 'kubectl'
    plan = manager.plan_setup()
    assert plan.failed
    helm = [check for check in plan.checks if check.name == 'helm available'][0]
    assert not helm.ok
    assert helm.level == 'required'
    assert any('NVIDIA device plugin' in action for action in plan.actions)


def test_gpu_none_does_not_generate_or_reconcile_gpu_profiles():
    fake = FakeCluster(
        releases=[{'name': 'kubeai', 'namespace': 'kubeai', 'chart': 'kubeai-0.22.0'}],
        values={},
    )
    manager = manager_for(fake)
    plan = manager.plan_setup(gpu='none')
    assert not plan.failed
    assert plan.resource_profiles == {}
    assert not any('resource profile' in action for action in plan.actions)


def test_apply_fresh_cluster_installs_gpu_then_kubeai(tmp_path, monkeypatch):
    fake = FakeCluster(
        nodes=[node(product=None, memory=None, count=None)],
        runtime_class=True,
        crd=False,
        namespace=False,
        service=False,
        releases=[],
    )
    manager = manager_for(fake)
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    fresh, values_path = manager.apply_setup(kubeai_version='0.22.0')
    assert not fresh.failed
    assert fresh.actions == []
    assert values_path is not None
    assert 'nvidia-rtx-test' in yaml.safe_load(values_path.read_text())['resourceProfiles']
    upgrades = [
        args for args, _ in fake.calls
        if args[:3] == ['helm', 'upgrade', '--install']
    ]
    assert upgrades[0][3] == 'nvdp'
    assert upgrades[1][3] == 'kubeai'


def test_existing_helm_secrets_are_preserved_but_not_persisted(tmp_path, monkeypatch):
    fake = FakeCluster(
        nodes=[
            node('a', product='RTX-A', memory='24576'),
            node('b', product='RTX-B', memory='49152'),
        ],
        releases=[{'name': 'kubeai', 'namespace': 'kubeai', 'chart': 'kubeai-0.22.0'}],
        values={
            'secrets': {'huggingface': {'token': 'existing-secret'}},
            'resourceProfiles': {
                'nvidia-rtx-a': {
                    'nodeSelector': {GPU_PRODUCT: 'RTX-A'},
                    'requests': {GPU_RESOURCE: '1'},
                    'limits': {GPU_RESOURCE: '1'},
                },
            },
        },
    )
    manager = manager_for(fake)
    monkeypatch.setenv('INFER_STACK_DATA_DIR', str(tmp_path))
    fresh, values_path = manager.apply_setup()
    assert not fresh.failed
    persisted = yaml.safe_load(values_path.read_text())
    assert 'secrets' not in persisted
    assert len(fake.applied_values_documents) == 2
    assert fake.applied_values_documents[1] == {
        'secrets': {'huggingface': {'token': 'existing-secret'}}
    }
    helm_upgrade = [
        args for args, _ in fake.calls
        if args[:4] == ['helm', 'upgrade', '--install', 'kubeai']
    ][-1]
    value_paths = [
        Path(helm_upgrade[idx + 1])
        for idx, arg in enumerate(helm_upgrade) if arg == '-f'
    ]
    assert value_paths[0] == values_path
    assert not value_paths[1].exists()


def test_k3s_bootstrap_existing_server_does_not_fetch_installer(monkeypatch):
    from infer_stack.kube import k3s

    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), dict(kwargs)))
        if args == ['systemctl', 'is-active', '--quiet', 'k3s-agent']:
            raise RuntimeError('inactive')
        if args == ['systemctl', 'is-active', '--quiet', 'k3s']:
            return 'active\n'
        if args == ['k3s', '--version']:
            return 'k3s version v1.34.3+k3s1 (test)\n'
        return ''

    monkeypatch.setattr(k3s, '_require_local_tool', lambda name: None)
    monkeypatch.setattr(k3s, '_ensure_helm', lambda run: None)
    monkeypatch.setattr(k3s, '_ensure_default_kubeconfig_link', lambda: True)
    ready = k3s.bootstrap(version='v1.34.3+k3s1', run=fake_run)
    assert ready is True
    assert not any(args and args[0] == 'curl' for args, _ in calls)
    assert not any('restart' in args for args, _ in calls)
    assert any(
        args[:3] == ['kubectl', '--kubeconfig', str(k3s.K3S_KUBECONFIG)]
        for args, _ in calls
    )


def test_generic_setup_missing_kubectl_hint_is_distribution_neutral():
    manager = KubeManager(run=lambda args, **kwargs: '')
    manager.command_exists = lambda name: False
    plan = manager.plan_setup()
    assert plan.failed
    details = '\n'.join(check.detail for check in plan.checks)
    assert 'k3s' not in details.lower()
    assert 'distribution-specific' in details


def test_generic_setup_runtime_hint_is_distribution_neutral():
    fake = FakeCluster(
        nodes=[node(product=None, memory=None, count=None)],
        runtime_class=False,
        crd=True,
        namespace=True,
        service=True,
        releases=[],
    )
    manager = manager_for(fake)
    plan = manager.plan_setup(gpu='nvidia')
    assert plan.failed
    details = '\n'.join(check.detail for check in plan.checks)
    assert 'k3s' not in details.lower()
    assert 'cluster/distribution' in details


def test_node_detach_is_reversible_and_keeps_daemonsets():
    fake = FakeCluster(
        nodes=[node('gpu-a', control_plane=True)],
        pods={
            'gpu-a': [
                pod('model', namespace='kubeai'),
                pod('nvidia-plugin', namespace='nvidia-device-plugin', owner='DaemonSet'),
                pod('kube-apiserver', namespace='kube-system', mirror=True),
            ],
        },
    )
    manager = manager_for(fake)
    before = manager.node_lifecycle_plan('gpu-a')
    assert before.schedulable
    assert before.control_plane
    assert before.workload_pods == ['kubeai/model']
    assert sorted(before.retained_pods) == [
        'kube-system/kube-apiserver',
        'nvidia-device-plugin/nvidia-plugin',
    ]

    detached = manager.detach_node_for_compose('gpu-a', timeout_seconds=42)
    assert detached.detached_for_compose
    assert not detached.schedulable
    assert detached.workload_pods == []
    assert any(
        args == [
            'kubectl', 'drain', 'gpu-a', '--ignore-daemonsets',
            '--delete-emptydir-data', '--timeout=42s',
        ]
        for args, _ in fake.calls
    )
    assert not any(
        args[:2] == ['systemctl', 'stop'] or args[:2] == ['sudo', 'systemctl']
        for args, _ in fake.calls
    )

    attached = manager.attach_node_from_compose('gpu-a', timeout_seconds=19)
    assert attached.schedulable
    assert not attached.detached_for_compose
    assert any(args == ['kubectl', 'uncordon', 'gpu-a'] for args, _ in fake.calls)


def test_node_detach_refuses_unmanaged_pods_instead_of_force_deleting():
    fake = FakeCluster(
        nodes=[node('gpu-a')],
        pods={'gpu-a': [pod('manual-debug', owner=None)]},
    )
    manager = manager_for(fake)
    plan = manager.node_lifecycle_plan('gpu-a')
    assert plan.unmanaged_pods == ['default/manual-debug']
    assert not plan.can_detach
    try:
        manager.detach_node_for_compose('gpu-a')
    except RuntimeError as ex:
        assert 'without --force' in str(ex)
    else:
        raise AssertionError('expected unmanaged pod safety refusal')
    assert not any(args[:2] == ['kubectl', 'drain'] for args, _ in fake.calls)


def test_node_attach_refuses_to_undo_foreign_cordon():
    fake = FakeCluster(nodes=[node('gpu-a', schedulable=False, detached=False)])
    manager = manager_for(fake)
    try:
        manager.attach_node_from_compose('gpu-a')
    except RuntimeError as ex:
        assert 'operator-owned maintenance cordon' in str(ex)
    else:
        raise AssertionError('expected foreign cordon safety refusal')
    assert not any(args[:2] == ['kubectl', 'uncordon'] for args, _ in fake.calls)


def test_interrupted_detach_marker_can_be_retried():
    interrupted = node('gpu-a', schedulable=False)
    interrupted['metadata']['annotations']['infer-stack.io/compose-detached'] = 'requested'
    fake = FakeCluster(
        nodes=[interrupted],
        pods={'gpu-a': [pod('model', namespace='kubeai')]},
    )
    manager = manager_for(fake)
    before = manager.node_lifecycle_plan('gpu-a')
    assert before.detach_owned
    assert not before.detached_for_compose
    assert before.workload_pods == ['kubeai/model']

    after = manager.detach_node_for_compose('gpu-a')
    assert after.detached_for_compose
    assert after.workload_pods == []


def test_node_rows_show_compose_detach_state():
    rows = KubeManager.node_rows([
        node('a'),
        node('b', schedulable=False, detached=True),
    ])
    assert rows[0]['schedulable'] is True
    assert rows[0]['detached_for_compose'] is False
    assert rows[1]['schedulable'] is False
    assert rows[1]['detached_for_compose'] is True


def test_detach_marker_on_schedulable_node_is_detectable_conflict():
    row = KubeManager.node_rows([
        node('gpu-a', schedulable=True, detached=True),
    ])[0]
    assert row['schedulable'] is True
    assert row['detached_for_compose'] is True
