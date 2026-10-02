"""Efficient Kubernetes dashboard and deliberate lifecycle controls."""
import asyncio
import json
import threading
import time

import pytest

pytest.importorskip('textual')

from test_kube_readiness import Cluster, manager, node
from test_leasing_kubeai import FakeHttp, FakeKubectl
from test_tui import _ctx
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
)

from infer_stack.backends.kubeai import KubeaiBackend
from infer_stack.kube.manage import KubeManager
from infer_stack.kube.monitor import gpu_request, node_rows, pod_rows, snapshot
from infer_stack.leasing import Controller, Ledger, SqliteStore
from infer_stack.tui import InferStackTUI, _AddEndpointScreen, _ConfirmScreen


def context(tmp_path):
    kubectl = FakeKubectl()
    backend = KubeaiBackend(state_dir=tmp_path, namespace='default', run=kubectl,
                           http=FakeHttp(kubectl))
    _, catalog = _ctx()
    return Controller(Ledger(SqliteStore(':memory:')), backend), catalog


def test_snapshot_budget_and_independent_failure():
    calls = []
    def run(args):
        calls.append(args)
        if 'nodes' in args:
            return json.dumps({'items': [node()]})
        if 'pods' in args:
            raise RuntimeError('Forbidden all-namespaces pods')
        return json.dumps({'items': [{'metadata': {'name': 'q'}}]})
    report = snapshot(KubeManager(run=run), namespace='default')
    assert len(calls) == 3
    assert report['nodes'] and report['models']
    assert report['pods'] is None
    assert node_rows(report)[0][4] == '?'
    assert node_rows(report)[0][-1] == 'unknown'
    assert 'Forbidden' in report['errors']['pods']
    assert not any('helm' in c or 'nvidia-smi' in c for c in calls)


def test_gpu_requests_and_pod_diagnosis_include_other_namespace_usage():
    pod = {'metadata': {'name': 'other', 'namespace': 'other'},
           'spec': {'nodeName': 'gpu-a', 'containers': [{'resources': {'limits': {'nvidia.com/gpu': 2}}}],
                    'initContainers': [{'resources': {'requests': {'nvidia.com/gpu': 3}}}]},
           'status': {'phase': 'Running'}}
    assert gpu_request(pod) == 3
    report = {'namespace': 'default', 'nodes': [node()], 'pods': [pod]}
    assert node_rows(report)[0][4] == '3'
    assert pod_rows(report) == []
    pod['metadata']['namespace'] = 'default'
    pod['status']['containerStatuses'] = [{'ready': False, 'restartCount': 4,
                                           'state': {'waiting': {'reason': 'ImagePullBackOff'}}}]
    row = pod_rows(report)[0]
    assert row[:3] == ('default/other', 'gpu-a', 'ImagePullBackOff')
    assert row[4] == '4'
    pod['status']['phase'] = 'Succeeded'
    assert node_rows(report)[0][4] == '0'


def test_restartable_init_sidecars_count_toward_scheduled_request():
    def container(count):
        return {'resources': {'requests': {'nvidia.com/gpu': count}}}
    pod = {'spec': {'containers': [container(2)],
                    'initContainers': [dict(container(1), restartPolicy='Always'), container(4)]}}
    assert gpu_request(pod) == 5


def test_cluster_visible_only_cache_manual_refresh_and_partial_data(tmp_path):
    controller, catalog = context(tmp_path)
    cluster = Cluster()
    m = manager(cluster)
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None, kube_manager=m)
        async with app.run_test(size=(180, 55)) as pilot:
            await pilot.pause()
            assert len(cluster.calls) == 1 and 'models.kubeai.org' in cluster.calls[0]  # one global readiness read; no node/pod inventory
            app.query_one('#docker', Collapsible).collapsed = False
            app.query_one('#docker-tabs', TabbedContent).active = 'tab-cluster'
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.query_one('#cluster-nodes', DataTable).row_count == 1
            count = len(cluster.calls)
            assert count == 4  # one global Model read plus three Cluster reads
            app.action_refresh()
            await app.workers.wait_for_complete()
            assert len(cluster.calls) == count
            app.action_cluster_refresh()
            await app.workers.wait_for_complete()
            assert len(cluster.calls) == count + 3
            app._cluster_at = None
            app.query_one('#top', TabbedContent).active = 'tab-api'
            await pilot.pause()
            app.action_refresh()
            await app.workers.wait_for_complete()
            assert len(cluster.calls) == count + 3
            app.query_one('#top', TabbedContent).active = 'tab-dashboard'
            app.query_one('#docker', Collapsible).collapsed = True
            await pilot.pause()
            app.action_refresh()
            await app.workers.wait_for_complete()
            assert len(cluster.calls) == count + 3
            report = snapshot(KubeManager(run=lambda args: (_ for _ in ()).throw(RuntimeError('unreachable'))), namespace='default')
            app._receive_cluster(report)
            assert 'unreachable' in str(app.query_one('#cluster-summary', Static).render())
            assert '(unavailable)' in str(app.query_one('#cluster-nodes', DataTable).get_row_at(0))
    asyncio.run(scenario())


def test_slow_cluster_probe_never_overlaps_or_blocks_ui(tmp_path):
    controller, catalog = context(tmp_path)
    gate, entered = threading.Event(), threading.Event()
    calls = []
    def run(args):
        calls.append(args)
        entered.set()
        gate.wait(5)
        return '{"items": []}'
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None,
                            kube_manager=KubeManager(run=run))
        async with app.run_test(size=(180, 55)) as pilot:
            app.action_cluster_refresh()
            await pilot.pause(0.1)
            assert entered.is_set()
            for _ in range(4):
                app.action_cluster_refresh()
            await pilot.press('2')
            await pilot.pause()
            assert app.query_one('#top', TabbedContent).active == 'tab-api'
            assert len(calls) == 2  # one global Model read and one slow Cluster read
            gate.set()
            await app.workers.wait_for_complete()
            assert len(calls) == 4
    try:
        asyncio.run(scenario())
    finally:
        gate.set()


def test_node_control_previews_confirms_and_reuses_manager(tmp_path):
    from infer_stack.kube.manage import NodeLifecyclePlan
    controller, catalog = context(tmp_path)
    calls = []
    m = KubeManager(run=lambda args: '{"items": []}')
    m.node_lifecycle_plan = lambda name: calls.append(('plan', name)) or NodeLifecyclePlan(
        name, True, True, None, False, workload_pods=['default/q'])
    m.detach_node_for_compose = lambda name: calls.append(('detach', name))
    m.attach_node_from_compose = lambda name: calls.append(('attach', name))
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None, kube_manager=m)
        async with app.run_test(size=(180, 55)) as pilot:
            await pilot.pause()
            app._node_names = ['namek']
            app.query_one('#cluster-nodes', DataTable).add_row('namek', *['-'] * 8)
            app.action_node_control('detach')
            await app.workers.wait_for_complete()
            assert isinstance(app.screen, _ConfirmScreen)
            assert 'default/q' in app.screen._message
            assert 'emptyDir' in app.screen._message
            assert calls == [('plan', 'namek')]
            app.screen.dismiss(False)
            await pilot.pause()
            assert calls == [('plan', 'namek')]
            app.action_node_control('attach')
            await app.workers.wait_for_complete()
            assert isinstance(app.screen, _ConfirmScreen)
            app.screen.dismiss(True)
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert calls[-1] == ('attach', 'namek')
            assert any('infer-stack kube node attach namek --yes' in line for line in app._app_log_lines)
    asyncio.run(scenario())


def test_kube_control_without_gateway_render_confirms_down(tmp_path):
    controller, catalog = context(tmp_path)
    calls = []
    controller.apply_now = lambda: calls.append('apply') or type('Result', (), {'publication_pending': False})()
    controller.backend.down = lambda: calls.append('down')
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None,
                            kube_manager=manager(Cluster()))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._compose_target() is None
            app.action_compose_up()
            await app.workers.wait_for_complete()
            assert calls == ['apply']
            app.action_compose_down()
            assert isinstance(app.screen, _ConfirmScreen)
            assert 'managed KubeAI Models in default' in app.screen._message
            assert calls == ['apply']
            app.screen.dismiss(True)
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert calls == ['apply', 'down']
    asyncio.run(scenario())


def test_kube_endpoint_editor_uses_resource_profiles(tmp_path):
    controller, catalog = context(tmp_path)
    results = []
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None)
        async with app.run_test(size=(100, 45)) as pilot:
            screen = _AddEndpointScreen(['qc'], kubeai=True, name='q',
                                        entry={'engine': 'vllm', 'model': 'qc',
                                               'runtime': {'resource_profile': 'gpu-single-default:1'}})
            app.push_screen(screen, results.append)
            await pilot.pause()
            assert not list(screen.query('#e-gpu-pin'))
            assert not list(screen.query('#e-image'))
            assert not list(screen.query('#e-command'))
            assert any('within one serving deployment' in str(label.render()) for label in screen.query(Label))
            assert any('runtime.min_replicas' in str(label.render()) for label in screen.query(Static))
            screen.query_one('#e-env', Input).value = 'MODE=fast'
            assert screen.query_one('#e-engine', Select).value == 'vllm'
            screen.query_one('#e-resource-profile', Input).value = 'nvidia-rtx-3090:2'
            screen.on_button_pressed(Button.Pressed(Button(id='ok')))
            await pilot.pause()
    asyncio.run(scenario())
    entry = InferStackTUI._endpoint_entry(results[0])
    assert entry['runtime']['resource_profile'] == 'nvidia-rtx-3090:2'
    assert entry['runtime']['env'] == {'MODE': 'fast'}
    assert 'image' not in entry['runtime'] and 'command' not in entry['runtime']
    assert 'placement' not in entry


def test_visible_runtime_instances_share_cadence_and_cluster_pod_data(tmp_path):
    controller, catalog = context(tmp_path)
    calls = []
    controller.backend.instances = lambda: calls.append('instances') or []
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None,
                            kube_manager=manager(Cluster()))
        async with app.run_test(size=(180, 55)) as pilot:
            await pilot.pause()
            app.query_one('#docker', Collapsible).collapsed = False
            app.query_one('#docker-tabs', TabbedContent).active = 'tab-containers'
            await pilot.pause()
            await app.workers.wait_for_complete()
            app._refresh_now()
            app._refresh_now()
            assert calls == ['instances']
            report = {'nodes': [], 'models': [], 'errors': {}, 'namespace': 'default',
                      'sampled_at': time.monotonic(),
                      'pods': [{'metadata': {'name': 'q-pod', 'namespace': 'default',
                            'labels': {'infer-stack/managed': 'true', 'infer-stack/deployment': 'g'}},
                          'spec': {'containers': [{'name': 'vllm'}]},
                          'status': {'containerStatuses': [{'state': {'running': {'startedAt': 'now'}}}]}}]}
            app._receive_cluster(report)
            app._instances_at = None
            data = app._collect()
            assert [i.name for i in data['instances']] == ['q-pod']
            assert calls == ['instances']
            app._refresh_now()
            assert app.query_one('#ps', DataTable).row_count == 1
    asyncio.run(scenario())


def test_refresh_does_not_spawn_duplicate_slow_observation(tmp_path):
    controller, catalog = context(tmp_path)
    gate = threading.Event()
    calls = []
    def observe(args, **kw):
        calls.append(args)
        gate.wait(5)
        return '{"items": []}'
    controller.backend.run = observe
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            for _ in range(4):
                app.action_refresh()
            await pilot.press('2')
            await pilot.pause()
            assert len(calls) == 1
            assert app.query_one('#top', TabbedContent).active == 'tab-api'
            gate.set()
            await app.workers.wait_for_complete()
    try:
        asyncio.run(scenario())
    finally:
        gate.set()


def test_log_followers_pause_while_cluster_or_other_top_tab_is_visible(tmp_path):
    controller, catalog = context(tmp_path)
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None,
                            kube_manager=manager(Cluster()))
        async with app.run_test(size=(180, 55)) as pilot:
            await pilot.pause()
            calls = []
            app._restart_logs = lambda svc: calls.append('start')
            app._terminate_logs = lambda: calls.append('stop')
            app.query_one('#docker', Collapsible).collapsed = False
            await pilot.pause()
            assert 'start' in calls
            calls.clear()
            app.query_one('#docker-tabs', TabbedContent).active = 'tab-cluster'
            await pilot.pause()
            assert calls == ['stop']
            calls.clear()
            app._sync_log_services()
            app._sync_pane_state()
            assert calls == []
            app._write_log_lines(['apply output captured while logs paused'])
            app.query_one('#docker-tabs', TabbedContent).active = 'tab-logs'
            await pilot.pause()
            assert calls == ['start']
            assert 'apply output captured while logs paused' in app._log_lines
            calls.clear()
            app.query_one('#top', TabbedContent).active = 'tab-api'
            await pilot.pause()
            assert calls == ['stop']
    asyncio.run(scenario())


@pytest.mark.parametrize('status,expected', [
    ({'replicas': {'all': 2, 'ready': 1}}, (2, 1)),
    ({'replicas': {'all': 2, 'ready': 0}, 'readyReplicas': 2}, (2, 0)),
    ({'replicas': 2, 'readyReplicas': 1}, (2, 1)),
    ({'readyReplicas': 1}, (None, 1)),
    ({}, (None, None)),
])
def test_actual_kubeai_replica_counts_and_compatibility(status, expected):
    from infer_stack.backends.kubeai import model_replica_counts
    assert model_replica_counts({'status': status}) == expected


def lifecycle_report(gid='grp-q'):
    return {'nodes': [], 'pods': [], 'namespace': 'default', 'errors': {},
            'sampled_at': time.monotonic(), 'models': [{
                'metadata': {'name': 'qwen-coder', 'uid': 'model-1', 'generation': 1,
                             'labels': {'infer-stack/deployment': gid}},
                'status': {'replicas': {'all': 2, 'ready': 0}}}]}


def test_kubeai_lifecycle_evidence_and_namespace_scope():
    from infer_stack.kube.monitor import model_states
    r = lifecycle_report()
    assert model_states(r)['grp-q']['stage'] == 'declared'
    pod = {'metadata': {'name': 'vllm', 'namespace': 'default', 'uid': 'pod-1',
                       'labels': {'infer-stack/deployment': 'grp-q'}},
           'spec': {'nodeName': 'namek'}, 'status': {'phase': 'Pending'}}
    r['pods'] = [pod]
    assert model_states(r)['grp-q']['stage'] == 'scheduled'
    pod['status'] = {'phase': 'Running', 'containerStatuses': [
        {'containerID': 'container-1', 'state': {'running': {'startedAt': 'now'}}}]}
    assert model_states(r)['grp-q']['stage'] == 'pod running'
    r['models'][0]['status']['replicas']['ready'] = 1
    assert not model_states(r)['grp-q']['replica_ready']  # stale CR counter cannot override unready pods
    pod['status']['conditions'] = [{'type': 'Ready', 'status': 'True'}]
    assert model_states(r)['grp-q']['stage'] == 'replica ready'
    before = model_states(r)['grp-q']['identity']
    pod['status']['containerStatuses'][0]['restartCount'] = 1
    assert model_states(r)['grp-q']['identity'] != before
    pod['metadata']['namespace'] = 'unrelated'
    assert not model_states(r)['grp-q']['replica_ready']
    r['pods'] = None  # Model-only polling uses the CR replica count, not guessed pod facts
    assert model_states(r)['grp-q']['replica_ready']
    r['models'][0]['metadata']['deletionTimestamp'] = 'now'
    assert not model_states(r)['grp-q']['replica_ready']


def test_pending_model_never_offered_as_ready_then_real_generation_is_distinct(tmp_path):
    from test_leasing_kubeai import vllm
    controller, catalog = context(tmp_path)
    deployment = vllm('grp-q', served='qwen-coder')
    deployment.served = {'qwen-coder': {'served_model_name': 'qwen-coder', 'protocol': 'chat'}}
    controller.ledger.store.insert_deployment(deployment)
    requests = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'choices': [{'message': {'content': 'ok'}}]}
    class HTTP:
        def post(self, *args, **kw):
            requests.append((args, kw))
            return Response()
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None,
                            kube_manager=manager(Cluster()), http=HTTP())
        app._litellm = lambda: ('http://gateway/v1', 'key')
        async with app.run_test(size=(180, 55)) as pilot:
            await pilot.pause()
            r = lifecycle_report()
            app._receive_cluster(r)
            assert app._api_models_wanted == []
            assert app._kube_states['grp-q']['stage'] == 'declared'
            assert '0/2 ready' in str(app.query_one('#cluster-summary', Static).render())
            app.query_one('#top', TabbedContent).active = 'tab-api'
            await pilot.pause()
            assert app._ready_endpoints == []
            r['pods'] = None
            r['models'][0]['status']['replicas']['ready'] = 1
            app._receive_cluster(r)
            assert app._ready_endpoints == ['qwen-coder']
            assert app._kube_verified == {}
            assert '1/2 ready' in str(app.query_one('#cluster-summary', Static).render())
            assert 'generation unverified' in str(app.query_one('#api-model', Select)._options)
            assert requests == []  # no generation during passive refresh
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert 'qwen-coder' in app._kube_verified
            assert 'Last successful generation' in str(app.query_one('#api-readiness', Static).render())
            assert len(requests) == 1
            old_post = app._http_client().post
            def fail(*args, **kwargs):
                raise RuntimeError('upstream loading')
            app._http_client().post = fail
            app.action_api_test_all()
            await app.workers.wait_for_complete()
            assert not app._kube_verified
            app._http_client().post = old_post
            app.action_api_test_all()
            await app.workers.wait_for_complete()
            assert 'qwen-coder' in app._kube_verified
            old_token = app._kube_tokens['qwen-coder']
            r['models'][0]['metadata']['uid'] = 'replacement-model'
            app._receive_cluster(r)
            assert not app._kube_verified
            app._record_kube_generation('qwen-coder', old_token, True)
            assert not app._kube_verified  # late response cannot verify a replacement
            r['models'][0]['status']['replicas']['ready'] = 0
            app._receive_cluster(r)
            assert app._ready_endpoints == []
    asyncio.run(scenario())


@pytest.mark.parametrize('launch', [
    {'image': 'my-vllm:foo'}, {'command': ['custom-server']},
    {'serve_recipe': 'hyperqwen-3090-single'},
])
def test_kubeai_editor_refuses_existing_unsupported_launch(tmp_path, launch):
    controller, catalog = context(tmp_path)
    results, errors = [], []
    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999, proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            screen = _AddEndpointScreen(['qc'], kubeai=True, name='q',
                                        entry={'engine': 'vllm', 'model': 'qc', 'runtime': launch})
            screen._error = errors.append
            app.push_screen(screen, results.append)
            await pilot.pause()
            screen.on_button_pressed(Button.Pressed(Button(id='ok')))
            await pilot.pause()
            assert app.screen is screen
            assert results == []
            assert 'does not support runtime.image/command' in errors[0]
    asyncio.run(scenario())
