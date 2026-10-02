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
            assert not cluster.calls  # no kube inventory at mount/hidden pane
            app.query_one('#docker', Collapsible).collapsed = False
            app.query_one('#docker-tabs', TabbedContent).active = 'tab-cluster'
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.query_one('#cluster-nodes', DataTable).row_count == 1
            count = len(cluster.calls)
            assert count == 3
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
            assert len(calls) == 1
            gate.set()
            await app.workers.wait_for_complete()
            assert len(calls) == 3
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
            assert screen.query_one('#e-engine', Select).value == 'vllm'
            screen.query_one('#e-resource-profile', Input).value = 'nvidia-rtx-3090:2'
            screen.on_button_pressed(Button.Pressed(Button(id='ok')))
            await pilot.pause()
    asyncio.run(scenario())
    entry = InferStackTUI._endpoint_entry(results[0])
    assert entry['runtime']['resource_profile'] == 'nvidia-rtx-3090:2'
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
    def observe():
        calls.append('observe')
        gate.wait(5)
        return set()
    controller.backend.observe = observe
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
