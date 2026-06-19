"""Headless tests for the optional Textual TUI (skipped if textual absent)."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip('textual')

CATALOG = {
    'models': {'qc': {'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct'}},
    'endpoints': {
        'qwen-coder': {'engine': 'vllm', 'model': 'qc'},
        'qwen-fast': {'engine': 'vllm', 'model': 'qc'},
    },
}


def _ctx():
    from infer_stack.leasing import (
        Catalog,
        Controller,
        Ledger,
        NullBackend,
        SqliteStore,
    )

    catalog = Catalog.from_dict(CATALOG)
    controller = Controller(Ledger(SqliteStore(':memory:')), NullBackend())
    return controller, catalog


class _FakeProc:
    """Stands in for a `docker compose logs -f` process."""

    def __init__(self, lines):
        self.stdout = iter(lines)
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _run(scenario):
    asyncio.run(scenario())


def test_tui_panes_list_catalog_leases_and_deployments():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    controller.acquire('alice', catalog.resolve_names(['qwen-coder']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            eps = app.query_one('#endpoints', DataTable)
            assert eps.row_count == 2                 # both catalog endpoints
            assert app._endpoint_names == ['qwen-coder', 'qwen-fast']
            assert app.query_one('#models', DataTable).row_count == 1
            assert app.query_one('#leases', DataTable).row_count == 1
            assert app.query_one('#deployments', DataTable).row_count == 1

    _run(scenario)


def test_tui_relationship_columns_link_lease_to_deployment():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    out = controller.acquire('alice', catalog.resolve_names(['qwen-coder']))
    gid = out.lease.deployment_ids[0]

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            leases = app.query_one('#leases', DataTable)
            deployments = app.query_one('#deployments', DataTable)
            assert 'deployment' in [str(c.label) for c in leases.columns.values()]
            glabels = [str(c.label) for c in deployments.columns.values()]
            assert 'leases' in glabels and 'held by' in glabels
            # the lease row names the deployment id; the deployment row counts
            # the lease (1) and shows the owner — the many-to-one join.
            assert gid in leases.get_row_at(0)
            assert '1' in deployments.get_row_at(0)
            assert 'alice' in deployments.get_row_at(0)
            # selecting a lease explains the link in the status bar
            app.query_one('#leases').focus()
            app.query_one('#leases', DataTable).move_cursor(row=0)
            await pilot.pause()
            status = str(app.query_one('#status').render())
            assert 'deployment' in status

    _run(scenario)


def test_tui_serve_from_catalog_creates_a_lease():
    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#endpoints').focus()
            await pilot.press('s')                    # serve the selected endpoint
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert len(leases) == 1
    assert leases[0].state == LeaseState.ACTIVE
    assert 'qwen-coder' in leases[0].endpoints      # first row, sorted


def test_tui_panes_are_keyboard_resizable():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            w0, h0 = app._sidebar_w, app._log_h
            await pilot.press('right_square_bracket')   # wider sidebar
            await pilot.press('minus')                  # shorter logs
            await pilot.pause()
            assert app._sidebar_w == w0 + 4
            assert app._log_h == h0 - 2

    _run(scenario)


def test_tui_docker_pane_has_logs_and_containers_tabs():
    from textual.widgets import Collapsible, DataTable, TabbedContent

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # docker is a collapsible pane with Logs + Containers tabs; system
            # and api are now their own (collapsed) panes.
            tabs = app.query_one('#docker-tabs', TabbedContent)
            assert {p.id for p in tabs.query('TabPane')} == {
                'tab-logs', 'tab-containers'
            }
            assert app.query_one('#docker', Collapsible)
            assert app.query_one('#system', Collapsible).collapsed
            assert app.query_one('#api', Collapsible).collapsed
            # the containers ps view carries the docker-ps columns
            ps = app.query_one('#ps', DataTable)
            labels = [str(c.label) for c in ps.columns.values()]
            assert 'status (uptime)' in labels
            assert 'created' in labels
            assert 'container id' in labels

    _run(scenario)


def test_tui_panes_drag_resize():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            w0, h0 = app._sidebar_w, app._log_h
            app._drag_sidebar(6)            # pull the vertical splitter right
            app._drag_logs(3)               # pull the horizontal splitter down
            assert app._sidebar_w == w0 + 6
            assert app._log_h == h0 - 3     # down = shorter logs

    _run(scenario)


def test_tui_dividers_have_a_grab_area():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # A 0-size divider can't be grabbed; both must span their cross-axis.
            assert app.query_one('#vsplit').region.height > 1
            assert app.query_one('#hsplit').region.width > 1

    _run(scenario)


def test_tui_drag_resizes_sidebar():
    from textual import events

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            div = app.query_one('#vsplit')
            w0 = app._sidebar_w
            await pilot.mouse_down('#vsplit')
            assert app.mouse_captured is div          # the bar grabbed the mouse
            div.post_message(events.MouseMove(
                widget=div, x=0, y=0, delta_x=5, delta_y=0, button=0,
                shift=False, meta=False, ctrl=False,
                screen_x=div.region.x + 5, screen_y=div.region.y,
            ))
            await pilot.pause()
            await pilot.mouse_up('#vsplit')
            assert app._sidebar_w == w0 + 5

    _run(scenario)


def test_tui_pane_scoped_action_buttons():
    from textual.widgets import Button

    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Serve lives under the catalog; release/evict under their tables.
            assert app.query_one('#btn-serve', Button)
            assert app.query_one('#btn-release', Button)
            assert app.query_one('#btn-evict', Button)
            app.query_one('#endpoints').move_cursor(row=0)
            await pilot.click('#btn-serve')
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert len(leases) == 1 and leases[0].state == LeaseState.ACTIVE


def test_tui_add_model_wizard_writes_catalog(tmp_path):
    import yaml

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    catalog_path = tmp_path / 'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump(CATALOG))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None,
                            catalog_path=str(catalog_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_add_model({'name': 'newmod',
                               'source': 'hf://org/NewModel'})
            await pilot.pause()
            assert 'newmod' in app.catalog.models        # reloaded in memory

    _run(scenario)
    on_disk = yaml.safe_load(catalog_path.read_text())
    assert on_disk['models']['newmod']['source'] == 'hf://org/NewModel'


def test_tui_endpoint_entry_builds_runtime_from_advanced_params():
    from infer_stack.tui import InferStackTUI

    entry = InferStackTUI._endpoint_entry({
        'model': 'qc', 'engine': 'vllm',
        'tensor_parallel': 2, 'max_model_len': 8192, 'gpu_mem': 0.4,
        'extra_args': '--dtype=half --enforce-eager', 'reclaim': 'keep-warm',
    })
    assert entry['runtime']['tensor_parallel_size'] == 2
    assert entry['runtime']['max_model_len'] == 8192
    assert entry['runtime']['gpu_memory_utilization'] == 0.4
    assert entry['runtime']['extra_args'] == ['--dtype=half', '--enforce-eager']
    assert entry['reclaim'] == {'policy': 'keep-warm'}


def test_tui_add_endpoint_writes_advanced_params(tmp_path):
    import yaml

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    catalog_path = tmp_path / 'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump(CATALOG))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None,
                            catalog_path=str(catalog_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_add_endpoint({
                'name': 'big', 'model': 'qc', 'engine': 'vllm',
                'tensor_parallel': 2, 'max_model_len': None, 'gpu_mem': None,
                'extra_args': '', 'reclaim': '',
            })
            await pilot.pause()

    _run(scenario)
    on_disk = yaml.safe_load(catalog_path.read_text())
    assert on_disk['endpoints']['big']['runtime']['tensor_parallel_size'] == 2


def test_tui_edit_blocked_while_served(tmp_path):
    import yaml
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    catalog_path = tmp_path / 'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump(CATALOG))
    controller.acquire('me', catalog.resolve_names(['qwen-coder']))  # serve it

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None,
                            catalog_path=str(catalog_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#endpoints', DataTable).move_cursor(row=0)  # qwen-coder
            app.action_edit_endpoint()
            await pilot.pause()
            assert 'served' in str(app.query_one('#status').render())

    _run(scenario)


def test_tui_remove_endpoint_writes_catalog(tmp_path):
    import yaml

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    catalog_path = tmp_path / 'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump(CATALOG))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None,
                            catalog_path=str(catalog_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_remove('endpoints', 'qwen-fast')   # confirm bypassed
            await pilot.pause()
            assert 'qwen-fast' not in app.catalog.endpoints

    _run(scenario)
    on_disk = yaml.safe_load(catalog_path.read_text())
    assert 'qwen-fast' not in on_disk['endpoints']


def test_tui_empty_catalog_shows_suggest_hint():
    from textual.widgets import Static

    from infer_stack.leasing import (
        Catalog,
        Controller,
        Ledger,
        NullBackend,
        SqliteStore,
    )
    from infer_stack.tui import InferStackTUI

    controller = Controller(Ledger(SqliteStore(':memory:')), NullBackend())
    catalog = Catalog.from_dict({'models': {}, 'endpoints': {}})

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            help_text = str(app.query_one('#catalog-help', Static).render())
            assert 'suggest' in help_text.lower()

    _run(scenario)


def test_tui_ps_rows_parse_status_created_and_id():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    sample = json.dumps([
        {'Service': 'litellm', 'Status': 'Up 3 minutes', 'State': 'running',
         'CreatedAt': '2026-06-19 00:00:00 -0400', 'ID': 'abcdef1234567890',
         'Publishers': [{'PublishedPort': 14042, 'TargetPort': 4000}]},
    ])

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # stub the backend's compose seam
            backend = controller.backend
            import tempfile
            f = tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False)
            f.write('services: {}\n')
            f.close()
            backend.compose_file = f.name
            backend.project = 'infer-stack'
            backend.run = lambda args: sample
            rows = app._compose_ps_rows()
            assert rows[0]['status'] == 'Up 3 minutes'
            assert rows[0]['created'].startswith('2026-06-19')
            assert rows[0]['id'] == 'abcdef123456'        # truncated to 12
            assert '14042->4000' in rows[0]['ports']

    _run(scenario)


def test_tui_system_tab_renders_without_nvidia_smi():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._fill_gpus(None)             # nvidia-smi unavailable -> hint row
            gpus = app.query_one('#gpus', DataTable)
            assert gpus.row_count == 1
            assert 'cpus' in app._system_line()

    _run(scenario)


def test_tui_collapsed_console_skips_expensive_polling():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._active_tab = 'tab-containers'
            app._collapsed['docker'] = True
            assert 'ps' not in app._collect()          # collapsed -> no ps poll
            app._collapsed['docker'] = False
            assert 'ps' in app._collect()              # visible -> polled
            app._active_tab = 'tab-logs'
            assert 'ps' not in app._collect()          # other tab -> no ps poll
            app._collapsed['system'] = True
            assert 'gpus' not in app._collect()        # system collapsed
            app._collapsed['system'] = False
            assert 'gpus' in app._collect()            # system expanded -> polled

    _run(scenario)


def test_tui_open_builds_openwebui_url():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            controller.backend.ui_port = 13000
            assert app._ui_url('qwen-coder') == (
                'http://localhost:13000/?models=qwen-coder'
            )

    _run(scenario)


def test_tui_api_tester_sends_via_injected_http():
    from textual.widgets import Select

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'choices': [{'message': {'content': 'pong'}}]}

    class _HTTP:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=None):
            self.calls.append((url, json))
            return _Resp()

    http = _HTTP()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None, http=http)
        async with app.run_test() as pilot:
            await pilot.pause()
            controller.backend.litellm_port = 14042
            # only ready (running) models are offered; simulate one being ready
            app._sync_api_models(['qwen-coder'])
            assert app.query_one('#api-model', Select).value == 'qwen-coder'
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('pong' in line for line in app._api_lines)

    _run(scenario)
    assert http.calls and http.calls[0][0].endswith('/v1/chat/completions')


def test_tui_api_lists_only_ready_models():
    from textual.widgets import Select

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # NullBackend observes nothing running -> no ready models listed,
            # even though the catalog has endpoints.
            assert app._ready_endpoints == []
            assert app.query_one('#api-model', Select).value is Select.NULL

    _run(scenario)


def test_tui_cleanup_prunes_released_and_stopped(tmp_path):
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    # an active lease, then released -> it becomes a RELEASED tail entry
    out = controller.acquire('bob', catalog.resolve_names(['qwen-coder']))
    controller.release(out.lease.id)
    leases, _ = controller.ledger.status()
    assert any(str(le.state) == 'released' for le in leases)

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_cleanup()
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert not any(str(le.state) == 'released' for le in leases)


def test_tui_logs_stream_from_injected_source():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    lines = ['litellm   | started', 'litellm   | ready']

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: _FakeProc(lines))
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()     # drain the log stream
            await pilot.pause()
            assert any('ready' in line for line in app._log_lines)

    _run(scenario)
