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


def test_tui_panes_list_catalog_leases_and_groups():
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
            assert app.query_one('#groups', DataTable).row_count == 1

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


def test_tui_has_docker_logs_and_ps_tabs():
    from textual.widgets import DataTable, TabbedContent

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one('#docker', TabbedContent)
            assert {p.id for p in tabs.query('TabPane')} == {
                'tab-logs', 'tab-ps', 'tab-system', 'tab-api'
            }
            # the ps table renders (empty -> a placeholder row) with the docker
            # ps columns the user asked for (status/uptime, created, id).
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
            app._active_tab = 'tab-ps'
            app._console_collapsed = True
            assert 'ps' not in app._collect()          # collapsed -> no ps poll
            app._console_collapsed = False
            assert 'ps' in app._collect()              # visible -> polled
            app._active_tab = 'tab-logs'
            assert 'ps' not in app._collect()          # other tab -> no ps poll

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
            # models from the catalog populate the API selector
            assert app.query_one('#api-model', Select).value == 'qwen-coder'
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('pong' in line for line in app._api_lines)

    _run(scenario)
    assert http.calls and http.calls[0][0].endswith('/v1/chat/completions')


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
