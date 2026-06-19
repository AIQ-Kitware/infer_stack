"""Headless tests for the optional Textual TUI (skipped if textual absent)."""

from __future__ import annotations

import asyncio

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
            assert {p.id for p in tabs.query('TabPane')} == {'tab-logs', 'tab-ps'}
            # the ps table renders (empty -> a placeholder row)
            assert app.query_one('#ps', DataTable).row_count >= 1

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
