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
