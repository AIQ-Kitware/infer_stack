"""Headless tests for the optional Textual TUI (skipped if textual absent)."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip('textual')

CATALOG = {
    'models': {'qc': {'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct'}},
    'endpoints': {'qwen-coder': {'engine': 'vllm', 'model': 'qc'}},
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


def test_tui_mounts_lists_and_opens_serve():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI, ServeModal

    controller, catalog = _ctx()
    controller.acquire('alice', catalog.resolve_names(['qwen-coder']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999)
        async with app.run_test() as pilot:
            await pilot.pause()
            leases = app.query_one('#leases', DataTable)
            groups = app.query_one('#groups', DataTable)
            assert leases.row_count == 1          # the active lease
            assert groups.row_count == 1          # its deployment group
            assert app._selected_lease() == app._lease_ids[0]
            assert app._selected_group() == app._group_ids[0]
            # `s` opens the serve picker listing catalog endpoints
            await pilot.press('s')
            await pilot.pause()
            assert isinstance(app.screen, ServeModal)
            await pilot.press('escape')
            await pilot.pause()

    asyncio.run(scenario())


def test_tui_release_action_drops_the_lease():
    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    controller.acquire('alice', catalog.resolve_names(['qwen-coder']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('d')                # release the selected lease
            await app.workers.wait_for_complete()  # let the mutate worker finish
            await pilot.pause()

    asyncio.run(scenario())
    leases, _ = controller.ledger.status()
    assert [le.state for le in leases] == [LeaseState.RELEASED]
