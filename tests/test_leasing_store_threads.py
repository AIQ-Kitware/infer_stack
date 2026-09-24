"""One ledger connection shared by threads must never return garbage rows.

The TUI reads the ledger from its refresh worker and from the UI thread at
the same moment. Python's sqlite3 connection is not safe for that on its own:
two threads stepping the same cached statement interleave, and a lease came
back with ``None`` among its deployment ids (TypeError in the leases table).
"""

from __future__ import annotations

import threading

from infer_stack.leasing import Catalog, Controller, Ledger, NullBackend, SqliteStore


def test_concurrent_readers_see_consistent_rows(tmp_path):
    catalog = Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/m'}},
        'endpoints': {f'ep-{i:02d}': {'engine': 'vllm', 'model': 'm'} for i in range(30)},
    })
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    controller = Controller(ledger, NullBackend())
    for i in range(30):
        controller.acquire(f'u{i:02d}', catalog.resolve_names([f'ep-{i:02d}']))

    problems: list[str] = []

    def read():
        try:
            for _ in range(150):
                leases, deployments = ledger.status(virtual_expiry=True)
                if len(leases) != 30 or len(deployments) != 30:
                    problems.append(f'{len(leases)} leases, {len(deployments)} deployments')
                for le in leases:
                    if not le.deployment_ids or None in le.deployment_ids:
                        problems.append(f'{le.id}: deployment ids {le.deployment_ids}')
        except Exception as ex:  # noqa: BLE001 - any error is the failure
            problems.append(repr(ex))

    threads = [threading.Thread(target=read) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert problems == []
