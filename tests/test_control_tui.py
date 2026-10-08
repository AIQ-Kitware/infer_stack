"""Remote dashboard is a transport client, including its mutation controls."""
import asyncio

import pytest

pytest.importorskip('textual')

from textual.widgets import DataTable, Input

from infer_stack.control.tui import RemoteApp


def test_remote_tui_uses_only_typed_transport():
    class Transport:
        url = 'https://authority.example'
        mutations = []

        def request(self, method, resource, *, payload=None):
            if method != 'GET':
                self.mutations.append((method, resource, payload))
                return {'id': 'op-1'}
            if resource == '/v1/catalog':
                return [{'alias': 'q', 'protocol': 'chat'}]
            if resource == '/v1/usage':
                return []
            if resource == '/v1/nodes':
                return {'nodes': [], 'errors': {}}
            return {'leases': [{'id': 'lease-1', 'owner': 'operator', 'state': 'active', 'endpoints': ['q']}],
                    'deployments': [], 'operations': []}

    async def exercise():
        transport = Transport()
        app = RemoteApp(transport, interval=60)
        async with app.run_test(size=(120, 55)) as pilot:
            await pilot.pause()
            assert app.query_one('#catalog', DataTable).row_count == 1
            assert app.query_one('#leases', DataTable).row_count == 1
            app.query_one('#endpoint', Input).value = 'q'
            await pilot.click('#acquire')
            await pilot.pause()
            await pilot.click('#release')
            await pilot.pause()
            await pilot.click('#reconcile')
            await pilot.pause()
            assert transport.mutations == [('POST', '/v1/access', {'endpoints': ['q'], 'ttl_seconds': 7200}),
                                           ('DELETE', '/v1/leases/lease-1', None), ('POST', '/v1/reconcile', {})]
            # A refresh can reorder rows while an operator is selecting a
            # lease. Preserve identity so the release target cannot change.
            second = {'id': 'lease-2', 'owner': 'operator', 'state': 'active', 'endpoints': ['q']}
            first = app.snapshot['leases'][0]
            app._paint({'leases': [first, second], 'deployments': [], 'operations': []}, [], [])
            app.query_one('#leases', DataTable).move_cursor(row=1)
            app._paint({'leases': [second, first], 'deployments': [], 'operations': []}, [], [])
            await pilot.click('#release')
            await pilot.pause()
            assert transport.mutations[-1] == ('DELETE', '/v1/leases/lease-2', None)
            app._paint({'identity': {'role': 'observer'}, 'leases': [], 'deployments': [], 'operations': []}, [], [])
            from textual.widgets import Button
            assert all(app.query_one('#' + name, Button).disabled for name in ['acquire', 'release', 'reconcile'])
    asyncio.run(exercise())
