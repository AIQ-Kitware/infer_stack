"""A remote dashboard that owns no controller, kubeconfig, or local ledger."""
from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Footer, Header, Input, Static


class RemoteApp(App):
    TITLE = 'infer-stack remote control'
    BINDINGS = [('q', 'quit', 'Quit'), ('r', 'refresh', 'Refresh')]
    CSS = '''
    DataTable { height: 1fr; }
    Horizontal { height: 3; }
    Input { width: 1fr; }
    #ttl { width: 15; }
    #message { height: auto; max-height: 4; }
    '''

    def __init__(self, client, *, interval=3, exit_after_paint=False):
        super().__init__()
        self.client, self.interval = client, max(.1, interval)
        self.exit_after_paint = exit_after_paint
        self.snapshot = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static('Connecting to canonical authority…', id='message')
        yield Static('Approved catalog')
        yield DataTable(id='catalog')
        with Horizontal():
            yield Input(placeholder='Endpoint or bundle names, separated by commas', id='endpoint')
            yield Input(value='2h', placeholder='Lease TTL', id='ttl')
            yield Button('Access endpoints', id='acquire')
        yield Static('Leases (select a row to release)')
        yield DataTable(id='leases', cursor_type='row')
        with Horizontal():
            yield Button('Release selected', id='release')
            yield Button('Reconcile', id='reconcile')
        yield Static('Deployments')
        yield DataTable(id='deployments')
        yield Static('Operations')
        yield DataTable(id='operations')
        yield Static('Cluster nodes (allocation is requested capacity, not utilization)')
        yield DataTable(id='nodes')
        yield Static('Recorded accelerator usage')
        yield DataTable(id='usage')
        yield Footer()

    def on_mount(self):
        columns = {'catalog': ('Alias', 'Protocol'),
                   'leases': ('ID', 'Owner', 'State', 'Endpoints'),
                   'deployments': ('ID', 'Engine', 'Demand', 'Desired', 'Observed', 'Endpoints'),
                   'operations': ('ID', 'State', 'Result / error'),
                   'nodes': ('Node', 'Ready', 'Scheduling', 'GPUs', 'Requested', 'Min GiB', 'Pool'),
                   'usage': ('Deployment', 'Node', 'GPU-hours', 'Wh', 'Method', 'Coverage')}
        for name, values in columns.items():
            self.query_one('#' + name, DataTable).add_columns(*values)
        self.set_interval(self.interval, self.action_refresh)
        self.action_refresh()

    @work(thread=True, exclusive=True, group='remote-refresh')
    def action_refresh(self):
        try:
            snapshot = self.client.request('GET', '/v1/status')
            catalog = self.client.request('GET', '/v1/catalog')
            usage = self.client.request('GET', '/v1/usage')
            try:
                nodes = self.client.request('GET', '/v1/nodes')
            except Exception:
                nodes = {'nodes': [], 'errors': {'nodes': 'Node inventory unavailable'}}
            self.call_from_thread(self._paint, snapshot, catalog, usage, nodes)
        except Exception as ex:
            self.call_from_thread(self._message, str(ex))

    def _message(self, value):
        self.query_one('#message', Static).update(value)

    def _paint(self, snapshot, catalog, usage, nodes=None):
        identity = snapshot.get('identity', {})
        if not self.snapshot and identity.get('max_ttl_seconds') is not None:
            self.query_one('#ttl', Input).value = f"{min(7200, identity['max_ttl_seconds']):g}s"
        self.snapshot = snapshot
        role = identity.get('role')
        if role is not None:
            self.query_one('#acquire', Button).disabled = role == 'observer'
            self.query_one('#release', Button).disabled = role == 'observer'
            self.query_one('#reconcile', Button).disabled = role not in {'operator', 'catalog-admin'}
        rows = {
            'catalog': [(row['alias'], row['protocol']) for row in catalog],
            'leases': [(row['id'], row['owner'], row['state'], ', '.join(row['endpoints'])) for row in snapshot['leases']],
            'deployments': [(row['id'], row['engine'], str(row['demand']), row['state'], (row.get('lifecycle') or {}).get('phase', 'unknown'), ', '.join(row['endpoints'])) for row in snapshot['deployments']],
            'operations': [(row['id'], row['state'], str(row['error'] or row['result'] or '')) for row in snapshot['operations']],
            'nodes': [(row['name'], str(row['ready']), 'yes' if row['schedulable'] else 'cordoned', str(row['gpu_count']),
                       str(row['gpu_requested']) if row['gpu_requested'] is not None else 'unknown',
                       str(row['gpu_memory_gib']) if row['gpu_memory_gib'] is not None else 'unknown',
                       row['labels'].get('infer-stack.ai/pool', '')) for row in (nodes or {}).get('nodes', [])],
            'usage': [(row['facts']['deployment'], row['facts']['node'], f"{row['energy']['gpu_hours']:.4g}",
                       f"{row['energy']['energy_wh']:.3g}" if row['energy']['energy_wh'] is not None else 'unknown',
                       row['energy']['method'], f"{row['energy']['coverage']:.0%}") for row in usage],
        }
        for name, values in rows.items():
            table = self.query_one('#' + name, DataTable)
            selected = table.get_row_at(table.cursor_row)[0] if table.row_count else None
            table.clear()
            for index, row in enumerate(values):
                table.add_row(*row, key=str(index) if name == 'usage' else row[0])
            if values:
                index = next((index for index, row in enumerate(values) if row[0] == selected), 0)
                table.move_cursor(row=index)
        observation = '; '.join((nodes or {}).get('errors', {}).values())
        self.query_one('#message', Static).update(f'Authority: {self.client.url}; authenticated control access' + (f'; {observation}' if observation else ''))
        if self.exit_after_paint:
            self.exit(0)

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == 'acquire':
            import math

            from ..cli.commands_leasing import _parse_duration

            names = [part.strip() for part in self.query_one('#endpoint', Input).value.split(',') if part.strip()]
            if names:
                try:
                    ttl = _parse_duration(self.query_one('#ttl', Input).value)
                    if ttl is not None and (not math.isfinite(ttl) or ttl <= 0):
                        raise ValueError('TTL must be positive or infinite')
                except ValueError:
                    self._message('Invalid TTL; use 2h, 30m, 90s or inf')
                    return
                self._mutate('POST', '/v1/access', {'endpoints': names, 'ttl_seconds': ttl})
        elif event.button.id == 'release':
            table = self.query_one('#leases', DataTable)
            if table.row_count:
                lease_id = table.get_row_at(table.cursor_row)[0]
                self._mutate('DELETE', '/v1/leases/' + str(lease_id), None)
        elif event.button.id == 'reconcile':
            self._mutate('POST', '/v1/reconcile', {})

    @work(thread=True, group='remote-mutation')
    def _mutate(self, method, path, payload):
        try:
            op = self.client.request(method, path, payload=payload)
            self.call_from_thread(self._message, f"Accepted operation {op['id']}; progress appears below")
            self.call_from_thread(self.action_refresh)
        except Exception as ex:
            self.call_from_thread(self._message, str(ex))


def run_tui(client, *, interval=3, exit_after_paint=False):
    return RemoteApp(client, interval=interval, exit_after_paint=exit_after_paint).run() or 0
