"""Optional Textual TUI to monitor and control the leasing stack.

A live view of the lease ledger + deployment groups (the same data as
``infer-stack leases``: desired *state* vs *running*, and which GPUs each model
is on), plus key-bound controls to serve a model, release a lease, or evict an
idle group — without leaving the terminal.

It is opt-in and only imported when ``infer-stack tui`` runs, so the rest of the
CLI never pays for textual. Launch with :func:`run_tui`; it reuses an already
built :class:`~infer_stack.leasing.controller.Controller` and
:class:`~infer_stack.leasing.catalog.Catalog`.
"""

from __future__ import annotations

from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Label, OptionList, Static

from .cli.commands_leasing import (
    _gpu_label,
    _lease_ttl,
    _placement_view,
    _running_label,
)
from .leasing import GroupState, LeaseState


class ServeModal(ModalScreen[str | None]):
    """Pick a catalog endpoint to serve (Enter to select, Esc to cancel)."""

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._names = names

    def compose(self) -> ComposeResult:
        with Vertical(id='serve-dialog'):
            yield Label('Serve which endpoint?  (Enter = select, Esc = cancel)')
            yield OptionList(*self._names, id='serve-options')

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        self.dismiss(str(event.option.prompt))

    def on_key(self, event) -> None:
        if event.key == 'escape':
            self.dismiss(None)


class InferStackTUI(App):
    """Monitor + control the leasing stack."""

    TITLE = 'infer-stack'
    SUB_TITLE = 'leasing monitor'

    CSS = """
    #status { dock: bottom; height: 1; padding: 0 1; color: $text-muted; }
    DataTable { height: 1fr; }
    #serve-dialog {
        width: 60; height: auto; padding: 1 2;
        border: round $accent; background: $surface;
        align: center middle;
    }
    #serve-options { height: auto; max-height: 16; }
    """

    BINDINGS = [
        ('s', 'serve', 'Serve'),
        ('d', 'release', 'Release'),
        ('a', 'release_all', 'Release all'),
        ('e', 'evict', 'Evict'),
        ('r', 'refresh', 'Refresh'),
        ('q', 'quit', 'Quit'),
    ]

    def __init__(self, controller, catalog, *, interval: float = 3.0) -> None:
        super().__init__()
        self.controller = controller
        self.catalog = catalog
        self.interval = interval
        self._lease_ids: list[str] = []
        self._group_ids: list[str] = []

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Label('leases', classes='section')
            yield DataTable(id='leases', cursor_type='row', zebra_stripes=True)
            yield Label('groups', classes='section')
            yield DataTable(id='groups', cursor_type='row', zebra_stripes=True)
        yield Static('', id='status')
        yield Footer()

    def on_mount(self) -> None:
        leases = self.query_one('#leases', DataTable)
        leases.add_columns('id', 'owner', 'state', 'ttl', 'endpoints')
        groups = self.query_one('#groups', DataTable)
        groups.add_columns(
            'id', 'engine', 'state', 'running', 'gpus', 'demand', 'served'
        )
        self.set_interval(self.interval, self.action_refresh)
        self.action_refresh()

    # -- data --------------------------------------------------------------

    def action_refresh(self) -> None:
        """Re-query the ledger + backend and repopulate the tables.

        Synchronous: the ledger is sqlite (instant) and the backend probe is
        best-effort, so a monitor refresh is cheap enough to run inline.
        """
        try:
            self.controller.ledger.sweep()
            leases, groups = self.controller.ledger.status()
            observed, assignments = _placement_view(self.controller)
        except Exception as ex:  # noqa: BLE001 - a monitor must never crash
            self._status(f'refresh error: {ex}')
            return
        self._fill_leases(leases)
        self._fill_groups(groups, observed, assignments)

    def _fill_leases(self, leases) -> None:
        table = self.query_one('#leases', DataTable)
        cursor = table.cursor_row
        table.clear()
        self._lease_ids = []
        for le in leases:
            table.add_row(
                le.id, le.owner, str(le.state), _lease_ttl(le),
                ','.join(le.endpoints) or '-',
            )
            self._lease_ids.append(le.id)
        self._restore_cursor(table, cursor)

    def _fill_groups(self, groups, observed, assignments) -> None:
        table = self.query_one('#groups', DataTable)
        cursor = table.cursor_row
        table.clear()
        self._group_ids = []
        for g in groups:
            table.add_row(
                g.id, g.engine, str(g.state),
                _running_label(g.id, observed),
                _gpu_label(g.id, observed, assignments),
                str(g.demand), ','.join(sorted(g.served)) or '-',
            )
            self._group_ids.append(g.id)
        self._restore_cursor(table, cursor)

    @staticmethod
    def _restore_cursor(table: DataTable, row: int) -> None:
        if table.row_count:
            table.move_cursor(row=min(max(row, 0), table.row_count - 1))

    # -- helpers -----------------------------------------------------------

    def _status(self, message: str) -> None:
        self.query_one('#status', Static).update(message)

    def _selected_lease(self) -> str | None:
        table = self.query_one('#leases', DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._lease_ids):
            return self._lease_ids[row]
        return None

    def _selected_group(self) -> str | None:
        table = self.query_one('#groups', DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._group_ids):
            return self._group_ids[row]
        return None

    # -- actions -----------------------------------------------------------

    def action_serve(self) -> None:
        names = sorted(self.catalog.endpoints)
        if not names:
            self._status('catalog has no endpoints to serve')
            return

        def _picked(name: str | None) -> None:
            if name:
                self._status(f'serving {name}…')
                self._do_serve(name)

        self.push_screen(ServeModal(names), _picked)

    def action_release(self) -> None:
        sid = self._selected_lease()
        if not sid:
            self._status('select a lease row to release')
            return
        self._status(f'releasing {sid}…')
        self._do_release(sid)

    def action_release_all(self) -> None:
        self._status('releasing all active leases…')
        self._do_release_all()

    def action_evict(self) -> None:
        gid = self._selected_group()
        if not gid:
            self._status('select a group row to evict')
            return
        self._status(f'evicting {gid}…')
        self._do_evict(gid)

    # Mutations converge the backend (docker up/down, possibly slow), so they
    # run off the UI thread; results + a refresh are marshalled back on.

    @work(thread=True, exclusive=True, group='mutate')
    def _do_serve(self, name: str) -> None:
        try:
            requests = self.catalog.resolve_names([name])
            self.controller.acquire(
                'manual', requests, ttl_seconds=None, wait=False, apply=True
            )
            msg = f'serving {name}'
        except Exception as ex:  # noqa: BLE001
            msg = f'serve {name} failed: {ex}'
        self._after_mutation(msg)

    @work(thread=True, exclusive=True, group='mutate')
    def _do_release(self, sid: str) -> None:
        try:
            self.controller.release(sid)
            msg = f'released {sid}'
        except Exception as ex:  # noqa: BLE001
            msg = f'release {sid} failed: {ex}'
        self._after_mutation(msg)

    @work(thread=True, exclusive=True, group='mutate')
    def _do_release_all(self) -> None:
        try:
            self.controller.ledger.sweep()
            leases, _ = self.controller.ledger.status()
            active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
            for sid in active:
                self.controller.ledger.release(sid)
            self.controller.reconcile()
            msg = f'released {len(active)} lease(s)'
        except Exception as ex:  # noqa: BLE001
            msg = f'release --all failed: {ex}'
        self._after_mutation(msg)

    @work(thread=True, exclusive=True, group='mutate')
    def _do_evict(self, gid: str) -> None:
        try:
            self.controller.ledger.sweep()
            group = self.controller.ledger.get_group(gid)
            if group is None or group.state != GroupState.IDLE:
                msg = f'{gid} is not idle — release it first'
            else:
                self.controller.evict([gid])
                msg = f'evicted {gid}'
        except Exception as ex:  # noqa: BLE001
            msg = f'evict {gid} failed: {ex}'
        self._after_mutation(msg)

    def _after_mutation(self, message: str) -> None:
        self.call_from_thread(self._status, message)
        self.call_from_thread(self.action_refresh)


def run_tui(controller, catalog, *, interval: float = 3.0) -> int:
    """Run the TUI against a built controller + catalog. Returns an exit code."""
    # The narration loguru sink writes to stderr, which would corrupt the
    # full-screen UI — silence it while the TUI owns the terminal.
    try:
        from ._log import logger

        logger.disable('infer_stack')
    except Exception:  # noqa: BLE001
        pass
    app: Any = InferStackTUI(controller, catalog, interval=interval)
    app.run()
    return 0
