"""Optional Textual TUI to monitor and control the leasing stack.

A multi-pane dashboard:

* **Catalog** (left) — the models + endpoints you can run; select an endpoint and
  press ``s`` / Enter to request a lease (serve it).
* **Leases** + **Groups** — the live ledger view (desired *state* vs *running*,
  and which GPUs each model is on), auto-refreshing.
* **Logs** — a live ``docker compose logs -f`` tail you can point at a specific
  service (or all of them).
* **Status bar** — the result of the last action.

It is opt-in and only imported when ``infer-stack tui`` runs, so the rest of the
CLI never pays for textual. The docker log source is injectable
(``proc_factory``) so the whole app is exercisable headless via Textual's pilot.
"""

from __future__ import annotations

import subprocess
from typing import Any, Callable, Iterable

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    RichLog,
    Select,
    Static,
)

from .cli.commands_leasing import (
    _gpu_label,
    _lease_ttl,
    _placement_view,
    _running_label,
)
from .leasing import GroupState, LeaseState

ALL_SERVICES = ''  # the Select value meaning "every service"


class _DockerLogProc:
    """A live ``docker compose logs -f`` process for one (or all) service(s)."""

    def __init__(self, project: str, compose_file: str, service: str | None):
        cmd = [
            'docker', 'compose', '-p', project, '-f', compose_file,
            'logs', '-f', '--tail', '200', '--no-color',
        ]
        if service:
            cmd.append(service)
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

    @property
    def stdout(self) -> Iterable[str]:
        return self._proc.stdout or iter(())

    def terminate(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self._proc.kill()


class InferStackTUI(App):
    """Monitor + control the leasing stack across panes."""

    TITLE = 'infer-stack'
    SUB_TITLE = 'leasing dashboard'

    # Calm + roomy: quiet (round $surface) borders that only brighten to $accent
    # on focus, generous padding, and margins between panes for breathing room.
    CSS = """
    Screen { layout: vertical; }

    #summary { height: 1; padding: 0 2; color: $text-muted; }
    #body { height: 1fr; padding: 0 1; }
    #sidebar { width: 38; min-width: 28; }
    #main { width: 1fr; }
    #tables { height: 1fr; }

    #endpoints, #models, #leases, #groups, #logbox {
        border: round $surface;
        background: $boost;
        padding: 0 1;
        margin: 0 0 1 0;
        border-title-color: $text-muted;
        border-title-align: left;
    }
    #endpoints:focus, #models:focus, #leases:focus, #groups:focus,
    #logbox:focus-within {
        border: round $accent;
        border-title-color: $accent;
    }

    #endpoints { height: 1fr; min-height: 6; }
    #models { height: 9; min-height: 5; }
    #leases { height: 1fr; min-height: 4; }
    #groups { height: 1fr; min-height: 4; }
    #logbox { height: 16; min-height: 6; }
    #logsvc { margin: 0 0 1 0; }
    #logs { height: 1fr; background: $surface; }

    #status { dock: bottom; height: 1; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [
        ('s', 'serve', 'Serve'),
        ('d', 'release', 'Release'),
        ('a', 'release_all', 'Release all'),
        ('e', 'evict', 'Evict'),
        ('r', 'refresh', 'Refresh'),
        ('tab', 'focus_next', 'Next pane'),
        ('q', 'quit', 'Quit'),
        Binding('left_square_bracket', 'sidebar_narrower', 'sidebar -', show=False),
        Binding('right_square_bracket', 'sidebar_wider', 'sidebar +', show=False),
        Binding('minus', 'logs_shorter', 'logs -', show=False),
        Binding('plus', 'logs_taller', 'logs +', show=False),
        Binding('equals_sign', 'logs_taller', 'logs +', show=False),
    ]

    def __init__(
        self,
        controller,
        catalog,
        *,
        interval: float = 3.0,
        proc_factory: Callable[[str | None], Any] | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.catalog = catalog
        self.interval = interval
        self._proc_factory = proc_factory or self._default_proc_factory()
        self._endpoint_names: list[str] = []
        self._lease_ids: list[str] = []
        self._group_ids: list[str] = []
        self._service_options: list[str] = []
        self._log_service: str = ALL_SERVICES
        self._log_proc: Any = None
        self._log_lines: list[str] = []  # mirror of the log pane, for tests
        self._sidebar_w = 38  # resizable via [ ]
        self._log_h = 16      # resizable via - +

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static('', id='summary')
        with Horizontal(id='body'):
            with Vertical(id='sidebar'):
                yield DataTable(id='endpoints', cursor_type='row',
                                zebra_stripes=True)
                yield DataTable(id='models', cursor_type='row',
                                zebra_stripes=True)
            with Vertical(id='main'):
                with Vertical(id='tables'):
                    yield DataTable(id='leases', cursor_type='row',
                                    zebra_stripes=True)
                    yield DataTable(id='groups', cursor_type='row',
                                    zebra_stripes=True)
                with Vertical(id='logbox'):
                    yield Select(
                        [('(all services)', ALL_SERVICES)],
                        value=ALL_SERVICES, allow_blank=False, id='logsvc',
                    )
                    yield RichLog(id='logs', highlight=False, markup=False,
                                  max_lines=2000, wrap=False)
        yield Static('', id='status')
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.theme = 'nord'  # a calm, muted palette
        except Exception:  # noqa: BLE001
            pass
        titles = {
            '#endpoints': 'catalog · endpoints', '#models': 'catalog · models',
            '#leases': 'leases', '#groups': 'groups', '#logbox': 'logs',
        }
        for sel, title in titles.items():
            self.query_one(sel).border_title = title
        self.query_one('#endpoints', DataTable).add_columns(
            'endpoint', 'model', 'engine', 'reclaim'
        )
        self.query_one('#models', DataTable).add_columns('model', 'source')
        self.query_one('#leases', DataTable).add_columns(
            'id', 'owner', 'state', 'ttl', 'endpoints'
        )
        self.query_one('#groups', DataTable).add_columns(
            'id', 'engine', 'state', 'running', 'gpus', 'demand', 'served'
        )
        self._apply_sizes()
        self._fill_catalog()
        self.action_refresh()
        self._restart_logs(self._log_service)
        self.set_interval(self.interval, self.action_refresh)
        self._status(
            'keys: s serve · d release · e evict · [ ] sidebar · - + logs'
        )
        self.query_one('#endpoints', DataTable).focus()

    def on_unmount(self) -> None:
        self._terminate_logs()

    # -- resizable panes ---------------------------------------------------

    def _apply_sizes(self) -> None:
        self.query_one('#sidebar').styles.width = self._sidebar_w
        self.query_one('#logbox').styles.height = self._log_h

    def action_sidebar_narrower(self) -> None:
        self._sidebar_w = max(26, self._sidebar_w - 4)
        self._apply_sizes()

    def action_sidebar_wider(self) -> None:
        self._sidebar_w = min(80, self._sidebar_w + 4)
        self._apply_sizes()

    def action_logs_shorter(self) -> None:
        self._log_h = max(6, self._log_h - 2)
        self._apply_sizes()

    def action_logs_taller(self) -> None:
        self._log_h = min(40, self._log_h + 2)
        self._apply_sizes()

    # -- catalog -----------------------------------------------------------

    def _fill_catalog(self) -> None:
        eps = self.query_one('#endpoints', DataTable)
        eps.clear()
        self._endpoint_names = []
        for name in sorted(self.catalog.endpoints):
            ep = self.catalog.endpoints[name]
            eps.add_row(
                name, getattr(ep, 'model', '') or '-',
                getattr(ep, 'engine', '') or '-',
                str(getattr(ep, 'reclaim', '') or '-'),
            )
            self._endpoint_names.append(name)
        models = self.query_one('#models', DataTable)
        models.clear()
        for name in sorted(self.catalog.models):
            models.add_row(name, getattr(self.catalog.models[name], 'source', ''))

    # -- monitor -----------------------------------------------------------

    def action_refresh(self) -> None:
        try:
            self.controller.ledger.sweep()
            leases, groups = self.controller.ledger.status()
            observed, assignments = _placement_view(self.controller)
        except Exception as ex:  # noqa: BLE001 - a monitor must never crash
            self._status(f'refresh error: {ex}')
            return
        self._fill_leases(leases)
        self._fill_groups(groups, observed, assignments)
        self._update_summary(leases, groups, observed)
        self._sync_log_services()

    def _update_summary(self, leases, groups, observed) -> None:
        active = sum(1 for le in leases if str(le.state) == 'active')
        running = sum(1 for g in groups if g.id in observed)
        self.query_one('#summary', Static).update(
            f'models {len(self.catalog.models)}   '
            f'endpoints {len(self.catalog.endpoints)}   ·   '
            f'leases {active} active / {len(leases)}   '
            f'groups {running} running / {len(groups)}'
        )

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

    # -- logs --------------------------------------------------------------

    def _service_names(self) -> list[str]:
        """Service names from the on-disk compose file (best-effort)."""
        backend = self.controller.backend
        path = getattr(backend, 'compose_file', None)
        if not path:
            return []
        try:
            from pathlib import Path

            import yaml
            data = yaml.safe_load(Path(path).read_text()) or {}
            return sorted((data.get('services') or {}).keys())
        except Exception:  # noqa: BLE001
            return []

    def _sync_log_services(self) -> None:
        names = self._service_names()
        if names == self._service_options:
            return
        self._service_options = names
        select = self.query_one('#logsvc', Select)
        options = [('(all services)', ALL_SERVICES)] + [(n, n) for n in names]
        select.set_options(options)
        # keep the current selection if it still exists, else fall back to all
        select.value = (
            self._log_service if self._log_service in names else ALL_SERVICES
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != 'logsvc':
            return
        service = '' if event.value is Select.BLANK else str(event.value)
        if service != self._log_service:
            self._log_service = service
            self._restart_logs(service)

    def _default_proc_factory(self) -> Callable[[str | None], Any]:
        def factory(service: str | None):
            backend = self.controller.backend
            path = getattr(backend, 'compose_file', None)
            project = getattr(backend, 'project', 'infer-stack')
            if not path:
                return None
            return _DockerLogProc(str(project), str(path), service)

        return factory

    def _restart_logs(self, service: str) -> None:
        self._terminate_logs()
        log = self.query_one('#logs', RichLog)
        log.clear()
        self._log_lines = []
        label = service or 'all services'
        log.write(f'— following logs: {label} —')
        self._stream_logs(service or None)

    def _terminate_logs(self) -> None:
        proc, self._log_proc = self._log_proc, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    @work(thread=True, exclusive=True, group='logs')
    def _stream_logs(self, service: str | None) -> None:
        proc = self._proc_factory(service)
        if proc is None:
            self.call_from_thread(
                self._append_log, '(no compose project yet — serve a model)'
            )
            return
        self._log_proc = proc
        try:
            for line in proc.stdout:
                self.call_from_thread(self._append_log, line.rstrip('\n'))
        except Exception:  # noqa: BLE001 - stream ends when the proc dies
            pass

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        self.query_one('#logs', RichLog).write(line)

    # -- helpers + actions -------------------------------------------------

    def _status(self, message: str) -> None:
        self.query_one('#status', Static).update(message)

    def _selected(self, table_id: str, ids: list[str]) -> str | None:
        row = self.query_one(f'#{table_id}', DataTable).cursor_row
        return ids[row] if 0 <= row < len(ids) else None

    def action_serve(self) -> None:
        name = self._selected('endpoints', self._endpoint_names)
        if not name:
            self._status('select an endpoint in the catalog to serve')
            return
        self._status(f'serving {name}…')
        self._do_serve(name)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter on the endpoints table serves that endpoint.
        if event.data_table.id == 'endpoints':
            self.action_serve()

    def action_release(self) -> None:
        sid = self._selected('leases', self._lease_ids)
        if not sid:
            self._status('select a lease row to release')
            return
        self._status(f'releasing {sid}…')
        self._do_release(sid)

    def action_release_all(self) -> None:
        self._status('releasing all active leases…')
        self._do_release_all()

    def action_evict(self) -> None:
        gid = self._selected('groups', self._group_ids)
        if not gid:
            self._status('select a group row to evict')
            return
        self._status(f'evicting {gid}…')
        self._do_evict(gid)

    # Mutations converge the backend (docker up/down, possibly slow) off the UI
    # thread; results + a refresh are marshalled back on.

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
