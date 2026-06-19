"""Optional Textual TUI to monitor and control the leasing stack.

A multi-pane dashboard built to be approachable for someone who has never
touched infer-stack:

* **Catalog** (left) — the models + endpoints you can run, with buttons to
  auto-suggest a set sized to your GPUs or add one by hand. Pick an endpoint and
  press ``s`` / Enter to request a lease (serve it).
* **Leases** + **Groups** (center) — the live ledger view (desired *state* vs
  what is actually *running*, and which GPUs each model is on), auto-refreshing.
* **Docker** (bottom) — a tabbed pane: a live ``logs -f`` tail you can point at a
  specific service, and a ``ps`` snapshot of what containers are up.
* **Status bar** — the result of the last action.

Design notes that keep it responsive and headless-testable:

* The catalog refresh and ``docker compose ps`` calls run on a worker thread, so
  shelling out to docker never freezes the UI.
* Docker's own progress (``up -d`` / ``down``) is captured and routed into the
  logs pane instead of bleeding onto the full-screen terminal.
* The docker log source is injectable (``proc_factory``) so the whole app is
  exercisable via Textual's pilot without docker or a GPU.

It is opt-in and only imported when ``infer-stack tui`` runs, so the rest of the
CLI never pays for textual.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .cli.commands_leasing import (
    _gpu_label,
    _lease_ttl,
    _placement_view,
    _running_label,
)
from .leasing import GroupState, LeaseState

ALL_SERVICES = ''  # the Select value meaning "every service"


# A calm, dark palette with one warm accent: very-dark-gray canvas, white text,
# orange highlights. Borders sit quiet on $surface and only warm to $accent
# (orange) on the focused pane, so the eye is drawn without the whole screen
# shouting.
INFER_THEME = Theme(
    name='infer-orange',
    primary='#ff8c1a',
    secondary='#ffb066',
    accent='#ff8c1a',
    foreground='#f2f2f2',
    background='#141414',
    surface='#1f1f1f',
    panel='#272727',
    success='#7bd88f',
    warning='#ffb454',
    error='#ff6b6b',
    dark=True,
)


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


class _Divider(Static):
    """A draggable splitter bar.

    Reports incremental drags to ``on_drag(delta)`` — ``delta`` is the pointer's
    movement along the bar's resize axis since the last event. Keyboard resize
    bindings remain the primary path; this just lets you grab and pull too.
    """

    def __init__(self, axis: str, on_drag: Callable[[int], None], **kw: Any):
        super().__init__('', **kw)
        self._axis = axis  # 'x' -> drag horizontally, 'y' -> drag vertically
        self._on_drag = on_drag
        self._dragging = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self.capture_mouse()
        event.stop()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging:
            return
        delta = event.delta_x if self._axis == 'x' else event.delta_y
        if delta:
            self._on_drag(int(delta))
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._dragging = False
        self.release_mouse()
        event.stop()


class _AddModelScreen(ModalScreen):
    """Wizard: add a model (a weight source vLLM will serve)."""

    CSS = """
    _AddModelScreen { align: center middle; }
    #dialog {
        width: 70; height: auto; padding: 1 2;
        border: round $accent; background: $surface;
    }
    #dialog .hint { color: $text-muted; margin: 0 0 1 0; }
    #dialog Input { margin: 0 0 1 0; }
    #dialog Horizontal { height: auto; align-horizontal: right; }
    #dialog Button { margin: 0 0 0 2; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id='dialog'):
            yield Label('Add a model', classes='title')
            yield Static(
                'A model is a set of weights (Hugging Face repo or local path). '
                'Endpoints point at it.',
                classes='hint',
            )
            yield Input(placeholder='name  (e.g. qwen-coder)', id='m-name')
            yield Input(
                placeholder='source  (hf://Qwen/Qwen2.5-Coder-7B-Instruct or /path)',
                id='m-source',
            )
            with Horizontal():
                yield Button('Cancel', id='cancel')
                yield Button('Add model', variant='primary', id='ok')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'cancel':
            self.dismiss(None)
            return
        name = self.query_one('#m-name', Input).value.strip()
        source = self.query_one('#m-source', Input).value.strip()
        if not name or not source:
            self.query_one(Label).update('name and source are both required')
            return
        self.dismiss({'name': name, 'source': source})


class _AddEndpointScreen(ModalScreen):
    """Wizard: add an endpoint (the served API name clients ask for)."""

    CSS = """
    _AddEndpointScreen { align: center middle; }
    #dialog {
        width: 70; height: auto; padding: 1 2;
        border: round $accent; background: $surface;
    }
    #dialog .hint { color: $text-muted; margin: 0 0 1 0; }
    #dialog Input, #dialog Select { margin: 0 0 1 0; }
    #dialog Horizontal { height: auto; align-horizontal: right; }
    #dialog Button { margin: 0 0 0 2; }
    """

    def __init__(self, models: list[str]):
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        with Vertical(id='dialog'):
            yield Label('Add an endpoint', classes='title')
            yield Static(
                'An endpoint is the served name (what Open WebUI shows and what '
                'clients request). It binds a model to an engine.',
                classes='hint',
            )
            yield Input(
                placeholder='name  (optional — defaults to <model>-N)',
                id='e-name',
            )
            model_opts = [(m, m) for m in self._models]
            if model_opts:
                yield Select(model_opts, prompt='model…', id='e-model')
            else:
                yield Input(placeholder='model name', id='e-model-text')
            yield Select(
                [('vllm', 'vllm'), ('ollama', 'ollama')],
                value='vllm', allow_blank=False, id='e-engine',
            )
            with Horizontal():
                yield Button('Cancel', id='cancel')
                yield Button('Add endpoint', variant='primary', id='ok')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'cancel':
            self.dismiss(None)
            return
        name = self.query_one('#e-name', Input).value.strip()
        try:
            value = self.query_one('#e-model', Select).value
            model = '' if value is Select.BLANK else str(value)
        except Exception:  # noqa: BLE001 - free-text fallback when no models yet
            model = self.query_one('#e-model-text', Input).value.strip()
        if not model:
            self.query_one(Label).update('pick (or type) a model')
            return
        engine = str(self.query_one('#e-engine', Select).value or 'vllm')
        self.dismiss({'name': name or None, 'model': model, 'engine': engine})


class InferStackTUI(App):
    """Monitor + control the leasing stack across panes."""

    TITLE = 'infer-stack'
    SUB_TITLE = 'leasing dashboard'

    CSS = """
    Screen { layout: vertical; }

    #intro { height: auto; padding: 0 2; color: $text-muted; }
    #body { height: 1fr; padding: 0 1; }

    #sidebar { width: 38; min-width: 26; }
    #vsplit { width: 1; background: $surface; }
    #vsplit:hover { background: $accent; }
    #main { width: 1fr; }
    #tables { height: 1fr; }
    #hsplit { height: 1; background: $surface; margin: 0 0 1 0; }
    #hsplit:hover { background: $accent; }

    #catalog-help { height: auto; color: $text-muted; padding: 0 1; }
    #catalog-buttons { height: auto; }
    #catalog-buttons Button { width: 1fr; margin: 1 0 0 0; }

    #endpoints, #models, #leases, #groups, #docker {
        border: round $surface;
        background: $boost;
        padding: 0 1;
        margin: 0 0 1 0;
        border-title-color: $text-muted;
        border-title-align: left;
        border-subtitle-color: $text-muted;
        border-subtitle-align: right;
    }
    #endpoints:focus, #models:focus, #leases:focus, #groups:focus,
    #docker:focus-within {
        border: round $accent;
        border-title-color: $accent;
    }

    #endpoints { height: 1fr; min-height: 5; }
    #models { height: 8; min-height: 4; }
    #leases { height: 1fr; min-height: 4; }
    #groups { height: 1fr; min-height: 4; }
    #docker { height: 16; min-height: 8; }
    #logsvc { margin: 0 0 1 0; }
    #logs { height: 1fr; background: $surface; }
    #ps { height: 1fr; background: $surface; }

    #status { dock: bottom; height: 1; padding: 0 2; color: $text-muted; }
    """

    BINDINGS = [
        ('s', 'serve', 'Serve'),
        ('d', 'release', 'Release'),
        ('e', 'evict', 'Evict'),
        ('g', 'suggest', 'Suggest'),
        ('m', 'add_model', 'Add model'),
        ('n', 'add_endpoint', 'Add endpoint'),
        ('r', 'refresh', 'Refresh'),
        ('a', 'release_all', 'Release all'),
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
        catalog_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.catalog = catalog
        self.interval = interval
        self.catalog_path = Path(catalog_path).expanduser() if catalog_path else None
        self._proc_factory = proc_factory or self._default_proc_factory()
        self._endpoint_names: list[str] = []
        self._lease_ids: list[str] = []
        self._group_ids: list[str] = []
        self._service_options: list[str] = []
        self._log_service: str = ALL_SERVICES
        self._log_proc: Any = None
        self._log_lines: list[str] = []  # mirror of the log pane, for tests
        self._sidebar_w = 38  # resizable via [ ] or dragging #vsplit
        self._log_h = 16      # resizable via - + or dragging #hsplit

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            'Request models from the [b]catalog[/b] (left) · watch '
            '[b]leases[/b] & GPUs (center) · tail [b]docker[/b] (below).  '
            'Keys: [b]s[/b] serve · [b]d[/b] release · [b]e[/b] evict · '
            '[b]g[/b] suggest · [b]m[/b]/[b]n[/b] add.',
            id='intro', markup=True,
        )
        with Horizontal(id='body'):
            with Vertical(id='sidebar'):
                yield Static('', id='catalog-help')
                yield DataTable(id='endpoints', cursor_type='row',
                                zebra_stripes=True)
                yield DataTable(id='models', cursor_type='row',
                                zebra_stripes=True)
                with Vertical(id='catalog-buttons'):
                    yield Button('✨  Suggest from my GPUs', id='btn-suggest')
                    yield Button('＋  Add model', id='btn-add-model')
                    yield Button('＋  Add endpoint', id='btn-add-endpoint')
            yield _Divider('x', self._drag_sidebar, id='vsplit')
            with Vertical(id='main'):
                with Vertical(id='tables'):
                    yield DataTable(id='leases', cursor_type='row',
                                    zebra_stripes=True)
                    yield DataTable(id='groups', cursor_type='row',
                                    zebra_stripes=True)
                yield _Divider('y', self._drag_logs, id='hsplit')
                with TabbedContent(id='docker'):
                    with TabPane('Logs', id='tab-logs'):
                        yield Select(
                            [('(all services)', ALL_SERVICES)],
                            value=ALL_SERVICES, allow_blank=False, id='logsvc',
                        )
                        yield RichLog(id='logs', highlight=False, markup=False,
                                      max_lines=2000, wrap=False)
                    with TabPane('Status · ps', id='tab-ps'):
                        yield DataTable(id='ps', cursor_type='row',
                                        zebra_stripes=True)
        yield Static('', id='status')
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.register_theme(INFER_THEME)
            self.theme = 'infer-orange'
        except Exception:  # noqa: BLE001
            pass
        titles = {
            '#endpoints': 'catalog · endpoints', '#models': 'catalog · models',
            '#leases': 'leases', '#groups': 'groups', '#docker': 'docker',
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
        self.query_one('#ps', DataTable).add_columns(
            'service', 'state', 'ports'
        )
        # Capture docker's own chatter (up/down progress on stderr) into the
        # logs pane instead of letting it bleed onto the full-screen terminal.
        self._install_quiet_docker()
        self._apply_sizes()
        self._fill_catalog()
        self._refresh_now()           # synchronous first paint (snappy + testable)
        self._restart_logs(self._log_service)
        self.set_interval(self.interval, self.action_refresh)
        self.query_one('#endpoints', DataTable).focus()

    def on_unmount(self) -> None:
        self._terminate_logs()

    # -- theming for docker output bleed ----------------------------------

    def _install_quiet_docker(self) -> None:
        """Route ``docker compose`` output to the logs pane, not the terminal.

        ``docker compose up -d`` / ``down`` print their progress to *stderr*,
        which the default backend runner lets through to the controlling
        terminal — corrupting the full-screen UI. Wrap the backend runner so
        every invocation is fully captured; the noisy verbs (up/down) get echoed
        into the logs pane, while quiet polling (``ps``) is swallowed.
        """
        backend = self.controller.backend
        if not hasattr(backend, 'run'):
            return

        def quiet_run(args: list[str]) -> str:
            proc = subprocess.run(args, capture_output=True, text=True)
            noisy = not any(a == 'ps' for a in args)
            if noisy:
                for line in (proc.stderr or '').splitlines():
                    if line.strip():
                        self.call_from_thread(self._append_log, line.rstrip())
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, args, proc.stdout, proc.stderr
                )
            return proc.stdout

        backend.run = quiet_run

    # -- resizable panes ---------------------------------------------------

    def _apply_sizes(self) -> None:
        self.query_one('#sidebar').styles.width = self._sidebar_w
        self.query_one('#docker').styles.height = self._log_h

    def _drag_sidebar(self, delta: int) -> None:
        self._sidebar_w = max(24, min(100, self._sidebar_w + delta))
        self._apply_sizes()

    def _drag_logs(self, delta: int) -> None:
        # Dragging the divider down (delta > 0) makes the logs pane shorter.
        self._log_h = max(6, min(60, self._log_h - delta))
        self._apply_sizes()

    def action_sidebar_narrower(self) -> None:
        self._drag_sidebar(-4)

    def action_sidebar_wider(self) -> None:
        self._drag_sidebar(4)

    def action_logs_shorter(self) -> None:
        self._drag_logs(2)

    def action_logs_taller(self) -> None:
        self._drag_logs(-2)

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
        self._update_catalog_help()

    def _update_catalog_help(self) -> None:
        help_ = self.query_one('#catalog-help', Static)
        if not self._endpoint_names:
            help_.update(
                'No endpoints yet. Press [b]g[/b] to suggest a set sized to '
                'your GPUs, or [b]m[/b]/[b]n[/b] to add a model/endpoint.'
            )
        else:
            help_.update(
                'Select an endpoint and press [b]s[/b] (or Enter) to serve it.'
            )

    def _reload_catalog(self) -> None:
        if not self.catalog_path or not self.catalog_path.exists():
            return
        try:
            from .leasing import Catalog
            self.catalog = Catalog.load(self.catalog_path)
        except Exception as ex:  # noqa: BLE001
            self._status(f'catalog reload failed: {ex}')
            return
        self._fill_catalog()

    # -- monitor -----------------------------------------------------------

    def _collect(self) -> dict[str, Any]:
        """Gather ledger + observed state. Safe to call off the UI thread."""
        try:
            self.controller.ledger.sweep()
            leases, groups = self.controller.ledger.status()
            observed, assignments = _placement_view(self.controller)
            ps_rows = self._compose_ps_rows()
            return {
                'leases': leases, 'groups': groups, 'observed': observed,
                'assignments': assignments, 'ps': ps_rows,
            }
        except Exception as ex:  # noqa: BLE001 - a monitor must never crash
            return {'error': str(ex)}

    def _render(self, data: dict[str, Any]) -> None:
        if 'error' in data:
            self._status(f'refresh error: {data["error"]}')
            return
        self._fill_leases(data['leases'])
        self._fill_groups(data['groups'], data['observed'], data['assignments'])
        self._fill_ps(data['ps'])
        self._update_summary(data['leases'], data['groups'], data['observed'])
        self._sync_log_services()

    def _refresh_now(self) -> None:
        """Synchronous collect + render (first paint; also what tests rely on)."""
        self._render(self._collect())

    @work(thread=True, exclusive=True, group='refresh')
    def _refresh_bg(self) -> None:
        data = self._collect()
        self.call_from_thread(self._render, data)

    def action_refresh(self) -> None:
        self._refresh_bg()

    def _update_summary(self, leases, groups, observed) -> None:
        active = sum(1 for le in leases if str(le.state) == 'active')
        running = sum(1 for g in groups if g.id in observed)
        self.query_one('#docker', TabbedContent).border_subtitle = (
            f'{running} running'
        )
        self.query_one('#leases', DataTable).border_subtitle = (
            f'{active} active / {len(leases)}'
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

    def _fill_ps(self, rows) -> None:
        table = self.query_one('#ps', DataTable)
        cursor = table.cursor_row
        table.clear()
        for row in rows:
            table.add_row(row['service'], row['state'], row['ports'] or '-')
        if not rows:
            table.add_row('(nothing running)', '-', '-')
        self._restore_cursor(table, cursor)

    @staticmethod
    def _restore_cursor(table: DataTable, row: int) -> None:
        if table.row_count:
            table.move_cursor(row=min(max(row, 0), table.row_count - 1))

    # -- docker ps ---------------------------------------------------------

    def _compose_ps_rows(self) -> list[dict[str, str]]:
        """Best-effort ``docker compose ps`` rows (service/state/ports)."""
        import json

        backend = self.controller.backend
        path = getattr(backend, 'compose_file', None)
        run = getattr(backend, 'run', None)
        if not path or not run or not Path(path).exists():
            return []
        project = getattr(backend, 'project', 'infer-stack')
        try:
            out = run([
                'docker', 'compose', '-p', str(project), '-f', str(path),
                'ps', '--format', 'json',
            ])
        except Exception:  # noqa: BLE001 - ps is best-effort
            return []
        rows: list[dict[str, Any]] = []
        out = (out or '').strip()
        if not out:
            return []
        try:
            parsed = json.loads(out)
            rows = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            for line in out.splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        result = []
        for row in rows:
            result.append({
                'service': str(row.get('Service') or row.get('Name') or '?'),
                'state': str(row.get('State') or row.get('Status') or '?'),
                'ports': str(row.get('Publishers') and _fmt_ports(row) or
                             row.get('Ports') or ''),
            })
        return sorted(result, key=lambda r: r['service'])

    # -- logs --------------------------------------------------------------

    def _service_names(self) -> list[str]:
        """Service names from the on-disk compose file (best-effort)."""
        backend = self.controller.backend
        path = getattr(backend, 'compose_file', None)
        if not path:
            return []
        try:
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
        self._status(f'serving {name}… (docker output appears in the logs pane)')
        self._do_serve(name)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter on the endpoints table serves that endpoint.
        if event.data_table.id == 'endpoints':
            self.action_serve()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'btn-suggest':
            self.action_suggest()
        elif event.button.id == 'btn-add-model':
            self.action_add_model()
        elif event.button.id == 'btn-add-endpoint':
            self.action_add_endpoint()

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

    # -- catalog editing (suggest + wizards) -------------------------------

    def action_add_model(self) -> None:
        if not self.catalog_path:
            self._status('no catalog path — launch the TUI with a catalog to edit')
            return
        self.push_screen(_AddModelScreen(), self._on_add_model)

    def _on_add_model(self, result: dict | None) -> None:
        if not result:
            return
        try:
            self._write_catalog('models', result['name'], {
                'source': result['source'],
            })
            self._status(f'added model {result["name"]}')
        except Exception as ex:  # noqa: BLE001
            self._status(f'add model failed: {ex}')
            return
        self._reload_catalog()

    def action_add_endpoint(self) -> None:
        if not self.catalog_path:
            self._status('no catalog path — launch the TUI with a catalog to edit')
            return
        self.push_screen(
            _AddEndpointScreen(sorted(self.catalog.models)), self._on_add_endpoint
        )

    def _on_add_endpoint(self, result: dict | None) -> None:
        if not result:
            return
        try:
            from .cli.commands_catalog import (
                _load_raw,
                _next_indexed_name,
                _slug_alias,
            )
            data = _load_raw(self.catalog_path)
            name = result['name']
            if not name:
                name = _next_indexed_name(
                    data['endpoints'], _slug_alias(result['model'])
                )
            entry = {'engine': result['engine'], 'model': result['model']}
            self._write_catalog('endpoints', name, entry)
            self._status(f'added endpoint {name} -> {result["model"]}')
        except Exception as ex:  # noqa: BLE001
            self._status(f'add endpoint failed: {ex}')
            return
        self._reload_catalog()

    def _write_catalog(self, section: str, name: str, entry: dict) -> None:
        from .cli.commands_catalog import _load_raw, _save_raw
        data = _load_raw(self.catalog_path)
        data[section][name] = entry
        _save_raw(self.catalog_path, data)  # validates; raises on a bad write

    def action_suggest(self) -> None:
        if not self.catalog_path:
            self._status('no catalog path — launch the TUI with a catalog to edit')
            return
        self._status('inspecting GPUs and suggesting a catalog…')
        self._do_suggest()

    @work(thread=True, exclusive=True, group='mutate')
    def _do_suggest(self) -> None:
        try:
            from .hardware import detect_inventory
            from .leasing.suggest import suggest_catalog
            inventory = detect_inventory()
            frag = suggest_catalog(inventory, reserve_display_gpu='auto')
            if not frag.get('models'):
                self._after_mutation(
                    'no pooled model fits the detected GPUs — add one with m/n'
                )
                return
            from .cli.commands_catalog import _load_raw, _save_raw
            data = _load_raw(self.catalog_path)
            added = 0
            for sec in ('models', 'endpoints'):
                for nm, val in frag.get(sec, {}).items():
                    if nm not in data[sec]:
                        data[sec][nm] = val
                        added += 1
            _save_raw(self.catalog_path, data)
            self.call_from_thread(self._reload_catalog)
            self._after_mutation(f'suggested catalog merged ({added} new entries)')
        except Exception as ex:  # noqa: BLE001
            self._after_mutation(f'suggest failed: {ex}')

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


def _fmt_ports(row: dict) -> str:
    """Compact published-ports string from a compose ps JSON row."""
    pubs = row.get('Publishers') or []
    bits = []
    for pub in pubs:
        published = pub.get('PublishedPort')
        target = pub.get('TargetPort')
        if published:
            bits.append(f'{published}->{target}')
    return ', '.join(bits)


def run_tui(
    controller,
    catalog,
    *,
    interval: float = 3.0,
    catalog_path: str | Path | None = None,
) -> int:
    """Run the TUI against a built controller + catalog. Returns an exit code."""
    # The narration loguru sink writes to stderr, which would corrupt the
    # full-screen UI — silence it while the TUI owns the terminal.
    try:
        from ._log import logger

        logger.disable('infer_stack')
    except Exception:  # noqa: BLE001
        pass
    app: Any = InferStackTUI(
        controller, catalog, interval=interval, catalog_path=catalog_path
    )
    app.run()
    return 0
