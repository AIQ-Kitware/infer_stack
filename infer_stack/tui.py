"""Optional Textual TUI to monitor and control the leasing stack.

A multi-pane dashboard built to be approachable for someone who has never
touched infer-stack. Each pane carries its own one-line description, its own
action buttons, and (for the heavy ones) collapses with a click so it isn't
polled while hidden:

* **Catalog** (left) — the models + endpoints you can run; Acquire the selected
  endpoint, Suggest a set sized to your GPUs, or add one by hand. Ctrl+click a
  served endpoint to open it in Open WebUI.
* **Leases** + **Deployments** (center) — the live ledger (desired *state* vs what's
  actually *running*, and which GPUs), with Release / Evict / Clean-up.
* **docker** — a collapsible pane with **Logs** and **Containers** (the
  ``docker ps`` view: status/uptime, created, id, ports) tabs.
* **system** — live ``nvidia-smi`` GPUs + host CPU/mem (collapsed by default).
* **api** — send a prompt to a *ready* model through the LiteLLM gateway
  (collapsed by default).

Responsiveness: refresh runs on a worker thread, only the visible tab / expanded
pane's expensive data is polled, and docker's own ``up``/``down`` output is
captured into the logs pane instead of bleeding onto the terminal. The log
source (``proc_factory``) and HTTP client (``http``) are injectable, so the app
is fully exercisable headless via Textual's pilot.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.coordinate import Coordinate
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    Collapsible,
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
from .leasing import LeaseState

ALL_SERVICES = ''  # the Select value meaning "every service"
SELECT_MARK = '✓'  # multi-select marker in the leases/deployments tables
DEFAULT_THEME = 'textual-dark'


# A warm alternative palette, kept registered (selectable from the command
# palette) even though the default is the stock dark theme.
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
    movement along the bar's resize axis since the last event. The bar must span
    its full cross-axis (see CSS) or there is nothing to grab.
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
    """Wizard: add or edit an endpoint (the served API name clients ask for).

    Exposes the runtime knobs that matter for serving — tensor-parallel size,
    max model length, GPU memory fraction, raw extra vLLM args (where data
    parallelism etc. go), and the reclaim policy — mirroring
    ``catalog endpoint add``. Pass ``entry``/``name`` to edit an existing one.
    """

    CSS = """
    _AddEndpointScreen { align: center middle; }
    #dialog {
        width: 78; max-height: 90%; height: auto; padding: 1 2;
        border: round $accent; background: $surface; overflow-y: auto;
    }
    #dialog .hint { color: $text-muted; margin: 0 0 1 0; }
    #dialog Label { margin: 1 0 0 0; color: $text-muted; }
    #dialog Input, #dialog Select { margin: 0 0 1 0; }
    #dialog #buttons { height: auto; align-horizontal: right; margin: 1 0 0 0; }
    #dialog #buttons Button { margin: 0 0 0 2; }
    #vllm-opts, #ollama-opts { height: auto; }
    """

    def __init__(self, models: list[str], *, name: str | None = None,
                 entry: dict | None = None):
        super().__init__()
        self._models = models
        self._edit_name = name
        self._entry = entry or {}

    def _rt(self, key):
        val = (self._entry.get('runtime') or {}).get(key)
        return '' if val is None else str(val)

    def compose(self) -> ComposeResult:
        editing = self._edit_name is not None
        rt = self._entry.get('runtime') or {}
        extra = rt.get('extra_args') or []
        extra_str = ' '.join(extra) if isinstance(extra, list) else str(extra)
        reclaim = (self._entry.get('reclaim') or {}).get('policy', '')
        cur_model = self._entry.get('model')
        cur_engine = self._entry.get('engine', 'vllm')
        with Vertical(id='dialog'):
            yield Label('Edit endpoint' if editing else 'Add an endpoint',
                        classes='title')
            yield Static(
                'The served name clients / Open WebUI request. Pick a model and '
                'engine; the runtime knobs size it on the GPU(s).', classes='hint',
            )
            yield Label('name')
            yield Input(value=self._edit_name or '',
                        placeholder='optional — defaults to <model>-N',
                        id='e-name', disabled=editing)
            yield Label('model')
            model_opts = [(m, m) for m in self._models]
            if model_opts:
                yield Select(model_opts, prompt='model…', id='e-model',
                             value=cur_model if cur_model in self._models
                             else Select.NULL)
            else:
                yield Input(value=cur_model or '', placeholder='model name',
                            id='e-model-text')
            yield Label('engine')
            yield Select([('vllm', 'vllm'), ('ollama', 'ollama')],
                         value=cur_engine, allow_blank=False, id='e-engine')
            with Vertical(id='vllm-opts'):
                yield Label('tensor-parallel size  (GPUs per replica)')
                yield Input(value=self._rt('tensor_parallel_size'),
                            placeholder='int, e.g. 2', id='e-tp')
                yield Label('data-parallel size  (replicas across GPUs)')
                yield Input(value=self._rt('data_parallel_size'),
                            placeholder='int, e.g. 2', id='e-dp')
                yield Label('max model len  (context window, tokens)')
                yield Input(value=self._rt('max_model_len'),
                            placeholder='int, e.g. 8192', id='e-mml')
                yield Label('GPU memory utilization  (0-1, per GPU)')
                yield Input(value=self._rt('gpu_memory_utilization'),
                            placeholder='float, e.g. 0.9', id='e-gpu')
                yield Label('max concurrent sequences')
                yield Input(value=self._rt('max_num_seqs'),
                            placeholder='int, optional', id='e-seqs')
                yield Label('prefix caching')
                yield Select([('default', ''), ('on', 'on'), ('off', 'off')],
                             value='on' if rt.get('enable_prefix_caching') else '',
                             allow_blank=False, id='e-prefix')
                yield Label('extra vLLM args  (raw flags — dtype etc. go here)')
                yield Input(value=extra_str,
                            placeholder='--dtype=half --enforce-eager',
                            id='e-extra')
            with Vertical(id='ollama-opts'):
                yield Label('host  (runtime host name from the catalog)')
                yield Input(value=self._entry.get('host', ''),
                            placeholder='e.g. ollama-local', id='e-host')
                yield Label('extra runtime  (KEY=VALUE, space-separated)')
                yield Input(value=self._kv_str(rt),
                            placeholder='num_ctx=8192 keep_alive=5m', id='e-orun')
            yield Label('reclaim policy  (when idle)')
            yield Select(
                [('default', ''), ('keep-warm', 'keep-warm'), ('stop', 'stop'),
                 ('scale-to-zero', 'scale-to-zero')],
                value=reclaim or '', allow_blank=False, id='e-reclaim',
            )
            with Horizontal(id='buttons'):
                yield Button('Cancel', id='cancel')
                yield Button('Save' if editing else 'Add endpoint',
                             variant='primary', id='ok')

    def on_mount(self) -> None:
        self._show_engine(self._entry.get('engine', 'vllm') or 'vllm')

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == 'e-engine':
            engine = 'vllm' if event.value is Select.NULL else str(event.value)
            self._show_engine(engine)

    def _show_engine(self, engine: str) -> None:
        try:
            self.query_one('#vllm-opts').display = engine == 'vllm'
            self.query_one('#ollama-opts').display = engine == 'ollama'
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _kv_str(runtime: dict) -> str:
        return ' '.join(
            f'{k}={v}' for k, v in (runtime or {}).items() if k != 'extra_args'
        )

    @staticmethod
    def _num(raw: str, cast):
        raw = raw.strip()
        return cast(raw) if raw else None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'cancel':
            self.dismiss(None)
            return
        try:
            value = self.query_one('#e-model', Select).value
            model = '' if value is Select.NULL else str(value)
        except Exception:  # noqa: BLE001 - free-text fallback when no models yet
            model = self.query_one('#e-model-text', Input).value.strip()
        if not model:
            self.query_one(Label).update('pick (or type) a model')
            return
        name = self.query_one('#e-name', Input).value.strip()
        engine = str(self.query_one('#e-engine', Select).value or 'vllm')
        reclaim = str(self.query_one('#e-reclaim', Select).value or '')
        result: dict[str, Any] = {
            'name': self._edit_name or name or None,
            'model': model, 'engine': engine, 'reclaim': reclaim,
        }
        try:
            if engine == 'vllm':
                result.update({
                    'tensor_parallel': self._num(self._v('e-tp'), int),
                    'data_parallel': self._num(self._v('e-dp'), int),
                    'max_model_len': self._num(self._v('e-mml'), int),
                    'gpu_mem': self._num(self._v('e-gpu'), float),
                    'max_num_seqs': self._num(self._v('e-seqs'), int),
                    'prefix_caching': str(
                        self.query_one('#e-prefix', Select).value or ''),
                    'extra_args': self._v('e-extra'),
                })
            else:
                result.update({
                    'host': self._v('e-host'),
                    'ollama_runtime': self._v('e-orun'),
                })
        except ValueError:
            self.query_one(Label).update(
                'parallel sizes / max-len / seqs must be ints, gpu-mem a float')
            return
        self.dismiss(result)

    def _v(self, wid: str) -> str:
        return self.query_one(f'#{wid}', Input).value


class _ConfirmScreen(ModalScreen):
    """A small yes/no confirmation for destructive actions."""

    CSS = """
    _ConfirmScreen { align: center middle; }
    #dialog {
        width: 64; height: auto; padding: 1 2;
        border: round $error; background: $surface;
    }
    #dialog Horizontal { height: auto; align-horizontal: right; }
    #dialog Button { margin: 0 0 0 2; }
    """

    def __init__(self, message: str):
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id='dialog'):
            yield Label(self._message)
            with Horizontal():
                yield Button('Cancel', id='cancel')
                yield Button('Remove', variant='error', id='ok')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == 'ok')


class InferStackTUI(App):
    """Monitor + control the leasing stack across panes."""

    TITLE = 'infer-stack'
    SUB_TITLE = 'leasing dashboard'

    CSS = """
    Screen { layout: vertical; }

    #body { height: 1fr; padding: 0 1; }
    #sidebar { width: 38; min-width: 10; }
    #vsplit { width: 1; height: 1fr; background: $panel; }
    #vsplit:hover { background: $accent; }
    #main { width: 1fr; }
    #tables { height: 1fr; layout: vertical; }
    #hsplit, #csplit, #tsplit {
        width: 1fr; height: 1; background: $panel; margin: 0 0 1 0;
    }
    #hsplit:hover, #csplit:hover, #tsplit:hover { background: $accent; }

    /* one-line, per-pane descriptions (replaces the old global intro) */
    .desc { height: auto; color: $text-muted; padding: 0 1; }
    #catalog-help { height: auto; color: $text-muted; padding: 0 1; }

    #endpoint-actions, #lease-actions, #deployment-actions, #model-actions {
        height: auto; margin: 0 0 1 0;
    }
    /* width: 1fr + min-width: 0 so several buttons share a narrow sidebar row
       instead of overflowing it (Button's default min-width is 16). */
    #endpoint-actions Button, #model-actions Button { width: 1fr; min-width: 0; }
    #lease-actions Button, #deployment-actions Button {
        margin: 0 1 0 0; min-width: 8;
    }
    /* compact buttons: 1 row, no border box, so action bars don't eat space.
       text-wrap: nowrap is load-bearing, not cosmetic: the endpoint/model
       buttons above are min-width: 0, so in a narrow sidebar their content box
       can shrink to ~2 cells. Textual's Button carries line-pad: 1, and its
       wrapping path folds the label at (width - line_pad*2). At width 2 that is
       0, and rich's chop_cells does range(0, n, 0) -> ValueError, crashing the
       render. nowrap skips that fold path entirely. (We can't just set
       line-pad: 0 -- Textual's integer CSS parser rejects 0.) */
    #endpoint-actions Button, #lease-actions Button, #deployment-actions Button,
    #model-actions Button, #api-controls Button {
        height: 1; border: none; padding: 0 1; text-wrap: nowrap;
    }
    #api { padding: 1 2; }

    #endpoints, #models, #leases-pane, #deployments-pane, #docker, #system, #api {
        border: round $surface;
        background: $boost;
        padding: 0 1;
        margin: 0 0 1 0;
        border-title-color: $text-muted;
        border-title-align: left;
        border-subtitle-color: $text-muted;
        border-subtitle-align: right;
    }
    #endpoints:focus, #models:focus, #leases-pane:focus-within,
    #deployments-pane:focus-within, #docker:focus-within, #system:focus-within,
    #api:focus-within {
        border: round $accent;
        border-title-color: $accent;
    }

    #endpoints { height: 1fr; min-height: 5; }
    #models { height: 8; min-height: 4; }   /* height set via _apply_sizes */
    /* leases/deployments are separate panes split by #tsplit: leases gets a
       resizable fixed height (_apply_sizes), deployments takes the rest. Their
       tables flex to fill each pane around the desc + action rows. */
    #leases-pane { height: 14; min-height: 6; }   /* height set via _apply_sizes */
    #deployments-pane { height: 1fr; min-height: 6; }
    #leases, #deployments { height: 1fr; min-height: 3; }
    #docker-tabs { height: 16; min-height: 8; }
    #logsvc { margin: 0 0 1 0; }
    #logs, #ps { height: 1fr; background: $surface; }
    #gpus, #api-out { height: 8; background: $surface; }
    #sysinfo { height: auto; color: $text-muted; padding: 0 1; }
    .hint { height: auto; color: $text-muted; padding: 0 1; }
    #api-controls { height: auto; margin: 0 0 1 0; }
    #api-controls Select { width: 1fr; }
    #api-controls Button { margin: 0 0 0 1; min-width: 10; }
    #api-prompt { margin: 0 0 1 0; }
    #api-urls { height: auto; color: $text-muted; padding: 0 1; margin: 0 0 1 0; }
    #api-curl { height: auto; color: $text-muted; padding: 0 1; }
    #api-extra { height: auto; margin: 0 0 1 0; }
    #api-extra Button { margin: 0 1 0 0; min-width: 12; }

    #status { dock: bottom; height: 1; padding: 0 2; color: $text-muted; }

    #settings { padding: 1 2; }
    #settings Label { margin: 1 0 0 0; color: $text-muted; }
    #settings Input, #settings Select { margin: 0 0 1 0; width: 64; }
    #settings-actions { height: auto; margin: 1 0 0 0; }
    #compose-path { height: auto; color: $text-muted; padding: 0 1; margin: 0 0 1 0; }
    #compose-actions { height: auto; }
    #compose-actions Button { margin: 0 1 0 0; }
    """

    BINDINGS = [
        # Truly global controls stay in the footer.
        ('r', 'refresh', 'Refresh'),
        ('tab', 'focus_next', 'Next pane'),
        ('q', 'quit', 'Quit'),
        # Pane-scoped actions: keys still work, but they live as buttons under
        # the pane they act on rather than in the global menu.
        Binding('s', 'acquire', 'Acquire', show=False),
        Binding('d', 'release', 'Release', show=False),
        Binding('e', 'evict', 'Evict', show=False),
        Binding('a', 'release_all', 'Release all', show=False),
        Binding('x', 'cleanup', 'Clean up', show=False),
        # Multi-select: space toggles the cursor row in the focused leases/
        # deployments table; release/evict then act on every checked row.
        Binding('space', 'toggle_select', 'Select row', show=False),
        Binding('g', 'suggest', 'Suggest', show=False),
        Binding('m', 'add_model', 'Add model', show=False),
        Binding('n', 'add_endpoint', 'Add endpoint', show=False),
        Binding('o', 'open', 'Open in browser', show=False),
        Binding('y', 'copy_status', 'Copy status', show=False),
        Binding('c', 'toggle_docker', 'Collapse docker', show=False),
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
        http: Any = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.catalog = catalog
        self.interval = interval
        self.catalog_path = Path(catalog_path).expanduser() if catalog_path else None
        self._http = http
        self._proc_factory = proc_factory or self._default_proc_factory()
        self._endpoint_names: list[str] = []
        self._model_names: list[str] = []
        self._lease_ids: list[str] = []
        self._deployment_ids: list[str] = []
        # Multi-select sets (by id) for the leases/deployments tables. Kept by id
        # so a selection survives a poll refresh; pruned to live rows on refill.
        # NOTE: this is a hand-rolled shim — Textual 8.x has no native row
        # multi-select (only text selection). If it gains one (see
        # github.com/Textualize/textual discussion #3606 / PR #6585), this whole
        # marker-column + sel-set + click machinery can be deleted in favour of
        # the native `selected_rows` / `selection_anchor` API.
        self._lease_sel: set[str] = set()
        self._dep_sel: set[str] = set()
        # Range anchor (row index, per table) for shift-click range selection.
        self._sel_anchor: dict[str, int] = {}
        self._last_leases: list[Any] = []   # row-aligned with the leases table
        self._last_deployments: list[Any] = []   # row-aligned with the deployments table
        self._service_options: list[str] = []
        self._log_service: str = ALL_SERVICES
        self._log_proc: Any = None
        self._log_lines: list[str] = []  # mirror of the log pane, for tests
        self._api_lines: list[str] = []  # mirror of the API output, for tests
        self._sidebar_w = 38  # resizable via [ ] or dragging #vsplit
        self._log_h = 16      # resizable via - + or dragging #hsplit
        self._models_h = 8    # resizable by dragging #csplit
        self._leases_h = 14   # resizable by dragging #tsplit
        self._active_tab = 'tab-logs'   # which docker tab is visible
        # heavy panes start collapsed (and therefore unpolled)
        self._collapsed = {'docker': False, 'system': True}
        self._ready_endpoints: list[str] = []

    # -- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id='top'):
            with TabPane('Dashboard', id='tab-dashboard'):
                yield from self._compose_dashboard()
            with TabPane('API', id='tab-api'):
                yield from self._compose_api()
            with TabPane('Settings', id='tab-settings'):
                yield from self._compose_settings()
        yield Static('', id='status')
        yield Footer()

    def _compose_dashboard(self) -> ComposeResult:
        with Horizontal(id='body'):
            with Vertical(id='sidebar'):
                yield Static(
                    'Endpoints — runnable model + engine configs. Acquire one to '
                    'serve it, or Suggest a set sized to your GPUs.', classes='desc',
                )
                yield Static('', id='catalog-help')
                yield DataTable(id='endpoints', cursor_type='row',
                                zebra_stripes=True)
                with Horizontal(id='endpoint-actions'):
                    yield Button('Acquire', id='btn-acquire', variant='primary')
                    yield Button('Suggest', id='btn-suggest')
                    yield Button('Add', id='btn-add-endpoint')
                    yield Button('Edit', id='btn-edit-endpoint')
                    yield Button('Remove', id='btn-remove-endpoint')
                yield _Divider('y', self._drag_models, id='csplit')
                yield Static(
                    'Models — weights an endpoint can serve. Add models here, '
                    'then point an endpoint at one.', classes='desc',
                )
                yield DataTable(id='models', cursor_type='row',
                                zebra_stripes=True)
                with Horizontal(id='model-actions'):
                    yield Button('Add', id='btn-add-model')
                    yield Button('Remove', id='btn-remove-model')
            yield _Divider('x', self._drag_sidebar, id='vsplit')
            with Vertical(id='main'):
                with Vertical(id='tables'):
                    with Vertical(id='leases-pane'):
                        yield Static(
                            'Reservations you hold. Each maps to one deployment '
                            'below (see the deployment column); many leases can '
                            'share one. Release acts on the cursor row, or on '
                            'every row you check (space, or ctrl/shift-click).',
                            classes='desc',
                        )
                        yield DataTable(id='leases', cursor_type='row',
                                        zebra_stripes=True)
                        with Horizontal(id='lease-actions'):
                            yield Button('Release', id='btn-release')
                            yield Button('Release all', id='btn-release-all')
                            yield Button('Clean up', id='btn-cleanup')
                    yield _Divider('y', self._drag_tables, id='tsplit')
                    with Vertical(id='deployments-pane'):
                        yield Static(
                            'Running model deployments and the GPUs they hold. '
                            "The 'leases' column is how many leases hold each. "
                            'Evict an idle one to free its GPU (cursor row, or '
                            'rows checked with space / ctrl/shift-click); Evict '
                            'all idle clears every kept-warm one; Clean up forgets '
                            'stopped ones.', classes='desc',
                        )
                        yield DataTable(id='deployments', cursor_type='row',
                                        zebra_stripes=True)
                        with Horizontal(id='deployment-actions'):
                            yield Button('Evict', id='btn-evict')
                            yield Button('Evict all idle', id='btn-evict-all')
                            yield Button('Clean up', id='btn-cleanup-deployments')
                yield _Divider('y', self._drag_logs, id='hsplit')
                with Collapsible(title='docker', collapsed=False, id='docker'):
                    with TabbedContent(id='docker-tabs'):
                        with TabPane('Logs', id='tab-logs'):
                            yield Select(
                                [('(all services)', ALL_SERVICES)],
                                value=ALL_SERVICES, allow_blank=False,
                                id='logsvc',
                            )
                            yield RichLog(id='logs', highlight=False,
                                          markup=False, max_lines=2000,
                                          wrap=False)
                        with TabPane('Containers', id='tab-containers'):
                            yield DataTable(id='ps', cursor_type='row',
                                            zebra_stripes=True)
                        with TabPane('Control', id='tab-control'):
                            yield Static(
                                'Bring the rendered compose project up or down. '
                                'Output appears in the Logs tab.', classes='hint',
                            )
                            yield Static('', id='compose-path')
                            with Horizontal(id='compose-actions'):
                                yield Button('Compose up', id='btn-compose-up',
                                             variant='primary')
                                yield Button('Compose down',
                                             id='btn-compose-down')
                with Collapsible(title='system', collapsed=True, id='system'):
                    yield Static(
                        'Live GPU utilization (nvidia-smi) and host CPU/memory.',
                        classes='desc',
                    )
                    yield Static('', id='sysinfo')
                    yield DataTable(id='gpus', cursor_type='row',
                                    zebra_stripes=True)

    def _compose_api(self) -> ComposeResult:
        with Vertical(id='api'):
            yield Static(
                'Talk to the LiteLLM gateway. Only models that are up and ready '
                'are listed. Ctrl+click a URL to open it; buttons copy to your '
                'clipboard.', classes='desc',
            )
            yield Static('', id='api-urls', markup=False)
            with Horizontal(id='api-controls'):
                yield Select([], prompt='model…', id='api-model')
                yield Button('Send', id='btn-api-send', variant='primary')
                yield Button('Test all', id='btn-api-test-all')
                yield Button('List models', id='btn-api-list')
            yield Input(placeholder='prompt (default: a short hello)',
                        id='api-prompt')
            yield Static('', id='api-curl')
            with Horizontal(id='api-extra'):
                yield Button('Copy curl', id='btn-api-copy-curl')
                yield Button('Open WebUI', id='btn-open-webui')
            yield RichLog(id='api-out', highlight=False, markup=False,
                          wrap=True, max_lines=500)

    def _compose_settings(self) -> ComposeResult:
        from .cli.commands_leasing import _coerce_bool
        from .paths import data_root, get_setting, settings_path

        def onoff(key, default):
            return 'on' if _coerce_bool(get_setting(key), default) else 'off'

        with Vertical(id='settings'):
            yield Static(
                'Durable settings (settings.yaml). Save writes them; backend / '
                'data-dir / proxy take effect on the next acquire.', classes='desc',
            )
            yield Static(f'file: {settings_path()}', classes='hint')
            yield Label('backend')
            yield Select(
                [('null (dry-run)', 'null'), ('compose', 'compose'),
                 ('kubeai', 'kubeai')],
                value=str(get_setting('backend') or 'null'),
                allow_blank=False, id='set-backend',
            )
            yield Label('data dir  (where weights + state live)')
            yield Input(value=str(get_setting('data_dir') or data_root()),
                        id='set-data-dir')
            yield Label('Open WebUI')
            yield Select([('on', 'on'), ('off', 'off')], value=onoff('ui', True),
                         allow_blank=False, id='set-ui')
            yield Label('reverse proxy  (single-port front door)')
            yield Select([('off', 'off'), ('on', 'on')],
                         value=onoff('reverse_proxy', False),
                         allow_blank=False, id='set-rp')
            yield Label('skip display GPUs during placement')
            yield Select([('off', 'off'), ('on', 'on')],
                         value=onoff('skip_display_gpus', False),
                         allow_blank=False, id='set-skip')
            with Horizontal(id='settings-actions'):
                yield Button('Save settings', variant='primary',
                             id='btn-save-settings')

    def on_mount(self) -> None:
        try:
            self.register_theme(INFER_THEME)
            self.theme = DEFAULT_THEME
        except Exception:  # noqa: BLE001
            pass
        titles = {
            '#endpoints': 'catalog · endpoints', '#models': 'catalog · models',
            '#leases-pane': 'leases', '#deployments-pane': 'deployments',
        }
        for sel, title in titles.items():
            self.query_one(sel).border_title = title
        self.query_one('#endpoints', DataTable).add_columns(
            'endpoint', 'model', 'engine', 'reclaim'
        )
        self.query_one('#models', DataTable).add_columns(
            'model', 'source', 'quant', 'cached'
        )
        leases_tbl = self.query_one('#leases', DataTable)
        leases_tbl.add_column('', width=2, key='sel')  # multi-select marker
        leases_tbl.add_columns(
            'id', 'owner', 'state', 'ttl', 'endpoints', 'deployment'
        )
        deployments_tbl = self.query_one('#deployments', DataTable)
        deployments_tbl.add_column('', width=2, key='sel')  # multi-select marker
        deployments_tbl.add_columns(
            'id', 'engine', 'state', 'running', 'gpus', 'served',
            'leases', 'held by'
        )
        self.query_one('#ps', DataTable).add_columns(
            'service', 'status (uptime)', 'created', 'container id', 'ports'
        )
        self.query_one('#gpus', DataTable).add_columns(
            'gpu', 'name', 'util%', 'mem (used/total)', 'temp'
        )
        compose_file = getattr(self.controller.backend, 'compose_file', None)
        self.query_one('#compose-path', Static).update(
            f'compose file: {compose_file or "(not rendered yet)"}'
        )
        # Capture docker's own chatter (up/down progress on stderr) into the
        # logs pane instead of letting it bleed onto the full-screen terminal.
        self._install_quiet_docker()
        self._apply_sizes()
        self._fill_catalog()
        self._update_api_urls()
        self._update_api_curl()
        self._refresh_now()           # synchronous first paint (snappy + testable)
        self._restart_logs(self._log_service)
        self.set_interval(self.interval, self.action_refresh)
        self.query_one('#endpoints', DataTable).focus()

    def on_unmount(self) -> None:
        self._terminate_logs()

    # -- theming for docker output bleed ----------------------------------

    def _install_quiet_docker(self) -> None:
        """Route ``docker compose`` output to the logs pane, not the terminal."""
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
        self.query_one('#docker-tabs').styles.height = self._log_h
        self.query_one('#models').styles.height = self._models_h
        self.query_one('#leases-pane').styles.height = self._leases_h

    def _drag_sidebar(self, delta: int) -> None:
        # Allow the full width range (down to a sliver, up to nearly all of it),
        # not just the middle — clamp against the actual terminal width.
        hi = max(20, self.size.width - 12)
        self._sidebar_w = max(10, min(hi, self._sidebar_w + delta))
        self._apply_sizes()

    def _drag_logs(self, delta: int) -> None:
        # Dragging the divider down (delta > 0) makes the docker pane shorter.
        self._log_h = max(6, min(60, self._log_h - delta))
        self._apply_sizes()

    def _drag_models(self, delta: int) -> None:
        # Divider above the models table (models fixed-height, endpoints flexes):
        # drag down = bar follows the cursor down = models shorter.
        self._models_h = max(3, min(40, self._models_h - delta))
        self._apply_sizes()

    def _drag_tables(self, delta: int) -> None:
        # Divider between the leases pane (fixed-height) and deployments pane
        # (flexes): drag down = bar follows the cursor down = leases taller.
        hi = max(8, self.size.height - 10)
        self._leases_h = max(6, min(hi, self._leases_h + delta))
        self._apply_sizes()

    def action_sidebar_narrower(self) -> None:
        self._drag_sidebar(-4)

    def action_sidebar_wider(self) -> None:
        self._drag_sidebar(4)

    def action_logs_shorter(self) -> None:
        self._drag_logs(2)

    def action_logs_taller(self) -> None:
        self._drag_logs(-2)

    def action_toggle_docker(self) -> None:
        col = self.query_one('#docker', Collapsible)
        col.collapsed = not col.collapsed

    def action_copy_status(self) -> None:
        """Copy the status line (often a URL/command) to the clipboard."""
        text = str(self.query_one('#status', Static).render())
        if not self._copy(text):
            self._status('copy failed — install wl-copy/xclip, or enable OSC 52')

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
        self._model_names = []
        hub = self._hf_hub_dir()
        for name in sorted(self.catalog.models):
            m = self.catalog.models[name]
            source = getattr(m, 'source', '') or ''
            quant = getattr(m, 'quantization', None) or '-'
            models.add_row(name, source, quant, self._cached_label(source, hub))
            self._model_names.append(name)
        self._update_catalog_help()

    @staticmethod
    def _hf_hub_dir() -> Path | None:
        """Host Hugging Face hub cache dir (best-effort) for a 'cached?' check."""
        try:
            from .config import default_state_paths
            hf = default_state_paths().get('hf_cache')
            return Path(hf).expanduser() / 'hub' if hf else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _cached_label(source: str, hub: Path | None) -> str:
        """'yes'/'no'/'-' whether a model's weights look present (no slow du)."""
        if not source:
            return '-'
        if not source.startswith('hf://'):
            try:
                return 'yes' if Path(source).expanduser().exists() else 'no'
            except Exception:  # noqa: BLE001
                return '-'
        if hub is None:
            return '?'
        repo = source[len('hf://'):].split('@', 1)[0]
        try:
            return 'yes' if (hub / f'models--{repo.replace("/", "--")}').exists() \
                else 'no'
        except Exception:  # noqa: BLE001
            return '-'

    def _sync_api_models(self, names: list[str]) -> None:
        """Point the API model selector at the currently-ready endpoints only."""
        if names == self._ready_endpoints:
            return
        self._ready_endpoints = names
        select = self.query_one('#api-model', Select)
        current = None if select.value is Select.NULL else select.value
        select.set_options([(n, n) for n in names])
        if current in names:
            select.value = current
        elif names:
            select.value = names[0]

    def _update_catalog_help(self) -> None:
        help_ = self.query_one('#catalog-help', Static)
        if not self._endpoint_names:
            help_.update(
                'No endpoints yet. Press [b]g[/b] to suggest a set sized to '
                'your GPUs, or [b]m[/b]/[b]n[/b] to add a model/endpoint.'
            )
        else:
            help_.update(
                'Select an endpoint and Acquire it. Ctrl+click a served one to '
                'open it in Open WebUI.'
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
        """Gather ledger + observed state. Safe to call off the UI thread.

        Only data for a *visible* (expanded, active) pane is polled — collapse
        the docker / system panes (or sit on another docker tab) and the
        expensive ``ps`` / nvidia-smi calls stop.
        """
        try:
            self.controller.ledger.sweep()
            leases, deployments = self.controller.ledger.status()
            observed, assignments = _placement_view(self.controller)
            data: dict[str, Any] = {
                'leases': leases, 'deployments': deployments, 'observed': observed,
                'assignments': assignments,
            }
            if not self._collapsed['docker'] and self._active_tab == 'tab-containers':
                data['ps'] = self._compose_ps_rows()
            if not self._collapsed['system']:
                data['gpus'] = self._gpu_rows()
                data['sysinfo'] = self._system_line()
            return data
        except Exception as ex:  # noqa: BLE001 - a monitor must never crash
            return {'error': str(ex)}

    def _render(self, data: dict[str, Any]) -> None:
        if 'error' in data:
            self._status(f'refresh error: {data["error"]}')
            return
        self._last_leases = data['leases']
        self._last_deployments = data['deployments']
        self._fill_leases(data['leases'])
        self._fill_deployments(
            data['deployments'], data['observed'], data['assignments'], data['leases']
        )
        if 'ps' in data:
            self._fill_ps(data['ps'])
        if 'gpus' in data:
            self._fill_gpus(data['gpus'])
            self.query_one('#sysinfo', Static).update(data.get('sysinfo', ''))
        # Ready = endpoints served by a deployment that is actually running.
        ready: set[str] = set()
        for g in data['deployments']:
            if g.id in data['observed']:
                ready.update(g.served)
        self._sync_api_models(sorted(ready))
        self._update_summary(data['leases'], data['deployments'], data['observed'])
        self._sync_log_services()

    def _sync_pane_state(self) -> None:
        """Read live collapse + active-tab state (UI thread) so the polling gate
        in ``_collect`` reflects reality — Collapsible.Toggled doesn't fire on
        every path, so don't depend on it alone."""
        try:
            self._collapsed['docker'] = \
                self.query_one('#docker', Collapsible).collapsed
            self._collapsed['system'] = \
                self.query_one('#system', Collapsible).collapsed
            self._active_tab = self.query_one('#docker-tabs', TabbedContent).active
        except Exception:  # noqa: BLE001
            pass

    def _refresh_now(self) -> None:
        """Synchronous collect + render (first paint; also what tests rely on)."""
        self._sync_pane_state()
        self._render(self._collect())

    @work(thread=True, exclusive=True, group='refresh')
    def _refresh_bg(self) -> None:
        data = self._collect()
        self.call_from_thread(self._render, data)

    def action_refresh(self) -> None:
        self._sync_pane_state()   # capture pane state on the UI thread first
        self._refresh_bg()

    def _update_summary(self, leases, deployments, observed) -> None:
        active = sum(1 for le in leases if str(le.state) == 'active')
        running = sum(1 for g in deployments if g.id in observed)
        try:
            self.query_one('#docker', Collapsible).title = (
                f'docker — {running} running'
            )
            self.query_one('#leases-pane').border_title = (
                f'leases — {active} active / {len(leases)}'
            )
            self.query_one('#deployments-pane').border_title = (
                f'deployments — {running} running / {len(deployments)}'
            )
        except Exception:  # noqa: BLE001
            pass

    def _fill_leases(self, leases) -> None:
        table = self.query_one('#leases', DataTable)
        cursor = table.cursor_row
        table.clear()
        self._lease_ids = []
        self._lease_sel &= {le.id for le in leases}  # drop selections for gone rows
        for le in leases:
            # The 'deployment' column is the join key: it lists the same deployment
            # id(s) shown in the deployments pane, so lease -> deployment is visible.
            table.add_row(
                SELECT_MARK if le.id in self._lease_sel else '',
                le.id, le.owner, str(le.state), _lease_ttl(le),
                ','.join(le.endpoints) or '-',
                ','.join(le.deployment_ids) or '-',
            )
            self._lease_ids.append(le.id)
        self._restore_cursor(table, cursor)

    def _fill_deployments(self, deployments, observed, assignments, leases) -> None:
        table = self.query_one('#deployments', DataTable)
        cursor = table.cursor_row
        table.clear()
        self._deployment_ids = []
        self._dep_sel &= {g.id for g in deployments}  # drop selections for gone rows
        # owners of the active leases holding each deployment (the "many" side)
        owners: dict[str, list[str]] = {}
        for le in leases:
            if le.state == LeaseState.ACTIVE:
                for gid in le.deployment_ids:
                    owners.setdefault(gid, []).append(le.owner)
        for g in deployments:
            table.add_row(
                SELECT_MARK if g.id in self._dep_sel else '',
                g.id, g.engine, str(g.state),
                _running_label(g.id, observed),
                _gpu_label(g.id, observed, assignments),
                ','.join(sorted(g.served)) or '-',
                str(g.demand), ','.join(owners.get(g.id, [])) or '-',
            )
            self._deployment_ids.append(g.id)
        self._restore_cursor(table, cursor)

    def _fill_ps(self, rows) -> None:
        table = self.query_one('#ps', DataTable)
        cursor = table.cursor_row
        table.clear()
        for row in rows:
            table.add_row(
                row['service'], row['status'], row['created'] or '-',
                row['id'] or '-', row['ports'] or '-',
            )
        if not rows:
            table.add_row('(nothing running)', '-', '-', '-', '-')
        self._restore_cursor(table, cursor)

    def _fill_gpus(self, rows) -> None:
        table = self.query_one('#gpus', DataTable)
        cursor = table.cursor_row
        table.clear()
        if rows is None:
            table.add_row('—', 'nvidia-smi not found on this host', '-', '-', '-')
        elif not rows:
            table.add_row('—', '(no GPUs reported)', '-', '-', '-')
        else:
            for r in rows:
                idx, name, util, used, total, temp = r
                table.add_row(idx, name, f'{util}%',
                              f'{used}/{total} MiB', f'{temp}°C')
        self._restore_cursor(table, cursor)

    @staticmethod
    def _restore_cursor(table: DataTable, row: int) -> None:
        if table.row_count:
            table.move_cursor(row=min(max(row, 0), table.row_count - 1))

    # -- system info -------------------------------------------------------

    def _gpu_rows(self) -> list[list[str]] | None:
        """nvidia-smi rows, or ``None`` when nvidia-smi is unavailable."""
        import shutil

        if not shutil.which('nvidia-smi'):
            return None
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,name,utilization.gpu,memory.used,'
                 'memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:  # noqa: BLE001
            return []
        rows = []
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                rows.append(parts[:6])
        return rows

    def _system_line(self) -> str:
        """A compact host CPU/mem/load line from /proc (best-effort)."""
        import os

        bits = []
        try:
            la = Path('/proc/loadavg').read_text().split()[:3]
            bits.append('load ' + ' '.join(la))
        except Exception:  # noqa: BLE001
            pass
        try:
            mem = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, _, val = line.partition(':')
                mem[key] = val.strip()
            total = int(mem['MemTotal'].split()[0])      # kB
            avail = int(mem['MemAvailable'].split()[0])
            used_g = (total - avail) / 1024 / 1024
            total_g = total / 1024 / 1024
            bits.append(f'mem {used_g:.1f}/{total_g:.1f} GiB')
        except Exception:  # noqa: BLE001
            pass
        bits.append(f'cpus {os.cpu_count()}')
        return '   ·   '.join(bits)

    # -- docker ps ---------------------------------------------------------

    def _compose_ps_rows(self) -> list[dict[str, str]]:
        """Best-effort ``docker compose ps`` rows."""
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
            ports = _fmt_ports(row) or str(row.get('Ports') or '')
            cid = str(row.get('ID') or '')[:12]
            # `Status` is the docker-ps STATUS column ("Up 3 minutes") — it
            # carries the uptime; `CreatedAt`/`RunningFor` give the age.
            created = str(row.get('CreatedAt') or row.get('RunningFor') or '')
            result.append({
                'service': str(row.get('Service') or row.get('Name') or '?'),
                'status': str(row.get('Status') or row.get('State') or '?'),
                'created': created,
                'id': cid,
                'ports': ports,
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
        select.value = (
            self._log_service if self._log_service in names else ALL_SERVICES
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == 'api-model':
            self._update_api_curl()
            return
        if event.select.id != 'logsvc':
            return
        service = '' if event.value is Select.NULL else str(event.value)
        if service != self._log_service:
            self._log_service = service
            self._restart_logs(service)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == 'api-prompt':
            self._update_api_curl()

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
                self._append_log, '(no compose project yet — acquire a model)'
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

    # -- docker tab / collapse plumbing ------------------------------------

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        # Only the docker sub-tabs gate polling; the top-level Dashboard/Settings
        # tabs fire this too — ignore those.
        if getattr(event.tabbed_content, 'id', None) != 'docker-tabs':
            return
        self._active_tab = self.query_one('#docker-tabs', TabbedContent).active
        self.action_refresh()   # fill the newly-shown tab right away

    def on_collapsible_toggled(self, event: Collapsible.Toggled) -> None:
        self._sync_pane_state()
        if self._collapsed['docker']:
            self._terminate_logs()
        elif self._log_proc is None:
            self._restart_logs(self._log_service)
        self.action_refresh()

    # -- helpers + actions -------------------------------------------------

    def _status(self, message: str) -> None:
        self.query_one('#status', Static).update(message)

    def _copy(self, text: str) -> bool:
        """Copy to the system clipboard. Prefer OS tools (reliable on a desktop:
        wl-copy / xclip / xsel / pbcopy) since many terminals don't honor the
        OSC 52 escape Textual uses; fall back to OSC 52."""
        import shutil
        import subprocess

        for cmd in (['wl-copy'], ['xclip', '-selection', 'clipboard'],
                    ['xsel', '--clipboard', '--input'], ['pbcopy']):
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, input=text.encode(), check=True,
                                   timeout=5)
                    return True
                except Exception:  # noqa: BLE001
                    continue
        try:
            self.copy_to_clipboard(text)   # OSC 52 (terminal must allow it)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _selected(self, table_id: str, ids: list[str]) -> str | None:
        row = self.query_one(f'#{table_id}', DataTable).cursor_row
        return ids[row] if 0 <= row < len(ids) else None

    def _target_ids(
        self, table_id: str, ids: list[str], sel: set[str]
    ) -> list[str]:
        """Ids an action should act on: every checked row, else the cursor row.

        Returned in table order. Multi-select (space) wins when anything is
        checked; otherwise we fall back to the single cursor row so the buttons
        behave exactly as before when nothing is checked.
        """
        if sel:
            return [i for i in ids if i in sel]
        one = self._selected(table_id, ids)
        return [one] if one else []

    def _table_sel(self, tid: str | None) -> tuple[list[str], set[str]] | None:
        """(ids, selection-set) for the leases/deployments table, else None."""
        if tid == 'leases':
            return self._lease_ids, self._lease_sel
        if tid == 'deployments':
            return self._deployment_ids, self._dep_sel
        return None

    def _repaint_marks(self, tid: str) -> None:
        """Redraw the marker column of a table from its current selection set."""
        res = self._table_sel(tid)
        if res is None:
            return
        ids, sel = res
        table = self.query_one(f'#{tid}', DataTable)
        for r, gid in enumerate(ids):
            if r < table.row_count:
                mark = SELECT_MARK if gid in sel else ''
                table.update_cell_at(Coordinate(r, 0), mark)

    def _click_select(self, tid: str, row: int, *, shift: bool, ctrl: bool) -> None:
        """Apply a (possibly modified) click on row ``row`` to the selection.

        Desktop semantics: ctrl/cmd toggles one row (discontiguous); shift
        extends a contiguous range from the anchor; a plain click collapses the
        multi-selection back to the single cursor row. Pure over (tid, row,
        modifiers) so it's unit-testable without synthesizing mouse events.
        """
        res = self._table_sel(tid)
        if res is None:
            return
        ids, sel = res
        if not (0 <= row < len(ids)):
            return
        if shift:
            anchor = self._sel_anchor.get(tid, row)
            lo, hi = sorted((anchor, min(row, len(ids) - 1)))
            sel.update(ids[lo:hi + 1])
        elif ctrl:
            gid = ids[row]
            sel.discard(gid) if gid in sel else sel.add(gid)
            self._sel_anchor[tid] = row
        else:  # plain click -> collapse selection to the cursor row
            sel.clear()
            self._sel_anchor[tid] = row
        self._repaint_marks(tid)
        if sel:
            self._status(f'{len(sel)} checked in {tid}')

    def action_toggle_select(self) -> None:
        """Space: toggle the cursor row of the focused leases/deployments table."""
        focused = self.focused
        tid = getattr(focused, 'id', None)
        res = self._table_sel(tid)
        if res is None:
            return
        ids, sel = res
        row = focused.cursor_row
        if not (0 <= row < len(ids)):
            return
        gid = ids[row]
        sel.discard(gid) if gid in sel else sel.add(gid)
        self._sel_anchor[tid] = row
        self._repaint_marks(tid)
        self._status(f'{len(sel)} checked in {tid} (space/ctrl-click toggles)')

    def action_acquire(self) -> None:
        name = self._selected('endpoints', self._endpoint_names)
        if not name:
            self._status('select an endpoint in the catalog to acquire')
            return
        self._status(f'acquiring {name}… (docker output appears in the logs pane)')
        self._do_acquire(name)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter on the endpoints table acquires that endpoint.
        if event.data_table.id == 'endpoints':
            self.action_acquire()

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        # Spell out the many-to-one link in the status bar as you move the
        # cursor: a lease points at one deployment; a deployment is held by many.
        tid = event.data_table.id
        row = event.cursor_row
        if tid == 'leases' and 0 <= row < len(self._last_leases):
            le = self._last_leases[row]
            deps = ', '.join(le.deployment_ids) or '—'
            held = {g.id: g.demand for g in self._last_deployments}
            n = max((held.get(gid, 0) for gid in le.deployment_ids), default=0)
            others = f' (1 of {n} lease(s) on it)' if n > 1 else ''
            self._status(f'lease {le.id} → deployment {deps}{others}')
        elif tid == 'deployments' and 0 <= row < len(self._last_deployments):
            g = self._last_deployments[row]
            owners = [
                le.owner for le in self._last_leases
                if g.id in le.deployment_ids and le.state == LeaseState.ACTIVE
            ]
            who = ', '.join(owners) or '—'
            self._status(
                f'deployment {g.id} ← held by {g.demand} lease(s): {who}'
            )

    def on_click(self, event: events.Click) -> None:
        # Resolve which dashboard element was clicked (walk up to a known id).
        zone = None
        try:
            node, _ = self.screen.get_widget_at(event.screen_x, event.screen_y)
        except Exception:  # noqa: BLE001
            node = None
        while node is not None:
            nid = getattr(node, 'id', None)
            if nid in ('leases', 'deployments', 'endpoints', 'api-urls'):
                zone = nid
                break
            node = getattr(node, 'parent', None)

        ctrl = getattr(event, 'ctrl', False) or getattr(event, 'meta', False)
        shift = getattr(event, 'shift', False)

        # Multi-select on the leases/deployments tables via ctrl/shift-click (the
        # DataTable moved its cursor to the clicked row first, so cursor_row is
        # the click target). A plain click collapses any multi-selection.
        if zone in ('leases', 'deployments'):
            table = self.query_one(f'#{zone}', DataTable)
            self._click_select(zone, table.cursor_row, shift=shift, ctrl=ctrl)
            return

        # Ctrl+click a served endpoint / the API URLs -> open in the browser.
        if ctrl and zone == 'endpoints':
            self.action_open()
            event.stop()
        elif ctrl and zone == 'api-urls':
            self.action_open_webui()
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            'btn-acquire': self.action_acquire,
            'btn-release': self.action_release,
            'btn-release-all': self.action_release_all,
            'btn-evict': self.action_evict,
            'btn-evict-all': self.action_evict_all,
            'btn-cleanup': self.action_cleanup,
            'btn-cleanup-deployments': self.action_cleanup,
            'btn-suggest': self.action_suggest,
            'btn-add-model': self.action_add_model,
            'btn-add-endpoint': self.action_add_endpoint,
            'btn-edit-endpoint': self.action_edit_endpoint,
            'btn-remove-endpoint': self.action_remove_endpoint,
            'btn-remove-model': self.action_remove_model,
            'btn-api-send': self.action_api_send,
            'btn-api-test-all': self.action_api_test_all,
            'btn-api-list': self.action_api_list_models,
            'btn-api-copy-curl': self.action_api_copy_curl,
            'btn-open-webui': self.action_open_webui,
            'btn-save-settings': self._on_save_settings,
            'btn-compose-up': self.action_compose_up,
            'btn-compose-down': self.action_compose_down,
        }
        handler = handlers.get(event.button.id or '')
        if handler:
            handler()

    def _on_save_settings(self) -> None:
        from .paths import load_settings, save_settings

        try:
            s = load_settings()
            s['backend'] = str(self.query_one('#set-backend', Select).value)
            data_dir = self.query_one('#set-data-dir', Input).value.strip()
            if data_dir:
                s['data_dir'] = data_dir
            s['ui'] = self.query_one('#set-ui', Select).value == 'on'
            s['reverse_proxy'] = self.query_one('#set-rp', Select).value == 'on'
            s['skip_display_gpus'] = (
                self.query_one('#set-skip', Select).value == 'on'
            )
            path = save_settings(s)
            self._status(f'saved settings → {path}')
        except Exception as ex:  # noqa: BLE001
            self._status(f'save settings failed: {ex}')

    def action_release(self) -> None:
        ids = self._target_ids('leases', self._lease_ids, self._lease_sel)
        if not ids:
            self._status('select a lease row (or check rows with space) to release')
            return
        self._status(f'releasing {len(ids)} lease(s)…')
        self._do_release(ids)

    def action_release_all(self) -> None:
        self._status('releasing all active leases…')
        self._do_release_all()

    def action_evict(self) -> None:
        ids = self._target_ids(
            'deployments', self._deployment_ids, self._dep_sel
        )
        if not ids:
            self._status(
                'select a deployment row (or check rows with space) to evict'
            )
            return
        self._status(f'evicting {len(ids)} deployment(s)…')
        self._do_evict(ids)

    def action_evict_all(self) -> None:
        self._status('evicting all idle deployments…')
        self._do_evict_all()

    def action_cleanup(self) -> None:
        self._status('cleaning up released/expired leases + stopped deployments…')
        self._do_cleanup()

    # -- docker compose control -------------------------------------------

    def _compose_target(self) -> tuple[Any, str, str] | None:
        """(run, project, compose_file) for the leasing project, or None."""
        backend = self.controller.backend
        path = getattr(backend, 'compose_file', None)
        run = getattr(backend, 'run', None)
        if not path or not run or not Path(path).exists():
            return None
        return run, str(getattr(backend, 'project', 'infer-stack')), str(path)

    def action_compose_up(self) -> None:
        if self._compose_target() is None:
            self._status('nothing rendered yet — acquire a model first')
            return
        self._status('docker compose up… (output in the Logs tab)')
        self._do_compose(['up', '-d', '--remove-orphans'], 'up')

    def action_compose_down(self) -> None:
        if self._compose_target() is None:
            self._status('nothing rendered yet — nothing to bring down')
            return
        self._status('docker compose down…')
        self._do_compose(['down', '--remove-orphans'], 'down')

    @work(thread=True, exclusive=True, group='mutate')
    def _do_compose(self, args: list[str], label: str) -> None:
        target = self._compose_target()
        if target is None:
            self._after_mutation('nothing rendered yet')
            return
        run, project, path = target
        try:
            run(['docker', 'compose', '-p', project, '-f', path, *args])
            msg = f'compose {label} done'
        except Exception as ex:  # noqa: BLE001
            msg = f'compose {label} failed: {ex}'
        self._after_mutation(msg)

    # -- open in browser ---------------------------------------------------

    def _ui_url(self, endpoint: str) -> str | None:
        base = self._openwebui_url()
        return f'{base}/?models={endpoint}' if base else None

    def _openwebui_url(self) -> str | None:
        port = getattr(self.controller.backend, 'ui_port', None)
        return f'http://localhost:{port}' if port else None

    def _served_endpoints(self) -> set[str]:
        try:
            leases, _ = self.controller.ledger.status()
        except Exception:  # noqa: BLE001
            return set()
        served: set[str] = set()
        for le in leases:
            if le.state == LeaseState.ACTIVE:
                served.update(le.endpoints)
        return served

    def action_open(self) -> None:
        name = self._selected('endpoints', self._endpoint_names)
        if not name:
            self._status('select an endpoint to open in the browser')
            return
        url = self._ui_url(name)
        if not url:
            self._status('no Open WebUI URL (compose backend only)')
            return
        served = name in self._served_endpoints()
        opened = False
        try:
            import webbrowser
            opened = webbrowser.open(url)
        except Exception:  # noqa: BLE001
            opened = False
        note = '' if served else '  (not served yet — start it first)'
        tail = '' if opened else '  [copy this URL into your browser]'
        self._status(f'open {url}{note}{tail}')

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

    def action_edit_endpoint(self) -> None:
        name = self._selected('endpoints', self._endpoint_names)
        if not name:
            self._status('select an endpoint to edit')
            return
        if not self.catalog_path:
            self._status('no catalog path — launch the TUI with a catalog to edit')
            return
        if name in self._served_endpoints():
            self._status(f'{name} is actively served — release it before editing')
            return
        from .cli.commands_catalog import _load_raw
        entry = _load_raw(self.catalog_path)['endpoints'].get(name, {})
        self.push_screen(
            _AddEndpointScreen(sorted(self.catalog.models), name=name, entry=entry),
            self._on_add_endpoint,
        )

    @staticmethod
    def _endpoint_entry(result: dict) -> dict:
        """Build a catalog endpoint entry from a wizard result (mirrors the CLI).

        vLLM and Ollama expose different knobs: vLLM gets the parallelism /
        context / memory runtime keys (dtype etc. via extra_args), Ollama gets a
        host plus free-form KEY=VALUE runtime.
        """
        import shlex

        entry: dict[str, Any] = {'engine': result['engine'],
                                 'model': result['model']}
        runtime: dict[str, Any] = {}
        if result['engine'] == 'vllm':
            keymap = {
                'tensor_parallel': 'tensor_parallel_size',
                'data_parallel': 'data_parallel_size',
                'max_model_len': 'max_model_len',
                'max_num_seqs': 'max_num_seqs',
            }
            for rk, ck in keymap.items():
                if result.get(rk) is not None:
                    runtime[ck] = result[rk]
            if result.get('gpu_mem') is not None:
                runtime['gpu_memory_utilization'] = result['gpu_mem']
            if result.get('prefix_caching') in ('on', 'off'):
                runtime['enable_prefix_caching'] = result['prefix_caching'] == 'on'
            if result.get('extra_args'):
                runtime['extra_args'] = shlex.split(result['extra_args'])
        else:  # ollama
            if result.get('host'):
                entry['host'] = result['host']
            runtime.update(_parse_kv_str(result.get('ollama_runtime', '')))
        if runtime:
            entry['runtime'] = runtime
        if result.get('reclaim'):
            entry['reclaim'] = {'policy': result['reclaim']}
        return entry

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
            self._write_catalog('endpoints', name, self._endpoint_entry(result))
            self._status(f'saved endpoint {name} -> {result["model"]}')
        except Exception as ex:  # noqa: BLE001
            self._status(f'save endpoint failed: {ex}')
            return
        self._reload_catalog()

    def action_remove_endpoint(self) -> None:
        name = self._selected('endpoints', self._endpoint_names)
        if not name or not self.catalog_path:
            self._status('select an endpoint to remove')
            return
        if name in self._served_endpoints():
            self._status(f'{name} is actively served — release it before removing')
            return
        self.push_screen(
            _ConfirmScreen(f"Remove endpoint '{name}' from the catalog?"),
            lambda ok: self._do_remove('endpoints', name) if ok else None,
        )

    def action_remove_model(self) -> None:
        name = self._selected('models', self._model_names)
        if not name or not self.catalog_path:
            self._status('select a model to remove')
            return
        self.push_screen(
            _ConfirmScreen(f"Remove model '{name}'? (endpoints using it will "
                           'block the write)'),
            lambda ok: self._do_remove('models', name) if ok else None,
        )

    def _do_remove(self, section: str, name: str) -> None:
        try:
            from .cli.commands_catalog import _load_raw, _save_raw
            data = _load_raw(self.catalog_path)
            data[section].pop(name, None)
            _save_raw(self.catalog_path, data)  # validates cross-refs
            self._status(f'removed {section[:-1]} {name}')
        except Exception as ex:  # noqa: BLE001
            self._status(f'remove failed: {ex}')
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

    # -- API tester --------------------------------------------------------

    def _litellm(self) -> tuple[str | None, str | None]:
        backend = self.controller.backend
        port = getattr(backend, 'litellm_port', None)
        if not port:
            return None, None
        key = None
        mk = getattr(backend, 'master_key', None)
        try:
            key = mk() if callable(mk) else None
        except Exception:  # noqa: BLE001
            key = None
        return f'http://localhost:{port}', key

    def _http_client(self) -> Any:
        if self._http is not None:
            return self._http
        import requests
        return requests

    def _api_chat(self, model: str, prompt: str) -> str:
        base, key = self._litellm()
        if not base:
            raise RuntimeError('no LiteLLM gateway (needs the compose backend)')
        headers = {'Authorization': f'Bearer {key}'} if key else {}
        resp = self._http_client().post(
            f'{base}/v1/chat/completions',
            json={
                'model': model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 128, 'temperature': 0,
            },
            headers=headers, timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']

    def _curl_for(self, model: str, prompt: str) -> str:
        """The equivalent ``curl`` for a chat-completion against the gateway."""
        import json as _json

        base, key = self._litellm()
        if not base:
            return '# acquire a model first — no LiteLLM gateway yet'
        auth = f" -H 'Authorization: Bearer {key}'" if key else ''
        body = _json.dumps({
            'model': model or '<model>',
            'messages': [{'role': 'user', 'content': prompt or 'hello'}],
        })
        return (f"curl -s {base}/v1/chat/completions{auth} "
                f"-H 'Content-Type: application/json' -d '{body}'")

    def _update_api_urls(self) -> None:
        # Plain text (no Textual [link] markup — it rejects the ':' in URLs);
        # ctrl+click this line opens Open WebUI, or use the Open WebUI button.
        base, _ = self._litellm()
        ui = self._openwebui_url()
        parts = []
        if base:
            parts.append(f'gateway: {base}/v1')
        if ui:
            parts.append(f'open webui: {ui}')
        text = '   ·   '.join(parts) or '(acquire a model to get a gateway URL)'
        try:
            self.query_one('#api-urls', Static).update(text)
        except Exception:  # noqa: BLE001
            pass

    def _update_api_curl(self) -> None:
        model = self._selected_api_model() or '<model>'
        try:
            prompt = self.query_one('#api-prompt', Input).value.strip() or 'hello'
            self.query_one('#api-curl', Static).update(
                self._curl_for(model, prompt))
        except Exception:  # noqa: BLE001
            pass

    def _api_log(self, line: str) -> None:
        self._api_lines.append(line)
        self.query_one('#api-out', RichLog).write(line)

    def _selected_api_model(self) -> str | None:
        value = self.query_one('#api-model', Select).value
        return None if value is Select.NULL else str(value)

    def action_api_send(self) -> None:
        model = self._selected_api_model()
        if not model:
            self._status('no ready models to query (acquire one first)')
            return
        prompt = (self.query_one('#api-prompt', Input).value.strip()
                  or 'Say hello in one short sentence.')
        self._api_log(f'> [{model}] {prompt}')
        self._do_api_send(model, prompt)

    def action_api_test_all(self) -> None:
        models = list(self._ready_endpoints)
        if not models:
            self._status('no ready models to test (acquire one first)')
            return
        self._api_log(f'— testing {len(models)} ready model(s) —')
        self._do_api_test_all(models)

    def action_api_list_models(self) -> None:
        self._api_log('> GET /v1/models  (what the gateway routes)')
        self._do_api_list()

    def action_api_copy_curl(self) -> None:
        text = str(self.query_one('#api-curl', Static).render())
        ok = self._copy(text)
        self._status('copied curl to clipboard' if ok else
                     'copy failed — install wl-copy/xclip, or enable OSC 52')

    def action_open_webui(self) -> None:
        url = self._openwebui_url()
        if not url:
            self._status('no Open WebUI URL (compose backend only)')
            return
        opened = False
        try:
            import webbrowser
            opened = webbrowser.open(url)
        except Exception:  # noqa: BLE001
            opened = False
        self._status(
            f'open {url}' + ('' if opened else '  [copy into your browser]'))

    @work(thread=True, group='api')
    def _do_api_send(self, model: str, prompt: str) -> None:
        try:
            out = self._api_chat(model, prompt)
            self.call_from_thread(self._api_log, f'  [{model}] {out}')
        except Exception as ex:  # noqa: BLE001
            self.call_from_thread(self._api_log, f'  [{model}] ERROR: {ex}')

    @work(thread=True, group='api')
    def _do_api_list(self) -> None:
        base, key = self._litellm()
        if not base:
            self.call_from_thread(self._api_log, '  ERROR: no LiteLLM gateway')
            return
        try:
            headers = {'Authorization': f'Bearer {key}'} if key else {}
            resp = self._http_client().get(
                f'{base}/v1/models', headers=headers, timeout=30)
            resp.raise_for_status()
            ids = [m.get('id') for m in (resp.json().get('data') or [])]
            self.call_from_thread(
                self._api_log, f'  routes: {", ".join(ids) or "(none)"}')
        except Exception as ex:  # noqa: BLE001
            self.call_from_thread(self._api_log, f'  ERROR: {ex}')

    @work(thread=True, group='api')
    def _do_api_test_all(self, models: list[str]) -> None:
        import time

        for model in models:
            start = time.perf_counter()
            try:
                self._api_chat(model, 'Reply with the single word: ok')
                dt = time.perf_counter() - start
                self.call_from_thread(self._api_log, f'  ✓ {model}  ({dt:.1f}s)')
            except Exception as ex:  # noqa: BLE001
                self.call_from_thread(self._api_log, f'  ✗ {model}  {ex}')
        self.call_from_thread(self._api_log, '— done —')

    # Mutations converge the backend (docker up/down, possibly slow) off the UI
    # thread; results + a refresh are marshalled back on.

    @work(thread=True, exclusive=True, group='mutate')
    def _do_acquire(self, name: str) -> None:
        try:
            requests = self.catalog.resolve_names([name])
            self.controller.acquire(
                'manual', requests, ttl_seconds=None, wait=False, apply=True
            )
            msg = f'acquiring {name}'
        except Exception as ex:  # noqa: BLE001
            msg = f'acquire {name} failed: {ex}'
        self._after_mutation(msg)

    @work(thread=True, exclusive=True, group='mutate')
    def _do_release(self, ids: list[str]) -> None:
        try:
            if len(ids) == 1:
                self.controller.release(ids[0])
            else:
                # Release every selected lease in the ledger, then converge once.
                for sid in ids:
                    self.controller.ledger.release(sid)
                self.controller.reconcile()
            self._lease_sel.clear()
            msg = f'released {len(ids)} lease(s)'
        except Exception as ex:  # noqa: BLE001
            msg = f'release failed: {ex}'
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
    def _do_evict(self, ids: list[str]) -> None:
        try:
            # evict() only touches IDLE deployments (others are skipped), so a
            # mixed selection still does the right thing; report what stuck.
            outcome = self.controller.evict(ids)
            n = len(outcome.evicted_deployment_ids)
            self._dep_sel.clear()
            if n:
                msg = f'evicted {n} of {len(ids)} deployment(s)'
            else:
                msg = (f'none of the {len(ids)} selected were idle — '
                       'release their leases first')
        except Exception as ex:  # noqa: BLE001
            msg = f'evict failed: {ex}'
        self._after_mutation(msg)

    @work(thread=True, exclusive=True, group='mutate')
    def _do_evict_all(self) -> None:
        try:
            outcome = self.controller.evict(None)  # every idle deployment
            n = len(outcome.evicted_deployment_ids)
            self._dep_sel.clear()
            msg = (f'evicted {n} idle deployment(s)' if n
                   else 'no idle deployments to evict')
        except Exception as ex:  # noqa: BLE001
            msg = f'evict all failed: {ex}'
        self._after_mutation(msg)

    @work(thread=True, exclusive=True, group='mutate')
    def _do_cleanup(self) -> None:
        try:
            n_leases, n_deployments = self.controller.ledger.prune()
            msg = (f'cleaned up {n_leases} released/expired lease(s) + '
                   f'{n_deployments} stopped deployment(s)')
        except Exception as ex:  # noqa: BLE001
            msg = f'cleanup failed: {ex}'
        self._after_mutation(msg)

    def _after_mutation(self, message: str) -> None:
        self.call_from_thread(self._status, message)
        self.call_from_thread(self.action_refresh)


def _parse_kv_str(text: str) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` pairs (space-separated); values are YAML-typed."""
    import yaml

    out: dict[str, Any] = {}
    for item in (text or '').split():
        if '=' in item:
            key, _, val = item.partition('=')
            out[key.strip()] = yaml.safe_load(val)
    return out


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
