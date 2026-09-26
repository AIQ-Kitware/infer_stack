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


def _front(backend, litellm_port=14042, ui_port=None):
    """Give a NullBackend a gateway front door at these ports."""
    import tempfile

    from infer_stack.leasing.gateway import Gateway

    ports = {'litellm': litellm_port or 0, 'open_webui': ui_port or 0}
    gateway = Gateway(tempfile.mkdtemp(), ports=ports,
                      litellm=bool(litellm_port), ui=bool(ui_port))
    backend.front_door = lambda: type('Front', (), {'gateway': gateway})()
    backend.master_key = lambda: 'sk-test'


class _FakeProc:
    """Stands in for a `docker compose logs -f` process."""

    def __init__(self, lines):
        self.stdout = iter(lines)
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _run(scenario):
    asyncio.run(scenario())


def test_tui_panes_list_catalog_leases_and_deployments():
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
            assert app.query_one('#deployments', DataTable).row_count == 1

    _run(scenario)


def test_tui_relationship_columns_link_lease_to_deployment():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    out = controller.acquire('alice', catalog.resolve_names(['qwen-coder']))
    gid = out.lease.deployment_ids[0]

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            leases = app.query_one('#leases', DataTable)
            deployments = app.query_one('#deployments', DataTable)
            assert 'deployment' in [str(c.label) for c in leases.columns.values()]
            glabels = [str(c.label) for c in deployments.columns.values()]
            assert 'leases' in glabels and 'held by' in glabels
            # the lease row names the deployment id; the deployment row counts
            # the lease (1) and shows the owner — the many-to-one join.
            assert gid in leases.get_row_at(0)
            assert '1' in deployments.get_row_at(0)
            assert 'alice' in deployments.get_row_at(0)
            # selecting a lease explains the link in the status bar
            app.query_one('#leases').focus()
            app.query_one('#leases', DataTable).move_cursor(row=0)
            await pilot.pause()
            status = str(app.query_one('#status').render())
            assert 'deployment' in status

    _run(scenario)


def test_tui_acquire_from_catalog_creates_a_lease():
    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#endpoints').focus()
            await pilot.press('s')                    # acquire the selected endpoint
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert len(leases) == 1
    assert leases[0].state == LeaseState.ACTIVE
    assert 'qwen-coder' in leases[0].endpoints      # first row, sorted


def _endpoint_row_points(app):
    """Screen (x, y) of each endpoint DataTable data row, keyed by row index.

    Scans the rendered cell metadata so it's independent of header height and
    scroll — the same mapping the click handler reads from ``event.style.meta``.
    """
    from textual.widgets import DataTable

    reg = app.query_one('#endpoints', DataTable).region
    points: dict[int, tuple[int, int]] = {}
    for y in range(reg.y, reg.y + reg.height):
        node, _ = app.screen.get_widget_at(reg.x + 2, y)
        if getattr(node, 'id', None) != 'endpoints':
            continue
        row = app.screen.get_style_at(reg.x + 2, y).meta.get('row')
        if isinstance(row, int) and row >= 0:
            points.setdefault(row, (reg.x + 2, y))
    return points


def test_tui_single_click_highlights_but_does_not_acquire():
    """A lone click must not bring an endpoint up (see double-click gate)."""
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            points = _endpoint_row_points(app)
            await pilot.click(offset=points[0])
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert leases == []                              # nothing acquired


def test_tui_double_click_acquires_from_any_row():
    """Two quick clicks on the same row acquire it, even from an unfocused row."""
    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            points = _endpoint_row_points(app)
            # row 1 is not the initial cursor row, so this also proves the
            # gesture works without a prior "highlight" click.
            await pilot.click(offset=points[1])
            await pilot.click(offset=points[1])
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert len(leases) == 1                          # exactly one, no double-fire
    assert leases[0].state == LeaseState.ACTIVE
    assert 'qwen-fast' in leases[0].endpoints        # second row, sorted


def test_tui_enter_still_acquires_selected_endpoint():
    """The keyboard path keeps its single-press Enter-to-acquire behaviour."""
    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#endpoints').focus()
            await pilot.press('enter')
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert len(leases) == 1
    assert leases[0].state == LeaseState.ACTIVE
    assert 'qwen-coder' in leases[0].endpoints


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


def test_tui_docker_pane_has_logs_and_containers_tabs():
    from textual.widgets import Collapsible, DataTable, TabbedContent

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # runtime is a collapsible pane with Logs/Instances/Control tabs;
            # system is its own collapsed pane; API is a top-level tab now.
            tabs = app.query_one('#docker-tabs', TabbedContent)
            assert {p.id for p in tabs.query('TabPane')} == {
                'tab-logs', 'tab-containers', 'tab-control'
            }
            assert app.query_one('#docker', Collapsible)
            assert app.query_one('#system', Collapsible).collapsed
            top = app.query_one('#top', TabbedContent)
            assert 'tab-api' in {p.id for p in top.query('TabPane')}
            # the instances view carries what `infer-stack ps` shows
            ps = app.query_one('#ps', DataTable)
            labels = [str(c.label) for c in ps.columns.values()]
            assert labels == ['name', 'status', 'serves', 'started', 'ports']

    _run(scenario)


def test_tui_endpoint_action_buttons_fit_the_sidebar():
    # Regression: 4 buttons at width:1fr were defeated by Button's default
    # min-width (16) and overflowed off the narrow sidebar — Edit/Remove only
    # appeared after widening. They must all fit at the default sidebar width.
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            sb = app.query_one('#sidebar').region
            for bid in ('#btn-acquire', '#btn-add-endpoint', '#btn-edit-endpoint',
                        '#btn-remove-endpoint'):
                r = app.query_one(bid).region
                assert r.width > 0 and r.x >= sb.x and \
                    r.x + r.width <= sb.x + sb.width, f'{bid} overflows sidebar'

    _run(scenario)


def test_tui_renders_when_action_buttons_are_squeezed_narrow():
    # Regression: the min-width:0 action buttons can shrink until their content
    # box is ~2 cells. Textual's Button carries line-pad:1 and folds the label
    # at (width - line_pad*2); at width 2 that is 0 and rich's chop_cells does
    # range(0, n, 0) -> ValueError, crashing the whole render. text-wrap:nowrap
    # on the compact buttons skips that fold path. Sweep the sidebar across the
    # crash zone (region width ~4 / content ~2) and force a real composite.
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            sidebar = app.query_one('#sidebar')
            for width in range(10, 25):
                app._sidebar_w = width
                sidebar.styles.width = width
                await pilot.pause()
                # export_screenshot drives the full StylesCache render path that
                # raised in the field; it must not blow up at any narrow width.
                app.export_screenshot()

    _run(scenario)


def test_tui_expanding_system_pane_polls_gpus():
    # Regression: the polling gate trusted Collapsible.Toggled (which doesn't
    # fire on every path), so expanding System never flipped the gate and the
    # GPU table stayed empty. _sync_pane_state reads the live state instead.
    from textual.widgets import Collapsible, DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._gpu_rows = lambda: [['0', 'RTX', '15', '5000', '24576', '56']]
            app.query_one('#system', Collapsible).collapsed = False  # expand
            app._sync_pane_state()
            assert app._collapsed['system'] is False
            assert 'gpus' in app._collect()       # now polled
            app._refresh_now()
            assert app.query_one('#gpus', DataTable).row_count == 1

    _run(scenario)


def test_tui_monitor_panes_are_collapsible():
    from textual.widgets import Collapsible

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # docker/system are collapsible auxiliary panes
            for pid in ('#docker', '#system'):
                pane = app.query_one(pid, Collapsible)
                pane.collapsed = True
            await pilot.pause()
            assert app.query_one('#docker', Collapsible).collapsed
            assert app.query_one('#system', Collapsible).collapsed

    _run(scenario)


def _split_layout():
    """The TUI with both pane pairs stacked around dividers instead of tabbed."""
    from infer_stack.tui import InferStackTUI

    class Split(InferStackTUI):
        TABBED_CATALOG = False
        TABBED_TABLES = False
    return Split


@pytest.mark.parametrize('tabbed', [True, False])
def test_tui_pane_pairs_are_tabs_or_split(tabbed):
    """Either layout has the same panes and ids; only the container differs."""
    from textual.containers import Vertical
    from textual.css.query import NoMatches
    from textual.widgets import TabbedContent

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    out = controller.acquire('bob', catalog.resolve_names(['qwen-coder']))
    assert out.lease
    cls = InferStackTUI if tabbed else _split_layout()

    async def scenario():
        app = cls(controller, catalog, interval=999, proc_factory=lambda svc: None)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.pause()
            app.action_refresh()
            await app.workers.wait_for_complete()
            await pilot.pause()
            for pane in ('#leases-pane', '#deployments-pane'):
                assert isinstance(app.query_one(pane), Vertical)
            for table in ('#endpoints', '#models', '#leases', '#deployments'):
                assert app.query_one(table)
            dividers = []
            for divider in ('#csplit', '#tsplit'):
                try:
                    dividers.append(app.query_one(divider))
                except NoMatches:
                    pass
            if tabbed:
                assert not dividers
                tabs = app.query_one('#table-tabs', TabbedContent)
                assert str(tabs.get_tab('pane-leases').label) == 'Leases 1/1'
                # the pane in front gets the whole height, not a fixed 14 rows
                assert app.query_one('#leases-pane').size.height > 14
            else:
                assert len(dividers) == 2

    _run(scenario)


def test_tui_panes_drag_resize():
    controller, catalog = _ctx()

    async def scenario():
        app = _split_layout()(controller, catalog, interval=999,
                              proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            w0, h0 = app._sidebar_w, app._log_h
            m0 = app._models_h
            l0 = app._leases_h
            app._drag_sidebar(6)            # pull the vertical splitter right
            app._drag_logs(3)               # pull the horizontal splitter down
            app._drag_models(2)             # catalog endpoints|models splitter
            app._drag_tables(2)             # leases|deployments splitter
            assert app._sidebar_w == w0 + 6
            assert app._log_h == h0 - 3     # down = shorter logs
            assert app._models_h == m0 - 2  # drag down = bar down = models shorter
            assert app._leases_h == l0 + 2  # drag down = bar down = leases taller

    _run(scenario)


def test_tui_dividers_have_a_grab_area():
    controller, catalog = _ctx()

    async def scenario():
        app = _split_layout()(controller, catalog, interval=999,
                              proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # A 0-size divider can't be grabbed; both must span their cross-axis.
            assert app.query_one('#vsplit').region.height > 1
            assert app.query_one('#hsplit').region.width > 1
            assert app.query_one('#csplit').region.width > 1   # endpoints|models
            assert app.query_one('#tsplit').region.width > 1   # leases|deployments

    _run(scenario)


def test_tui_drag_resizes_sidebar():
    from textual import events

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            div = app.query_one('#vsplit')
            w0 = app._sidebar_w
            await pilot.mouse_down('#vsplit')
            assert app.mouse_captured is div          # the bar grabbed the mouse
            div.post_message(events.MouseMove(
                widget=div, x=0, y=0, delta_x=5, delta_y=0, button=0,
                shift=False, meta=False, ctrl=False,
                screen_x=div.region.x + 5, screen_y=div.region.y,
            ))
            await pilot.pause()
            await pilot.mouse_up('#vsplit')
            assert app._sidebar_w == w0 + 5

    _run(scenario)


def test_tui_pane_scoped_action_buttons():
    from textual.widgets import Button

    from infer_stack.leasing import LeaseState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Acquire lives under the catalog; release/evict under their tables.
            assert app.query_one('#btn-acquire', Button)
            assert app.query_one('#btn-release', Button)
            assert app.query_one('#btn-evict', Button)
            app.query_one('#endpoints').move_cursor(row=0)
            await pilot.click('#btn-acquire')
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not app._acquire_inflight
            status = str(app.query_one('#status').render())
            assert 'acquired qwen-coder' in status and 'lease ' in status

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert len(leases) == 1 and leases[0].state == LeaseState.ACTIVE


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


def test_tui_has_top_level_dashboard_and_settings_tabs():
    from textual.widgets import TabbedContent

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            top = app.query_one('#top', TabbedContent)
            ids = {p.id for p in top.query('TabPane')}
            assert {'tab-dashboard', 'tab-settings'} <= ids
            # dashboard widgets still resolve (composed via the helper)
            assert app.query_one('#endpoints')
            assert app.query_one('#set-backend')   # settings form present

    _run(scenario)


def test_tui_settings_save_writes_yaml(tmp_path):
    import yaml
    from textual.widgets import Input, Select

    from infer_stack import paths
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    paths.set_config_root(tmp_path)
    paths.set_data_root(tmp_path)

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#set-backend', Select).value = 'compose'
            app.query_one('#set-data-dir', Input).value = str(tmp_path / 'data')
            app.query_one('#set-ui', Select).value = 'off'
            app._on_save_settings()
            await pilot.pause()

    try:
        _run(scenario)
        saved = yaml.safe_load((tmp_path / 'settings.yaml').read_text())
        assert saved['backend'] == 'compose'
        assert saved['data_dir'] == str(tmp_path / 'data')
        assert saved['ui'] is False
    finally:
        paths.set_config_root(None)
        paths.set_data_root(None)


def test_tui_endpoint_entry_builds_runtime_from_advanced_params():
    from infer_stack.tui import InferStackTUI

    entry = InferStackTUI._endpoint_entry({
        'model': 'qc', 'engine': 'vllm',
        'tensor_parallel': 2, 'max_model_len': 8192, 'gpu_mem': 0.4,
        'extra_args': '--dtype=half --enforce-eager', 'reclaim': 'keep-warm',
    })
    assert entry['runtime']['tensor_parallel_size'] == 2
    assert entry['runtime']['max_model_len'] == 8192
    assert entry['runtime']['gpu_memory_utilization'] == 0.4
    assert entry['runtime']['extra_args'] == ['--dtype=half', '--enforce-eager']
    assert entry['reclaim'] == {'policy': 'keep-warm'}


def test_tui_endpoint_entry_data_parallel_and_ollama():
    from infer_stack.tui import InferStackTUI

    v = InferStackTUI._endpoint_entry({
        'model': 'm', 'engine': 'vllm', 'data_parallel': 2,
        'prefix_caching': 'on', 'max_num_seqs': 64,
    })
    assert v['runtime']['data_parallel_size'] == 2
    assert v['runtime']['enable_prefix_caching'] is True
    assert v['runtime']['max_num_seqs'] == 64

    pinned = InferStackTUI._endpoint_entry({
        'model': 'm', 'engine': 'vllm',
        'placement': {'min_vram_gib': 24, 'gpu_indices': [1]},
    })
    assert pinned['placement'] == {'min_vram_gib': 24, 'gpu_indices': [1]}

    o = InferStackTUI._endpoint_entry({
        'model': 'qwen', 'engine': 'ollama', 'host': 'oll',
        'ollama_runtime': 'num_ctx=8192 keep_alive=5m',
    })
    assert o['engine'] == 'ollama' and o['host'] == 'oll'
    assert o['runtime']['num_ctx'] == 8192
    assert o['runtime']['keep_alive'] == '5m'


def test_tui_endpoint_wizard_is_engine_adaptive_and_labeled():
    from textual.widgets import Input, Label, Select

    from infer_stack.tui import InferStackTUI, _AddEndpointScreen

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = _AddEndpointScreen(['qc'])
            app.push_screen(screen)
            await pilot.pause()
            # vLLM knobs visible by default; ollama hidden
            assert screen.query_one('#vllm-opts').display is True
            assert screen.query_one('#ollama-opts').display is False
            assert screen.query_one('#e-dp')        # data-parallel field exists
            assert screen.query_one('#e-gpu-pin', Input).value == 'auto'
            # fields are labeled (the "blank page" complaint)
            labels = [str(lbl.render()) for lbl in screen.query(Label)]
            assert any('tensor-parallel' in x for x in labels)
            assert any('data-parallel' in x for x in labels)
            assert any('GPU placement' in x for x in labels)
            # switching engine swaps the field groups
            screen.query_one('#e-engine', Select).value = 'ollama'
            await pilot.pause()
            assert screen.query_one('#vllm-opts').display is False
            assert screen.query_one('#ollama-opts').display is True

    _run(scenario)


def test_tui_add_endpoint_writes_advanced_params(tmp_path):
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
            app._on_add_endpoint({
                'name': 'big', 'model': 'qc', 'engine': 'vllm',
                'tensor_parallel': 2, 'max_model_len': None, 'gpu_mem': None,
                'extra_args': '', 'reclaim': '',
            })
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    on_disk = yaml.safe_load(catalog_path.read_text())
    assert on_disk['endpoints']['big']['runtime']['tensor_parallel_size'] == 2


def test_tui_edit_blocked_while_served(tmp_path):
    import yaml
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    catalog_path = tmp_path / 'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump(CATALOG))
    controller.acquire('me', catalog.resolve_names(['qwen-coder']))  # serve it

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None,
                            catalog_path=str(catalog_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#endpoints', DataTable).move_cursor(row=0)  # qwen-coder
            app.action_edit_endpoint()
            await pilot.pause()
            assert 'served' in str(app.query_one('#status').render())

    _run(scenario)


def test_tui_gpu_pin_writes_catalog_and_updates_gpu_column(tmp_path):
    import yaml
    from textual.widgets import DataTable, Input

    from infer_stack.tui import InferStackTUI, _AddEndpointScreen

    controller, catalog = _ctx()
    catalog_path = tmp_path / 'catalog.yaml'
    catalog_path.write_text(yaml.safe_dump(CATALOG))
    old = controller.acquire('me', catalog.resolve_names(['qwen-coder']))
    old_gid = old.lease.deployment_ids[0]
    controller.release(old.lease.id)  # leave a keep-warm idle deployment behind
    inventory = {
        'gpu_count': 2,
        'gpus': [
            {'index': 0, 'name': 'Big GPU', 'memory_gib': 48,
             'display_active': False},
            {'index': 1, 'name': 'Small GPU', 'memory_gib': 24,
             'display_active': False},
        ],
    }

    async def scenario():
        app = InferStackTUI(
            controller, catalog, interval=999, proc_factory=lambda svc: None,
            catalog_path=str(catalog_path),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            raw = yaml.safe_load(catalog_path.read_text())
            app._show_endpoint_editor(
                'qwen-coder', raw['endpoints']['qwen-coder'], inventory
            )
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, _AddEndpointScreen)
            screen.query_one('#e-gpu-pin', Input).value = '1'
            await pilot.click('#ok')
            await app.workers.wait_for_complete()
            await pilot.pause()
            ep = app.catalog.endpoints['qwen-coder']
            assert ep.placement['gpu_indices'] == [1]
            assert controller.ledger.get_deployment(old_gid).state == 'stopped'
            table = app.query_one('#endpoints', DataTable)
            columns = [str(c.label) for c in table.columns.values()]
            assert 'gpu' in columns

    _run(scenario)
    on_disk = yaml.safe_load(catalog_path.read_text())
    assert on_disk['endpoints']['qwen-coder']['placement']['gpu_indices'] == [1]


def test_tui_endpoint_edit_keeps_a_custom_launch_and_changes_gpu_pin():
    from infer_stack.tui import InferStackTUI

    base = {
        'engine': 'vllm',
        'model': 'q38',
        'protocol': 'chat',
        'placement': {'min_vram_gib': 24, 'gpu_indices': [0]},
        'runtime': {
            'serve_recipe': 'hyperqwen-3090-single',
            'image': 'ghcr.io/syv-ai/hyperqwen:sha-684e927',
            'pipeline_parallel_size': 1,
            'max_model_len': 65536,
        },
    }
    entry = InferStackTUI._endpoint_entry({
        'name': 'q38-hq',
        'model': 'q38',
        'engine': 'vllm',
        'base_entry': base,
        'placement': {'min_vram_gib': 24, 'gpu_indices': [1]},
        'tensor_parallel': None,
        'data_parallel': None,
        'max_model_len': 65536,
        'gpu_mem': None,
        'max_num_seqs': None,
        'prefix_caching': '',
        'extra_args': '',
        'reclaim': '',
    })
    assert entry['placement'] == {'min_vram_gib': 24, 'gpu_indices': [1]}
    assert entry['protocol'] == 'chat'
    # The legacy recipe name is saved as the generic launch it means.
    assert 'serve_recipe' not in entry['runtime']
    assert entry['runtime']['command'] == ['single']
    assert entry['runtime']['mounts']['/cache'] == 'hyperqwen/qwen3.8-27b/cache'
    assert entry['runtime']['image'] == 'ghcr.io/syv-ai/hyperqwen:sha-684e927'
    assert entry['runtime']['pipeline_parallel_size'] == 1


def test_tui_endpoint_edit_can_switch_a_launcher_mode_and_keeps_what_it_hides():
    """Fast -> long from the form alone; mounts (not in the form) survive."""
    from infer_stack.tui import InferStackTUI, _AddEndpointScreen

    base = {'engine': 'vllm', 'model': 'q38', 'runtime': {
        'image': 'img:1', 'command': ['single'], 'max_model_len': 65536,
        'env': {'SPEC': 'dflash2', 'MAX_LEN': '{max_model_len}', 'PREFIX_CACHE': 1},
        'mounts': {'/cache': 'x/cache'},
    }}
    env = _AddEndpointScreen._parse_env(
        "SPEC=mtp CTX=long MAX_LEN='{max_model_len}' PREFIX_CACHE=1")
    entry = InferStackTUI._endpoint_entry({
        'name': 'q38', 'model': 'q38', 'engine': 'vllm', 'base_entry': base,
        'placement': {}, 'tensor_parallel': None, 'data_parallel': None,
        'max_model_len': 150000, 'gpu_mem': 0.93, 'max_num_seqs': None,
        'prefix_caching': 'on', 'extra_args': '', 'reclaim': '',
        'image': 'img:1', 'command': 'single', 'env': env,
    })
    rt = entry['runtime']
    assert rt['max_model_len'] == 150000
    assert rt['env'] == {'SPEC': 'mtp', 'CTX': 'long', 'MAX_LEN': '{max_model_len}',
                         'PREFIX_CACHE': '1'}
    assert rt['command'] == ['single'] and rt['mounts'] == {'/cache': 'x/cache'}
    # Clearing the command field returns the endpoint to stock vLLM.
    entry = InferStackTUI._endpoint_entry({
        'name': 'q38', 'model': 'q38', 'engine': 'vllm', 'base_entry': entry,
        'placement': {}, 'tensor_parallel': None, 'data_parallel': None,
        'max_model_len': None, 'gpu_mem': None, 'max_num_seqs': None,
        'prefix_caching': '', 'extra_args': '', 'reclaim': '',
        'image': '', 'command': '', 'env': {},
    })
    assert not {'command', 'env', 'image'} & set(entry.get('runtime') or {})


def test_tui_env_field_rejects_a_word_without_equals():
    from infer_stack.tui import _AddEndpointScreen

    with pytest.raises(ValueError, match='KEY=VALUE'):
        _AddEndpointScreen._parse_env('SPEC=mtp long')


def test_tui_remove_endpoint_writes_catalog(tmp_path):
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
            app._do_remove('endpoints', 'qwen-fast')   # confirm bypassed
            await pilot.pause()
            assert 'qwen-fast' not in app.catalog.endpoints

    _run(scenario)
    on_disk = yaml.safe_load(catalog_path.read_text())
    assert 'qwen-fast' not in on_disk['endpoints']


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


def test_tui_instance_rows_say_what_each_serves():
    from types import SimpleNamespace

    from infer_stack.leasing.instances import Instance
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    instances = [
        Instance('vllm-qwen', 'abcdef1234567890', 'grp-1', 'restarting',
                 restarts=2, reason='CrashLoopBackOff',
                 started='2026-06-19T00:00:00.123Z'),
        Instance('litellm', 'fedcba', '', 'running', ports='14042->4000/tcp'),
    ]

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._last_deployments = [SimpleNamespace(id='grp-1', served={'qwen': {}})]
            engine, gateway = app._ps_rows(instances)
            assert engine['serves'] == 'qwen'
            assert engine['status'] == 'restarting (CrashLoopBackOff, 2 restarts)'
            # Shown in local time, like every other time a person reads here.
            from infer_stack.leasing.instances import local_time
            assert engine['started'] == local_time('2026-06-19T00:00:00Z')
            assert gateway['serves'] == '(front door)'
            assert gateway['ports'] == '14042->4000/tcp'
            assert app._ps_rows(None) is None          # the runtime was unreadable

    _run(scenario)


def test_tui_system_tab_renders_without_nvidia_smi():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._fill_gpus(None)             # nvidia-smi unavailable -> hint row
            gpus = app.query_one('#gpus', DataTable)
            assert gpus.row_count == 1
            assert 'cpus' in app._system_line()

    _run(scenario)


def test_tui_collapsed_console_skips_expensive_polling():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._active_tab = 'tab-containers'
            app._collapsed['docker'] = True
            assert 'instances' not in app._collect()   # collapsed -> no poll
            app._collapsed['docker'] = False
            assert 'instances' in app._collect()       # visible -> polled
            app._active_tab = 'tab-logs'
            assert 'instances' in app._collect()       # the log picker needs names
            app._collapsed['system'] = True
            assert 'gpus' not in app._collect()        # system collapsed
            app._collapsed['system'] = False
            assert 'gpus' in app._collect()            # system expanded -> polled

    _run(scenario)


def test_tui_open_builds_openwebui_url():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            _front(controller.backend, None, 13000)
            assert app._ui_url('qwen-coder') == (
                'http://127.0.0.1:13000/?models=qwen-coder'
            )

    _run(scenario)


def test_tui_api_tester_sends_via_injected_http():
    from textual.widgets import Select

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'choices': [{'message': {'content': 'pong'}}]}

    class _HTTP:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=None):
            self.calls.append((url, json))
            return _Resp()

    http = _HTTP()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None, http=http)
        async with app.run_test() as pilot:
            await pilot.pause()
            _front(controller.backend, 14042)
            # only ready (running) models are offered; simulate one being ready
            app._sync_api_models(['qwen-coder'])
            assert app.query_one('#api-model', Select).value == 'qwen-coder'
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('pong' in line for line in app._api_lines)

    _run(scenario)
    assert http.calls and http.calls[0][0].endswith('/v1/chat/completions')


def test_tui_api_send_surfaces_http_error_body():
    """A gateway 400 must show its BODY (e.g. 'Invalid model name'), not just a
    bare '400 Client Error' — the missing body is what made the route-stripping
    incident hard to diagnose from the TUI."""
    import requests
    from textual.widgets import Select

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    class _Resp:
        status_code = 400
        url = 'http://x:14042/v1/chat/completions'
        text = '{"error":{"message":"Invalid model name passed in"}}'

        def raise_for_status(self):
            raise requests.HTTPError('400 Client Error: Bad Request')

    class _HTTP:
        def post(self, url, json=None, headers=None, timeout=None):
            return _Resp()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None, http=_HTTP())
        async with app.run_test() as pilot:
            await pilot.pause()
            _front(controller.backend, 14042)
            app._sync_api_models(['qwen-coder'])
            assert app.query_one('#api-model', Select).value == 'qwen-coder'
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('Invalid model name' in line for line in app._api_lines)

    _run(scenario)


def test_tui_api_urls_render_without_markup_error():
    # Regression: the URLs were rendered with Textual [link=URL] markup, which
    # rejects the ':' in http:// and crashed on get_content_height.
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            _front(controller.backend, 14042, 13000)
            app._update_api_urls()
            await pilot.pause()
            text = str(app.query_one('#api-urls').render())  # must not raise
            assert '14042' in text and '13000' in text

    _run(scenario)


def test_tui_api_list_models_and_curl():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'data': [{'id': 'qwen-coder'}, {'id': 'qwen-fast'}]}

    class _HTTP:
        def get(self, url, headers=None, timeout=None):
            self.url = url
            return _Resp()

    http = _HTTP()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None, http=http)
        async with app.run_test() as pilot:
            await pilot.pause()
            _front(controller.backend, 14042)
            app._sync_api_models(['qwen-coder'])
            app._update_api_curl()
            curl = str(app.query_one('#api-curl').render())
            assert 'curl' in curl and '/v1/chat/completions' in curl
            assert 'qwen-coder' in curl
            app.action_api_copy_curl()           # must not raise
            app.action_api_list_models()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('qwen-coder' in line for line in app._api_lines)

    _run(scenario)
    assert http.url.endswith('/v1/models')


def test_tui_api_tester_respects_completions_protocol():
    # A completions-only endpoint must be probed on /v1/completions with a
    # `prompt` body (not /v1/chat/completions with `messages`), and its `text`
    # response must be surfaced.
    from textual.widgets import Select

    from infer_stack.leasing import Catalog, Controller, Ledger, NullBackend, SqliteStore
    from infer_stack.tui import InferStackTUI

    catalog = Catalog.from_dict({
        'models': {'qc': {'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct'}},
        'endpoints': {
            'legacy-completions': {
                'engine': 'vllm', 'model': 'qc', 'protocol': 'completions',
            },
        },
    })
    controller = Controller(Ledger(SqliteStore(':memory:')), NullBackend())

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'choices': [{'text': 'pong'}]}

    class _HTTP:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, headers=None, timeout=None):
            self.calls.append((url, json))
            return _Resp()

    http = _HTTP()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None, http=http)
        async with app.run_test() as pilot:
            await pilot.pause()
            _front(controller.backend, 14042)
            app._sync_api_models(['legacy-completions'])
            assert app.query_one('#api-model', Select).value == 'legacy-completions'
            # curl preview reflects the completions surface
            app._update_api_curl()
            curl = str(app.query_one('#api-curl').render())
            assert '/v1/completions' in curl and '/v1/chat/completions' not in curl
            assert '"prompt"' in curl
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('pong' in line for line in app._api_lines)

    _run(scenario)
    assert http.calls
    url, body = http.calls[0]
    assert url.endswith('/v1/completions')
    assert 'prompt' in body and 'messages' not in body


def test_tui_api_lists_only_ready_models():
    from textual.widgets import Select

    from infer_stack.tui import SELECT_BLANK, InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # NullBackend observes nothing running -> no ready models listed,
            # even though the catalog has endpoints.
            assert app._ready_endpoints == []
            assert app.query_one('#api-model', Select).value is SELECT_BLANK

    _run(scenario)


def test_tui_cleanup_prunes_released_and_stopped(tmp_path):
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    # an active lease, then released -> it becomes a RELEASED tail entry
    out = controller.acquire('bob', catalog.resolve_names(['qwen-coder']))
    controller.release(out.lease.id)
    leases, _ = controller.ledger.status()
    assert any(str(le.state) == 'released' for le in leases)

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_cleanup()
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert not any(str(le.state) == 'released' for le in leases)


def test_clean_up_is_one_footer_key_that_logs_its_cli_command():
    """One Clean up, on `x` in the footer, and each action names its command."""
    from textual.widgets import Button

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            seen['buttons'] = [b.id for b in app.query(Button)
                               if 'cleanup' in (b.id or '')]
            seen['footer'] = [b for b in app.BINDINGS if isinstance(b, tuple)]
            await pilot.press('x')
            await app.workers.wait_for_complete()
            await pilot.press('r')
            await pilot.pause()
            seen['applog'] = '\n'.join(app._app_log_lines)

    _run(scenario)
    assert seen['buttons'] == []
    assert ('x', 'cleanup', 'Clear finished') in seen['footer']    # tuples show in the footer
    assert 'CLI: infer-stack gc --forget' in seen['applog']
    assert 'CLI: infer-stack status' in seen['applog']


def test_tui_evict_all_idle_button():
    from infer_stack.leasing import DeploymentState
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    out = controller.acquire('alice', catalog.resolve_names(['qwen-coder']))
    gid = out.lease.deployment_ids[0]
    controller.release(out.lease.id)          # deployment -> IDLE (keep-warm)
    assert controller.ledger.get_deployment(gid).state == DeploymentState.IDLE

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_evict_all()                # one action clears every idle one
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    # the idle keep-warm deployment is now STOPPED (so Clean up can forget it)
    assert controller.ledger.get_deployment(gid).state == DeploymentState.STOPPED


def test_tui_multiselect_releases_checked_leases():
    from textual.widgets import DataTable

    from infer_stack.leasing import LeaseState
    from infer_stack.tui import SELECT_MARK, InferStackTUI

    controller, catalog = _ctx()
    controller.acquire('alice', catalog.resolve_names(['qwen-coder']))
    controller.acquire('bob', catalog.resolve_names(['qwen-fast']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one('#leases', DataTable)
            assert table.row_count == 2
            table.focus()
            table.move_cursor(row=0)
            await pilot.press('space')            # check row 0
            table.move_cursor(row=1)
            await pilot.press('space')            # check row 1
            await pilot.pause()
            assert len(app._lease_sel) == 2
            # the marker column (col 0) shows the check on a selected row
            assert table.get_row_at(0)[0] == SELECT_MARK
            app.action_release()                  # acts on both checked rows
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._lease_sel == set()        # selection cleared after action

    _run(scenario)
    leases, _ = controller.ledger.status()
    assert leases and all(le.state == LeaseState.RELEASED for le in leases)


def test_tui_space_toggles_selection_off_again():
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    controller.acquire('alice', catalog.resolve_names(['qwen-coder']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one('#leases', DataTable)
            table.focus()
            table.move_cursor(row=0)
            await pilot.press('space')            # select
            assert len(app._lease_sel) == 1
            await pilot.press('space')            # toggle back off
            assert app._lease_sel == set()
            assert table.get_row_at(0)[0] == ''   # marker cleared

    _run(scenario)


def test_tui_click_select_ctrl_toggle_shift_range_plain_clear():
    from textual.widgets import DataTable

    from infer_stack.tui import SELECT_MARK, InferStackTUI

    controller, catalog = _ctx()
    controller.acquire('a', catalog.resolve_names(['qwen-coder']))
    controller.acquire('b', catalog.resolve_names(['qwen-fast']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            ids = app._lease_ids
            assert len(ids) == 2
            table = app.query_one('#leases', DataTable)
            # ctrl-click toggles one row on, then off (discontiguous pick)
            app._click_select('leases', 0, shift=False, ctrl=True)
            assert app._lease_sel == {ids[0]}
            app._click_select('leases', 0, shift=False, ctrl=True)
            assert app._lease_sel == set()
            # ctrl-click sets the anchor; shift-click extends a contiguous range
            app._click_select('leases', 0, shift=False, ctrl=True)
            app._click_select('leases', 1, shift=True, ctrl=False)
            assert app._lease_sel == {ids[0], ids[1]}
            assert table.get_row_at(0)[0] == SELECT_MARK
            assert table.get_row_at(1)[0] == SELECT_MARK
            # a plain click collapses the selection back to the cursor row
            app._click_select('leases', 1, shift=False, ctrl=False)
            assert app._lease_sel == set()
            assert table.get_row_at(0)[0] == ''

    _run(scenario)




def test_tui_model_cached_label(tmp_path):
    from infer_stack.tui import InferStackTUI

    hub = tmp_path / 'hub'
    (hub / 'models--Org--Model').mkdir(parents=True)
    assert InferStackTUI._cached_label('hf://Org/Model', hub) == 'yes'
    assert InferStackTUI._cached_label('hf://Org/Other', hub) == 'no'
    assert InferStackTUI._cached_label('hf://Org/Model', None) == '?'
    assert InferStackTUI._cached_label('', hub) == '-'


def test_tui_up_applies_through_the_controller(tmp_path):
    """The Up action is `infer-stack apply` (serialised, selective), never a raw
    `docker compose up --remove-orphans`."""
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    compose_file = tmp_path / 'docker-compose.yml'
    compose_file.write_text('services: {}\n')
    calls = []
    applied = []
    controller.apply_now = lambda: applied.append(1) or type('R', (), {'publication_pending': False})()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            backend = controller.backend
            backend.rendered_file = compose_file
            backend.run = lambda args: calls.append(args) or ''
            app.action_compose_up()
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    assert applied == [1]
    assert not any('up' in c for c in calls)


def test_tui_logs_stream_from_injected_source():
    from infer_stack.tui import ALL_SERVICES, InferStackTUI

    controller, catalog = _ctx()
    lines = ['litellm   | started', 'litellm   | ready']

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: _FakeProc(lines))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._restart_logs(ALL_SERVICES)           # the gateway's own lines
            await app.workers.wait_for_complete()     # the stream has ended
            app._drain_logs()                         # lines are drawn in batches
            await pilot.pause()
            assert any('ready' in line for line in app._log_lines)

    _run(scenario)


def test_tui_compacts_registered_litellm_traceback():
    from infer_stack.tui import ALL_SERVICES, InferStackTUI

    controller, catalog = _ctx()
    p = 'litellm-1 | '
    lines = [
        p + 'Traceback (most recent call last):\n',
        p + '  File "/usr/lib/python3.13/site-packages/aiohttp/connector.py", line 1298, in _wrap_create_connection\n',
        p + '  File "/usr/lib/python3.13/site-packages/aiohappyeyeballs/impl.py", line 122, in start_connection\n',
        p + '  File "uvloop/loop.pyx", line 2633, in sock_connect\n',
        p + "ConnectionRefusedError: [Errno 111] Connect call failed ('172.18.0.4', 8000)\n",
    ]

    async def scenario():
        app = InferStackTUI(
            controller, catalog, interval=999,
            proc_factory=lambda svc: _FakeProc(lines),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app._restart_logs(ALL_SERVICES)           # the gateway's own lines
            await app.workers.wait_for_complete()
            app._drain_logs()                         # lines are drawn in batches
            await pilot.pause()
            text = '\n'.join(app._log_lines)
            assert 'Traceback (most recent call last):' not in text
            assert '  File "' not in text
            assert 'ConnectionRefusedError: [Errno 111]' in text

    _run(scenario)


def test_tui_observe_is_throttled_between_ledger_ticks():
    """The expensive observe()/plan() view is cached between ledger polls and
    only refreshed once observe_interval has elapsed."""
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    calls = {'n': 0}
    real_observe = controller.backend.observe

    def counting_observe():
        calls['n'] += 1
        return real_observe()

    controller.backend.observe = counting_observe

    async def scenario():
        # observe_interval huge -> after the first poll it must not run again
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        app.observe_interval = 10_000
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()      # first observe (mount)
            await pilot.pause()
            assert calls['n'] >= 1
            seen = calls['n']
            app._collect()                             # extra polls within window
            app._collect()
            assert calls['n'] == seen                  # served from cache

    _run(scenario)


def test_tui_apply_ui_settings_retunes_and_persists(tmp_path, monkeypatch):
    from textual.widgets import Input

    from infer_stack import paths
    from infer_stack.tui import InferStackTUI

    monkeypatch.setattr(paths, '_config_root_override', tmp_path)
    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#set-ledger-interval', Input).value = '2'
            app.query_one('#set-observe-interval', Input).value = '8'
            app._on_apply_ui_settings()
            await pilot.pause()
            assert app.ledger_interval == 2.0
            assert app.observe_interval == 8.0

    _run(scenario)
    # persisted to the TUI's own file, not the CLI settings.yaml
    saved = paths.load_tui_settings()
    assert saved['ledger_interval'] == 2.0 and saved['observe_interval'] == 8.0
    assert not (tmp_path / paths.SETTINGS_FILENAME).exists()


def test_tui_observe_interval_never_below_refresh():
    from textual.widgets import Input

    from infer_stack import paths
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one('#set-ledger-interval', Input).value = '5'
            app.query_one('#set-observe-interval', Input).value = '1'
            app._on_apply_ui_settings()
            await pilot.pause()
            assert app.observe_interval >= app.ledger_interval

    _run(scenario)


def test_tui_table_rebuild_preserves_scroll_offset():
    """A refresh that rebuilds a table (rows added / removed / reordered) must
    not yank the viewport back to the top — the user's scroll offset survives.

    Regression: ``_diff_fill`` used to restore only the cursor after a
    clear()+rebuild, so a scrolled-away viewport snapped to row 0 on the next
    poll. ``_restore_view`` now restores the scroll offset too.
    """
    from textual.widgets import DataTable

    from infer_stack.leasing import (
        Catalog,
        Controller,
        Ledger,
        NullBackend,
        SqliteStore,
    )
    from infer_stack.tui import InferStackTUI

    # Many endpoints -> many lease rows, so the leases table actually scrolls.
    cat = {
        'models': {'qc': {'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct'}},
        'endpoints': {
            f'ep-{i:02d}': {'engine': 'vllm', 'model': 'qc'} for i in range(40)
        },
    }
    catalog = Catalog.from_dict(cat)
    controller = Controller(Ledger(SqliteStore(':memory:')), NullBackend())
    for i in range(40):
        controller.acquire(f'u{i:02d}', catalog.resolve_names([f'ep-{i:02d}']))

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # The mount-time background refresh must land first, or it can
            # repaint all 40 rows after the rebuild below (an intermittent fail).
            await app.workers.wait_for_complete()
            await pilot.pause()
            table = app.query_one('#leases', DataTable)
            assert table.row_count == 40
            # Scroll a few rows down, away from the top.
            table.scroll_to(y=6, animate=False)
            await pilot.pause()
            before = table.scroll_offset.y
            assert before > 0, 'precondition: table must be scrolled off the top'
            # Force the rebuild path: dropping one lease changes the row count,
            # so _diff_fill clear()+rebuilds rather than patching cells.
            app._fill_leases(app._last_leases[1:])
            await pilot.pause()
            assert table.row_count == 39
            assert table.scroll_offset.y == before  # not reset to the top

    _run(scenario)


def test_tui_subprocess_backed_panes_note_loading():
    # The containers/GPU tables are filled by subprocess observes (docker
    # compose ps / nvidia-smi) that run on their own beat in workers. Until a
    # pane's first poll lands it must say it is loading, not sit silently
    # empty. Deterministic here: the default docker tab is logs (ps never
    # polled) and the system pane starts collapsed (gpus never polled).
    from textual.widgets import DataTable

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            ps = app.query_one('#ps', DataTable)
            assert ps.row_count == 1
            assert ps.get_row_at(0)[0] == '(loading…)'
            gpus = app.query_one('#gpus', DataTable)
            assert gpus.row_count == 1
            assert gpus.get_row_at(0)[1] == '(loading…)'

    _run(scenario)


def test_tui_titles_say_observing_before_first_docker_observe():
    # Before the first (worker-run) docker observe lands, the running counts
    # are unknown — the titles must say 'observing…' rather than a confident
    # '0 running'.
    from textual.widgets import Collapsible

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._observed_at = None
            app._update_summary([], [], set())
            assert 'observing…' in app.query_one('#docker', Collapsible).title
            assert 'observing…' in str(
                app.query_one('#deployments-pane').border_title
            )
            app._observed_at = 123.0
            app._update_summary([], [], set())
            assert '0 running' in app.query_one('#docker', Collapsible).title

    _run(scenario)


def test_gateway_services_are_excluded_from_the_default_log_view():
    """The logs pane defaults to engines, not everything.

    LiteLLM logs a line per proxied request, so on a busy host it scrolls the
    engine output -- where errors actually appear -- out of the pane. An
    engine is an instance that serves a deployment; the gateway, UI, database
    and proxy serve none, whatever they are named (a name hint used to let
    Open WebUI and Postgres into the engines view).
    """
    from infer_stack.leasing.instances import Instance
    from infer_stack.tui import ALL_SERVICES, ENGINE_SERVICES

    engine = Instance('vllm-a', 'c1', 'grp-1', 'running')
    assert engine.is_engine
    for name in ('litellm', 'infer-stack-litellm-1', 'open-webui', 'postgres'):
        assert not Instance(name, 'c2', '', 'running').is_engine
    # The two sentinels must stay distinguishable from each other and from any
    # real instance name.
    assert ENGINE_SERVICES != ALL_SERVICES
    assert ENGINE_SERVICES not in ('litellm', 'vllm-a')


def test_named_log_process_follows_only_that_service(monkeypatch):
    """A named log view must never read the gateway's output."""
    import os
    import subprocess
    import time

    from infer_stack.leasing.instances import Instance
    from infer_stack.tui import InferStackTUI

    service = 'vllm-qwen3-8-27b-dbirks-hyperqwen'
    touched = []

    class _Done:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(cmd, **kwargs):
        touched.append(cmd[-1])                   # docker logs --tail N <id>
        return _Done(b'')

    class _Proc:
        def __init__(self, cmd):
            touched.append(cmd[-1])               # docker attach <id>
            self._r, w = os.pipe()
            os.close(w)                           # immediate EOF
            self.stdout = os.fdopen(self._r, 'rb')

        def poll(self):
            return 0

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    controller, catalog = _ctx()
    controller.backend.instances = lambda: [
        Instance(service, 'c1', 'grp-1', 'running'),
        Instance('litellm', 'c2', '', 'running'),
    ]
    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **kw: _Proc(cmd))
    app = InferStackTUI(controller, catalog, interval=999)
    proc = app._default_proc_factory()(service)
    deadline = time.monotonic() + 5
    while len(touched) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    proc.terminate()
    assert touched[:2] == ['c1', 'c1']           # its history, then its live output
    assert 'c2' not in touched                   # the gateway, never


def test_stale_log_stream_cannot_bleed_into_new_service_selection():
    """Buffered output from a terminated stream belongs to its old generation.

    Regression: switching from LiteLLM to a named vLLM service could still show
    LiteLLM lines because the old docker-compose process/worker drained buffered
    stdout after the pane had already been cleared and relabelled.
    """
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._log_lines = []
            app._log_generation = 12
            app._append_log_if_current(11, 'litellm | stale request spam')
            assert app._log_lines == []
            app._append_log_if_current(12, 'vllm-qwen | current engine line')
            assert app._log_lines == ['vllm-qwen | current engine line']

    _run(scenario)


def test_log_target_resolves_the_engines_sentinel_to_service_names():
    from infer_stack.tui import ALL_SERVICES, ENGINE_SERVICES, InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            from infer_stack.leasing.instances import Instance
            app._last_instances = [Instance('litellm', 'c0', '', 'running'),
                                   Instance('vllm-a', 'c1', 'g1', 'running'),
                                   Instance('vllm-b', 'c2', 'g2', 'running')]

            target, label = app._resolve_log_target(ENGINE_SERVICES)
            assert target == ['vllm-a', 'vllm-b']
            assert 'litellm' not in label

            # A named service is passed straight through.
            assert app._resolve_log_target('litellm') == ('litellm', 'litellm')

            # "everything" stays None: the follower takes every instance.
            target, label = app._resolve_log_target(ALL_SERVICES)
            assert target is None and label == 'everything'

            # With no engines there is nothing to follow. Falling back to
            # every instance would show the gateway under the engines label.
            from infer_stack.tui import NO_LOG_TARGET
            app._last_instances = [Instance('litellm', 'c0', '', 'running')]
            target, label = app._resolve_log_target(ENGINE_SERVICES)
            assert target is NO_LOG_TARGET and 'no engines running' in label

    _run(scenario)


def test_engines_view_follows_engines_that_appear_after_it_opened():
    """Regression: the engines view showed litellm lines.

    The pane opened while no engine existed, fell back to every service, and
    kept that stream after engines appeared -- the selection had not changed,
    so nothing restarted it.
    """
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    started = []

    def factory(service):
        started.append(service)
        return _FakeProc([])

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=factory)
        async with app.run_test() as pilot:
            await pilot.pause()
            from infer_stack.leasing.instances import Instance
            app._last_instances = [Instance('litellm', 'c0', '', 'running')]
            app._collapsed['docker'] = False
            app._sync_log_services()
            app._restart_logs(app._log_service)       # the pane opens
            await pilot.pause(0.2)
            assert started == []                      # nothing to follow yet

            app._last_instances = [*app._last_instances,   # an engine is deployed
                                   Instance('vllm-a', 'c1', 'g1', 'running')]
            app._sync_log_services()
            await pilot.pause(0.2)
            await pilot.pause()
            assert started == [['vllm-a']]
            assert None not in started                # never every service

    _run(scenario)


def test_tui_reports_button_handler_failures(tmp_path):
    """A failing action must be impossible to miss: a sticky status line, the
    TUI log tab (turned red, with a count), a toast, and a file on disk.

    Textual runs handlers on its own message pump, so an uncaught exception
    otherwise leaves the click looking like it did nothing at all.
    """
    from textual.widgets import Button, Static, TabbedContent

    from infer_stack.tui import APP_LOG_TAB_TITLE, InferStackTUI

    controller, catalog = _ctx()
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            def boom():
                raise RuntimeError('editor exploded')

            app.action_edit_endpoint = boom
            app.on_button_pressed(Button.Pressed(Button(id='btn-edit-endpoint')))
            for _ in range(5):                      # survives refresh ticks
                await pilot.pause()
            seen['status'] = str(app.query_one('#status', Static).render())
            seen['applog'] = '\n'.join(app._app_log_lines)
            seen['docker'] = '\n'.join(app._log_lines)
            tabs = app.query_one('#top', TabbedContent)
            seen['label'] = str(tabs.get_tab('tab-applog').label)
            seen['path'] = app.error_log_path()
            app.action_show_app_log()
            for _ in range(3):
                await pilot.pause()
            seen['active'] = tabs.active

    _run(scenario)
    assert 'RuntimeError: editor exploded' in seen['status']
    assert APP_LOG_TAB_TITLE in seen['status']          # says where to look
    assert 'btn-edit-endpoint pressed' in seen['applog']
    assert 'editor exploded' in seen['applog'] and 'Traceback' in seen['applog']
    assert '⚠' in seen['label'] and '(1)' in seen['label']   # rendered red
    assert 'editor exploded' not in seen['docker']      # never in the docker logs
    assert seen['path'].exists() and 'editor exploded' in seen['path'].read_text()
    assert str(seen['path']) in seen['applog']          # the file is discoverable
    assert seen['active'] == 'tab-applog'


def test_tui_reports_background_worker_failures(tmp_path):
    """The same for a thread worker: the catalog editor runs in one, so a
    failure there used to be entirely silent."""
    from textual.widgets import Static

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            def failing():
                raise ValueError('no inventory for you')

            app.run_worker(failing, thread=True, group='catalog-editor',
                           name='endpoint editor', exit_on_error=False)
            for _ in range(200):                      # until the ERROR state lands
                await pilot.pause()
                if app._app_log_errors:
                    break
            seen['status'] = str(app.query_one('#status', Static).render())
            seen['applog'] = '\n'.join(app._app_log_lines)

    _run(scenario)
    assert 'ValueError: no inventory for you' in seen['status']
    assert 'endpoint editor failed' in seen['applog']


def test_tui_log_records_what_an_action_decided(tmp_path):
    """An action that declines to act must say why in the TUI log, so 'nothing
    happened' is never the whole story."""
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.catalog_path = None                   # the editor has nowhere to write
            app.action_edit_endpoint()
            await pilot.pause()
            seen['applog'] = '\n'.join(app._app_log_lines)

    _run(scenario)
    assert 'no catalog path' in seen['applog']


def test_tui_header_shows_the_running_version():
    from infer_stack import __version__
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            seen['title'] = app.title
            seen['sub'] = app.sub_title

    _run(scenario)
    assert seen['title'] == 'infer-stack'
    assert seen['sub'].startswith(__version__)      # version, then the description
    assert 'leasing dashboard' in seen['sub']


def test_tui_refusals_pop_up_instead_of_doing_nothing():
    """`Edit` on an actively served endpoint must say so in a popup.

    This is the case that looked like a dead button: the refusal was a status
    line only, and the next refresh tick wiped it.
    """
    from textual.widgets import Static

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    out = controller.acquire('alice', catalog.resolve_names(['qwen-coder']))
    assert out.lease.endpoints                    # the endpoint is now served
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.catalog_path = 'catalog.yaml'     # editing is otherwise refused earlier
            notifications = []
            app.notify = lambda message, **kw: notifications.append((message, kw))
            app.query_one('#endpoints').focus()
            app.action_edit_endpoint()
            for _ in range(5):                    # survives refresh ticks
                await pilot.pause()
            seen['notifications'] = notifications
            seen['status'] = str(app.query_one('#status', Static).render())
            seen['applog'] = '\n'.join(app._app_log_lines)

    _run(scenario)
    assert seen['notifications'], 'a refused action must raise a popup'
    message, kwargs = seen['notifications'][0]
    assert 'actively served' in message and 'release it before editing' in message
    assert kwargs.get('severity') == 'warning'
    assert 'actively served' in seen['status']    # and the status line keeps it
    assert 'warn: ' in seen['applog'] and 'actively served' in seen['applog']


def test_tui_log_shows_the_cli_command_for_an_action_and_backend_progress():
    """The dashboard teaches the CLI, and a long pull is visible while it runs."""
    import threading

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_release_all()
            # A worker thread reporting, as ComposeBackend._pull_missing does.
            worker = threading.Thread(
                target=app._backend_progress,
                args=('pulling img: 1 of 4 layers downloaded, 2.0 GB of 9.0 GB',))
            worker.start()
            while worker.is_alive():               # join() would block the loop it needs
                await pilot.pause(0.05)
            await pilot.pause()
            seen['applog'] = '\n'.join(app._app_log_lines)

    _run(scenario)
    assert 'CLI: infer-stack release --all --yes' in seen['applog']
    assert 'pulling img: 1 of 4 layers' in seen['applog']


def test_a_ui_thread_stall_is_reported_with_where_it_happened(tmp_path, monkeypatch):
    """"The TUI froze" must come with what it was doing."""
    import time

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    monkeypatch.setattr(InferStackTUI, 'error_log_path', lambda self: tmp_path / 'e.log')
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause(0.3)
            original = app._update_summary

            def slow(*args, **kwargs):
                time.sleep(0.9)                   # a blocking call on the UI thread
                return original(*args, **kwargs)

            app._update_summary = slow
            app._refresh_now()
            app._update_summary = original
            await pilot.pause(0.5)                # the watchdog reports after it ends
            seen['applog'] = '\n'.join(app._app_log_lines)

    _run(scenario)
    assert 'UI stalled' in seen['applog']
    assert 'in slow' in seen['applog'] or 'tui.py' in seen['applog']
    assert 'time.sleep(0.9)' in (tmp_path / 'e.log').read_text()


def test_a_log_flood_is_drawn_in_bounded_batches():
    from infer_stack.tui import ALL_SERVICES, LOG_PANE_LINES, InferStackTUI

    controller, catalog = _ctx()
    lines = [f'vllm-x  | loading shard {i}\n' for i in range(5000)]

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: _FakeProc(lines))
        async with app.run_test() as pilot:
            await pilot.pause()
            app._restart_logs(ALL_SERVICES)
            await app.workers.wait_for_complete()
            for _ in range(20):                   # a bounded batch per tick
                app._drain_logs()
            await pilot.pause()
            shown = list(app._log_lines)
            assert len(shown) <= 2 * LOG_PANE_LINES
            assert shown[-1].endswith('loading shard 4999')        # newest kept
            assert any('earlier line(s) not shown' in s for s in shown)

    _run(scenario)


def test_a_running_action_shows_in_the_activity_line_until_it_ends():
    """A slow action is visibly working: spinner, what, and for how long."""
    import threading

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    name = next(iter(catalog.endpoints))
    gate = threading.Event()
    real_acquire = controller.acquire

    def slow_acquire(*args, **kwargs):
        gate.wait(5)
        return real_acquire(*args, **kwargs)

    controller.acquire = slow_acquire
    seen = {}

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._start_acquire(name)
            await pilot.pause(0.3)
            line = app.query_one('#activity')
            seen['during'] = (line.display, str(line.render()))
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause(0.3)
            seen['after'] = line.display

    _run(scenario)
    shown, text = seen['during']
    assert shown and f'acquiring {name}' in text and 's' in text
    assert seen['after'] is False


def test_an_edit_made_outside_the_tui_appears_on_the_next_refresh(tmp_path):
    """The catalog file is reread when it changes (and on `r`); a broken save
    is reported once, not on every refresh."""
    import copy

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
            edited = copy.deepcopy(CATALOG)
            edited['endpoints']['qwen-extra'] = dict(edited['endpoints']['qwen-fast'])
            catalog_path.write_text(yaml.safe_dump(edited))
            app.action_refresh()
            await pilot.pause()
            assert 'qwen-extra' in app._endpoint_names

            refused = []
            app._refuse = lambda msg, **kw: refused.append(msg)
            catalog_path.write_text('endpoints: [not, a, mapping\n')
            app.action_refresh()
            app.action_refresh()
            assert len(refused) == 1 and 'catalog reload failed' in refused[0]
            assert 'qwen-extra' in app._endpoint_names       # the last good one stays

    _run(scenario)


def test_an_80x24_terminal_shows_logs_and_the_tables_when_the_runtime_opens():
    """At 80x24 the runtime pane used to take every row (the tables vanished),
    and then, capped naively, left none for the log itself."""
    from textual.widgets import Collapsible

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app.has_class('compact')           # descriptions give way
            app.query_one('#docker', Collapsible).collapsed = False
            await pilot.pause()
            await pilot.pause()
            assert app.query_one('#logs').region.height >= 3
            assert app.query_one('#tables').region.height >= 1

    _run(scenario)


def test_the_api_tab_never_shows_the_master_key():
    """The curl on screen reads the key when run; the clipboard gets it."""
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._litellm = lambda: ('http://127.0.0.1:14042/v1', 'sk-secret-key')
            app._sync_api_models(['qwen-coder'])
            app._update_api_curl()
            shown = str(app.query_one('#api-curl').render())
            assert 'sk-secret-key' not in shown
            assert '$(infer-stack env LITELLM_MASTER_KEY)' in shown
            copied = []
            app._copy = lambda text: copied.append(text) or True
            app.action_api_copy_curl()
            assert copied and 'sk-secret-key' in copied[0]

    _run(scenario)


def test_colored_engine_output_reads_cleanly_in_the_logs_pane():
    """vLLM colors its "(APIServer pid=1)" prefix; the escape codes used to
    garble the line in the pane."""
    from textual.widgets import RichLog

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import Collapsible

            app.query_one('#docker', Collapsible).collapsed = False
            await pilot.pause()
            app._write_log_lines(['\x1b[1;36m(APIServer pid=1)\x1b[0;0m INFO engines: model loaded'])
            await pilot.pause()
            text = '\n'.join(strip.text for strip in app.query_one('#logs', RichLog).lines)
            assert '(APIServer pid=1) INFO engines: model loaded' in text
            assert '\x1b' not in text and '[1;36m' not in text

    _run(scenario)


def test_tui_sidebar_follows_terminal_width_until_resized_by_hand():
    # Pass 5 of the UX audit: at 200 columns the catalog sidebar stayed 38
    # wide and cut its gpu column to "aut" beside 160 columns of empty table.
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.pause()
            assert app.query_one('#sidebar').size.width == 64
            await pilot.press('right_square_bracket')
            await pilot.pause()
            assert app._sidebar_w == 68
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app._sidebar_w == 68          # a hand-set width is kept

    _run(scenario)


def test_tui_number_keys_and_palette_reach_every_top_tab():
    # Pass 5 of the UX audit: API, UI and Settings were reachable only by
    # mouse or by tabbing into the tab bar; the palette knew none of them.
    from textual.widgets import TabbedContent

    from infer_stack.tui import TOP_TABS, InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            top = app.query_one('#top', TabbedContent)
            for i, (_, pane) in reversed(list(enumerate(TOP_TABS, 1))):
                await pilot.press(str(i))
                await pilot.pause()
                assert top.active == pane
            titles = [c.title for c in app.get_system_commands(app.screen)]
            assert {f'Go to {label}' for label, _ in TOP_TABS} <= set(titles)

    _run(scenario)


def test_tui_log_pane_grows_when_the_terminal_does():
    # Pass 5 of the UX audit: started at 80x24 and enlarged to 200x50, the
    # runtime pane kept its small-screen height (one log line) because the
    # resize handler read the app's size before it updated.
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            small = app.query_one('#docker-tabs').styles.height.value
            await pilot.resize_terminal(200, 50)
            await pilot.pause()
            assert app.query_one('#docker-tabs').styles.height.value > small
            assert app.query_one('#docker-tabs').styles.height.value == app._log_height(50)

    _run(scenario)
