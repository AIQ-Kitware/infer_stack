"""Headless tests for the optional Textual TUI (skipped if textual absent)."""

from __future__ import annotations

import asyncio
import json

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
            # docker is a collapsible pane with Logs/Containers/Control tabs;
            # system is its own collapsed pane; API is a top-level tab now.
            tabs = app.query_one('#docker-tabs', TabbedContent)
            assert {p.id for p in tabs.query('TabPane')} == {
                'tab-logs', 'tab-containers', 'tab-control'
            }
            assert app.query_one('#docker', Collapsible)
            assert app.query_one('#system', Collapsible).collapsed
            top = app.query_one('#top', TabbedContent)
            assert 'tab-api' in {p.id for p in top.query('TabPane')}
            # the containers ps view carries the docker-ps columns
            ps = app.query_one('#ps', DataTable)
            labels = [str(c.label) for c in ps.columns.values()]
            assert 'status (uptime)' in labels
            assert 'created' in labels
            assert 'container id' in labels

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


def test_tui_leases_deployments_are_separate_panes():
    from textual.containers import Vertical

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # leases/deployments are their own panes split by a drag handle,
            # not collapsibles.
            assert isinstance(app.query_one('#leases-pane'), Vertical)
            assert isinstance(app.query_one('#deployments-pane'), Vertical)
            assert app.query_one('#tsplit')

    _run(scenario)


def test_tui_panes_drag_resize():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
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
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
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

    o = InferStackTUI._endpoint_entry({
        'model': 'qwen', 'engine': 'ollama', 'host': 'oll',
        'ollama_runtime': 'num_ctx=8192 keep_alive=5m',
    })
    assert o['engine'] == 'ollama' and o['host'] == 'oll'
    assert o['runtime']['num_ctx'] == 8192
    assert o['runtime']['keep_alive'] == '5m'


def test_tui_endpoint_wizard_is_engine_adaptive_and_labeled():
    from textual.widgets import Label, Select

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
            # fields are labeled (the "blank page" complaint)
            labels = [str(lbl.render()) for lbl in screen.query(Label)]
            assert any('tensor-parallel' in x for x in labels)
            assert any('data-parallel' in x for x in labels)
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


def test_tui_ps_rows_parse_status_created_and_id():
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    sample = json.dumps([
        {'Service': 'litellm', 'Status': 'Up 3 minutes', 'State': 'running',
         'CreatedAt': '2026-06-19 00:00:00 -0400', 'ID': 'abcdef1234567890',
         'Publishers': [{'PublishedPort': 14042, 'TargetPort': 4000}]},
    ])

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # stub the backend's compose seam
            backend = controller.backend
            import tempfile
            f = tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False)
            f.write('services: {}\n')
            f.close()
            backend.compose_file = f.name
            backend.project = 'infer-stack'
            backend.run = lambda args: sample
            rows = app._compose_ps_rows()
            assert rows[0]['status'] == 'Up 3 minutes'
            assert rows[0]['created'].startswith('2026-06-19')
            assert rows[0]['id'] == 'abcdef123456'        # truncated to 12
            assert '14042->4000' in rows[0]['ports']

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
            assert 'ps' not in app._collect()          # collapsed -> no ps poll
            app._collapsed['docker'] = False
            assert 'ps' in app._collect()              # visible -> polled
            app._active_tab = 'tab-logs'
            assert 'ps' not in app._collect()          # other tab -> no ps poll
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
            controller.backend.ui_port = 13000
            assert app._ui_url('qwen-coder') == (
                'http://localhost:13000/?models=qwen-coder'
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
            controller.backend.litellm_port = 14042
            # only ready (running) models are offered; simulate one being ready
            app._sync_api_models(['qwen-coder'])
            assert app.query_one('#api-model', Select).value == 'qwen-coder'
            app.action_api_send()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any('pong' in line for line in app._api_lines)

    _run(scenario)
    assert http.calls and http.calls[0][0].endswith('/v1/chat/completions')


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
            controller.backend.litellm_port = 14042
            controller.backend.ui_port = 13000
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
            controller.backend.litellm_port = 14042
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


def test_tui_api_lists_only_ready_models():
    from textual.widgets import Select

    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # NullBackend observes nothing running -> no ready models listed,
            # even though the catalog has endpoints.
            assert app._ready_endpoints == []
            assert app.query_one('#api-model', Select).value is Select.NULL

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


def test_tui_compose_control_runs_up(tmp_path):
    from infer_stack.tui import InferStackTUI

    controller, catalog = _ctx()
    compose_file = tmp_path / 'docker-compose.yml'
    compose_file.write_text('services: {}\n')
    calls = []

    async def scenario():
        app = InferStackTUI(controller, catalog, interval=999,
                            proc_factory=lambda svc: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            backend = controller.backend
            backend.compose_file = compose_file
            backend.project = 'infer-stack'
            backend.run = lambda args: calls.append(args) or ''
            app.action_compose_up()
            await app.workers.wait_for_complete()
            await pilot.pause()

    _run(scenario)
    assert calls and calls[0][:2] == ['docker', 'compose'] and 'up' in calls[0]


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
