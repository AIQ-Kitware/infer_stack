"""Gateway routes: derived at render, one layer per owner.

The gateway's static ``model_list`` is derived at every render from the
published catalog union (every runbook's endpoints) and the placed
deployments, over the route registry, which now keeps only the routes of
deployments no published catalog defines (remembered past release, so
releasing one does not recreate the gateway) plus any rows from before.
The headline property: the rendered config depends on endpoint definitions,
not on which models are up or which runbook converged, so the gateway is not
recreated as models come and go.

Driven by the same stateful fake-docker seam as ``test_leasing_compose.py``;
no real docker, GPUs, or network. Converges run ``apply=False``: only the
render half (which reads the registry and remembers rows) executes.
"""

from __future__ import annotations

import contextlib
import json

import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Catalog, ComposeBackend
from infer_stack.leasing.gateway import (
    CONFIG_HASH_LABEL,
    LITELLM_CONFIG_FILENAME,
    LITELLM_REGISTRY_FILENAME,
    LITELLM_ROUTES_FILENAME,
    catalog_routes,
    deployment_routes,
    front_door_routes,
    registry_routes,
    remembered_rows,
)
from infer_stack.leasing.models import (
    RESERVED_ENGINE,
    Deployment,
    DeploymentState,
)
from infer_stack.leasing.profile import CatalogUnion
from infer_stack.leasing.routes import GatewayRoute, route_table

STATE = {'hf_cache': '/cache/hf', 'ollama': '/cache/ollama'}
IMAGES = {
    'vllm': 'vllm/vllm-openai:test',
    'ollama': 'ollama/ollama:test',
    'litellm': 'ghcr.io/berriai/litellm:test',
}
PORTS = {'ollama': 11434}


# -- fixtures / builders ---------------------------------------------------


def vllm_ep(endpoint, *, served=None, t=0.0):
    """A vLLM deployment whose *endpoint alias* (served-map key) is ``endpoint``.

    Matches how a catalog endpoint acquired live coalesces: the endpoint name is
    the alias, so its registry row keys line up with the catalog's — which is
    what keeps live-vs-released status from oscillating the rendered bytes.
    """
    served_name = served or endpoint
    return Deployment(
        'grp-' + endpoint, 'ck-' + endpoint, 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': 'org/model',
            'served_model_name': served_name,
            'runtime': {'tensor_parallel_size': 1, 'max_model_len': 32768},
            'reclaim': 'keep-warm',
        },
        {endpoint: {'served_model_name': served_name, 'protocol': 'chat'}},
        DeploymentState.LIVE, t, t,
    )


def _catalog_dict(endpoint, *, model='m', served=None):
    spec = {'engine': 'vllm', 'model': model}
    if served:
        spec['served_name'] = served
    # The same runtime as ``vllm_ep`` below: a live-acquired catalog endpoint
    # carries the endpoint's own runtime, so its deployment and its catalog
    # route agree on the window they advertise and the rendered bytes never
    # move as the model comes and goes.
    spec['runtime'] = {'tensor_parallel_size': 1, 'max_model_len': 32768}
    return {'models': {model: {'source': f'hf://org/{model}'}},
            'endpoints': {endpoint: spec}}


def _catalog(endpoint, *, model='m', served=None):
    return Catalog.from_dict(_catalog_dict(endpoint, model=model, served=served))


def _write_registry(tmp_path, entries, *, version=1):
    path = tmp_path / LITELLM_REGISTRY_FILENAME
    path.write_text(json.dumps({'version': version, 'entries': entries},
                               sort_keys=True, indent=2) + '\n')
    return path


def _litellm_service(tmp_path):
    return yaml.safe_load((tmp_path / 'docker-compose.yml').read_text())['services']['litellm']


class FakeDocker:
    """Stateful docker compose stand-in (unused under apply=False, but the
    backend still needs a ``run`` seam)."""

    def __init__(self):
        self.running: list[str] = []

    def __call__(self, args):  # pragma: no cover - not exercised at apply=False
        return ''


def make_backend(tmp_path, *, catalog=None, dynamic_routing=False, spec='4x80'):
    return ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory(spec),
        run=FakeDocker(),
        images=IMAGES, ports=PORTS, state=STATE,
        catalog=catalog, dynamic_routing=dynamic_routing,
    )


def _model_list(tmp_path):
    cfg = tmp_path / LITELLM_CONFIG_FILENAME
    data = yaml.safe_load(cfg.read_text()) or {}
    return data.get('model_list', [])


def _aliases(tmp_path):
    return sorted(e['model_name'] for e in _model_list(tmp_path))


def _registry(tmp_path):
    return json.loads((tmp_path / LITELLM_REGISTRY_FILENAME).read_text())


@contextlib.contextmanager
def capture_warnings():
    """Capture infer-stack's loguru WARNING narration (the library is
    ``logger.disable``d by default, and loguru does not route to pytest's
    caplog, so we attach a sink for the duration)."""
    from infer_stack._log import logger

    msgs: list[str] = []
    logger.enable('infer_stack')
    sink = logger.add(
        lambda m: msgs.append(m.record['message']), level='WARNING'
    )
    try:
        yield msgs
    finally:
        logger.remove(sink)
        logger.disable('infer_stack')


# -- 1. precedence: registry < catalog < deployment < upstream -------------


def test_route_table_precedence():
    old = GatewayRoute('a', 'openai', 'old', 'http://old/v1', origin='registry')
    cat = GatewayRoute('a', 'openai', 'cat', 'http://cat/v1')
    dep = GatewayRoute('a', 'openai', 'dep', 'http://dep/v1', origin='deployment')
    up = GatewayRoute('a', 'openai', 'up', 'http://up/v1', origin='upstream')
    assert route_table([old], [cat], [dep], [up]) == [up]
    assert route_table([up], [old]) == [old]           # later layer wins
    assert [r.alias for r in route_table([cat], [old.__class__('b', 'openai', 'x', 'y')])] == ['a', 'b']


def test_catalog_endpoints_are_not_stored(tmp_path):
    """The published union is the one store of endpoint definitions: a render
    derives their routes, and the registry keeps none of them."""
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    be.converge([vllm_ep('alpha')], apply=False)
    assert _aliases(tmp_path) == ['alpha']
    assert not (tmp_path / LITELLM_REGISTRY_FILENAME).exists()


# -- 2. byte stability (the headline property) ------------------------------


def test_config_is_byte_stable_as_models_come_and_go(tmp_path):
    union = CatalogUnion.from_sources([_catalog_dict('alpha'), _catalog_dict('beta')])
    be = make_backend(tmp_path, catalog=union)
    be.converge([vllm_ep('alpha')], apply=False)
    cfg_1 = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()
    svc_1 = _litellm_service(tmp_path)
    be.converge([vllm_ep('beta')], apply=False)
    be.converge([], apply=False)
    assert (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes() == cfg_1
    assert _litellm_service(tmp_path)['labels'][CONFIG_HASH_LABEL] == \
        svc_1['labels'][CONFIG_HASH_LABEL]
    assert _aliases(tmp_path) == ['alpha', 'beta']


def test_hash_stable_across_runbook_alternation(tmp_path):
    """Runbook A, then B, then A again on one ledger: the published union holds
    both catalogs, so renders 2 and 3 are byte-identical."""
    from test_leasing_profile import cat, controller

    a, b = Catalog.from_dict(cat('alpha')), Catalog.from_dict(cat('beta'))
    _, ctl_a = controller(tmp_path, catalog=a)
    ctl_a.acquire('a', a.resolve_names(['alpha']), wait=False)
    _, ctl_b = controller(tmp_path, catalog=b)
    ctl_b.acquire('b', b.resolve_names(['beta']), wait=False)
    config = tmp_path / 'state' / LITELLM_CONFIG_FILENAME
    cfg_2 = config.read_bytes()
    _, ctl_a2 = controller(tmp_path, catalog=a)
    ctl_a2.acquire('a2', a.resolve_names(['alpha']), wait=False)
    assert config.read_bytes() == cfg_2
    assert sorted(e['model_name'] for e in yaml.safe_load(cfg_2)['model_list']) == \
        ['alpha', 'beta']


# -- 3. a non-catalog deployment is remembered past its release -------------


def test_live_non_catalog_deployment_stays_routed(tmp_path):
    """A deployment no published catalog defines is remembered in the registry,
    so it stays routed after release and releasing it does not recreate the
    gateway."""
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    be.converge([vllm_ep('alpha'), vllm_ep('extra')], apply=False)
    before = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()
    assert set(_registry(tmp_path)['entries']) == {'extra'}   # alpha is derived
    be.converge([vllm_ep('alpha')], apply=False)              # 'extra' released
    assert (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes() == before
    assert _aliases(tmp_path) == ['alpha', 'extra']


def test_a_published_definition_wins_over_a_registry_row(tmp_path):
    """A registry row (e.g. one an older binary wrote for a catalog endpoint)
    is the lowest layer: the published definition renders instead."""
    _write_registry(tmp_path, {'alpha': {'engine': 'vllm', 'served': 'v1'}})
    be = make_backend(tmp_path, catalog=_catalog('alpha', served='v2'))
    be.converge([], apply=False)
    (entry,) = _model_list(tmp_path)
    assert entry['litellm_params']['model'] == 'openai/v2'


# -- 4. registry rows from an older binary are still routed -----------------


def test_catalog_less_converge_renders_existing_registry_rows(tmp_path):
    """An upgraded state dir whose registry holds catalog rows: a catalog-less
    converge still routes them (nothing is lost on upgrade), and the file is
    not rewritten."""
    _write_registry(tmp_path, {'alpha': {'engine': 'vllm', 'served': 'alpha'},
                               'tiny': {'engine': 'ollama', 'model': 'tinyllama',
                                        'host': 'gpuhost'}})
    original = (tmp_path / LITELLM_REGISTRY_FILENAME).read_bytes()
    bare = make_backend(tmp_path, catalog=None)
    bare.converge([], apply=False)
    assert _aliases(tmp_path) == ['alpha', 'tiny']
    assert (tmp_path / LITELLM_REGISTRY_FILENAME).read_bytes() == original


def test_corrupt_registry_is_ignored(tmp_path):
    (tmp_path / LITELLM_REGISTRY_FILENAME).write_text('{ this is not json')
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    be.converge([vllm_ep('alpha')], apply=False)  # must not raise
    assert _aliases(tmp_path) == ['alpha']
    (tmp_path / LITELLM_REGISTRY_FILENAME).write_text('{"entries": [1, 2, 3]}')
    be.converge([vllm_ep('alpha')], apply=False)
    assert _aliases(tmp_path) == ['alpha']


def test_unknown_version_is_read_and_never_rewritten(tmp_path):
    """A registry from a newer schema is rendered as-is with a warning and
    never rewritten, even when a render has a row to remember."""
    path = _write_registry(tmp_path, {'old': {'engine': 'vllm', 'served': 'old'}},
                           version=99)
    original = path.read_bytes()
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    with capture_warnings() as warnings:
        be.converge([vllm_ep('alpha'), vllm_ep('extra')], apply=False)
    assert path.read_bytes() == original
    assert _aliases(tmp_path) == ['alpha', 'extra', 'old']
    assert any('unknown schema version' in m for m in warnings)


# -- 5. dynamic routing ------------------------------------------------------


def test_dynamic_routing_creates_no_registry(tmp_path):
    be = make_backend(tmp_path, dynamic_routing=True)
    be.converge([vllm_ep('alpha')], apply=False)
    assert not (tmp_path / LITELLM_REGISTRY_FILENAME).exists()
    assert (tmp_path / LITELLM_ROUTES_FILENAME).exists()


# -- 6. derivations agree ----------------------------------------------------


def test_reserved_engine_routes_nowhere():
    reserved = Deployment(
        'grp-r', 'ck-r', RESERVED_ENGINE, 'dedicated', {},
        {'engine': RESERVED_ENGINE, 'reserved_gpu_count': 1},
        {'reserved-gpu': {}}, DeploymentState.LIVE, 0.0, 0.0,
    )
    placed = {'grp-r': [0], 'grp-alpha': [1]}
    assert [r.alias for r in deployment_routes([reserved, vllm_ep('alpha')], placed)] == ['alpha']
    assert set(remembered_rows([reserved, vllm_ep('alpha')], placed, defined=set())) == {'alpha'}


def test_remembered_rows_render_back_to_the_live_route():
    """A remembered row renders exactly the route its deployment had, so a
    release never moves the rendered bytes (vLLM, multi-alias, and Ollama)."""
    multi = Deployment(
        'grp-m', 'ck-m', 'vllm', 'shared-compatible', {},
        {'engine': 'vllm', 'hf_model_id': 'org/model', 'served_model_name': 'shared',
         'runtime': {'tensor_parallel_size': 1}, 'reclaim': 'keep-warm'},
        {'ep1': {'served_model_name': 'shared'}, 'ep2': {'served_model_name': 'shared'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )
    tiny = Deployment(
        'grp-o', 'ck-o', 'ollama', 'shared-compatible', {},
        {'engine': 'ollama', 'host': 'gpuhost'}, {'tiny': {'model': 'tinyllama'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )
    placed = {'grp-m': [0], 'grp-o': [1]}
    live = deployment_routes([multi, tiny], placed)
    rows = remembered_rows([multi, tiny], placed, defined=set())
    assert registry_routes({'entries': rows}) == sorted(live, key=lambda r: r.alias)
    upstream = GatewayRoute('k', 'openai', 'model-k', 'http://kubeai/openai/v1',
                            origin='upstream')
    rows = remembered_rows([], {}, defined=set(), extra=[upstream])
    assert registry_routes({'entries': rows}) == [upstream]


def test_catalog_and_live_routes_coincide():
    assert catalog_routes(_catalog('alpha')) == deployment_routes(
        [vllm_ep('alpha')], {'grp-alpha': [0]})


def test_dynamic_routes_are_one_per_managed_id():
    external = Catalog.from_dict({'endpoints': {'box': {'external': {
        'api_base': 'http://box:9000/v1', 'model': 'Org/Big', 'api_key_env': 'BOX_KEY'}}}})
    static, dynamic = front_door_routes([vllm_ep('alpha')], {'grp-alpha': [0]},
                                        catalog=external, dynamic=True)
    assert static == []
    assert [r.alias for r in dynamic] == ['alpha', 'box']
    assert len({r.route_id for r in dynamic}) == 2


# -- 7. concurrency smoke ----------------------------------------------------


def test_concurrency_smoke_no_lost_update(tmp_path):
    """Two backends sharing one state dir remember different ad-hoc
    deployments (serialized by the converge flock): both rows survive."""
    be_a = make_backend(tmp_path, catalog=_catalog('alpha'))
    be_b = make_backend(tmp_path, catalog=_catalog('beta'))
    be_a.converge([vllm_ep('one')], apply=False)
    be_b.converge([vllm_ep('two')], apply=False)
    assert set(_registry(tmp_path)['entries']) == {'one', 'two'}
