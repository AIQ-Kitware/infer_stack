"""Tests for the append-only LiteLLM route registry (static-superset mode).

The registry accumulates the semantic route inputs (served name / engine / host)
of every catalog *and* live deployment ever merged across the runbooks that share
one state dir, and the gateway ``model_list`` is rendered from the whole registry.
The headline property under test: once every catalog has been merged once, a
converge under any runbook renders a byte-identical gateway config, so the gateway
is never recreated and no cross-catalog converge can strip another's live routes.

Driven by the same stateful fake-docker seam as ``test_leasing_compose.py``; no
real docker, GPUs, or network. Converges run ``apply=False`` so only the render
half (which owns the registry read-merge-write) executes — deterministic and fast.
"""

from __future__ import annotations

import contextlib
import json

import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Catalog, ComposeBackend, render_compose
from infer_stack.leasing.compose import (
    CONFIG_HASH_LABEL,
    LITELLM_CONFIG_FILENAME,
    LITELLM_REGISTRY_FILENAME,
    LITELLM_ROUTES_FILENAME,
    _litellm_model_list,
    _litellm_model_list_from_registry,
    _merge_route_registry,
    _registry_incoming_from_catalog,
    _registry_incoming_from_deployments,
)
from infer_stack.leasing.models import (
    RESERVED_ENGINE,
    Deployment,
    DeploymentState,
)

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


def _catalog(endpoint, *, model='m'):
    return Catalog.from_dict(
        {
            'models': {model: {'source': f'hf://org/{model}'}},
            'endpoints': {endpoint: {'engine': 'vllm', 'model': model}},
        }
    )


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


# -- 1. merge idempotence --------------------------------------------------


def test_merge_idempotent_byte_identical(tmp_path):
    cat = _catalog('alpha')
    be = make_backend(tmp_path, catalog=cat)
    be.converge([vllm_ep('alpha')], apply=False)
    first = (tmp_path / LITELLM_REGISTRY_FILENAME).read_bytes()
    be.converge([vllm_ep('alpha')], apply=False)
    second = (tmp_path / LITELLM_REGISTRY_FILENAME).read_bytes()
    assert first == second  # merging the same inputs twice is a no-op


def test_merge_incoming_wins_on_conflict_pure():
    existing = {'version': 1, 'entries': {'a': {'engine': 'vllm', 'served': 'a'}}}
    merged, warnings = _merge_route_registry(
        existing, {'a': {'engine': 'vllm', 'served': 'a2'}}
    )
    assert merged['entries']['a'] == {'engine': 'vllm', 'served': 'a2'}
    assert warnings and 'redefined' in warnings[0]


def test_merge_additive_never_removes_pure():
    existing = {'version': 1, 'entries': {'a': {'engine': 'vllm', 'served': 'a'}}}
    merged, _ = _merge_route_registry(
        existing, {'b': {'engine': 'vllm', 'served': 'b'}}
    )
    assert set(merged['entries']) == {'a', 'b'}


# -- 2. hash stability across catalog alternation (the headline property) ---


def test_hash_stable_across_alternation(tmp_path):
    """Converge catalog A, then B, then A again on one shared state dir. Renders
    2 and 3 must produce a byte-identical ``litellm_config.yaml`` AND an equal
    litellm service dict / CONFIG_HASH_LABEL — the label is what drives recreate.
    """
    be_a = make_backend(tmp_path, catalog=_catalog('alpha'))
    be_b = make_backend(tmp_path, catalog=_catalog('beta'))

    be_a.converge([vllm_ep('alpha')], apply=False)  # render 1: {alpha}
    be_b.converge([vllm_ep('beta')], apply=False)   # render 2: {alpha, beta}
    cfg_2 = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()
    svc_2 = yaml.safe_load(
        (tmp_path / 'docker-compose.yml').read_text()
    )['services']['litellm']

    be_a.converge([vllm_ep('alpha')], apply=False)  # render 3: still {alpha, beta}
    cfg_3 = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()
    svc_3 = yaml.safe_load(
        (tmp_path / 'docker-compose.yml').read_text()
    )['services']['litellm']

    assert cfg_2 == cfg_3  # gateway config byte-stable across the alternation
    assert svc_2 == svc_3
    assert (
        svc_2['labels'][CONFIG_HASH_LABEL] == svc_3['labels'][CONFIG_HASH_LABEL]
    )


# -- 3. union correctness --------------------------------------------------


def test_union_routes_both_catalogs_sorted(tmp_path):
    be_a = make_backend(tmp_path, catalog=_catalog('alpha'))
    be_b = make_backend(tmp_path, catalog=_catalog('beta'))
    be_a.converge([vllm_ep('alpha')], apply=False)
    be_b.converge([vllm_ep('beta')], apply=False)
    assert _aliases(tmp_path) == ['alpha', 'beta']


# -- 4. live non-catalog deployment persists across a foreign converge ------


def test_live_non_catalog_deployment_stays_routed(tmp_path):
    """A deployment absent from the invoking catalog is merged from the desired
    set and remains routed on a later converge under a different catalog."""
    be_a = make_backend(tmp_path, catalog=_catalog('alpha'))
    # 'extra' is not in catalog A — it enters the registry via the desired set.
    be_a.converge([vllm_ep('alpha'), vllm_ep('extra')], apply=False)
    assert 'extra' in _aliases(tmp_path)

    be_b = make_backend(tmp_path, catalog=_catalog('beta'))
    be_b.converge([vllm_ep('beta')], apply=False)  # 'extra' no longer live
    # Persisted past its live window and past a foreign converge.
    assert _aliases(tmp_path) == ['alpha', 'beta', 'extra']


# -- 5. conflict: incoming wins, warning, bytes change ---------------------


def test_conflict_incoming_wins_and_changes_bytes(tmp_path):
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    be.converge([vllm_ep('alpha', served='v1')], apply=False)
    before = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()

    with capture_warnings() as warnings:
        be.converge([vllm_ep('alpha', served='v2')], apply=False)
    after = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()

    assert before != after  # a genuinely changed definition => one recreate
    assert _registry(tmp_path)['entries']['alpha']['served'] == 'v2'
    assert any('redefined' in m for m in warnings)


# -- 6. seeding from a pre-existing litellm_config.yaml (upgrade migration) -


def test_seed_from_existing_config_preserves_vllm_routes(tmp_path):
    """A state dir upgraded in place: a rendered ``litellm_config.yaml`` exists
    but no registry. The first converge under a *disjoint* catalog must still
    route the old vLLM aliases (recovered by seeding). Ollama rows are skipped."""
    legacy = {
        'model_list': [
            {
                'model_name': 'old-vllm',
                'litellm_params': {
                    'model': 'openai/old-served',
                    'api_base': 'http://vllm-old-served:8000/v1',
                    'api_key': 'EMPTY',
                },
            },
            {
                'model_name': 'old-ollama',
                'litellm_params': {
                    'model': 'ollama/llama3:8b',
                    'api_base': 'http://ollama-gpuhost:11434',
                },
            },
        ]
    }
    (tmp_path / LITELLM_CONFIG_FILENAME).write_text(yaml.safe_dump(legacy))
    assert not (tmp_path / LITELLM_REGISTRY_FILENAME).exists()

    be = make_backend(tmp_path, catalog=_catalog('beta'))
    be.converge([vllm_ep('beta')], apply=False)

    aliases = _aliases(tmp_path)
    assert 'old-vllm' in aliases   # vLLM row recovered exactly
    assert 'beta' in aliases       # the new catalog merged in
    assert 'old-ollama' not in aliases  # Ollama row skipped (host not invertible)


# -- 7. corrupt registry: fail-open, rebuilt from seed + catalog -----------


def test_corrupt_registry_is_rebuilt(tmp_path):
    (tmp_path / LITELLM_REGISTRY_FILENAME).write_text('{ this is not json')
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    be.converge([vllm_ep('alpha')], apply=False)  # must not raise
    assert _aliases(tmp_path) == ['alpha']
    # A non-map ``entries`` is likewise structurally unusable -> rebuilt.
    (tmp_path / LITELLM_REGISTRY_FILENAME).write_text('{"entries": [1, 2, 3]}')
    be.converge([vllm_ep('alpha')], apply=False)
    assert _aliases(tmp_path) == ['alpha']


# -- 8. dynamic-routing isolation ------------------------------------------


def test_dynamic_routing_creates_no_registry(tmp_path):
    be = make_backend(tmp_path, dynamic_routing=True)
    be.converge([vllm_ep('alpha')], apply=False)
    assert not (tmp_path / LITELLM_REGISTRY_FILENAME).exists()
    # dynamic routing still renders its own desired route set file.
    assert (tmp_path / LITELLM_ROUTES_FILENAME).exists()


# -- 11. catalog-less converge renders from the accumulated registry --------


def test_catalog_less_converge_renders_full_registry(tmp_path):
    """Seed a registry via a catalog-full converge, then converge with
    ``catalog=None`` (as a bare release/gc does). The render must still contain
    the full registry (no fall-through to the legacy empty-when-no-deployments
    branch), bytes unchanged."""
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    be.converge([vllm_ep('alpha')], apply=False)
    before = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()

    bare = make_backend(tmp_path, catalog=None)
    bare.converge([], apply=False)  # no catalog, no live deployments
    after = (tmp_path / LITELLM_CONFIG_FILENAME).read_bytes()

    assert _aliases(tmp_path) == ['alpha']  # NOT [] -> registry, not legacy
    assert before == after  # byte-stable: no strip, no blip


# -- 12. engine filter: RESERVED_ENGINE contributes no row ------------------


def test_reserved_engine_contributes_no_row():
    reserved = Deployment(
        'grp-r', 'ck-r', RESERVED_ENGINE, 'dedicated', {},
        {'engine': RESERVED_ENGINE, 'reserved_gpu_count': 1},
        {'reserved-gpu': {}}, DeploymentState.LIVE, 0.0, 0.0,
    )
    incoming = _registry_incoming_from_deployments(
        [reserved, vllm_ep('alpha')], {'grp-r': [0], 'grp-alpha': [1]}
    )
    assert set(incoming) == {'alpha'}  # reserved skipped, mirrors render loop


# -- 13. multi-alias deployment: two rows, one shared upstream --------------


def test_multi_alias_deployment_matches_legacy_render():
    dep = Deployment(
        'grp-m', 'ck-m', 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': 'org/model',
            'served_model_name': 'shared',
            'runtime': {'tensor_parallel_size': 1},
            'reclaim': 'keep-warm',
        },
        {
            'ep1': {'served_model_name': 'shared'},
            'ep2': {'served_model_name': 'shared'},
        },
        DeploymentState.LIVE, 0.0, 0.0,
    )
    assignments = {'grp-m': [0]}
    incoming = _registry_incoming_from_deployments([dep], assignments)
    assert incoming == {
        'ep1': {'engine': 'vllm', 'served': 'shared'},
        'ep2': {'engine': 'vllm', 'served': 'shared'},
    }
    registry = {'version': 1, 'entries': incoming}
    rendered = _litellm_model_list_from_registry(registry)
    legacy = _litellm_model_list([dep], assignments)
    assert sorted(rendered, key=lambda e: e['model_name']) == sorted(
        legacy, key=lambda e: e['model_name']
    )


# -- 14. unknown-version tolerance -----------------------------------------


def test_unknown_version_preserved_not_reseeded(tmp_path):
    """A registry from a newer schema (``version: 99``) with a valid ``entries``
    map is rendered as-is with a warning and NOT rewritten/reseeded — a binary
    rollback must not discard the accumulated union."""
    registry = {
        'version': 99,
        'entries': {'alpha': {'engine': 'vllm', 'served': 'alpha'}},
    }
    path = tmp_path / LITELLM_REGISTRY_FILENAME
    path.write_text(json.dumps(registry, sort_keys=True, indent=2) + '\n')
    original = path.read_bytes()

    # Converge with the same catalog+deployment already in the registry, so the
    # merge is a no-op and the file must not be rewritten (version stays 99).
    be = make_backend(tmp_path, catalog=_catalog('alpha'))
    with capture_warnings() as warnings:
        be.converge([vllm_ep('alpha')], apply=False)

    assert path.read_bytes() == original  # not rewritten/reseeded
    assert _registry(tmp_path)['version'] == 99
    assert 'alpha' in _aliases(tmp_path)
    assert any('unknown schema version' in m for m in warnings)


# -- extra: catalog vs live reduce to the identical row --------------------


def test_catalog_and_live_rows_coincide():
    cat = _catalog('alpha')
    from_cat = _registry_incoming_from_catalog(cat)
    from_live = _registry_incoming_from_deployments(
        [vllm_ep('alpha')], {'grp-alpha': [0]}
    )
    assert from_cat == from_live == {'alpha': {'engine': 'vllm', 'served': 'alpha'}}
