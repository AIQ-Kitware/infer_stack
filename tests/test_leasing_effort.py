"""Per-request Qwen effort is a catalog capability, not a serve-time budget."""

from __future__ import annotations

import pytest
import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing.catalog import Catalog, CatalogError
from infer_stack.leasing.gateway import (
    Gateway, compose_catalog_route, deployment_routes, remembered_rows,
    registry_route,
)
from infer_stack.leasing.routes import GatewayRoute
from infer_stack.leasing.suggest import suggest_catalog


def _catalog(levels=('low', 'medium', 'xhigh')):
    return Catalog.from_dict({
        'models': {'q': {'source': 'hf://dbirks/Qwen3.8-27B-W4A16-AutoRound'}},
        'endpoints': {'hq': {
            'engine': 'vllm', 'model': 'q',
            'reasoning_efforts': list(levels),
            'runtime': {'max_model_len': 65536},
        }},
    })


def test_suggested_hyperqwen_variants_publish_exact_native_efforts():
    out = suggest_catalog(simulate_inventory('1x96'))
    for name, spec in out['endpoints'].items():
        if name.startswith('qwen3.8-27b-dbirks-hyperqwen'):
            assert spec['reasoning_efforts'] == ['low', 'medium', 'xhigh']
    Catalog.from_dict(out)


@pytest.mark.parametrize('levels', [
    [], ['ultra'], ['bogus'], ['low', 'low'], ['low', 1], 'low', None,
])
def test_reject_unknown_or_ill_formed_efforts(levels):
    source = {
        'models': {'q': {'source': 'hf://org/q'}},
        'endpoints': {'q': {'engine': 'vllm', 'model': 'q',
                            'reasoning_efforts': levels}},
    }
    with pytest.raises(CatalogError, match='reasoning_efforts'):
        Catalog.from_dict(source)


def test_catalog_persists_capability_per_alias_without_deployment_identity_change():
    with_levels = _catalog()
    raw = {'models': {'q': {'source': 'hf://dbirks/Qwen3.8-27B-W4A16-AutoRound'}},
           'endpoints': {'hq': {'engine': 'vllm', 'model': 'q',
                                'runtime': {'max_model_len': 65536}}}}
    without = Catalog.from_dict(raw)
    a = with_levels.resolve_endpoint('hq').to_request()
    b = without.resolve_endpoint('hq').to_request()
    assert a.structural == b.structural
    assert a.spec == b.spec
    assert a.served['reasoning_efforts'] == ['low', 'medium', 'xhigh']
    assert 'reasoning_efforts' not in b.served


def test_litellm_route_opt_in_explicitly_allows_request_field():
    route = GatewayRoute('hq', 'openai', 'qwen', 'http://vllm:8000/v1',
                         reasoning_efforts=('low', 'medium', 'xhigh'))
    entry = route.entry()
    assert entry['litellm_params']['allowed_openai_params'] == ['reasoning_effort']
    assert entry['model_info']['supports_reasoning'] is True
    assert entry['model_info']['reasoning_effort_levels'] == [
        'low', 'medium', 'xhigh']
    assert 'allowed_openai_params' not in GatewayRoute(
        'no-effort', 'openai', 'm', 'http://vllm:8000/v1').entry()['litellm_params']
    assert 'allowed_openai_params' not in GatewayRoute(
        'ollama', 'ollama', 'q', 'http://ollama:11434').entry()['litellm_params']


def test_static_catalog_route_advertises_efforts():
    cat = _catalog()
    route = compose_catalog_route('hq', cat.resolve_endpoint('hq').to_request())
    assert route.entry()['model_info']['reasoning_effort_levels'] == [
        'low', 'medium', 'xhigh']


def test_dynamic_route_and_registry_keep_per_alias_capabilities():
    from infer_stack.leasing.models import Deployment, DeploymentState
    cat = _catalog()
    request = cat.resolve_endpoint('hq').to_request()
    dep = Deployment('d', 'ck', 'vllm', 'shared-compatible', {}, request.spec,
                     {'hq': request.served}, DeploymentState.LIVE, 1, 1)
    routes = deployment_routes([dep], {'d': [0]}, dynamic=True)
    assert routes[0].entry()['model_info']['reasoning_effort_levels'] == [
        'low', 'medium', 'xhigh']
    rows = remembered_rows([dep], {'d': [0]}, defined=set())
    assert registry_route('hq', rows['hq']).entry()['model_info'][
        'reasoning_effort_levels'] == ['low', 'medium', 'xhigh']
    # An already-running deployment created before the new metadata remains
    # visible when the catalog's current definition now supplies the levels.
    dep.served['hq'] = dict(request.served)
    dep.served['hq'].pop('reasoning_efforts')
    routes = deployment_routes([dep], {'d': [0]}, dynamic=True, catalog=cat)
    assert routes[0].reasoning_efforts == ('low', 'medium', 'xhigh')


def test_dynamic_reconcile_compares_positive_capability_without_false_churn():
    enabled = GatewayRoute('hq', 'openai', 'm', 'http://vllm:8000/v1',
                           route_id='isr-test',
                           reasoning_efforts=('low', 'medium', 'xhigh')).entry()
    disabled = GatewayRoute('hq', 'openai', 'm', 'http://vllm:8000/v1',
                            route_id='isr-test').entry()
    assert not Gateway._route_semantics_match(
        Gateway._route_semantics(enabled), Gateway._route_semantics(disabled))
    assert Gateway._route_semantics_match(
        Gateway._route_semantics(enabled), Gateway._route_semantics(enabled))
    synthesized = Gateway._route_semantics(disabled)
    synthesized['supports_reasoning'] = False
    assert Gateway._route_semantics_match(
        Gateway._route_semantics(disabled), synthesized)


def test_static_compose_serializes_native_effort_metadata(tmp_path):
    from test_leasing_compose import IMAGES, PORTS, STATE, render_compose

    rc = render_compose([], {}, images=IMAGES, ports=PORTS, state=STATE,
                        litellm=True, litellm_port=14042, aux_dir=tmp_path,
                        catalog=_catalog())
    (entry,) = yaml.safe_load(rc.litellm_config)['model_list']
    assert entry['model_name'] == 'hq'
    assert entry['model_info']['supports_reasoning'] is True
    assert entry['model_info']['reasoning_effort_levels'] == [
        'low', 'medium', 'xhigh']
    assert entry['litellm_params']['allowed_openai_params'] == ['reasoning_effort']


def test_dynamic_gateway_publishes_effort_and_repairs_capability_drift(tmp_path):
    from test_leasing_dynamic_routing import (
        RecordingGateway, dep, make_backend, _route_id,
    )

    d = dep('grp-aaaaaa', served='hyperqwen')
    d.served['hyperqwen']['reasoning_efforts'] = ['low', 'medium', 'xhigh']
    gw = RecordingGateway()
    backend = make_backend(tmp_path, gw)
    backend.converge([d], apply=True)
    route_id = _route_id(d.id, 'hyperqwen')
    body = gw.models[route_id]
    assert body['model_info']['reasoning_effort_levels'] == [
        'low', 'medium', 'xhigh']
    assert body['litellm_params']['allowed_openai_params'] == ['reasoning_effort']

    # If an older route loses the capability metadata, recreate it.
    del gw.models[route_id]['litellm_params']['allowed_openai_params']
    before = len(gw.calls)
    assert backend.gateway._reconcile_routes() is True
    assert gw.calls[before:] == [('delete', route_id), ('new', route_id)]
    assert gw.models[route_id]['litellm_params']['allowed_openai_params'] == [
        'reasoning_effort']

    # Correctly declared capabilities must not trigger spurious route churn.
    before = len(gw.calls)
    assert backend.gateway._reconcile_routes() is True
    assert len(gw.calls) == before
