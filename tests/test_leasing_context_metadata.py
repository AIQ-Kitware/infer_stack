"""The advertised context window.

The invariant this change implements: for every managed vLLM endpoint, the
effective ``max_model_len`` its process launches with equals the
``model_info.max_input_tokens`` the front door advertises about it, on every
route path (static compose, dynamic /model/new, KubeAI, upstream, and the
remembered registry). Clients that read ``/v1/model/info`` (such as pi) then
learn the real context window instead of guessing from the model family.
"""

from __future__ import annotations

import json

import pytest
import yaml

from infer_stack.leasing.compose import VLLM_DEFAULTS, vllm_service_dict
from infer_stack.leasing.gateway import Gateway, registry_routes, upstream_route
from infer_stack.leasing.launch import effective_max_model_len
from infer_stack.leasing.models import Deployment, DeploymentState
from infer_stack.leasing.routes import GatewayRoute

from test_leasing_dynamic_routing import RecordingGateway, dep


def _vllm_dep(gid, *, served, max_len=32768, t=0.0, extra_endpoints=()):
    """A vLLM deployment whose spec carries an explicit ``max_model_len``."""
    endpoints = {served: {'served_model_name': served, 'protocol': 'chat'}}
    for alias in extra_endpoints:
        endpoints[alias] = {'served_model_name': served, 'protocol': 'chat'}
    return Deployment(
        gid, 'ck-' + served, 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': 'org/model',
            'served_model_name': served,
            'runtime': {'tensor_parallel_size': 1, 'max_model_len': max_len},
            'reclaim': 'keep-warm',
        },
        endpoints, DeploymentState.LIVE, t, t,
    )


# -- A. GatewayRoute serialization ------------------------------------------


def test_entry_advertises_the_context_window():
    route = GatewayRoute('alpha', 'openai', 'a-model', 'http://host/v1',
                         route_id='isr-test', max_input_tokens=262144)
    assert route.entry()['model_info'] == {
        'id': 'isr-test',
        'max_input_tokens': 262144,
    }


def test_entry_without_route_id_still_advertises():
    route = GatewayRoute('alpha', 'openai', 'a-model', 'http://host/v1',
                         max_input_tokens=8192)
    assert route.entry()['model_info'] == {'max_input_tokens': 8192}


def test_entry_without_context_or_id_has_no_model_info():
    entry = GatewayRoute('x', 'openai', 'm', 'http://host/v1').entry()
    assert 'model_info' not in entry


def test_routes_with_different_windows_are_different_routes():
    same = GatewayRoute('alpha', 'openai', 'm', 'http://host/v1',
                        max_input_tokens=1)
    assert same == GatewayRoute('alpha', 'openai', 'm', 'http://host/v1',
                                max_input_tokens=1)
    assert same != GatewayRoute('alpha', 'openai', 'm', 'http://host/v1',
                                max_input_tokens=2)


# -- B. the static compose path ---------------------------------------------


def _catalog_with_window(tmp_path, *, window=262144):
    from infer_stack.leasing.catalog import Catalog

    return Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/m'}},
        'endpoints': {'big': {'engine': 'vllm', 'model': 'm',
                              'runtime': {'tensor_parallel_size': 1,
                                          'max_model_len': window}}},
    })


def test_static_superset_advertises_the_catalog_window(tmp_path):
    from test_leasing_compose import IMAGES, PORTS, STATE, render_compose

    # No deployment running: the standing superset route already advertises
    # the window the catalog says the endpoint launches with.
    rc = render_compose([], {}, images=IMAGES, ports=PORTS, state=STATE,
                        litellm=True, litellm_port=14042, aux_dir=tmp_path,
                        catalog=_catalog_with_window(tmp_path))
    (entry,) = yaml.safe_load(rc.litellm_config)['model_list']
    assert entry['model_name'] == 'big'
    assert entry['model_info']['max_input_tokens'] == 262144


def test_live_deployment_advertises_what_it_launches_with(tmp_path):
    from test_leasing_compose import IMAGES, PORTS, STATE, render_compose

    catalog = _catalog_with_window(tmp_path, window=262144)
    d = _vllm_dep('grp-a', served='big', max_len=262144)
    rc = render_compose([d], {d.id: [0]}, images=IMAGES, ports=PORTS,
                        state=STATE, litellm=True, litellm_port=14042,
                        aux_dir=tmp_path, catalog=catalog)
    (entry,) = yaml.safe_load(rc.litellm_config)['model_list']
    # The deployment route wins on the shared alias and carries the launched
    # window; here it agrees with the catalog's, so the bytes never move.
    assert entry['model_info']['max_input_tokens'] == 262144


def test_external_endpoints_advertise_nothing(tmp_path):
    from infer_stack.leasing.catalog import Catalog
    from test_leasing_compose import IMAGES, PORTS, STATE, render_compose

    catalog = Catalog.from_dict({
        'endpoints': {'box': {'external': {
            'api_base': 'http://box:9000/v1', 'model': 'Org/Big',
            'api_key_env': 'BOX_KEY'}}},
    })
    rc = render_compose([], {}, images=IMAGES, ports=PORTS, state=STATE,
                        litellm=True, litellm_port=14042, aux_dir=tmp_path,
                        catalog=catalog)
    (entry,) = yaml.safe_load(rc.litellm_config)['model_list']
    assert entry['model_name'] == 'box'
    assert 'model_info' not in entry


# -- C. one value, launch and advertisement ----------------------------------


def test_the_default_window_is_the_same_value_in_both():
    d = _vllm_dep('g', served='s', max_len=2048)
    assert effective_max_model_len(d.spec['runtime']) == 2048
    assert vllm_service_dict(d)['max_model_len'] == 2048

    # No max_model_len at all: launch and advertisement both fall back to
    # the same documented default.
    d2 = _vllm_dep('g2', served='s2', max_len=8192)
    d2.spec['runtime'] = {'tensor_parallel_size': 1}
    assert effective_max_model_len(d2.spec['runtime']) == VLLM_DEFAULTS['max_model_len']
    assert vllm_service_dict(d2)['max_model_len'] == VLLM_DEFAULTS['max_model_len']


# -- D. the dynamic /model/new path ------------------------------------------


def test_dynamic_model_new_body_carries_the_window(tmp_path):
    from test_leasing_dynamic_routing import make_backend, _route_id

    a = dep('grp-aaaaaa', served='smol', t=0)
    gw = RecordingGateway()
    be = make_backend(tmp_path, gw)
    be.converge([a], apply=True)

    rid = _route_id(a.id, 'smol')
    # The /model/new body (recorded verbatim by the fake gateway) ...
    assert gw.models[rid]['model_info']['max_input_tokens'] == 2048
    # ... and the desired-set file the render half wrote, so a later
    # reconcile judges drift on the window as well.
    routes = json.loads((tmp_path / 'litellm_routes.json').read_text())
    (body,) = [r for r in routes if r['model_info']['id'] == rid]
    assert body['model_info']['max_input_tokens'] == 2048


def test_window_change_is_semantic_drift(tmp_path):
    """Same route id, different window: reconcile must replace, not keep."""
    from test_leasing_dynamic_routing import make_backend, _route_id

    a = dep('grp-aaaaaa', served='smol', t=0)
    gw = RecordingGateway()
    be = make_backend(tmp_path, gw)
    be.converge([a], apply=True)
    rid = _route_id(a.id, 'smol')

    # Simulate an earlier definition with the same id and a different window.
    gw.models[rid]['model_info']['max_input_tokens'] = 999999
    before = len(gw.calls)
    assert be.gateway._reconcile_routes() is True
    assert gw.models[rid]['model_info']['max_input_tokens'] == 2048
    assert gw.calls[before:] == [('delete', rid), ('new', rid)]


def test_route_semantics_include_the_window():
    # _route_semantics judges a LiteLLM model-info entry (what the gateway
    # reports), not a route object.
    lo = GatewayRoute('a', 'openai', 'm', 'http://h/v1',
                      max_input_tokens=1).entry()
    hi = GatewayRoute('a', 'openai', 'm', 'http://h/v1',
                      max_input_tokens=2).entry()
    neither = GatewayRoute('a', 'openai', 'm', 'http://h/v1').entry()
    assert Gateway._route_semantics(lo) != Gateway._route_semantics(hi)
    assert Gateway._route_semantics(neither) == dict(
        Gateway._route_semantics(lo), max_input_tokens=None)
    assert Gateway._route_semantics(lo) == Gateway._route_semantics(lo)


# -- F/G. the remembered registry --------------------------------------------


def test_remembered_vllm_row_advertises_its_window():
    (route,) = registry_routes({'version': 1, 'entries': {
        'big': {'engine': 'vllm', 'served': 'big-model',
                'max_input_tokens': 262144}}})
    assert route.max_input_tokens == 262144
    assert route.entry()['model_info']['max_input_tokens'] == 262144


def test_remembered_upstream_row_carries_its_window():
    (route,) = registry_routes({'version': 1, 'entries': {
        'ext': {'engine': 'upstream', 'served': 'Org/Big',
                'api_base': 'http://box/v1', 'api_key_env': 'BOX_KEY',
                'max_input_tokens': 131072}}})
    assert route.max_input_tokens == 131072


@pytest.mark.parametrize('bad', ['262144', 262144.0 + 1e-9, -3, True, None])
def test_unsound_or_absent_window_is_not_advertised(bad):
    (route,) = registry_routes({'version': 1, 'entries': {
        'big': {'engine': 'vllm', 'served': 'big-model',
                'max_input_tokens': bad}}})
    assert route.max_input_tokens is None
    assert 'model_info' not in route.entry()


def test_legacy_registry_rows_load_without_the_field():
    (route,) = registry_routes({'version': 1, 'entries': {
        'big': {'engine': 'vllm', 'served': 'big-model'}}})
    assert route.max_input_tokens is None
    assert 'model_info' not in route.entry()


def test_converge_remembers_the_window_in_the_registry(tmp_path):
    from infer_stack.hardware import simulate_inventory
    from test_leasing_compose import IMAGES, PORTS, STATE
    from infer_stack.leasing.compose import ComposeBackend

    be = ComposeBackend(
        state_dir=tmp_path, inventory=simulate_inventory('1x96'),
        run=lambda args: '', images=IMAGES, ports=PORTS,
        state=STATE, litellm=True,
    )
    be.converge([_vllm_dep('grp-a', served='ad-hoc', max_len=131072)],
                apply=False)
    registry = json.loads((tmp_path / 'litellm_registry.json').read_text())
    assert registry['entries']['ad-hoc']['max_input_tokens'] == 131072


# -- H. one deployment, several aliases --------------------------------------


def test_two_aliases_of_one_deployment_advertise_the_same_window():
    from infer_stack.leasing.gateway import deployment_routes

    d = _vllm_dep('grp-a', served='shared-model', max_len=65536,
                  extra_endpoints=['alias-b'])
    routes = {r.alias: r for r in
              deployment_routes([d], {'grp-a': [0]})}
    assert {r.max_input_tokens for r in routes.values()} == {65536}
    assert routes['alias-b'].entry()['model_info'] == \
        routes['shared-model'].entry()['model_info'] == {
            'max_input_tokens': 65536,
        }


def test_upstream_route_advertises_only_when_told():
    r = upstream_route('dep-1', 'e1', 'm', 'http://h/v1',
                       max_input_tokens=123)
    assert r.max_input_tokens == 123
    assert upstream_route('dep-1', 'e1', 'm', 'http://h/v1').max_input_tokens is None


# -- I. the KubeAI path -------------------------------------------------------


def test_kubeai_static_routes_advertise_the_window(tmp_path):
    from infer_stack.leasing import Catalog
    from test_leasing_kubeai import make_front_door_backend, vllm

    be, _ = make_front_door_backend(tmp_path)
    be.catalog = Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/m'}},
        # Same runtime as the ``vllm`` fixture, as a live acquisition would
        # carry: the catalog route and the deployment serving it agree.
        'endpoints': {'tiny': {'engine': 'vllm', 'model': 'm',
                               'runtime': {'tensor_parallel_size': 1,
                                           'max_model_len': 4096}}},
    })
    be.converge([], apply=False)
    config = yaml.safe_load((be.gateway.state_dir / 'litellm_config.yaml')
                            .read_text())
    (entry,) = [e for e in config['model_list'] if e['model_name'] == 'tiny']
    assert entry['model_info']['max_input_tokens'] == 4096

    dep = vllm('grp-a', served='tiny')
    dep.served = {'tiny': {'served_model_name': 'tiny', 'protocol': 'chat'}}
    be.converge([dep], apply=False)
    config = yaml.safe_load((be.gateway.state_dir / 'litellm_config.yaml')
                            .read_text())
    (entry,) = [e for e in config['model_list'] if e['model_name'] == 'tiny']
    assert entry['model_info']['max_input_tokens'] == 4096


def test_kubeai_dynamic_routes_advertise_the_window(tmp_path):
    from test_leasing_kubeai import make_front_door_backend, vllm

    be, _ = make_front_door_backend(tmp_path, dynamic_routing=True)
    dep = vllm('grp-a', served='tiny')
    dep.served = {'tiny': {'served_model_name': 'tiny', 'protocol': 'chat'}}
    be.converge([dep], apply=False)

    routes = json.loads((be.gateway.state_dir / 'litellm_routes.json')
                        .read_text())
    (body,) = [r for r in routes if r['model_name'] == 'tiny']
    assert body['model_info']['max_input_tokens'] == 4096
