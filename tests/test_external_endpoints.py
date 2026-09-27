"""External endpoints: an alias fulfilled by a server infer-stack does not run.

Campaign 2 (docs/planning/external-endpoints.md). An external endpoint is a
catalog definition with an ``external:`` target; it has no lease, deployment
or model entry, and it is reached through the front door like any other.
"""

from __future__ import annotations

import pytest
import yaml

from infer_stack.leasing import Catalog
from infer_stack.leasing.catalog import CatalogError
from infer_stack.leasing.endpoints import ExternalTarget

REMOTE = {'external': {'api_base': 'http://box:8000/v1', 'model': 'Qwen/Qwen3-32B',
                       'api_key_env': 'REMOTE_QWEN_KEY'}, 'protocol': 'chat'}


def catalog(**endpoints):
    return {'models': {'m': {'source': 'hf://org/m'}},
            'endpoints': {'local': {'engine': 'vllm', 'model': 'm'}, **endpoints},
            'bundles': {'pair': ['local', 'remote']} if 'remote' in endpoints else {}}


# -- 1. parses and round-trips semantically --------------------------------------


def test_an_external_endpoint_parses_and_round_trips():
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    resolved = cat.resolve_endpoint('remote')
    assert not resolved.managed
    assert resolved.target == ExternalTarget('http://box:8000/v1', 'Qwen/Qwen3-32B',
                                             'REMOTE_QWEN_KEY')
    again = Catalog.from_dict(yaml.safe_load(yaml.safe_dump(cat.source)))
    assert again.resolve_endpoint('remote').semantic_key() == resolved.semantic_key()
    assert 'remote' not in cat.models          # no managed model artifact


# -- 2. invalid mixtures fail clearly ----------------------------------------------


@pytest.mark.parametrize('extra, needle', [
    ({'runtime': {'max_model_len': 8}}, "'runtime' describes a runtime"),
    ({'engine': 'vllm'}, "'engine' describes a runtime"),
    ({'reclaim': 'stop'}, "'reclaim' describes a runtime"),
    ({'placement': {'gpu_indices': [0]}}, "'placement' describes a runtime"),
    ({'host': 'h'}, "'host' describes a runtime"),
])
def test_managed_only_keys_beside_external_are_refused(extra, needle):
    with pytest.raises(CatalogError, match=needle):
        Catalog.from_dict(catalog(remote={**REMOTE, **extra}))


@pytest.mark.parametrize('external, needle', [
    ({'api_base': 'box:8000', 'model': 'm'}, 'must be an http'),
    ({'api_base': 'http://box/v1'}, 'external.model is required'),
    ({'api_base': 'http://box/v1', 'model': 'm', 'api_key_env': 'not a name'},
     'environment variable name'),
    ({'api_base': 'http://box/v1', 'model': 'm', 'api_key_env': 'LITELLM_MASTER_KEY'},
     "infer-stack's own secrets"),
    ({'api_base': 'http://box/v1', 'model': 'm', 'api_key': 'sk-literal'},
     'unknown external key'),
])
def test_bad_external_blocks_are_refused(external, needle):
    with pytest.raises(CatalogError, match=needle):
        Catalog.from_dict(catalog(remote={'external': external}))


# -- acquire stays lease-only ------------------------------------------------------


def test_asking_for_a_lease_on_an_external_endpoint_points_at_access():
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    with pytest.raises(CatalogError, match='does not require a lease.*infer-stack access remote'):
        cat.resolve_requests(['remote'])
    with pytest.raises(CatalogError, match='does not require a lease'):
        cat.resolve_requests(['pair'])       # a mixed bundle goes through access too
    assert [r.endpoint for r in cat.resolve_requests(['local'])] == ['local']


def test_the_cli_adds_an_external_endpoint_and_refuses_runtime_options(tmp_path, capsys):
    from infer_stack.cli.commands_catalog import EndpointAddCLI

    path = tmp_path / 'catalog.yaml'
    path.write_text('models: {}\nendpoints: {}\n')
    common = ['--catalog', str(path), 'qwen', '--external-api-base', 'http://box/v1',
              '--external-model', 'Q/Q']
    assert EndpointAddCLI.main(argv=[*common, '--external-api-key-env', 'QKEY']) == 0
    entry = yaml.safe_load(path.read_text())['endpoints']['qwen']
    assert entry == {'external': {'api_base': 'http://box/v1', 'model': 'Q/Q',
                                  'api_key_env': 'QKEY'}}
    for flags in (['--engine', 'vllm'], ['--gpu', '0'], ['--reclaim', 'stop']):
        with pytest.raises(SystemExit, match='runs nothing'):
            EndpointAddCLI.main(argv=[*common, '--force', *flags])


# -- routes: one GatewayRoute per alias, whoever runs it ----------------------------


def _compose(tmp_path, cat, **kw):
    from test_leasing_profile import backend

    return backend(tmp_path / 'state', catalog=cat, **kw)


def test_an_external_endpoint_routes_to_its_own_server_with_its_key_by_name(tmp_path):
    be = _compose(tmp_path, Catalog.from_dict(catalog(remote=REMOTE)))
    be.converge([], apply=False)
    config = yaml.safe_load((be.state_dir / 'litellm_config.yaml').read_text())
    routes = {e['model_name']: e['litellm_params'] for e in config['model_list']}
    assert routes['remote'] == {'model': 'openai/Qwen/Qwen3-32B',
                                'api_base': 'http://box:8000/v1',
                                'api_key': 'os.environ/REMOTE_QWEN_KEY'}
    assert routes['local']['api_key'] == 'EMPTY'     # managed entries unchanged
    assert not (be.state_dir / 'litellm_registry.json').exists()


def test_a_dynamic_external_route_has_a_stable_id_from_its_alias(tmp_path):
    import json

    from infer_stack.leasing.gateway import _route_id

    moved = {**REMOTE, 'external': {**REMOTE['external'], 'api_base': 'http://other:8000/v1'}}
    ids = []
    for target in (REMOTE, moved):
        be = _compose(tmp_path, Catalog.from_dict(catalog(remote=target)), dynamic_routing=True)
        be.converge([], apply=False)
        (route,) = json.loads((be.state_dir / 'litellm_routes.json').read_text())
        ids.append(route['model_info']['id'])
        assert route['model_info']['infer_stack_key_env'] == 'REMOTE_QWEN_KEY'
    # Redefining the server replaces one route under one id.
    assert ids[0] == ids[1] == _route_id('external', 'remote')


def test_route_semantics_cover_the_key_name():
    from infer_stack.leasing.gateway import Gateway
    from infer_stack.leasing.routes import GatewayRoute

    a = GatewayRoute('r', 'openai', 'm', 'http://b/v1', key_env='K1', route_id='isr-x').entry()
    b = GatewayRoute('r', 'openai', 'm', 'http://b/v1', key_env='K2', route_id='isr-x').entry()
    assert Gateway._route_semantics(a) != Gateway._route_semantics(b)


def test_kubeai_routes_an_external_endpoint_directly_not_through_the_cluster(tmp_path):
    from test_leasing_kubeai import UPSTREAM, make_front_door_backend

    be, _ = make_front_door_backend(tmp_path)
    be.catalog = Catalog.from_dict(catalog(remote=REMOTE))
    routes = {r.alias: r for r in be.routes([])}
    assert routes['remote'].api_base == 'http://box:8000/v1'
    assert routes['remote'].origin == 'external'
    assert routes['local'].api_base == UPSTREAM


# -- publication: routes seed publishes, routes prune unpublishes -------------------


def _ctl(tmp_path, cat, docker=None):
    from test_leasing_profile import controller

    return controller(tmp_path, catalog=cat, docker=docker)


def _published(ledger):
    return sorted(n for s in (ledger.profile() or {}).get('catalogs') or []
                  for n in (s.get('endpoints') or {}))


def test_seed_publishes_an_external_endpoint_and_prune_unpublishes_it(tmp_path):
    mine = Catalog.from_dict(catalog())
    ledger, ctl = _ctl(tmp_path, mine)
    other = Catalog.from_dict({'endpoints': {'remote': REMOTE}})
    plan = ctl.plan_route_seed([other])
    assert list(plan.added) == ['remote']
    ctl.commit_route_seed(plan)
    assert _published(ledger) == ['local', 'remote']
    view = {r.alias: r.origin for r in ctl.route_view()}
    assert view == {'local': 'catalog', 'remote': 'external'}

    plan = ctl.plan_route_prune()                 # the invocation knows only 'local'
    assert plan.dropped == ['remote']
    dropped, _ = ctl.commit_route_prune(plan)
    assert dropped == ['remote']
    assert _published(ledger) == ['local']
    assert [r.alias for r in ctl.route_view()] == ['local']


def test_prune_keeps_what_a_live_deployment_serves(tmp_path):
    from test_leasing_compose import FakeDocker

    docker = FakeDocker()
    both = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _ctl(tmp_path, both, docker)
    ctl.acquire('w', both.resolve_names(['local']), wait=False)
    _, other = _ctl(tmp_path, Catalog.from_dict({'endpoints': {'remote': REMOTE}}), docker)
    assert other.plan_route_prune().dropped == []   # local is live, remote is mine


def test_seed_never_redefines_an_endpoint_a_resident_deployment_runs(tmp_path):
    from infer_stack.leasing.profile import ProfileMismatch
    from test_leasing_compose import FakeDocker

    docker = FakeDocker()
    mine = Catalog.from_dict(catalog())
    _, ctl = _ctl(tmp_path, mine, docker)
    ctl.acquire('w', mine.resolve_names(['local']), wait=False)
    changed = catalog()
    changed['endpoints']['local']['runtime'] = {'max_model_len': 8}
    plan = ctl.plan_route_seed([Catalog.from_dict(changed)])
    assert list(plan.conflicted) == ['local']
    with pytest.raises(ProfileMismatch, match='resident deployment'):
        ctl.commit_route_seed(plan, replace=True)


def test_kubeai_catalog_routes_are_not_remembered_but_ad_hoc_models_are(tmp_path):
    import json

    from test_leasing_kubeai import make_front_door_backend, vllm

    be, _ = make_front_door_backend(tmp_path)
    be.catalog = Catalog.from_dict(catalog())
    local = vllm('grp-l', served='local')
    local.served = {'local': {'served_model_name': 'local', 'protocol': 'chat'}}
    adhoc = vllm('grp-a', served='Org/Adhoc')
    adhoc.served = {'adhoc': {'served_model_name': 'Org/Adhoc', 'protocol': 'chat'}}
    be.converge([local, adhoc], apply=False)
    registry = json.loads((tmp_path / 'gateway' / 'litellm_registry.json').read_text())
    assert set(registry['entries']) == {'adhoc'}
