"""External endpoints: an alias fulfilled by a server infer-stack does not run.

Campaign 2 (docs/planning/external-endpoints.md). An external endpoint is a
catalog definition with an ``external:`` target; it has no lease, deployment
or model entry, and it is reached through the front door like any other.
"""

from __future__ import annotations

import json

import pytest
import yaml

from infer_stack.env_utils import write_env_file
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


def _compose(tmp_path, cat, *, key=True, **kw):
    """A compose backend; ``key`` writes the remote's key into its .env."""
    from test_leasing_profile import backend

    be = backend(tmp_path / 'state', catalog=cat, **kw)
    if key:
        write_env_file(be.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    return be


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


def _ctl(tmp_path, cat, docker=None, *, key=True):
    """A controller on a compose stack; ``key`` writes the remote's key."""
    from test_leasing_profile import controller

    ledger, ctl = controller(tmp_path, catalog=cat, docker=docker)
    if key:
        write_env_file(ctl.backend.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    return ledger, ctl


def _published(ledger):
    return sorted(n for s in (ledger.profile() or {}).get('catalogs') or []
                  for n in (s.get('endpoints') or {}))


def test_seed_publishes_an_external_endpoint_and_prune_unpublishes_it(tmp_path):
    from infer_stack.env_utils import write_env_file
    from infer_stack.leasing.profile import ProfileMismatch

    mine = Catalog.from_dict(catalog())
    ledger, ctl = _ctl(tmp_path, mine, key=False)
    other = Catalog.from_dict({'endpoints': {'remote': REMOTE}})
    with pytest.raises(ProfileMismatch, match=r'infer-stack env REMOTE_QWEN_KEY='):
        ctl.plan_route_seed([other])              # no key: nothing to send
    write_env_file(ctl.backend.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    plan = ctl.plan_route_seed([other])
    assert list(plan.added) == ['remote']
    ctl.commit_route_seed(plan)
    # The seed publishes what it names; the invocation's own catalog is
    # published by the operations that publish it (acquire, access, apply).
    assert _published(ledger) == ['remote']
    view = {r.alias: r.origin for r in ctl.route_view()}
    assert view['remote'] == 'external'

    plan = ctl.plan_route_prune()                 # the invocation knows only 'local'
    assert plan.dropped == ['remote']
    dropped, _ = ctl.commit_route_prune(plan)
    assert dropped == ['remote']
    assert _published(ledger) == []
    assert 'remote' not in [r.alias for r in ctl.route_view()]


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


# -- item 33: an external route is standing desired state ---------------------------


def _dynamic(tmp_path, cat, http):
    from test_leasing_dynamic_routing import FakeDocker, IMAGES, PORTS, STATE

    from infer_stack.hardware import simulate_inventory
    from infer_stack.leasing.compose import ComposeBackend

    be = ComposeBackend(state_dir=tmp_path, inventory=simulate_inventory('4x80'),
                        run=FakeDocker(), http=http, images=IMAGES, ports=PORTS,
                        state=STATE, ui=False, dynamic_routing=True, catalog=cat)
    write_env_file(be.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    return be


def test_a_dynamic_external_route_survives_deployments_coming_and_going(tmp_path):
    from test_leasing_dynamic_routing import RecordingGateway, _managed, dep

    from infer_stack.leasing.gateway import _route_id

    gw = RecordingGateway()
    be = _dynamic(tmp_path, Catalog.from_dict({'endpoints': {'remote': REMOTE}}), gw)
    ext = _route_id('external', 'remote')
    a, b = dep('grp-aaaaaa', served='smol', t=0), dep('grp-bbbbbb', served='smol', t=1)
    be.converge([a, b], apply=True)                   # two dedicated deployments
    assert _managed(gw) == {ext, _route_id(a.id, 'smol'), _route_id(b.id, 'smol')}
    gw.calls.clear()
    be.converge([a], apply=True)                      # a dedicated one goes away
    be.converge([], apply=True)                       # zero deployments
    assert _managed(gw) == {ext}
    assert all(rid != ext for _, rid in gw.calls)     # never touched


def test_redefining_or_unpublishing_an_external_endpoint_changes_one_route(tmp_path):
    from test_leasing_dynamic_routing import RecordingGateway, _managed

    from infer_stack.leasing.gateway import _route_id

    gw = RecordingGateway()
    ext = _route_id('external', 'remote')
    _dynamic(tmp_path, Catalog.from_dict({'endpoints': {'remote': REMOTE}}), gw).converge(
        [], apply=True)
    moved = {'remote': {**REMOTE, 'external': {**REMOTE['external'],
                                               'api_base': 'http://other:8000/v1'}}}
    gw.calls.clear()
    _dynamic(tmp_path, Catalog.from_dict({'endpoints': moved}), gw).converge([], apply=True)
    assert gw.calls == [('delete', ext), ('new', ext)]            # replaced in place
    assert gw.models[ext]['litellm_params']['api_base'] == 'http://other:8000/v1'
    gw.calls.clear()
    _dynamic(tmp_path, None, gw).converge([], apply=True)          # unpublished
    assert gw.calls == [('delete', ext)] and _managed(gw) == set()


def test_release_and_gc_leave_an_external_route_published(tmp_path):
    from test_leasing_compose import FakeDocker

    both = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _ctl(tmp_path, both, FakeDocker())
    out = ctl.acquire('w', both.resolve_names(['local']), wait=False)
    ctl.release(out.lease.id)
    ctl.gc()
    assert {r.alias: r.origin for r in ctl.route_view()}['remote'] == 'external'
    assert _published(ledger) == ['local', 'remote']


# -- item 34: credentials by reference -----------------------------------------------


def test_the_host_gateway_gets_the_key_by_name_and_recreates_when_it_changes(tmp_path):
    from infer_stack.env_utils import write_env_file
    from infer_stack.leasing.compose import FINGERPRINT_LABEL

    be = _compose(tmp_path, Catalog.from_dict(catalog(remote=REMOTE)))
    write_env_file(be.gateway._env_path, {'REMOTE_QWEN_KEY': 'one'})
    be.converge([], apply=False)
    text = be.compose_file.read_text()
    litellm = yaml.safe_load(text)['services']['litellm']
    assert litellm['environment']['REMOTE_QWEN_KEY'] == '${REMOTE_QWEN_KEY}'
    assert 'one' not in text                                  # the name, never the value
    before = litellm['labels'][FINGERPRINT_LABEL]
    write_env_file(be.gateway._env_path, {'REMOTE_QWEN_KEY': 'two'})
    be.converge([], apply=False)
    after = yaml.safe_load(be.compose_file.read_text())['services']['litellm']
    assert after['labels'][FINGERPRINT_LABEL] != before       # recreated on apply


def test_managed_only_gateways_render_exactly_as_before(tmp_path):
    be = _compose(tmp_path, Catalog.from_dict(catalog()))
    be.converge([], apply=False)
    env = yaml.safe_load(be.compose_file.read_text())['services']['litellm']['environment']
    assert set(env) == {'LITELLM_MASTER_KEY'}


def test_a_render_refuses_a_route_whose_key_has_no_value(tmp_path):
    from infer_stack.leasing.gateway import MissingRouteKey

    be = _compose(tmp_path, Catalog.from_dict(catalog(remote=REMOTE)), key=False)
    with pytest.raises(MissingRouteKey, match='infer-stack env REMOTE_QWEN_KEY='):
        be.converge([], apply=False)
    assert not be.compose_file.exists()                  # nothing written


def test_the_cluster_gateway_puts_keys_in_its_secret_and_rolls_on_a_change(tmp_path):
    from infer_stack.backends.kubeai_gateway import ClusterGateway
    from infer_stack.env_utils import write_env_file
    from infer_stack.leasing.gateway import catalog_routes

    calls = []
    gw = ClusterGateway(state_dir=tmp_path, namespace='kubeai', url='http://gw:30442',
                        run=lambda argv: calls.append(argv) or '')
    gw.extra_routes = catalog_routes(Catalog.from_dict({'endpoints': {'remote': REMOTE}}))

    def rendered(value):
        write_env_file(gw.gateway._env_path, {'REMOTE_QWEN_KEY': value})
        text = gw._render_documents()[gw.manifests_file]
        assert value not in text                                  # never in a manifest
        deploy = [d for d in yaml.safe_load_all(text) if d['kind'] == 'Deployment'][0]
        return deploy['spec']['template']['metadata']['annotations']['infer-stack/key-hash']

    assert rendered('one') != rendered('two')
    gw.converge()
    secret = yaml.safe_load((tmp_path / 'gateway-secret.yaml').read_text())
    assert secret['stringData']['REMOTE_QWEN_KEY'] == 'two'
    assert 'LITELLM_MASTER_KEY' in secret['stringData']


def test_env_says_when_a_published_key_takes_effect(tmp_path, monkeypatch, capsys):
    from infer_stack.cli import commands_leasing as cl
    from infer_stack.leasing import Ledger, SqliteStore
    from infer_stack.leasing.profile import catalog_sources

    db = tmp_path / 'ledger.db'
    ledger = Ledger(SqliteStore(str(db)))
    ledger.set_profile({'backend': 'compose', 'catalogs': catalog_sources(
        Catalog.from_dict(catalog(remote=REMOTE)))})
    monkeypatch.setattr(cl, 'default_ledger_path', lambda: db)
    monkeypatch.setattr(cl, '_secret_env_path', lambda config=None: tmp_path / '.env')
    assert cl.EnvCLI.main(argv=['REMOTE_QWEN_KEY=sk-x']) == 0
    out = capsys.readouterr().out
    assert "remote send(s) $REMOTE_QWEN_KEY" in out and 'infer-stack apply' in out
    assert cl.EnvCLI.main(argv=['UNRELATED=1']) == 0
    assert 'apply' not in capsys.readouterr().out


# -- items 35-36: access above leasing ------------------------------------------------


def _keyed(tmp_path, cat, docker=None):
    """A controller on a compose stack whose managed .env holds the remote key."""
    from infer_stack.env_utils import write_env_file

    ledger, ctl = _ctl(tmp_path, cat, docker)
    write_env_file(ctl.backend.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    return ledger, ctl


def _demand(ledger):
    leases, deployments = ledger.status()
    return leases, deployments


def test_external_only_access_publishes_a_route_and_takes_no_lease(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)
    result = ctl.access('me', cat.resolve(['remote']))
    assert result.lease is None and result.external == ['remote']
    assert result.ready and result.front_door_ready is True
    assert result.request_names == {'remote': 'remote'}
    assert _demand(ledger) == ([], [])                        # zero ledger demand
    assert ledger.publication_pending() is None
    config = yaml.safe_load((ctl.backend.state_dir / 'litellm_config.yaml').read_text())
    assert 'remote' in [e['model_name'] for e in config['model_list']]


def test_a_mixed_bundle_leases_only_its_managed_member(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))       # bundle 'pair' = local + remote
    ledger, ctl = _keyed(tmp_path, cat)
    result = ctl.access('me', cat.resolve(['pair']))
    assert result.endpoints == ['local', 'remote'] and result.external == ['remote']
    assert result.lease is not None and result.lease.endpoints == ['local']
    leases, deployments = _demand(ledger)
    assert len(leases) == 1 and [sorted(d.served) for d in deployments] == [['local']]
    ctl.release(result.lease.id)                          # releases only the real lease
    assert {r.alias: r.origin for r in ctl.route_view()}['remote'] == 'external'


def test_access_to_an_external_endpoint_needs_the_front_door_and_its_key(tmp_path):
    from infer_stack.leasing.profile import ProfileMismatch

    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _ctl(tmp_path, cat, key=False)
    with pytest.raises(ProfileMismatch, match='infer-stack env REMOTE_QWEN_KEY='):
        ctl.access('me', cat.resolve(['remote']))
    assert ledger.profile() is None                    # refused before any commit
    _, lean = _ctl(tmp_path / 'lean', cat)
    lean.backend.litellm = False
    with pytest.raises(ProfileMismatch, match='LiteLLM front door'):
        lean.access('me', cat.resolve(['remote']))


def test_a_front_door_that_never_answers_releases_the_lease_access_took(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _keyed(tmp_path, cat)
    ctl.FRONT_DOOR_WAIT_S = 0.0
    ctl.backend.gateway_accepts = lambda key, wait=0.0: None
    result = ctl.access('me', cat.resolve(['pair']))
    assert result.front_door_ready is False and not result.ready
    leases, _ = ledger.status()
    assert [le.state for le in leases] == ['released']


def test_a_live_managed_endpoint_is_not_silently_redefined_external(tmp_path):
    from infer_stack.leasing.profile import ProfileMismatch
    from test_leasing_compose import FakeDocker

    docker = FakeDocker()
    managed = Catalog.from_dict(catalog())
    ledger, ctl = _ctl(tmp_path, managed, docker)
    held = ctl.acquire('me', managed.resolve_requests(['local']), wait=False)
    moved = Catalog.from_dict({'endpoints': {'local': REMOTE}})
    _, other = _keyed(tmp_path, moved, docker)
    with pytest.raises(ProfileMismatch, match="'local'"):
        other.access('me', moved.resolve(['local']))
    ctl.release(held.lease.id)
    ctl.evict()                                                  # the keep-warm one too
    result = other.access('me', moved.resolve(['local']))       # unpinned: it moves
    assert result.external == ['local'] and result.lease is None
    back = Catalog.from_dict(catalog())                          # and back to managed
    _, again = _ctl(tmp_path, back, docker)
    out = again.access('me', back.resolve(['local']))
    assert out.lease is not None and out.external == []


def test_external_routes_come_back_from_durable_state(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    _, ctl = _keyed(tmp_path, cat)
    ctl.access('me', cat.resolve(['remote']))
    _, fresh = _ctl(tmp_path, None)                  # a new process, no catalog
    routes = {r.alias: r for r in fresh.route_view()}
    assert routes['remote'].api_base == 'http://box:8000/v1'


def test_routes_list_shows_what_the_gateway_renders(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    _, ctl = _keyed(tmp_path, cat)
    ctl.access('me', cat.resolve(['remote']))
    config = yaml.safe_load((ctl.backend.state_dir / 'litellm_config.yaml').read_text())
    assert [r.entry() for r in ctl.route_view()] == config['model_list']


# -- the CLI: access, run, acquire ------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """The CLI on a compose stack (fake docker, shared between commands)."""
    from types import SimpleNamespace

    from infer_stack.cli import commands_leasing as cl
    from infer_stack.env_utils import write_env_file
    from test_leasing_compose import FakeDocker
    from test_leasing_profile import backend

    docker = FakeDocker()
    cat_path = tmp_path / 'catalog.yaml'
    cat_path.write_text(yaml.safe_dump(catalog(remote=REMOTE)))
    state = tmp_path / 'state'

    def make(config, *, interactive=False):
        be = backend(state, catalog=Catalog.load(cat_path), docker=docker)
        write_env_file(be.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
        return be

    monkeypatch.setattr(cl, '_make_backend', make)
    base = ['--ledger', str(tmp_path / 'ledger.db'), '--catalog', str(cat_path)]
    return SimpleNamespace(cl=cl, base=base, tmp=tmp_path)


def test_access_cli_writes_a_descriptor_without_a_fake_lease_id(cli, capsys):
    envf = cli.tmp / 'remote.env'
    assert cli.cl.AccessCLI.main(argv=['remote', *cli.base, '--env-file', str(envf)]) == 0
    out = capsys.readouterr().out
    assert 'no lease: every endpoint is external' in out
    text = envf.read_text()
    assert 'INFER_STACK_LEASE_ID' not in text
    assert 'INFER_STACK_ENDPOINT_REMOTE=remote' in text and 'OPENAI_BASE_URL=' in text
    with pytest.raises(SystemExit) as info:                     # nothing to release
        cli.cl.ReleaseCLI.main(argv=[*cli.base[:2], '--env-file', str(envf)])
    assert info.value.code == 0


def test_access_cli_mixed_bundle_and_release_by_env_file(cli, capsys):
    envf = cli.tmp / 'pair.env'
    assert cli.cl.AccessCLI.main(argv=['pair', *cli.base, '--env-file', str(envf),
                                       '--json']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['lease_id'] and data['external'] == ['remote']
    assert set(data['descriptor']['endpoints']) == {'local', 'remote'}
    assert f"INFER_STACK_LEASE_ID={data['lease_id']}" in envf.read_text()
    cli.cl.ReleaseCLI.main(argv=[*cli.base[:2], '--env-file', str(envf)])


def test_acquire_points_an_external_endpoint_at_access(cli):
    with pytest.raises(SystemExit, match=r'does not require a lease; use `infer-stack access remote`'):
        cli.cl.AcquireCLI.main(argv=['remote', *cli.base])


@pytest.mark.parametrize('names', ['remote', 'pair'])
def test_run_works_external_only_and_mixed(cli, names):
    import sys

    probe = ('import os,sys; sys.exit(0 if os.environ.get("INFER_STACK_ENDPOINT_REMOTE")'
             ' == "remote" and os.environ.get("OPENAI_BASE_URL") else 3)')
    rc = cli.cl.RunCLI.main(argv=[*cli.base, '--endpoint', names, '--', sys.executable, '-c', probe])
    assert rc == 0
    from infer_stack.leasing import Ledger, SqliteStore

    leases, _ = Ledger(SqliteStore(cli.base[1])).status()
    assert all(le.state == 'released' for le in leases)         # run released its lease
    assert len(leases) == (1 if names == 'pair' else 0)


# -- item 37: views --------------------------------------------------------------------


def test_status_lists_published_external_endpoints_apart_from_deployments(tmp_path, monkeypatch):
    from infer_stack.cli import commands_runtime as rt
    from infer_stack.leasing import Ledger, SqliteStore
    from infer_stack.leasing.profile import catalog_sources

    db = tmp_path / 'ledger.db'
    ledger = Ledger(SqliteStore(str(db)))
    ledger.set_profile({'backend': 'compose', 'catalogs': catalog_sources(
        Catalog.from_dict(catalog(remote=REMOTE)))})
    import infer_stack.leasing as leasing
    monkeypatch.setattr(leasing, 'default_ledger_path', lambda: db)
    status = rt._leasing_status()
    assert status['external'] == [('remote', 'Qwen/Qwen3-32B', 'http://box:8000/v1')]
    assert status['deployments'] == []                  # no pseudo-deployment
    assert 'remote' in '\n'.join(rt._external_lines(status['external']))


def test_two_catalogs_agree_on_an_external_alias_or_conflict():
    from infer_stack.leasing.profile import CatalogConflict, CatalogUnion

    a = {'endpoints': {'remote': REMOTE}}
    b = {'models': {'x': {'source': 'hf://o/x'}}, 'endpoints': {'remote': REMOTE}}
    union = CatalogUnion.from_sources([a, b])          # same alias, same target
    assert list(union.endpoints) == ['remote']
    other = {'endpoints': {'remote': {**REMOTE, 'external': {
        **REMOTE['external'], 'model': 'Other/Model'}}}}
    with pytest.raises(CatalogConflict, match="'remote'"):
        CatalogUnion.from_sources([a, other])


def test_moving_a_bundled_endpoint_to_external_keeps_the_union_valid():
    """Found by dev/external_e2e.sh phase 8: redefining a bundle member left the
    published copy of the bundle naming an endpoint its source no longer had."""
    from infer_stack.leasing.profile import CatalogUnion, adopt_catalog_sources

    before = catalog(remote=REMOTE)                          # pair = local + remote
    after = catalog(remote=REMOTE)
    after['endpoints']['local'] = {'external': dict(REMOTE['external'])}
    sources = adopt_catalog_sources([before], [after], pinned=set())
    union = CatalogUnion.from_sources(sources)
    assert not union.resolve_endpoint('local').managed
    assert union.bundles['pair'] == ['local', 'remote']


# -- item 43: access readiness includes route publication -----------------------------


class _LiteLLM:
    """A healthy LiteLLM (it accepts its key) whose admin API can refuse adds."""

    def __init__(self):
        from test_leasing_dynamic_routing import RecordingGateway

        self.admin = RecordingGateway()
        self.refuse = True

    def get(self, url, **kw):
        from test_leasing_dynamic_routing import FakeResp

        if url.endswith('/v1/models'):
            return FakeResp(200, {'data': []})
        return self.admin.get(url, **kw)

    def post(self, url, **kw):
        from test_leasing_dynamic_routing import FakeResp

        if self.refuse and url.endswith('/model/new'):
            return FakeResp(500, {'detail': 'db unavailable'})
        return self.admin.post(url, **kw)


def test_external_access_is_not_ready_until_its_route_is_published(tmp_path):
    from test_leasing_dynamic_routing import FakeTime, _timed_backend

    from infer_stack.leasing import Controller, Ledger, SqliteStore

    time, http = FakeTime(), _LiteLLM()
    be = _timed_backend(tmp_path / 'state', http, time)
    be.catalog = Catalog.from_dict(catalog(remote=REMOTE))
    write_env_file(be.gateway._env_path, {'REMOTE_QWEN_KEY': 'sk-remote'})
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    ctl = Controller(ledger, be, clock=time.clock, sleep=time.sleep)
    result = ctl.access('me', be.catalog.resolve(['remote']))
    assert result.ready is False and not result.published
    assert ledger.publication_pending() is not None            # stays pending
    http.refuse = False                                         # the DB is back
    again = ctl.access('me', be.catalog.resolve(['remote']))
    assert again.ready is True and ledger.publication_pending() is None


def test_unchecked_access_is_not_called_ready(tmp_path):
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    _, ctl = _keyed(tmp_path, cat)
    assert ctl.access('me', cat.resolve(['remote']), wait=False).ready is None



# -- item 44: a missing key refuses every publication --------------------------------


def _remove_key(ctl):
    path = ctl.backend.gateway._env_path
    path.write_text(''.join(line + '\n' for line in path.read_text().splitlines()
                            if not line.startswith('REMOTE_QWEN_KEY=')))


@pytest.mark.parametrize('operation', ['acquire', 'apply', 'seed', 'config publish'])
def test_an_unrelated_publication_refuses_when_a_published_key_is_gone(tmp_path, operation):
    """A working external endpoint, then its key removed: no later publication
    (whatever it is for) may recreate the gateway without it."""
    from infer_stack.leasing.gateway import MissingRouteKey

    cat = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, ctl = _ctl(tmp_path, cat)
    ctl.access('me', cat.resolve(['remote']))
    config = (ctl.backend.state_dir / 'litellm_config.yaml').read_text()
    _remove_key(ctl)
    before = ledger.status()
    profile = ledger.profile()
    with pytest.raises(MissingRouteKey):
        if operation == 'acquire':
            ctl.acquire('me', cat.resolve_requests(['local']), wait=False)
        elif operation == 'apply':
            ctl.apply_now()
        elif operation == 'seed':
            extra = Catalog.from_dict({'models': {'x': {'source': 'hf://o/x'}},
                                       'endpoints': {'x': {'engine': 'vllm', 'model': 'x'}}})
            ctl.commit_route_seed(ctl.plan_route_seed([extra]))
        else:
            ctl.publish_profile({**ledger.profile(), 'ui': False})
    assert (ctl.backend.state_dir / 'litellm_config.yaml').read_text() == config
    assert ledger.status() == before and ledger.profile() == profile
    if operation != 'apply':                             # apply retries a pending one
        assert ledger.publication_pending() is None


def test_the_cluster_gateway_refuses_a_route_whose_key_has_no_value(tmp_path):
    from infer_stack.backends.kubeai_gateway import ClusterGateway
    from infer_stack.leasing.gateway import MissingRouteKey, catalog_routes

    gw = ClusterGateway(state_dir=tmp_path, namespace='kubeai', url='http://gw:30442',
                        run=lambda argv: '')
    gw.extra_routes = catalog_routes(Catalog.from_dict({'endpoints': {'remote': REMOTE}}))
    with pytest.raises(MissingRouteKey):
        gw.converge()
    assert not gw.manifests_file.exists()
