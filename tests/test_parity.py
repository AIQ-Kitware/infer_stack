"""Every row of docs/backend-parity.md that reads *same*, run on both backends.

One harness builds the same stack twice: the compose backend over a fake
Docker, and the kubeai backend (with its LiteLLM gateway) over a fake
kubectl. Each test is one row of the matrix, named after it, and runs the
same scenario on both. A row whose test is here and passes on both is what
*same* means; a behaviour that only one backend can show belongs in that
backend's own tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
import yaml

from infer_stack.leasing import Catalog, Controller, LeaseState, Ledger, SqliteStore
from infer_stack.leasing.backend import PlacementError
from infer_stack.leasing.residency import DEPLOYMENT_LABEL, ResidencyUnknown

CATALOG = {
    'models': {
        'tiny': {'source': 'hf://org/tiny'},
        'big': {'source': 'hf://org/big'},
    },
    'endpoints': {
        'one': {'engine': 'vllm', 'model': 'tiny', 'reclaim': {'policy': 'stop'},
                'runtime': {'max_model_len': 2048, 'extra_args': ['--seed', '7']}},
        'wide': {'engine': 'vllm', 'model': 'big',
                 'runtime': {'tensor_parallel_size': 2, 'max_model_len': 4096}},
        'warm': {'engine': 'vllm', 'model': 'big', 'reclaim': {'policy': 'keep-warm'},
                 'runtime': {'max_model_len': 1024}},
    },
}
CAT = Catalog.from_dict(CATALOG)
UPSTREAM = 'http://10.43.0.9/openai/v1'
CRASH_LOG = ('INFO vLLM API server version 0.25.1\n'
             'ValueError: The checkpoint has model type `example_moe_v1` but '
             'Transformers does not recognize this architecture. If the model is '
             'custom, set trust_remote_code=True.\n')


@dataclass
class Stack:
    """One backend under test, and the handles a scenario needs."""

    kind: str
    ctl: Controller
    backend: Any
    runtime: Any                 # the fake Docker or the fake kubectl
    front: Any                   # the ComposeBackend holding the gateway

    def acquire(self, *names, dedicated: bool = False, **kw):
        from infer_stack.leasing import Sharing

        kw.setdefault('wait', False)
        sharing = Sharing.DEDICATED if dedicated else None
        requests = CAT.resolve_names(list(names), sharing=sharing)
        return self.ctl.acquire('alice', requests, **kw)

    def running(self) -> set[str]:
        return {gid for gid, found in self.backend.residency().by_deployment.items()
                if any(c.warm for c in found)}

    def crash(self, gid: str) -> None:
        """Make the deployment's engine crash-loop with CRASH_LOG."""
        if self.kind == 'compose':
            from infer_stack.leasing.compose import CRASH_LOOP_RESTARTS

            for c in self.runtime.containers.values():
                if c['labels'].get(DEPLOYMENT_LABEL) == gid:
                    c.update(state='restarting', restart_count=CRASH_LOOP_RESTARTS,
                             exit_code=1)
        else:
            self.runtime.crash[gid] = CRASH_LOG
            # A crash-looping pod answers nothing, as a restarting container.
            post = self.backend.http.post
            self.backend.http.post = lambda url, **kw: (
                self.backend.http._Resp(503, {'detail': 'upstream down'})
                if url.startswith('http://127.0.0.1:') else post(url, **kw))
        self.backend.deployment_logs = lambda deployment, tail=400: CRASH_LOG

    def break_runtime(self) -> None:
        """The runtime stops answering (docker or kubectl down)."""
        def broken(args, **kw):
            raise RuntimeError('connection refused')
        self.backend.run = broken

    def gateway_routes(self) -> dict[str, dict]:
        cfg = self.front.state_dir / 'litellm_config.yaml'
        doc = yaml.safe_load(cfg.read_text()) if cfg.exists() else {}
        return {e['model_name']: e['litellm_params'] for e in doc.get('model_list') or []}


def _compose(tmp_path, **gateway_kw) -> Stack:
    from infer_stack.hardware import simulate_inventory
    from infer_stack.leasing.compose import ComposeBackend
    from test_leasing_compose import IMAGES, PORTS, STATE, FakeDocker, FakeHttp

    docker = FakeDocker()
    backend = ComposeBackend(
        state_dir=tmp_path / 'compose', inventory=simulate_inventory('4x80'),
        run=docker, http=FakeHttp(tmp_path / 'compose'),
        images={**IMAGES, 'open-webui': 'owui:test', 'nginx': 'nginx:test',
                'postgres': 'pg:test'},
        ports=PORTS, state=STATE, litellm=True, catalog=CAT, **gateway_kw,
    )
    ctl = Controller(Ledger(SqliteStore(str(tmp_path / 'ledger.db'))), backend)
    return Stack('compose', ctl, backend, docker, backend)


class _Kubectl:
    """kubectl over applied Model CRs; each Model runs one pod."""

    def __init__(self):
        from test_leasing_kubeai import FakeKubectl

        self.inner = FakeKubectl()
        self.crash: dict[str, str] = {}

    @property
    def applied(self):
        return self.inner.applied

    def __call__(self, args: list[str]) -> str:
        from test_leasing_kubeai import _pod

        if len(args) > 4 and args[3] == 'get' and args[4] == 'pods':
            pods = []
            for name, doc in self.inner.applied.items():
                gid = doc['metadata']['labels']['infer-stack/deployment']
                if gid in self.crash:
                    pods.append(_pod(f'model-{name}-0', gid, restarts=3,
                                     state={'waiting': {'reason': 'CrashLoopBackOff'}},
                                     last={'exitCode': 1, 'reason': 'Error'}))
                else:
                    pods.append(_pod(f'model-{name}-0', gid, ready=True))
            return json.dumps({'items': pods})
        return self.inner(args)


def _kubeai(tmp_path, **gateway_kw) -> Stack:
    from infer_stack.backends.kubeai import KubeaiBackend
    from infer_stack.leasing.compose import ComposeBackend
    from test_leasing_compose import FakeDocker
    from test_leasing_kubeai import GatewayHttp

    kubectl = _Kubectl()
    gateway = ComposeBackend(
        state_dir=tmp_path / 'gateway', inventory={'gpu_count': 0, 'gpus': []},
        run=FakeDocker(), http=GatewayHttp(kubectl.inner, tmp_path / 'gateway'),
        project='infer-stack-gateway', litellm=True,
        images={'litellm': 'litellm:test', 'open-webui': 'owui:test',
                'nginx': 'nginx:test', 'postgres': 'pg:test'},
        **gateway_kw,
    )
    backend = KubeaiBackend(state_dir=tmp_path / 'kubeai', run=kubectl,
                            http=gateway.http, gateway=gateway,
                            gateway_upstream=UPSTREAM, default_resource_profile='gpu')
    backend.catalog = CAT
    ctl = Controller(Ledger(SqliteStore(str(tmp_path / 'ledger.db'))), backend)
    return Stack('kubeai', ctl, backend, kubectl, gateway)


BUILDERS = {'compose': _compose, 'kubeai': _kubeai}


@pytest.fixture(params=sorted(BUILDERS))
def make_stack(request, tmp_path):
    def make(**gateway_kw) -> Stack:
        return BUILDERS[request.param](tmp_path, **gateway_kw)
    return make


# -- client contract -----------------------------------------------------------


def test_one_base_url_the_managed_key_and_the_alias(make_stack):
    stack = make_stack()
    stack.acquire('one')
    info = stack.backend.access(['one'])
    assert info['base_url'].startswith('http://127.0.0.1:')
    assert info['api_key'] == stack.front.master_key()
    assert info['request_names'] == {'one': 'one'}


def test_env_file_has_the_same_keys(make_stack):
    from types import SimpleNamespace

    from infer_stack.cli.commands_leasing import _descriptor_for
    from infer_stack.leasing.envfile import descriptor_env

    stack = make_stack()
    out = stack.acquire('one')
    config = SimpleNamespace(base_url='unused', api_key_env='LITELLM_MASTER_KEY')
    env = descriptor_env(_descriptor_for(stack.ctl, out.lease, out.deployments, config))
    assert set(env) >= {'INFER_STACK_LEASE_ID', 'OPENAI_BASE_URL',
                        'INFER_STACK_ENDPOINT_ONE', 'INFER_STACK_MODELS'}
    assert env['INFER_STACK_ENDPOINT_ONE'] == 'one'     # the alias, on both


def test_readiness_is_a_generation_through_the_front_door(make_stack):
    stack = make_stack()
    out = stack.acquire('one', wait=True, timeout=10, interval=1)
    assert out.wait is not None and out.wait.ready


def test_secrets_rotate(make_stack):
    stack = make_stack()
    old = stack.front.master_key()
    stack.ctl.rotate_gateway_key()
    assert stack.front.master_key() != old


# -- lifecycle -------------------------------------------------------------------


def test_acquire_release_evict_gc_renew(make_stack):
    stack = make_stack()
    one = stack.acquire('one')
    warm = stack.acquire('warm')
    gid_one, gid_warm = one.deployments[0].id, warm.deployments[0].id
    assert {gid_one, gid_warm} <= stack.running()
    stack.ctl.renew(one.lease.id, ttl_seconds=3600)
    stack.ctl.release(one.lease.id)
    stack.ctl.release(warm.lease.id)
    assert gid_one not in stack.running()          # reclaim: stop
    assert gid_warm in stack.running()             # keep-warm stays resident
    stack.ctl.gc()
    assert gid_warm in stack.running()             # plain gc leaves it
    stack.ctl.evict(None)
    assert stack.running() == set()


def test_a_refused_acquire_writes_nothing(make_stack):
    stack = make_stack()
    # What each backend cannot serve: more GPUs than the host has, or a
    # Model with no resource profile.
    if stack.kind == 'compose':
        stack.backend.inventory = {'gpu_count': 1, 'gpus': [
            {'index': 0, 'memory_total_mib': 81920}]}
    else:
        stack.backend.default_resource_profile = None
    with pytest.raises(PlacementError):
        stack.acquire('wide')
    assert stack.ctl.ledger.status() == ([], [])
    assert stack.running() == set()


def test_config_publish_previews_then_commits(make_stack):
    stack = make_stack()
    profile = stack.backend.render_profile()
    stack.ctl.publish_profile(profile)
    assert stack.ctl.ledger.profile() == profile


def test_no_apply_stages_and_apply_brings_it_up(make_stack):
    stack = make_stack()
    out = stack.acquire('one', apply=False)
    gid = out.deployments[0].id
    assert gid not in stack.running()
    assert stack.ctl.ledger.publication_pending() is not None
    stack.ctl.apply_now()
    assert gid in stack.running()


def test_a_crash_looping_engine_fails_fast_with_its_error(make_stack):
    stack = make_stack()
    out = stack.acquire('one')
    gid = out.deployments[0].id
    stack.crash(gid)
    probe = stack.backend.probe_ready(stack.ctl.ledger.get_deployment(gid), 'one')
    assert probe.ready is False and probe.fatal is True
    assert 'trust_remote_code' in probe.detail


def test_strict_residency_and_lenient_observe(make_stack):
    stack = make_stack()
    stack.acquire('one')
    stack.break_runtime()
    with pytest.raises(ResidencyUnknown):
        stack.backend.residency()
    assert stack.backend.observe() == set()


# -- placement and the catalog -----------------------------------------------------


def test_gpu_count_from_tp_pp_dp(make_stack):
    stack = make_stack()
    out = stack.acquire('wide')
    gid = out.deployments[0].id
    if stack.kind == 'compose':
        assert len(stack.ctl.ledger.get_deployment(gid).assigned_gpus) == 2
    else:
        (doc,) = [d for d in stack.runtime.applied.values()
                  if d['metadata']['labels']['infer-stack/deployment'] == gid]
        assert doc['spec']['resourceProfile'] == 'gpu:2'


def _engine_args(stack, gid) -> list[str]:
    if stack.kind == 'compose':
        doc = yaml.safe_load(stack.backend.compose_file.read_text())
        (svc,) = [s for s in doc['services'].values()
                  if s.get('labels', {}).get(DEPLOYMENT_LABEL) == gid]
        return [str(a) for a in svc['command']]
    (doc,) = [d for d in stack.runtime.applied.values()
              if d['metadata']['labels']['infer-stack/deployment'] == gid]
    return [str(a) for a in doc['spec']['args']]


def test_runtime_flags_and_extra_args_reach_the_engine(make_stack):
    stack = make_stack()
    out = stack.acquire('one')
    args = ' '.join(_engine_args(stack, out.deployments[0].id))
    assert '--max-model-len=2048' in args
    assert '--seed 7' in args


def test_served_names_come_from_one_rule(make_stack):
    from infer_stack.leasing.naming import dns_slug

    stack = make_stack()
    out = stack.acquire('one')
    gid = out.deployments[0].id
    if stack.kind == 'compose':
        doc = yaml.safe_load(stack.backend.compose_file.read_text())
        names = [n for n, s in doc['services'].items()
                 if s.get('labels', {}).get(DEPLOYMENT_LABEL) == gid]
        assert names == [f'vllm-{dns_slug("one")}']
    else:
        assert f'{dns_slug("one")}' in stack.runtime.applied


# -- the gateway ------------------------------------------------------------------


def test_catalog_routes_exist_before_models_run_and_do_not_churn(make_stack):
    stack = make_stack()
    stack.ctl.apply_now()
    before = stack.gateway_routes()
    assert {'one', 'wide', 'warm'} <= set(before)
    stack.acquire('one')
    assert stack.gateway_routes() == before


def test_routes_list(make_stack, monkeypatch, capsys):
    from infer_stack.cli import commands_leasing

    stack = make_stack()
    stack.acquire('one')
    monkeypatch.setattr(commands_leasing, '_open_controller', lambda *a, **k: stack.ctl)
    capsys.readouterr()
    assert commands_leasing.RoutesListCLI.main(argv=['--json']) == 0
    rows = {r['name']: r for r in json.loads(capsys.readouterr().out)['routes']}
    assert rows['one']['live'] is True and rows['one']['upstream'] != '?'


def test_dynamic_routing_gives_dedicated_deployments_their_own_upstreams(make_stack):
    stack = make_stack(dynamic_routing=True)
    first = stack.acquire('one', dedicated=True, apply=False)
    second = stack.acquire('one', dedicated=True, apply=False)
    assert first.deployments[0].id != second.deployments[0].id
    routes = json.loads((stack.front.state_dir / 'litellm_routes.json').read_text())
    ones = [r for r in routes if r['model_name'] == 'one']
    assert len(ones) == 2                               # one alias, two upstreams
    assert len({(r['litellm_params']['model'], r['litellm_params']['api_base'])
                for r in ones}) == 2

def test_open_webui_and_the_reverse_proxy(make_stack):
    from infer_stack.leasing.gateway import NGINX_SERVICE, OPEN_WEBUI_SERVICE

    stack = make_stack(ui=True, reverse_proxy=True)
    stack.acquire('one', apply=False)
    doc = yaml.safe_load(stack.front.compose_file.read_text())
    assert {OPEN_WEBUI_SERVICE, NGINX_SERVICE} <= set(doc['services'])


def test_the_gateways_changes_are_approved_with_the_acquire(make_stack, monkeypatch):
    from infer_stack import diff_prompt as dp

    stack = make_stack()
    for b in {id(stack.backend): stack.backend, id(stack.front): stack.front}.values():
        b.assume_yes = False
    asked = []
    monkeypatch.setattr(dp, 'confirm_writes',
                        lambda changed, **kw: asked.append(sorted(p.name for p in changed))
                        or True)
    out = stack.acquire('one')
    assert out.lease.state == LeaseState.ACTIVE
    shown = [name for names in asked for name in names]
    assert 'litellm_config.yaml' in shown               # the gateway's change too
    # Each file was shown once, before the commit, never again at the render.
    assert len(shown) == len(set(shown))


# -- day-2 -------------------------------------------------------------------------


def test_instances_have_one_shape(make_stack):
    stack = make_stack()
    out = stack.acquire('one')
    gid = out.deployments[0].id
    instances = stack.backend.instances()
    engines = [i for i in instances if i.deployment_id == gid]
    assert len(engines) == 1 and engines[0].state == 'running'
    assert any(not i.is_engine for i in instances)     # the gateway


def test_stack_down_stops_everything(make_stack):
    stack = make_stack()
    stack.acquire('one')
    stack.backend.down()
    assert stack.running() == set()


def test_the_tui_reads_either_backend(make_stack):
    """Leases, deployments, the Instances tab and the engines log view."""
    pytest.importorskip('textual')
    import asyncio

    from infer_stack.tui import ENGINE_SERVICES, InferStackTUI

    stack = make_stack()
    out = stack.acquire('one')
    gid = out.deployments[0].id

    async def scenario():
        app = InferStackTUI(stack.ctl, CAT, interval=999, proc_factory=lambda s: None)
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()       # the startup refresh
            app._collapsed['docker'] = False
            data = app._collect()
            assert [le.id for le in data['leases']] == [out.lease.id]
            assert gid in {g.id for g in data['deployments']}
            # What _render keeps, without racing the app's own refresh.
            app._last_deployments = data['deployments']
            app._last_instances = data['instances']
            (engine,) = [r for r in app._ps_rows(data['instances'])
                         if r['serves'] == 'one']
            assert engine['status'] == 'running'
            target, _ = app._resolve_log_target(ENGINE_SERVICES)
            assert target == [engine['name']]

    asyncio.run(scenario())
