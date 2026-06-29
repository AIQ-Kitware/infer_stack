"""Dynamic LiteLLM routing (admin API + Postgres).

These cover the *proper* fix for the same-model ``--dedicated`` collision: in
static-superset mode every deployment of one served model collapses onto a
single ``vllm-<served>`` container (one GPU); dynamic routing gives each
deployment its **own** upstream and manages the gateway's routes live via the
admin API, so N dedicated deployments become N containers on N GPUs with no
gateway recreation (no blip).

Rendering is exercised with the pure ``render_compose`` helper; the
list/diff/add/delete route reconcile is exercised against a ``RecordingGateway``
fake that models LiteLLM's admin API in memory — no real docker, DB, or network.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import ComposeBackend, render_compose
from infer_stack.leasing.compose import (
    POSTGRES_SERVICE,
    ROUTE_ID_PREFIX,
    _litellm_routes,
    _route_id,
    vllm_service_name,
)
from infer_stack.leasing.models import Deployment, DeploymentState

STATE = {'hf_cache': '/cache/hf', 'ollama': '/cache/ollama'}
IMAGES = {
    'vllm': 'vllm/vllm-openai:test',
    'ollama': 'ollama/ollama:test',
    'litellm': 'ghcr.io/berriai/litellm:test',
    'postgres': 'postgres:test',
    'open_webui': 'ghcr.io/open-webui/open-webui:test',
}
PORTS = {'ollama': 11434, 'litellm': 14042, 'open_webui': 13000}


def dep(gid, *, served='smol', endpoint=None, hf='org/smol', tp=1,
        sharing='dedicated', t=0.0):
    """A vLLM deployment serving one ``endpoint`` (default == served name)."""
    endpoint = endpoint or served
    return Deployment(
        gid, 'ck-' + served, 'vllm', sharing, {},
        {
            'engine': 'vllm',
            'hf_model_id': hf,
            'served_model_name': served,
            'runtime': {'tensor_parallel_size': tp, 'max_model_len': 2048},
            'reclaim': 'keep-warm',
        },
        {endpoint: {'served_model_name': served, 'protocol': 'chat'}},
        DeploymentState.LIVE, t, t,
    )


class FakeDocker:
    """Stateful docker compose stand-in: ``up`` reflects the compose file."""

    def __init__(self):
        self.running: list[str] = []
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        compose_file = args[args.index('-f') + 1] if '-f' in args else None
        if 'up' in args:
            data = yaml.safe_load(Path(compose_file).read_text()) or {}
            self.running = sorted((data.get('services') or {}).keys())
            return ''
        if 'down' in args:
            self.running = []
            return ''
        if 'ps' in args:
            return json.dumps(
                [{'Service': s, 'State': 'running'} for s in self.running]
            )
        return ''


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class RecordingGateway:
    """In-memory fake of LiteLLM's admin API (model_info / model/new / delete)."""

    def __init__(self, seed_ids=()):
        # id -> entry. Seeded ids model models added out-of-band (e.g. the UI).
        self.models: dict[str, dict] = {
            i: {'model_name': i, 'model_info': {'id': i}} for i in seed_ids
        }
        self.calls: list[tuple[str, str]] = []

    def get(self, url, **kw):
        if url.endswith('/v1/model/info'):
            data = [
                {'model_name': e['model_name'], 'model_info': {'id': i}}
                for i, e in self.models.items()
            ]
            return FakeResp(200, {'data': data})
        return FakeResp(404, {'detail': 'not found'})

    def post(self, url, **kw):
        body = kw.get('json') or {}
        if url.endswith('/model/new'):
            rid = body['model_info']['id']
            self.models[rid] = body
            self.calls.append(('new', rid))
            return FakeResp(200, {'model_id': rid})
        if url.endswith('/model/delete'):
            rid = body['id']
            self.models.pop(rid, None)
            self.calls.append(('delete', rid))
            return FakeResp(200, {})
        return FakeResp(200, {'choices': [{'message': {'content': 'ok'}}]})


def _managed(gateway):
    return {i for i in gateway.models if i.startswith(ROUTE_ID_PREFIX)}


# -- the core fix: dedicated same-model -> distinct services ----------------


def test_static_mode_collapses_dedicated_same_model():
    """The bug this feature fixes: in static-superset mode two dedicated
    deployments of one served model collapse onto ONE compose service."""
    a, b = dep('grp-aaaaaa', served='smol', t=0), dep('grp-bbbbbb', served='smol', t=1)
    rc = render_compose(
        [a, b], {a.id: [0], b.id: [1]},
        images=IMAGES, ports=PORTS, state=STATE,  # dynamic_routing=False
    )
    vllm_services = [s for s in rc.compose['services'] if s.startswith('vllm-')]
    assert vllm_services == ['vllm-smol']  # collapsed -> only one container/GPU


def test_dynamic_mode_gives_dedicated_distinct_services():
    """The fix: dynamic routing gives each deployment its own upstream, so two
    dedicated deployments of one model become two containers on two GPUs."""
    a, b = dep('grp-aaaaaa', served='smol', t=0), dep('grp-bbbbbb', served='smol', t=1)
    rc = render_compose(
        [a, b], {a.id: [0], b.id: [1]},
        images=IMAGES, ports=PORTS, state=STATE, aux_dir='.', dynamic_routing=True,
    )
    vllm_services = sorted(s for s in rc.compose['services'] if s.startswith('vllm-'))
    assert len(vllm_services) == 2  # two distinct containers
    assert vllm_service_name(a, unique=True) in vllm_services
    assert vllm_service_name(b, unique=True) in vllm_services
    # distinct GPUs (one each), proving the placement actually spreads
    gpus = [
        rc.compose['services'][s]['deploy']['resources']['reservations'][
            'devices'
        ][0]['device_ids']
        for s in vllm_services
    ]
    assert sorted(gpus) == [['0'], ['1']]


# -- the route set ---------------------------------------------------------


def test_litellm_routes_deterministic_and_per_deployment():
    a, b = dep('grp-aaaaaa', served='smol', t=0), dep('grp-bbbbbb', served='smol', t=1)
    routes = _litellm_routes([a, b], {a.id: [0], b.id: [1]})
    # one route per (deployment, endpoint); same public model_name, distinct
    # upstreams and distinct deterministic ids.
    assert [r['model_name'] for r in routes] == ['smol', 'smol']
    bases = {r['litellm_params']['api_base'] for r in routes}
    assert len(bases) == 2
    ids = {r['model_info']['id'] for r in routes}
    assert ids == {_route_id(a.id, 'smol'), _route_id(b.id, 'smol')}
    assert all(i.startswith(ROUTE_ID_PREFIX) for i in ids)
    # deterministic: re-rendering yields byte-identical routes
    assert _litellm_routes([a, b], {a.id: [0], b.id: [1]}) == routes


# -- no blip + DB wiring ---------------------------------------------------


def test_dynamic_config_and_service_invariant_across_model_set(tmp_path):
    """No blip: the LiteLLM config + service spec (and Postgres) do NOT change
    when the live model set changes — routes move through the API/DB, not the
    file — so ``docker compose up`` never recreates the gateway."""
    a = dep('grp-aaaaaa', served='alpha')
    b = dep('grp-bbbbbb', served='beta')
    rc_a = render_compose(
        [a], {a.id: [0]}, images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, ui=True, aux_dir=tmp_path, dynamic_routing=True,
    )
    rc_b = render_compose(
        [b], {b.id: [0]}, images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, ui=True, aux_dir=tmp_path, dynamic_routing=True,
    )
    # static base config (empty model_list) -> identical regardless of live set
    assert rc_a.litellm_config == rc_b.litellm_config
    assert yaml.safe_load(rc_a.litellm_config)['model_list'] == []
    # gateway + DB service specs byte-identical across the two states (no blip)
    assert rc_a.compose['services']['litellm'] == rc_b.compose['services']['litellm']
    assert rc_a.compose['services'][POSTGRES_SERVICE] == (
        rc_b.compose['services'][POSTGRES_SERVICE]
    )


def test_dynamic_render_wires_postgres_and_db_env(tmp_path):
    a = dep('grp-aaaaaa', served='alpha')
    rc = render_compose(
        [a], {a.id: [0]}, images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, aux_dir=tmp_path, dynamic_routing=True,
    )
    assert POSTGRES_SERVICE in rc.compose['services']
    litellm = rc.compose['services']['litellm']
    env = litellm['environment']
    assert env['STORE_MODEL_IN_DB'] == 'True'
    assert env['DATABASE_URL'].startswith('postgresql://litellm:')
    assert f'@{POSTGRES_SERVICE}:5432/litellm' in env['DATABASE_URL']
    # gateway waits for the DB to be healthy, not on per-model upstreams
    assert litellm['depends_on'] == {POSTGRES_SERVICE: {'condition': 'service_healthy'}}
    # the desired route set is carried out-of-band for the apply half
    assert rc.litellm_routes == _litellm_routes([a], {a.id: [0]})


# -- the apply half: reconcile via the admin API ---------------------------


def make_backend(tmp_path, http, *, spec='4x80'):
    return ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory(spec),
        run=FakeDocker(),
        http=http,
        images=IMAGES, ports=PORTS, state=STATE,
        ui=False, dynamic_routing=True,
    )


def test_converge_writes_routes_file_and_reconciles(tmp_path):
    a, b = dep('grp-aaaaaa', served='smol', t=0), dep('grp-bbbbbb', served='smol', t=1)
    gw = RecordingGateway()
    be = make_backend(tmp_path, gw)

    be.converge([a, b], apply=True)

    # render half wrote the desired route set
    routes = json.loads((tmp_path / 'litellm_routes.json').read_text())
    assert {r['model_info']['id'] for r in routes} == {
        _route_id(a.id, 'smol'), _route_id(b.id, 'smol')
    }
    # apply half added exactly those routes to the gateway, via /model/new
    assert _managed(gw) == {_route_id(a.id, 'smol'), _route_id(b.id, 'smol')}
    assert sorted(c[0] for c in gw.calls) == ['new', 'new']


def test_reconcile_deletes_routes_no_longer_desired(tmp_path):
    a, b = dep('grp-aaaaaa', served='smol', t=0), dep('grp-bbbbbb', served='smol', t=1)
    gw = RecordingGateway()
    be = make_backend(tmp_path, gw)
    be.converge([a, b], apply=True)
    assert len(_managed(gw)) == 2

    # b released -> desired set shrinks to {a}; reconcile must DELETE b's route.
    gw.calls.clear()
    be.converge([a], apply=True)
    assert _managed(gw) == {_route_id(a.id, 'smol')}
    assert gw.calls == [('delete', _route_id(b.id, 'smol'))]


def test_reconcile_is_idempotent(tmp_path):
    a = dep('grp-aaaaaa', served='smol')
    gw = RecordingGateway()
    be = make_backend(tmp_path, gw)
    be.converge([a], apply=True)
    gw.calls.clear()
    # nothing changed -> a redundant apply diffs to the same set and is a no-op
    be.converge([a], apply=True)
    assert gw.calls == []
    assert _managed(gw) == {_route_id(a.id, 'smol')}


def test_reconcile_leaves_unmanaged_models_alone(tmp_path):
    """A model added by hand (no isr- id) is never deleted by reconcile."""
    a = dep('grp-aaaaaa', served='smol')
    gw = RecordingGateway(seed_ids=['hand-added-by-ui'])
    be = make_backend(tmp_path, gw)
    be.converge([a], apply=True)
    # our route added; the foreign model untouched
    assert _route_id(a.id, 'smol') in gw.models
    assert 'hand-added-by-ui' in gw.models
    assert ('delete', 'hand-added-by-ui') not in gw.calls


def test_db_password_is_persisted_and_reused(tmp_path):
    be = make_backend(tmp_path, RecordingGateway())
    pw1 = be.db_password()
    assert pw1
    # rewritten only if missing -> stable across calls and a fresh backend
    assert be.db_password() == pw1
    be2 = make_backend(tmp_path, RecordingGateway())
    assert be2.db_password() == pw1


# -- no blip on the UPSTREAMS too (the readiness-killing churn) -------------


def test_dynamic_upstreams_have_no_host_ports_and_survive_set_change(tmp_path):
    """The blip that broke readiness mid-request: each vLLM upstream published a
    host port assigned by *position* in the live set (BASE + i), so adding or
    removing any deployment renumbered the survivors' ports — which changed their
    service specs and made ``docker compose up -d`` recreate unrelated, in-flight
    containers. Behind the gateway an upstream is internal (reached by
    compose-network DNS), so it publishes NO host port and each survivor's spec
    is byte-identical when the set changes -> nothing to recreate."""
    a = dep('grp-aaaaaa', served='smol', t=0)
    b = dep('grp-bbbbbb', served='smol', t=1)
    c = dep('grp-cccccc', served='smol', t=2)
    full = render_compose(
        [a, b, c], {a.id: [0], b.id: [1], c.id: [2]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, aux_dir=tmp_path, dynamic_routing=True,
    )
    # upstreams behind the gateway publish no host port (only the gateway does)
    for name, svc in full.compose['services'].items():
        if name.startswith('vllm-'):
            assert 'ports' not in svc, f'{name} should not publish a host port'
    # drop `a`; b and c keep their GPUs (sticky placement)
    shrunk = render_compose(
        [b, c], {b.id: [1], c.id: [2]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, aux_dir=tmp_path, dynamic_routing=True,
    )
    for d in (b, c):
        name = vllm_service_name(d, unique=True)
        assert shrunk.compose['services'][name] == full.compose['services'][name], (
            f'{name} spec changed when an unrelated deployment was removed -> '
            'docker compose would recreate it (a blip)'
        )


def test_no_gateway_still_publishes_upstream_host_port(tmp_path):
    """Without a gateway there is no front door, so the upstream MUST publish a
    host port (the readiness probe hits it directly). Guards against the no-blip
    change above accidentally dropping ports in the no-gateway path too."""
    a = dep('grp-aaaaaa', served='smol', t=0)
    rc = render_compose(
        [a], {a.id: [0]}, images=IMAGES, ports=PORTS, state=STATE,
        litellm=False,
    )
    svc = rc.compose['services'][vllm_service_name(a)]
    assert svc['ports'] == ['18000:8000']


def test_reconcile_delete_tolerates_already_gone(tmp_path, monkeypatch):
    """A shared gateway lets another converge delete a route between this one's
    list and delete; LiteLLM then answers the delete with 'not found in db'.
    That IS the desired end-state (route gone), so it must be swallowed, not
    warned. A real failure (or a delete without the flag) still warns."""
    import infer_stack._log as _log

    class NotFoundOnDelete(RecordingGateway):
        def post(self, url, **kw):
            if url.endswith('/model/delete'):
                rid = (kw.get('json') or {}).get('id')
                self.calls.append(('delete', rid))
                return FakeResp(
                    400, {'error': f'Model with id={rid} not found in db'}
                )
            return super().post(url, **kw)

    be = make_backend(tmp_path, NotFoundOnDelete())
    warnings: list = []
    monkeypatch.setattr(
        _log.logger, 'warning', lambda *a, **k: warnings.append((a, k))
    )
    be._post_route('/model/delete', {'id': 'isr-x'}, 'isr-x', ok_if_missing=True)
    assert warnings == []  # already gone -> no warning
    be._post_route('/model/delete', {'id': 'isr-x'}, 'isr-x')
    assert warnings  # same response without the flag -> warns
