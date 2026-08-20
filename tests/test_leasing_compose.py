"""Tests for the Compose backend, driven by a stateful fake docker seam.

These exercise render + converge + observe + teardown logic without real docker
or GPUs. The real docker/GPU path is validated separately on a GPU host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import re
from pathlib import Path

import pytest
import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import (
    Catalog,
    ComposeBackend,
    Controller,
    Ledger,
    SqliteStore,
    plan_placement,
    render_compose,
)
from infer_stack.leasing.compose import vllm_service_name
from infer_stack.leasing.models import Deployment, DeploymentState

STATE = {'hf_cache': '/cache/hf', 'ollama': '/cache/ollama'}
IMAGES = {
    'vllm': 'vllm/vllm-openai:test',
    'ollama': 'ollama/ollama:test',
    'litellm': 'ghcr.io/berriai/litellm:test',
}
PORTS = {'ollama': 11434}


def vllm(gid, *, hf='org/model', served=None, tp=1, max_len=32768, reclaim='keep-warm',
         protocol='chat', t=0.0):
    # endpoint (public alias) is the deployment id; served_model_name is the upstream
    served_name = served or gid
    return Deployment(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': hf,
            'served_model_name': served_name,
            'runtime': {'tensor_parallel_size': tp, 'max_model_len': max_len},
            'reclaim': reclaim,
        },
        {gid: {'served_model_name': served_name, 'protocol': protocol}},
        DeploymentState.LIVE, t, t,
    )


def ollama(gid, *, tag='m:1b', t=0.0):
    return Deployment(
        gid, 'ck-' + gid, 'ollama', 'shared-compatible', {},
        {'engine': 'ollama', 'gpu_indices': [], 'settings': {'keep_alive': '2m'}},
        {gid: {'model': tag}}, DeploymentState.LIVE, t, t,
    )


class FakeDocker:
    """Stateful docker compose stand-in: `up` reflects the compose file."""

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


class FakeHttp:
    """Fake requests-like client whose /models lists the LiteLLM config aliases."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)

    def _model_names(self):
        cfg = self.state_dir / 'litellm_config.yaml'
        if not cfg.exists():
            return []
        data = yaml.safe_load(cfg.read_text()) or {}
        return [e['model_name'] for e in data.get('model_list', [])]

    def get(self, url, **kw):
        if url.endswith('/models'):
            data = [{'id': n} for n in self._model_names()]
            return FakeResp(200, {'data': data})
        return FakeResp(404, {'detail': 'not found'})

    def post(self, url, **kw):
        return FakeResp(200, {'choices': [{'message': {'content': 'ok'}}]})


# -- pure render -----------------------------------------------------------


def test_render_vllm_service():
    deployment = vllm('grp-a', hf='Qwen/Q', served='qwen', tp=2)
    rc = render_compose(
        [deployment],
        {'grp-a': [0, 1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    # service name is deterministic from the served model (no deployment-id
    # suffix) so the gateway's static route table addresses it stably
    name = vllm_service_name(deployment)
    assert name == 'vllm-qwen'
    svc = rc.compose['services'][name]
    assert svc['image'] == 'vllm/vllm-openai:test'
    assert svc['command'][0] == 'Qwen/Q'
    assert '--tensor-parallel-size=2' in svc['command']
    assert '--served-model-name=qwen' in svc['command']
    assert svc['ports'] == ['18000:8000']
    devs = svc['deploy']['resources']['reservations']['devices'][0]
    assert devs['device_ids'] == ['0', '1']
    # Compose schema: capabilities is a list of *strings* (not [["gpu"]]).
    assert devs['capabilities'] == ['gpu']
    assert svc['labels']['infer-stack.deployment'] == 'grp-a'
    assert rc.services == {name: 'grp-a'}


def test_render_vllm_serving_knobs_reach_the_command():
    """Regression: revision/quantization/dtype/pp/chat_template/trust_remote_code
    /image were hashed into the compat key (distinct deployments) but silently
    dropped from the rendered command — `quantization: awq` served the
    full-precision `main` revision and OOM'd with no warning."""
    from infer_stack.leasing.models import DeploymentState

    deployment = Deployment(
        'grp-k', 'ck-k', 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': 'Qwen/Q-AWQ',
            'served_model_name': 'qwen-awq',
            'revision': 'v1.2',
            'quantization': 'awq',
            'dtype': 'half',
            'runtime': {
                'tensor_parallel_size': 1,
                'pipeline_parallel_size': 2,
                'chat_template': '/templates/chatml.jinja',
                'trust_remote_code': True,
                'image': 'vllm/vllm-openai:nightly',
            },
            'reclaim': 'keep-warm',
        },
        {'grp-k': {'served_model_name': 'qwen-awq'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )
    rc = render_compose(
        [deployment], {'grp-k': [0, 1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    svc = rc.compose['services'][vllm_service_name(deployment)]
    cmd = svc['command']
    assert '--revision=v1.2' in cmd
    assert '--quantization=awq' in cmd
    assert '--dtype=half' in cmd
    assert '--pipeline-parallel-size=2' in cmd
    assert '--chat-template=/templates/chatml.jinja' in cmd
    assert '--trust-remote-code' in cmd
    assert svc['image'] == 'vllm/vllm-openai:nightly'


def test_render_vllm_omits_unset_serving_knobs():
    """A knob-less service keeps vLLM's own defaults: no --revision /
    --quantization / --dtype / --chat-template / --trust-remote-code emitted."""
    deployment = vllm('grp-a', hf='Qwen/Q', served='qwen')
    rc = render_compose(
        [deployment], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    svc = rc.compose['services'][vllm_service_name(deployment)]
    cmd = svc['command']
    assert not any(a.startswith('--revision') for a in cmd)
    assert not any(a.startswith('--quantization') for a in cmd)
    assert not any(a.startswith('--dtype') for a in cmd)
    assert not any(a.startswith('--chat-template') for a in cmd)
    assert '--trust-remote-code' not in cmd
    assert '--pipeline-parallel-size=1' in cmd
    # attention backend is a vLLM env var; unset => not in the environment.
    assert 'VLLM_ATTENTION_BACKEND' not in svc['environment']


def test_render_vllm_attention_backend_reaches_environment_not_command():
    """attention_backend is a vLLM env var (VLLM_ATTENTION_BACKEND), not a CLI
    flag — it must land in the service environment, never in the command."""
    from infer_stack.leasing.models import DeploymentState

    deployment = Deployment(
        'grp-attn', 'ck-attn', 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': 'org/model',
            'served_model_name': 'm-sdpa',
            'runtime': {'attention_backend': 'TORCH_SDPA'},
            'reclaim': 'keep-warm',
        },
        {'grp-attn': {'served_model_name': 'm-sdpa'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )
    rc = render_compose(
        [deployment], {'grp-attn': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    svc = rc.compose['services'][vllm_service_name(deployment)]
    assert svc['environment']['VLLM_ATTENTION_BACKEND'] == 'TORCH_SDPA'
    assert not any('attention' in a.lower() for a in svc['command'])


def test_render_reports_service_name_collisions():
    """Regression: two live deployments sharing a served name rendered to ONE
    compose service (dict overwrite) — the earlier deployment's container never
    existed and its probes failed until lease timeout with no error anywhere.
    The oldest deployment keeps the name; later ones are reported."""
    a = vllm('grp-old', served='qwen', t=0)
    b = vllm('grp-new', served='qwen', t=1)     # same served name, later
    rc = render_compose(
        [a, b], {'grp-old': [0], 'grp-new': [1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert rc.services == {'vllm-qwen': 'grp-old'}   # oldest wins
    assert rc.unrenderable == {'grp-new'}
    assert any(
        e.startswith('grp-new') and 'grp-old' in e for e in rc.errors
    )


def test_render_dynamic_routing_avoids_same_served_collision():
    """Dynamic routing appends the deployment-id tail, so same-served-name
    dedicated deployments get distinct services and nothing collides."""
    a = vllm('grp-old', served='qwen', t=0)
    b = vllm('grp-new', served='qwen', t=1)
    rc = render_compose(
        [a, b], {'grp-old': [0], 'grp-new': [1]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, dynamic_routing=True,
    )
    assert rc.unrenderable == set()
    assert rc.errors == []
    assert sorted(rc.services.values()) == ['grp-new', 'grp-old']


def test_converge_surfaces_collision_as_unplaced(tmp_path):
    """The backend folds render collisions into last_unplaced/last_errors, so
    the controller treats the colliding acquire like a placement failure
    (fail loudly + roll back) instead of leasing a container that never runs."""
    be = make_backend(tmp_path)
    be.converge([
        vllm('grp-old', served='qwen', t=0),
        vllm('grp-new', served='qwen', t=1),
    ])
    assert 'grp-new' in be.last_unplaced
    assert any('grp-new' in e for e in be.last_errors)
    assert be.observe() == {'grp-old'}   # only the survivor is up


def test_render_two_vllm_distinct_ports():
    a, b = vllm('a', served='aa', t=0), vllm('b', served='bb', t=1)
    rc = render_compose(
        [a, b],
        {'a': [0], 'b': [1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert rc.compose['services'][vllm_service_name(a)]['ports'] == ['18000:8000']
    assert rc.compose['services'][vllm_service_name(b)]['ports'] == ['18001:8000']


def test_render_ollama_service():
    rc = render_compose(
        [ollama('daemon')], {'daemon': [1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    svc = rc.compose['services']['ollama-daemon']
    assert svc['image'] == 'ollama/ollama:test'
    assert svc['environment']['OLLAMA_KEEP_ALIVE'] == '2m'
    assert svc['ports'] == ['11434:11434']
    # GPU pinning is via the device reservation, which renumbers the reserved
    # GPU to 0 inside the container. Setting CUDA_VISIBLE_DEVICES to the *host*
    # index (1) would point at a non-existent in-container device -> CPU
    # fallback. So it must pin by device_ids and NOT set the host-index env var.
    assert svc['deploy']['resources']['reservations']['devices'][0][
        'device_ids'
    ] == ['1']
    assert 'CUDA_VISIBLE_DEVICES' not in svc['environment']


def test_render_skips_unplaced_deployments():
    a, b = vllm('a', served='aa'), vllm('b', served='bb')
    rc = render_compose(
        [a, b], {'a': [0]},  # b not placed
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert set(rc.compose['services']) == {vllm_service_name(a)}


def test_render_bakes_project_name():
    # a top-level `name:` makes a plain `docker compose -f file up` (no -p) land
    # in the same project infer-stack uses, so the manual path == apply.
    rc = render_compose(
        [vllm('grp-a', served='qwen')], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert rc.compose['name'] == 'infer-stack'   # default project
    rc2 = render_compose(
        [vllm('grp-a', served='qwen')], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE, project='custom-proj',
    )
    assert rc2.compose['name'] == 'custom-proj'


def test_vllm_service_name_is_model_led_and_dns_safe():
    import re as _re

    g = vllm('grp-098ed1', hf='Qwen/Qwen2.5-0.5B-Instruct', served='Qwen2.5-0.5B')
    name = vllm_service_name(g)
    # deterministic from the served model (slugified), with NO deployment-id
    # suffix, so a catalog endpoint and its live deployment resolve to the same
    # on-network DNS host (what makes the gateway's static route table work)
    assert name == 'vllm-qwen2-5-0-5b'
    # usable as a compose service *and* an on-network DNS host (LiteLLM routes
    # to it), so it must be lowercase [a-z0-9-]
    assert _re.fullmatch(r'[a-z0-9-]+', name)


# -- ComposeBackend converge/observe --------------------------------------


def make_backend(tmp_path, *, spec='4x80', **kw):
    return ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory(spec),
        run=FakeDocker(),
        http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS, state=STATE,
        **kw,
    )


def test_converge_writes_and_observes(tmp_path):
    be = make_backend(tmp_path)
    be.converge([vllm('grp-a', tp=2)])
    assert be.compose_file.exists()
    sidecar = json.loads((tmp_path / 'leasing-compose-state.json').read_text())
    assert sidecar['assignments'] == {'grp-a': [0, 1]}
    assert be.observe() == {'grp-a'}


def test_converge_is_idempotent_and_pins(tmp_path):
    be = make_backend(tmp_path)
    be.converge([vllm('a', tp=2, t=0)])
    be.converge([vllm('a', tp=2, t=0), vllm('b', t=1)])  # add b, keep a put
    sidecar = json.loads((tmp_path / 'leasing-compose-state.json').read_text())
    assert sidecar['assignments']['a'] == [0, 1]   # unchanged
    assert sidecar['assignments']['b'] == [2]
    assert be.observe() == {'a', 'b'}


def test_converge_removes_dropped_deployment(tmp_path):
    be = make_backend(tmp_path)
    be.converge([vllm('a', t=0), vllm('b', t=1)])
    assert be.observe() == {'a', 'b'}
    be.converge([vllm('a', t=0)])                  # b gone -> --remove-orphans
    assert be.observe() == {'a'}


def test_converge_to_empty_keeps_the_front_door(tmp_path):
    """Releasing the last model leaves the gateway + UI up (an empty front door).

    The front door is a standing entry point, not a per-model service: zero
    models -> empty model_list, but litellm/open-webui keep running. Only
    `stack down` takes everything off.
    """
    be = make_backend(tmp_path)            # litellm + ui on by default
    be.converge([vllm('a', t=0)])
    assert be.observe() == {'a'}
    fake = be.run
    fake.calls.clear()
    be.converge([])                        # last model released
    verbs = [c[c.index('-f') + 2] if '-f' in c else c[0] for c in fake.calls]
    assert 'up' in verbs and 'down' not in verbs   # front door stays up
    # no model deployments running, but the compose project still has the gateway/UI
    assert be.observe() == set()
    compose = yaml.safe_load(be.compose_file.read_text())
    assert set(compose['services']) == {'litellm', 'open-webui'}


def test_converge_to_empty_downs_when_gateway_off(tmp_path):
    """With no gateway (litellm=False), an empty desired set has nothing to run,
    so converge `down`s rather than `up`-ing a services-less file."""
    be = make_backend(tmp_path, litellm=False)
    be.converge([vllm('a', t=0)])
    fake = be.run
    fake.calls.clear()
    be.converge([])
    verbs = [c[c.index('-f') + 2] if '-f' in c else c[0] for c in fake.calls]
    assert 'down' in verbs and 'up' not in verbs


def test_render_reverse_proxy(tmp_path):
    images = {**IMAGES, 'open_webui': 'owui:test', 'nginx': 'nginx:test'}
    rc = render_compose(
        [vllm('grp-a', served='chat')], {'grp-a': [0]},
        images=images, ports=PORTS, state=STATE,
        litellm=True, ui=True, reverse_proxy=True, reverse_proxy_port=8080,
        aux_dir=tmp_path,
    )
    svc = rc.compose['services']['reverse-proxy']
    assert svc['image'] == 'nginx:test'
    assert svc['ports'] == ['8080:80']
    assert svc['depends_on'] == ['litellm', 'open-webui']
    # generated conf path-routes /v1 -> gateway, / -> UI
    conf = rc.nginx_config
    assert 'location /v1/' in conf and 'litellm:4000' in conf
    assert 'location /' in conf and 'open-webui:8080' in conf
    assert 'connection_upgrade' in conf            # websockets for the UI


def test_render_reverse_proxy_byo_config(tmp_path):
    images = {**IMAGES, 'open_webui': 'owui:test', 'nginx': 'nginx:test'}
    rc = render_compose(
        [vllm('grp-a', served='chat')], {'grp-a': [0]},
        images=images, ports=PORTS, state=STATE,
        litellm=True, ui=True, reverse_proxy=True,
        reverse_proxy_config='/etc/my/nginx.conf', aux_dir=tmp_path,
    )
    svc = rc.compose['services']['reverse-proxy']
    # BYO: mount the operator's file verbatim, generate nothing
    assert svc['volumes'] == ['/etc/my/nginx.conf:/etc/nginx/conf.d/default.conf:ro']
    assert rc.nginx_config is None


def test_reverse_proxy_needs_the_gateway(tmp_path):
    images = {**IMAGES, 'nginx': 'nginx:test'}
    rc = render_compose(
        [vllm('grp-a', served='chat')], {'grp-a': [0]},
        images=images, ports=PORTS, state=STATE,
        litellm=False, reverse_proxy=True, aux_dir=tmp_path,
    )
    assert 'reverse-proxy' not in rc.compose['services']   # no gateway -> no proxy


def test_converge_writes_nginx_conf_and_access_reports_proxy(tmp_path):
    be = make_backend(tmp_path, reverse_proxy=True, reverse_proxy_port=8080)
    be.converge([vllm('a', served='aa')])
    assert (tmp_path / 'nginx.conf').exists()
    assert 'reverse-proxy' in yaml.safe_load(be.compose_file.read_text())['services']
    info = be.access(['aa'])
    assert info['proxy_url'] == 'http://127.0.0.1:8080'


def test_render_gateway_with_zero_models(tmp_path):
    state = {**STATE, 'open_webui': '/cache/open-webui'}
    images = {**IMAGES, 'open_webui': 'ghcr.io/open-webui/open-webui:test'}
    rc = render_compose(
        [], {}, images=images, ports=PORTS, state=state,
        litellm=True, ui=True, aux_dir=tmp_path,
    )
    # gateway + UI render even with no models; the model_list is just empty
    assert set(rc.compose['services']) == {'litellm', 'open-webui'}
    assert yaml.safe_load(rc.litellm_config)['model_list'] == []
    assert 'depends_on' not in rc.compose['services']['litellm']


def test_observe_tolerates_unreadable_compose_file(tmp_path):
    # A stale/invalid file makes `docker compose ps` raise; observe must not
    # propagate it (else acquire bricks before converge can overwrite the file).
    class RaisingPs(FakeDocker):
        def __call__(self, args):
            if 'ps' in args:
                raise RuntimeError('compose schema error on a stale file')
            return super().__call__(args)

    be = ComposeBackend(
        state_dir=tmp_path, inventory=simulate_inventory('4x80'),
        run=RaisingPs(), http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS, state=STATE,
    )
    be.converge([vllm('a')])         # writes a good compose file
    assert be.observe() == set()     # ps raises -> lenient empty, no crash


def test_probe_ready_tracks_running(tmp_path):
    be = make_backend(tmp_path)
    g = vllm('a')
    assert be.probe_ready(g, 'a').ready is False   # nothing converged yet
    be.converge([g])
    assert be.probe_ready(g, 'a').ready is True


def test_placement_error_skips_deployment(tmp_path):
    be = make_backend(tmp_path, spec='2x80')
    plan = be.converge([vllm('big', tp=4)])        # needs 4, only 2
    assert not plan.ok
    assert be.last_errors
    assert be.observe() == set()


def test_plan_is_read_only(tmp_path):
    be = make_backend(tmp_path)
    p = be.plan([vllm('a', served='aa'), vllm('b', served='bb')])
    assert p.assignments == {'a': [0], 'b': [1]}    # placement computed
    assert not be.compose_file.exists()             # but nothing written
    assert be.run.calls == []                       # and docker never invoked


def test_converge_render_only_writes_without_applying(tmp_path):
    be = make_backend(tmp_path)
    plan = be.converge([vllm('a', served='aa')], apply=False)
    assert be.compose_file.exists()                 # on-disk project rendered
    assert plan.assignments == {'a': [0]}
    assert be.last_assignments == {'a': [0]}
    # neither `up` nor `down` ran — only render happened
    verbs = [c[c.index('-f') + 2] for c in be.run.calls if '-f' in c]
    assert 'up' not in verbs and 'down' not in verbs
    # ...and applying afterwards (default) brings exactly that file up
    be.converge([vllm('a', served='aa')], apply=True)
    assert be.observe() == {'a'}


# -- controller + ledger + catalog integration ----------------------------


CATALOG = {
    'models': {
        'qc': {'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct'},
        'rr': {'source': 'hf://BAAI/bge-reranker-base'},
    },
    'endpoints': {
        'qwen-coder': {'engine': 'vllm', 'model': 'qc',
                       'runtime': {'max_model_len': 32768}},
        'reranker': {'engine': 'vllm', 'model': 'rr',
                     'reclaim': {'policy': 'stop'}},
    },
}


def test_controller_compose_end_to_end(tmp_path):
    catalog = Catalog.from_dict(CATALOG)
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path)
    ctl = Controller(ledger, backend)

    out = ctl.acquire('alice', catalog.resolve_names(['qwen-coder', 'reranker']))
    assert out.wait.ready is True
    assert set(backend.observe()) == set(g.id for g in out.deployments)
    # two vLLM services + the LiteLLM front door
    compose = yaml.safe_load(backend.compose_file.read_text())
    deployment_services = [s for s in compose['services'] if s.startswith('vllm-')]
    assert len(deployment_services) == 2
    assert 'litellm' in compose['services']

    # release the lease: reranker is stop-policy -> torn down; qwen keep-warm stays
    qwen_gid = next(g.id for g in out.deployments
                    if 'qwen-coder' in g.served)
    ctl.release(out.lease.id)
    assert backend.observe() == {qwen_gid}


def test_evict_tears_down_keep_warm_idle_deployment(tmp_path):
    catalog = Catalog.from_dict(CATALOG)
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path)
    ctl = Controller(ledger, backend)

    out = ctl.acquire('alice', catalog.resolve_names(['qwen-coder']))
    qwen_gid = out.deployments[0].id
    ctl.release(out.lease.id)
    assert backend.observe() == {qwen_gid}        # keep-warm stays resident

    ev = ctl.evict(None)                           # force-evict idle deployments
    assert qwen_gid in ev.evicted_deployment_ids
    assert backend.observe() == set()             # GPU freed: service gone
    # ledger records it as stopped, not idle
    _, deployments = ledger.status()
    assert {g.state for g in deployments} == {DeploymentState.STOPPED}


# -- LiteLLM front door ----------------------------------------------------


def test_render_litellm_front_door(tmp_path):
    rc = render_compose(
        [vllm('grp-a', served='qwen-served')], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, litellm_port=14042, aux_dir=tmp_path,
    )
    assert 'litellm' in rc.compose['services']
    litellm = rc.compose['services']['litellm']
    assert litellm['ports'] == ['14042:4000']
    cfg = yaml.safe_load(rc.litellm_config)
    entry = cfg['model_list'][0]
    assert entry['model_name'] == 'grp-a'                  # alias = endpoint
    assert entry['litellm_params']['model'] == 'openai/qwen-served'
    # api_base host == the (deterministic) vLLM service name, so LiteLLM routes
    # over the network. This legacy per-deployment path (no catalog) keeps the
    # served-name-derived host; see the catalog-superset tests for no-blip.
    assert entry['litellm_params']['api_base'] == (
        'http://vllm-qwen-served:8000/v1'
    )


def _two_endpoint_catalog():
    from infer_stack.leasing.catalog import Catalog

    return Catalog.from_dict(
        {
            'models': {
                'ma': {'source': 'hf://org/a'},
                'mb': {'source': 'hf://org/b'},
            },
            'endpoints': {
                'alpha': {'engine': 'vllm', 'model': 'ma'},
                'beta': {'engine': 'vllm', 'model': 'mb'},
            },
        }
    )


def test_litellm_superset_config_is_invariant_across_model_set(tmp_path):
    """The no-blip property at the rendering level: with a catalog, the LiteLLM
    config + service spec do NOT change when the live model set changes, so
    `docker compose up` would not recreate the gateway (no blip)."""
    cat = _two_endpoint_catalog()
    # alpha live, then beta live -- two different desired sets, same catalog.
    rc_a = render_compose(
        [vllm('grp-a', served='alpha')], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, litellm_port=14042, aux_dir=tmp_path, catalog=cat,
    )
    rc_b = render_compose(
        [vllm('grp-b', served='beta')], {'grp-b': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, litellm_port=14042, aux_dir=tmp_path, catalog=cat,
    )

    # 1. Gateway config is identical regardless of which model is live.
    assert rc_a.litellm_config == rc_b.litellm_config

    # 2. It routes BOTH catalog endpoints (a superset), to deterministic hosts
    #    that match the live vLLM service names.
    cfg = yaml.safe_load(rc_a.litellm_config)
    routes = {
        e['model_name']: e['litellm_params']['api_base']
        for e in cfg['model_list']
    }
    assert routes == {
        'alpha': 'http://vllm-alpha:8000/v1',
        'beta': 'http://vllm-beta:8000/v1',
    }

    # 3. The litellm SERVICE SPEC is byte-identical across the two renders (same
    #    config_hash label, and no per-model depends_on) -> `docker compose up`
    #    leaves the gateway container running. This is the no-blip guarantee.
    assert rc_a.compose['services']['litellm'] == rc_b.compose['services']['litellm']
    assert 'depends_on' not in rc_a.compose['services']['litellm']

    # 4. The live upstream uses exactly the host its route points at.
    assert 'vllm-alpha' in rc_a.compose['services']
    assert 'vllm-beta' in rc_b.compose['services']


def _three_endpoint_catalog():
    from infer_stack.leasing.catalog import Catalog

    return Catalog.from_dict(
        {
            'models': {
                'mc': {'source': 'hf://org/c'},
                'ma': {'source': 'hf://org/a'},
                'mb': {'source': 'hf://org/b'},
            },
            'endpoints': {
                'cee': {'engine': 'vllm', 'model': 'mc'},
                'alpha': {'engine': 'vllm', 'model': 'ma'},
                'beta': {'engine': 'vllm', 'model': 'mb'},
            },
        }
    )


def test_untouched_model_stable_when_another_swaps_gpu(tmp_path):
    """No-blip for the concurrent-pipeline case: when one process releases its
    GPU and another acquires it for a DIFFERENT model, every *untouched* service
    must be byte-identical so `docker compose up -d` recreates only the swapped
    one. Models that nobody touched stay resident while others go up/down.

    Models C (untouched) + A are live; then A is released and B takes its freed
    GPU. C stays pinned on its own GPU (placement rule 1: pinned deployments keep
    their GPUs), so no reshuffle.
    """
    cat = _three_endpoint_catalog()
    rc1 = render_compose(
        [vllm('grp-c', served='cee'), vllm('grp-a', served='alpha')],
        {'grp-c': [2], 'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, litellm_port=14042, aux_dir=tmp_path, catalog=cat,
    )
    rc2 = render_compose(
        [vllm('grp-c', served='cee'), vllm('grp-b', served='beta')],
        {'grp-c': [2], 'grp-b': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, litellm_port=14042, aux_dir=tmp_path, catalog=cat,
    )
    s1, s2 = rc1.compose['services'], rc2.compose['services']

    # The untouched model's service is byte-identical (same GPU, same spec) ->
    # not recreated. This is the property the concurrent pipelines depend on.
    assert s1['vllm-cee'] == s2['vllm-cee']
    # The standing front door (gateway) is byte-identical too -> no gateway blip.
    assert s1['litellm'] == s2['litellm']
    # Only the swapped model's service differs between the two states.
    assert set(s1) ^ set(s2) == {'vllm-alpha', 'vllm-beta'}


def test_litellm_legacy_config_churns_without_catalog(tmp_path):
    """Contrast: without a catalog, the config is per-live-deployment, so it
    DOES change when the model set changes (the old, blip-causing behavior)."""
    rc_a = render_compose(
        [vllm('grp-a', served='alpha')], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, aux_dir=tmp_path,
    )
    rc_b = render_compose(
        [vllm('grp-b', served='beta')], {'grp-b': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, aux_dir=tmp_path,
    )
    assert rc_a.litellm_config != rc_b.litellm_config


def test_litellm_config_hash_label_tracks_model_list(tmp_path):
    """The litellm service must change when its routing config changes.

    Regression: a second alias coalesced onto a live deployment rewrote the config
    file, but `docker compose up -d` left the old litellm container running
    (spec unchanged), so the new alias never became routable. Stamping the
    config hash onto a label makes converge recreate litellm on a config change.
    """
    from infer_stack.leasing.compose import CONFIG_HASH_LABEL

    def label(deployment):
        rc = render_compose(
            [deployment], {deployment.id: [0]}, images=IMAGES, ports=PORTS, state=STATE,
            litellm=True, litellm_port=14042, aux_dir=tmp_path,
        )
        return rc.compose['services']['litellm']['labels'][CONFIG_HASH_LABEL]

    one = vllm('grp-a', served='qwen-served')
    # same deployment serving two aliases (the coalesced case)
    two = vllm('grp-a', served='qwen-served')
    two.served['extra-alias'] = {'served_model_name': 'qwen-served'}

    h_one, h_two = label(one), label(two)
    assert h_one and h_two
    assert h_one != h_two            # config changed -> spec (label) changed
    assert label(one) == h_one       # stable for identical input (idempotent)


def test_render_open_webui_default_and_stable(tmp_path):
    """Open WebUI renders with the gateway and is byte-stable across switches."""
    state = {**STATE, 'open_webui': '/cache/open-webui'}
    images = {**IMAGES, 'open_webui': 'ghcr.io/open-webui/open-webui:test'}

    def render(deployments, assigns):
        return render_compose(
            deployments, assigns, images=images, ports=PORTS, state=state,
            litellm=True, ui=True, ui_port=13000,
            litellm_master_key='sk-x', aux_dir=tmp_path,
        )

    one = render([vllm('grp-a', served='a')], {'grp-a': [0]})
    ow = one.compose['services']['open-webui']
    assert ow['ports'] == ['13000:8080']
    assert ow['environment']['OPENAI_API_BASE_URL'] == 'http://litellm:4000/v1'
    # The secret is referenced, not inlined — its value lives in the sidecar .env.
    assert ow['environment']['OPENAI_API_KEY'] == '${LITELLM_MASTER_KEY}'
    assert ow['depends_on'] == ['litellm']

    # Adding a second model recreates litellm (routing changed) but must NOT
    # touch open-webui — that is what keeps the UI from blinking on a switch.
    two = render([vllm('grp-a', served='a'), vllm('grp-b', served='b', t=1.0)],
                 {'grp-a': [0], 'grp-b': [1]})
    assert two.compose['services']['open-webui'] == ow
    assert two.compose['services']['litellm'] != one.compose['services']['litellm']


def test_render_no_open_webui_when_ui_off(tmp_path):
    state = {**STATE, 'open_webui': '/cache/open-webui'}
    images = {**IMAGES, 'open_webui': 'ow:test'}
    rc = render_compose(
        [vllm('grp-a', served='a')], {'grp-a': [0]},
        images=images, ports=PORTS, state=state,
        litellm=True, ui=False, aux_dir=tmp_path,
    )
    assert 'open-webui' not in rc.compose['services']


def _render_ui(deployments, assigns, tmp_path, **kw):
    state = {**STATE, 'open_webui': '/cache/open-webui'}
    images = {**IMAGES, 'open_webui': 'ow:test'}
    return render_compose(
        deployments, assigns, images=images, ports=PORTS, state=state,
        ui=True, ui_port=13000, litellm_master_key='sk-x', aux_dir=tmp_path,
        **kw,
    )


def test_open_webui_connects_ollama_daemon_directly(tmp_path):
    """No gateway: the UI talks straight to the Ollama daemon's native API, so
    you can pull/run models from the UI — a drop-in for hand-run ollama+webui."""
    rc = _render_ui([ollama('daemon')], {'daemon': [0]}, tmp_path, litellm=False)
    assert 'litellm' not in rc.compose['services']
    env = rc.compose['services']['open-webui']['environment']
    assert env['ENABLE_OLLAMA_API'] == 'True'
    assert env['OLLAMA_BASE_URL'] == 'http://ollama-daemon:11434'
    # nothing OpenAI to point at -> that connection is off, not dangling
    assert env['ENABLE_OPENAI_API'] == 'False'
    assert 'OPENAI_API_BASE_URL' not in env
    # the per-model upstream comes and goes, so the UI does not hard-depend on it
    assert 'depends_on' not in rc.compose['services']['open-webui']


def test_open_webui_enables_ollama_api_alongside_litellm(tmp_path):
    """Gateway on AND an Ollama daemon: chat routes through LiteLLM (one base_url
    for every alias) while the native Ollama connection still gives the UI its
    pull/manage panel."""
    rc = _render_ui([ollama('daemon')], {'daemon': [0]}, tmp_path, litellm=True)
    env = rc.compose['services']['open-webui']['environment']
    assert env['OPENAI_API_BASE_URL'] == 'http://litellm:4000/v1'
    assert env['OPENAI_API_KEY'] == '${LITELLM_MASTER_KEY}'
    assert env['ENABLE_OLLAMA_API'] == 'True'
    assert env['OLLAMA_BASE_URL'] == 'http://ollama-daemon:11434'
    assert rc.compose['services']['open-webui']['depends_on'] == ['litellm']


def test_open_webui_points_at_vllm_v1_without_litellm(tmp_path):
    """No gateway, a single vLLM upstream: the UI's OpenAI connection points at
    that process's own /v1."""
    rc = _render_ui([vllm('grp-a', served='a')], {'grp-a': [0]}, tmp_path,
                    litellm=False)
    name = vllm_service_name(vllm('grp-a', served='a'))
    env = rc.compose['services']['open-webui']['environment']
    assert env['OPENAI_API_BASE_URL'] == f'http://{name}:8000/v1'
    assert env['ENABLE_OLLAMA_API'] == 'False'


def test_no_open_webui_without_litellm_or_upstreams(tmp_path):
    """No gateway and nothing to point at -> no standing UI (an empty desired
    set then has nothing to run and converge tears the project down)."""
    rc = _render_ui([], {}, tmp_path, litellm=False)
    assert 'open-webui' not in rc.compose['services']


def test_litellm_router_settings_present(tmp_path):
    rc = render_compose(
        [vllm('grp-a', served='a')], {'grp-a': [0]},
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=True, aux_dir=tmp_path,
    )
    cfg = yaml.safe_load(rc.litellm_config)
    # transient upstream connection errors during warmup are retried, not 500s
    assert cfg['router_settings']['num_retries'] >= 1


def test_access_includes_ui_url_when_ui_on(tmp_path):
    be = make_backend(tmp_path, ui=True)
    assert be.access(['a'])['ui_url'] == 'http://127.0.0.1:13000'
    be_noui = make_backend(tmp_path, ui=False)
    assert 'ui_url' not in be_noui.access(['a'])


def test_converge_diff_decline_aborts(tmp_path, monkeypatch):
    import infer_stack.diff_prompt as dp
    from infer_stack.leasing.backend import ConvergeAborted

    monkeypatch.setattr(dp, 'confirm_writes', lambda *a, **k: False)
    be = make_backend(tmp_path, assume_yes=False)
    with pytest.raises(ConvergeAborted):
        be.converge([vllm('grp-a', served='a')])
    # Declined -> nothing written, no `docker compose up`.
    assert not be.compose_file.exists()
    assert not any('up' in call for call in be.run.calls)


def test_converge_diff_accept_applies(tmp_path, monkeypatch):
    import infer_stack.diff_prompt as dp

    monkeypatch.setattr(dp, 'confirm_writes', lambda *a, **k: True)
    be = make_backend(tmp_path, assume_yes=False)
    be.converge([vllm('grp-a', served='a')])
    assert be.compose_file.exists()
    assert any('up' in call for call in be.run.calls)


def test_converge_no_change_does_not_prompt(tmp_path, monkeypatch):
    import infer_stack.diff_prompt as dp

    be = make_backend(tmp_path, assume_yes=False)
    monkeypatch.setattr(dp, 'confirm_writes', lambda *a, **k: True)
    be.converge([vllm('grp-a', served='a')])     # first apply

    # An identical desired set renders byte-identical files -> no diff, no prompt.
    seen = {'n': 0}

    def boom(*a, **k):
        seen['n'] += 1
        return True

    monkeypatch.setattr(dp, 'confirm_writes', boom)
    be.converge([vllm('grp-a', served='a')])
    assert seen['n'] == 0


def test_acquire_rolls_back_lease_on_decline(tmp_path):
    from infer_stack.leasing import EndpointRequest, LeaseState, vllm_structural
    from infer_stack.leasing.backend import ConvergeAborted

    class DeclineBackend:
        def observe(self):
            return set()

        def converge(self, desired):
            raise ConvergeAborted('declined')

    led = Ledger(SqliteStore(tmp_path / 'ledger.db'))
    ctrl = Controller(led, DeclineBackend())
    with pytest.raises(ConvergeAborted):
        ctrl.acquire(
            'me', [EndpointRequest('a', 'vllm', vllm_structural(model_ref='a'))]
        )
    # The just-created lease must not linger as active after a decline.
    leases, _ = led.status()
    assert not [le for le in leases if le.state == LeaseState.ACTIVE]


def test_acquire_rolls_back_when_deployment_cannot_be_placed(tmp_path):
    """An acquire whose deployment can't be placed must fail fast, not hang on ready.

    Regression: with the only GPU already held, `acquire <other>` left the new
    deployment LIVE in the ledger (so `leases` showed a phantom "live" deployment) and
    then blocked forever waiting on a container placement skipped. Acquire now
    rolls the lease back and raises PlacementError.
    """
    from infer_stack.leasing import LeaseState
    from infer_stack.leasing.backend import PlacementError

    catalog = Catalog.from_dict(CATALOG)
    ledger = Ledger(SqliteStore(tmp_path / 'ledger.db'))
    backend = make_backend(tmp_path, spec='1x80')      # exactly one GPU
    ctl = Controller(ledger, backend)

    first = ctl.acquire('alice', catalog.resolve_names(['qwen-coder']))
    assert first.wait.ready                            # claims the only GPU
    qwen_gid = first.deployments[0].id

    with pytest.raises(PlacementError) as ei:
        ctl.acquire('bob', catalog.resolve_names(['reranker']))
    assert ei.value.reasons                            # carries the planner reason

    leases, deployments = ledger.status()
    # the rolled-back request leaves no second active lease, and the reranker
    # never lingers as a LIVE deployment with no container behind it
    assert [le.owner for le in leases if le.state == LeaseState.ACTIVE] == ['alice']
    live_served = {ep for g in deployments
                   if g.state == DeploymentState.LIVE for ep in g.served}
    assert 'reranker' not in live_served
    assert backend.observe() == {qwen_gid}             # only qwen is running


def test_controller_acquire_render_only_stages(tmp_path):
    """`acquire --no-apply` records intent + writes on-disk state, no `up`."""
    from infer_stack.leasing import LeaseState

    catalog = Catalog.from_dict(CATALOG)
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path)
    ctl = Controller(ledger, backend)

    out = ctl.acquire('alice', catalog.resolve_names(['qwen-coder']), apply=False)
    assert out.applied is False
    assert out.wait is None                       # never blocked on readiness
    assert backend.compose_file.exists()          # on-disk project written
    assert backend.observe() == set()             # but nothing brought up
    assert out.reconcile.assignments              # placement was computed
    leases, _ = ledger.status()
    assert [le.state for le in leases] == [LeaseState.ACTIVE]   # staged intent

    # applying (the default) brings up exactly what was staged
    ctl.reconcile()
    assert backend.observe() == {out.deployments[0].id}


def test_access_reports_litellm_base_url(tmp_path):
    be = make_backend(tmp_path)
    info = be.access(['qwen-coder', 'reranker'])
    assert info['base_url'] == 'http://127.0.0.1:14042/v1'
    assert info['api_key_env'] == 'LITELLM_MASTER_KEY'
    assert info['api_key'].startswith('sk-')      # infer-stack manages the key
    assert info['request_names'] == {'qwen-coder': 'qwen-coder',
                                     'reranker': 'reranker'}


def test_master_key_managed_stable_and_persisted(tmp_path):
    be = make_backend(tmp_path)
    k1 = be.master_key()
    assert k1.startswith('sk-')
    assert be.master_key() == k1                  # reused, not regenerated
    # a fresh backend over the same state dir recovers the same key
    assert make_backend(tmp_path).master_key() == k1


def test_converge_references_master_key_via_env_not_baked(tmp_path):
    be = make_backend(tmp_path)
    be.converge([vllm('a')])
    raw = be.compose_file.read_text()
    key = be.master_key()
    assert key.startswith('sk-')
    # The compose YAML references the var, it does NOT contain the secret value.
    compose = yaml.safe_load(raw)
    assert (
        compose['services']['litellm']['environment']['LITELLM_MASTER_KEY']
        == '${LITELLM_MASTER_KEY}'
    )
    assert key not in raw
    # The value lives in the sidecar .env next to the compose file, which
    # `docker compose --env-file` loads for interpolation.
    env_path = be.compose_file.parent / '.env'
    assert env_path.exists()
    assert f'LITELLM_MASTER_KEY={key}' in env_path.read_text()


def test_envfile_carries_managed_api_key(tmp_path):
    from infer_stack.leasing.envfile import build_descriptor, render_env_file
    from infer_stack.leasing.models import Lease

    be = make_backend(tmp_path)
    info = be.access(['qwen-coder'])
    lease = Lease('sess-x', 'me', 'active', 0.0, None, None, 0.0,
                  endpoints=['qwen-coder'])
    g = Deployment('g', 'ck', 'vllm', 'shared-compatible', {}, {},
                        {'qwen-coder': {'served_model_name': 'qwen-coder'}},
                        DeploymentState.LIVE, 0.0, 0.0)
    d = build_descriptor(lease, [g], base_url=info['base_url'],
                         api_key_env=info['api_key_env'], api_key=info['api_key'],
                         request_names=info['request_names'])
    env = render_env_file(d)
    assert 'OPENAI_API_KEY=sk-' in env            # source-and-go: key is in the env-file
    assert f"OPENAI_BASE_URL={info['base_url']}" in env


def test_access_none_without_litellm(tmp_path):
    # No gateway and no UI -> no single access point.
    be = make_backend(tmp_path, litellm=False, ui=False)
    assert be.access(['x']) is None


def test_access_reports_ui_url_without_litellm(tmp_path):
    # No gateway but a managed UI -> the UI is still a useful access point.
    be = make_backend(tmp_path, litellm=False, ui=True)
    info = be.access(['x'])
    assert info == {'ui_url': 'http://127.0.0.1:13000'}


def test_probe_ready_requires_routable_alias(tmp_path):
    be = make_backend(tmp_path)
    g = vllm('a', served='a-served')
    be.converge([g])                       # writes litellm config listing 'a'
    assert be.probe_ready(g, 'a').ready is True
    # an endpoint the gateway doesn't list is not ready
    assert be.probe_ready(g, 'ghost').ready is False


# -- Ollama pull / warmup readiness ----------------------------------------


def test_ollama_probe_pulls_tag_then_ready(tmp_path):
    be = make_backend(tmp_path)
    g = ollama('daemon', tag='qwen3.5:4b')
    be.converge([g])
    r = be.probe_ready(g, 'daemon')
    assert r.ready is True
    # the tag was pulled into the daemon via `docker compose exec ... ollama pull`
    pulls = [c for c in be.run.calls if 'pull' in c]
    assert pulls and 'qwen3.5:4b' in pulls[0]


def test_ollama_probe_pulls_tag_without_litellm(tmp_path):
    # A lean --no-litellm Ollama stack must still pull its declared anchor tag;
    # readiness for that daemon *is* the pull (no gateway to probe).
    be = make_backend(tmp_path, litellm=False)
    g = ollama('daemon', tag='qwen3.5:4b')
    be.converge([g])
    r = be.probe_ready(g, 'daemon')
    assert r.ready is True
    pulls = [c for c in be.run.calls if 'pull' in c]
    assert pulls and 'qwen3.5:4b' in pulls[0]


def test_ollama_pull_is_idempotent(tmp_path):
    be = make_backend(tmp_path)
    g = ollama('daemon', tag='m:1b')
    be.converge([g])
    be.probe_ready(g, 'daemon')
    be.probe_ready(g, 'daemon')
    assert len([c for c in be.run.calls if 'pull' in c]) == 1


class _FailingExecDocker(FakeDocker):
    def __call__(self, args):
        if 'exec' in args and 'pull' in args:
            raise RuntimeError('daemon not ready')
        return super().__call__(args)


def test_ollama_not_ready_when_pull_fails(tmp_path):
    be = ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory('4x80'),
        run=_FailingExecDocker(),
        http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS, state=STATE,
    )
    g = ollama('daemon', tag='m:1b')
    be.converge([g])
    r = be.probe_ready(g, 'daemon')
    assert r.ready is False
    assert 'pulling' in r.detail


def test_ollama_pull_uses_host_service_name_not_deployment_id(tmp_path):
    """Regression: the tag pull must exec into the daemon by the SAME name it is
    rendered/observed under — ``ollama-<host>`` — not ``ollama-<deployment.id>``.

    A catalog Ollama endpoint carries a ``host`` distinct from the generated
    deployment id, so the daemon renders as e.g. ``ollama-local-ollama`` while the
    pull was exec'ing into ``ollama-grp-...`` — a service that "is not running" —
    so the tag was never pulled and ``--require-generation`` timed out. The shared
    ``ollama()`` helper hid this because it sets no host (id == host).
    """
    from infer_stack.leasing.compose import ollama_service_name

    g = Deployment(
        'grp-xyz', 'ck', 'ollama', 'shared-compatible', {},
        {'engine': 'ollama', 'host': 'local-ollama', 'gpu_indices': [],
         'settings': {'keep_alive': '2m'}},
        {'smol-ollama': {'model': 'smollm2:135m'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )
    be = make_backend(tmp_path)
    be.converge([g])
    assert ollama_service_name(g) == 'ollama-local-ollama'   # host-derived name
    be.probe_ready(g, 'smol-ollama')
    pulls = [c for c in be.run.calls if 'exec' in c and 'pull' in c]
    assert pulls, 'a tag pull should have been attempted'
    assert 'ollama-local-ollama' in pulls[0]      # the rendered/observed service
    assert 'ollama-grp-xyz' not in pulls[0]       # NOT the deployment-id name


# -- protocol-aware, generation-gated readiness ----------------------------


class RecordingHttp:
    """A requests-like fake that advertises a fixed model set, records every
    GET/POST URL, and returns a configurable status for generation POSTs."""

    def __init__(self, models, post_status=200):
        self.models = list(models)
        self.post_status = post_status
        self.get_urls: list[str] = []
        self.post_urls: list[str] = []

    def get(self, url, **kw):
        self.get_urls.append(url)
        if url.endswith('/models'):
            return FakeResp(200, {'data': [{'id': m} for m in self.models]})
        return FakeResp(404, {'detail': 'not found'})

    def post(self, url, **kw):
        self.post_urls.append(url)
        payload = (
            {'choices': [{'message': {'content': 'ok'}}]}
            if self.post_status < 400 else {'error': 'not ready'}
        )
        return FakeResp(self.post_status, payload)


def _backend_with_http(tmp_path, http, **kw):
    return ComposeBackend(
        state_dir=tmp_path, inventory=simulate_inventory('4x80'),
        run=FakeDocker(), http=http, images=IMAGES, ports=PORTS, state=STATE, **kw,
    )


def test_probe_uses_completions_endpoint_for_completions_protocol(tmp_path):
    """A completions-only model must be probed at /completions, not the chat
    endpoint it never answers (which would block readiness forever)."""
    http = RecordingHttp(models=['e'])
    be = _backend_with_http(tmp_path, http)
    g = vllm('e', protocol='completions')
    be.converge([g])
    assert be.probe_ready(g, 'e').ready is True
    assert http.post_urls[-1].endswith('/v1/completions')


def test_probe_uses_chat_endpoint_by_default(tmp_path):
    http = RecordingHttp(models=['e'])
    be = _backend_with_http(tmp_path, http)
    g = vllm('e')  # default protocol chat
    be.converge([g])
    assert be.probe_ready(g, 'e').ready is True
    assert http.post_urls[-1].endswith('/v1/chat/completions')


def test_probe_not_ready_until_generation_succeeds(tmp_path):
    """The static superset gateway advertises the alias from the start, so a
    listed alias is NOT proof of readiness — only a successful generation is.
    A model whose generation 503s must report NOT ready."""
    http = RecordingHttp(models=['e'], post_status=503)
    be = _backend_with_http(tmp_path, http)
    g = vllm('e')
    be.converge([g])
    r = be.probe_ready(g, 'e')
    assert r.ready is False               # listed, but not serving -> not ready
    assert http.post_urls                  # we did attempt the generation


def test_probe_without_gateway_hits_published_upstream(tmp_path):
    """No gateway: readiness probes the vLLM upstream's own published /v1 with a
    real generation, not just 'container running'."""
    http = RecordingHttp(models=['srv'])
    be = _backend_with_http(tmp_path, http, litellm=False)
    g = vllm('e', served='srv')
    be.converge([g])
    assert be.probe_ready(g, 'e').ready is True
    # probed the host-published upstream port, and asked for the served name
    assert any('127.0.0.1:18000/v1' in u for u in http.get_urls + http.post_urls)


def test_catalog_parses_endpoint_protocol(tmp_path):
    from infer_stack.leasing.catalog import Catalog, CatalogError

    base = {'models': {'m': {'source': 'hf://org/m'}}}
    cat = Catalog.from_dict({**base, 'endpoints': {
        'chatty': {'engine': 'vllm', 'model': 'm'},
        'compl': {'engine': 'vllm', 'model': 'm', 'protocol': 'completions'},
    }})
    assert cat.resolve_endpoint('chatty').served['protocol'] == 'chat'
    assert cat.resolve_endpoint('compl').served['protocol'] == 'completions'
    with pytest.raises(CatalogError):
        Catalog.from_dict({**base, 'endpoints': {
            'bad': {'engine': 'vllm', 'model': 'm', 'protocol': 'embeddings'},
        }})


# -- docker compose schema validation --------------------------------------
#
# The fake-docker tests above check the dict we *build*, not whether docker
# accepts it (which is how `capabilities: [["gpu"]]` slipped through to real
# hardware). `docker compose config -q` runs Compose's schema validation
# without pulling images, starting containers, or needing a GPU, so it is fast
# and catches that whole class of bug. Skipped where docker compose is absent.


def _docker_compose_available() -> bool:
    if shutil.which('docker') is None:
        return False
    try:
        subprocess.run(
            ['docker', 'compose', 'version'],
            capture_output=True,
            timeout=15,
            check=True,
        )
        return True
    except Exception:
        return False


requires_compose = pytest.mark.skipif(
    not _docker_compose_available(),
    reason='docker compose not available',
)


def _render(deployments, *, litellm, tmp_path):
    plan = plan_placement(deployments, simulate_inventory('2x48'))
    return render_compose(
        deployments, plan.assignments,
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=litellm, litellm_port=14042, aux_dir=tmp_path,
    )


@requires_compose
@pytest.mark.parametrize(
    'name, deployments, litellm',
    [
        ('vllm-single', [vllm('grp-v', served='qwen')], True),
        ('vllm-tp2', [vllm('grp-v', served='qwen', tp=2)], True),
        ('vllm-and-ollama',
         [vllm('grp-v', served='qwen', t=0), ollama('grp-o', tag='m:1b', t=1)],
         True),
        ('no-litellm', [vllm('grp-v', served='qwen')], False),
    ],
)
def test_rendered_compose_passes_docker_schema(name, deployments, litellm, tmp_path):
    rc = _render(deployments, litellm=litellm, tmp_path=tmp_path)
    if rc.litellm_config is not None:
        (tmp_path / 'litellm_config.yaml').write_text(rc.litellm_config)
    compose_file = tmp_path / 'docker-compose.yml'
    compose_file.write_text(yaml.safe_dump(rc.compose, sort_keys=False))
    result = subprocess.run(
        ['docker', 'compose', '-p', 'infer-stack-validate',
         '-f', str(compose_file), 'config', '-q'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f'{name}: {result.stderr}'


def test_inventory_detected_lazily_not_at_construction(tmp_path, monkeypatch):
    # Startup paths (the TUI especially) construct the backend with
    # inventory=None; nothing may wait on the nvidia-smi subprocess until the
    # first placement actually needs the inventory.
    calls = []

    def fake_detect():
        calls.append(1)
        return simulate_inventory('2x48')

    import infer_stack.hardware as hardware
    monkeypatch.setattr(hardware, 'detect_inventory', fake_detect)

    be = ComposeBackend(
        state_dir=tmp_path, inventory=None,
        run=FakeDocker(), http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert calls == []                       # construction paid nothing
    assert be.inventory['gpu_count'] == 2    # first access detects...
    assert be.inventory['gpu_count'] == 2    # ...and caches
    assert calls == [1]


def test_explicit_inventory_never_detects(tmp_path, monkeypatch):
    import infer_stack.hardware as hardware
    monkeypatch.setattr(
        hardware, 'detect_inventory',
        lambda: (_ for _ in ()).throw(AssertionError('must not detect')),
    )
    be = ComposeBackend(
        state_dir=tmp_path, inventory=simulate_inventory('4x80'),
        run=FakeDocker(), http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert be.inventory['gpu_count'] == 4


def test_vllm_service_mounts_persistent_compile_caches(tmp_path):
    # Regression: only the HF cache was mounted, so with `reclaim: stop` every
    # re-acquire cold-started the container and re-paid the full
    # torch.compile / Triton / CUDA-jit pass (~10-20 min on big models). The
    # state dirs existed in default_state_paths all along — assert they are
    # actually mounted now.
    be = ComposeBackend(
        state_dir=tmp_path, inventory=simulate_inventory('2x48'),
        run=FakeDocker(), http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS,
        state={**STATE,
               'vllm_cache': '/cache/vllm', 'torch_cache': '/cache/torch',
               'triton_cache': '/cache/triton', 'cuda_cache': '/cache/cuda'},
    )
    be.converge([vllm('a')])
    import yaml as _yaml
    doc = _yaml.safe_load(be.compose_file.read_text())
    svc = next(v for k, v in doc['services'].items() if k.startswith('vllm-'))
    vllm_mounts = [v for v in svc['volumes'] if v.endswith(':/root/.cache/vllm')]
    assert len(vllm_mounts) == 1
    # Keyed per serve config: any arg change (e.g. --limit-mm-per-prompt)
    # starts a fresh compile cache instead of reloading a graph traced under
    # different inputs (vLLM's own cache key misses such knobs).
    assert re.match(r'/cache/vllm/cfg-[0-9a-f]{12}$', vllm_mounts[0].split(':')[0])
    assert '/cache/torch:/root/.cache/torch' in svc['volumes']
    assert '/cache/triton:/root/.triton' in svc['volumes']
    assert '/cache/cuda:/root/.nv' in svc['volumes']


def test_partial_state_dict_merges_over_defaults(tmp_path):
    be = ComposeBackend(
        state_dir=tmp_path, inventory=simulate_inventory('2x48'),
        run=FakeDocker(), http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS, state=STATE,  # partial: no cache keys
    )
    assert be.state['hf_cache'] == STATE['hf_cache']   # explicit wins
    assert 'vllm_cache' in be.state                    # defaults fill the rest


def test_vllm_compile_cache_rekeys_on_config_change(tmp_path):
    # Same model, different extra_args (the observed --limit-mm-per-prompt
    # change) -> different compile-cache subdir; identical config -> same.
    def _mount(extra_args):
        be = ComposeBackend(
            state_dir=tmp_path / str(len(extra_args)), inventory=simulate_inventory('2x48'),
            run=FakeDocker(), http=FakeHttp(tmp_path),
            images=IMAGES, ports=PORTS, state=STATE,
        )
        dep = vllm('a')
        dep.spec['runtime'] = {**dep.spec.get('runtime', {}), 'extra_args': extra_args}
        be.converge([dep])
        doc = yaml.safe_load(be.compose_file.read_text())
        svc = next(v for k, v in doc['services'].items() if k.startswith('vllm-'))
        return next(v for v in svc['volumes'] if v.endswith(':/root/.cache/vllm'))

    plain = _mount([])
    limited = _mount(['--limit-mm-per-prompt', '{"image": 0}'])
    assert plain != limited
    assert _mount([]) == plain  # deterministic for identical config


# ---------------------------------------------------------------------------
# Plan-time VRAM enrichment (docs/planning/vram-aware-placement.md Phase 3).
# ---------------------------------------------------------------------------


def _fake_weights(hf_cache: Path, model_id: str, mib: int):
    snap = (
        hf_cache / 'hub'
        / ('models--' + model_id.replace('/', '--'))
        / 'snapshots' / 'rev0'
    )
    snap.mkdir(parents=True)
    (snap / 'model.safetensors').write_bytes(b'x' * (mib * 1024 ** 2))



def test_plan_on_idle_host_sees_the_aggregate_that_cannot_fit(tmp_path):
    """Two deployments each placeable alone, impossible together.

    The Incubilate shape: a tensor-parallel-4 answerer plus a 1-GPU extractor
    on a 4-GPU host. Neither trips the planner's permanent-failure branch --
    each fits by itself -- so only planning them as a set reveals that the
    host can never hold both. The controller uses this to fail such a lease
    immediately instead of queueing for capacity that cannot exist.
    """
    be = make_backend(tmp_path, spec='4x80')
    plan = be.plan_on_idle_host([vllm('big', tp=4), vllm('ext')])
    assert set(plan.assignments) == {'big'}
    assert not plan.ok
    assert any(e.startswith('ext') for e in plan.errors)

    # ... and the pair that does fit reports no error, on the same host.
    plan = be.plan_on_idle_host([vllm('big', tp=2), vllm('ext')])
    assert set(plan.assignments) == {'big', 'ext'}
    assert plan.ok


def test_plan_on_idle_host_ignores_what_is_running(tmp_path):
    """"Idle host" means idle: converged deployments and their pins do not
    shrink the pool. Answering "is there room now" here would defeat the
    point -- that is what the ordinary plan/converge path already reports."""
    be = make_backend(tmp_path, spec='4x80')
    be.converge([vllm('a'), vllm('b'), vllm('c'), vllm('d')])  # every GPU busy
    assert len(be.observe()) == 4

    plan = be.plan_on_idle_host([vllm('e', tp=4)])
    assert plan.assignments == {'e': [0, 1, 2, 3]}
    assert plan.ok


def test_plan_on_idle_host_reuses_a_shared_deployment_pinned_elsewhere(tmp_path):
    """A requested deployment already running outside our allow-list is reusable.

    The Slurm shape: job A owns GPU 2 and started the shared extractor there;
    job B owns GPUs [0, 1] and wants a 2-GPU answerer plus that same extractor.
    B should queue for its two cards, not be rejected.

    Dropping every pin asked the wrong question -- "do a 2-GPU answerer AND a
    1-GPU extractor fit inside [0, 1]" -- and answered no. The extractor is
    already placed and does not need a card from B's slice at all; placement
    validates pins against the whole host for exactly this reason.
    """
    job_a = make_backend(tmp_path, spec='4x80', allowed_gpus=[2])
    job_a.converge([vllm('ext')])
    assert job_a.plan([vllm('ext')]).assignments == {'ext': [2]}

    job_b = make_backend(tmp_path, spec='4x80', allowed_gpus=[0, 1])
    plan = job_b.plan_on_idle_host([vllm('ans', tp=2), vllm('ext')])
    assert plan.assignments == {'ans': [0, 1], 'ext': [2]}
    assert plan.ok


def test_plan_on_idle_host_still_rejects_when_a_kept_pin_fills_the_host(tmp_path):
    """Keeping the requested deployments' pins must not weaken the detection.

    The answerer already holds all four cards, so the extractor has nowhere to
    go and never will. Honoring the pin is what makes that obvious.
    """
    be = make_backend(tmp_path, spec='4x80')
    be.converge([vllm('ans', tp=4)])
    plan = be.plan_on_idle_host([vllm('ans', tp=4), vllm('ext')])
    assert set(plan.assignments) == {'ans'}
    assert not plan.ok
    assert any(e.startswith('ext') for e in plan.errors)

def test_plan_enriches_floor_from_hf_cache(tmp_path):
    # A 20-GiB-weights model with NO declaration must still never plan onto
    # the 16-GiB card: the weight-bytes floor is attached automatically once
    # the weights are in the local HF cache.
    hf_cache = tmp_path / 'hf'
    _fake_weights(hf_cache, 'org/big', 20 * 1024)   # 20 GiB
    be = ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory('16,48'),
        run=FakeDocker(),
        http=FakeHttp(tmp_path),
        images=IMAGES, ports=PORTS,
        state={**STATE, 'hf_cache': str(hf_cache)},
    )
    plan = be.plan([vllm('big', hf='org/big')])
    assert plan.assignments == {'big': [1]}


def test_plan_enrichment_absent_cache_is_legacy(tmp_path):
    # No weights downloaded, nothing declared -> exactly the legacy plan.
    be = make_backend(tmp_path, spec='16,48')
    plan = be.plan([vllm('a', hf='org/never-downloaded')])
    assert plan.assignments == {'a': [0]}


def test_plan_uses_measured_overlay_when_undeclared(tmp_path):
    # A recorded measurement fills min_vram_gib for an undeclared endpoint:
    # 24 GiB measured -> only the 48 GiB card is eligible.
    be = make_backend(tmp_path, spec='16,48')
    g = vllm('m', hf='org/measured', max_len=4096)
    from infer_stack.leasing.vram import measurement_key_for_spec
    be.measurements.record(measurement_key_for_spec(g.spec), 24.0)
    plan = be.plan([g])
    assert plan.assignments == {'m': [1]}


def test_plan_declared_beats_measured_overlay(tmp_path):
    # A catalog declaration wins over the overlay (declared > measured).
    be = make_backend(tmp_path, spec='16,48')
    g = vllm('m', hf='org/measured', max_len=4096)
    g.spec['placement'] = {'min_vram_gib': 4.0}
    from infer_stack.leasing.vram import measurement_key_for_spec
    be.measurements.record(measurement_key_for_spec(g.spec), 24.0)
    plan = be.plan([g])
    # declared 4 GiB -> both cards eligible -> best-fit takes the 16er.
    assert plan.assignments == {'m': [0]}
