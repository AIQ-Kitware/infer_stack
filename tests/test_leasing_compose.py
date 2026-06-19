"""Tests for the Compose backend, driven by a stateful fake docker seam.

These exercise render + converge + observe + teardown logic without real docker
or GPUs. The real docker/GPU path is validated separately on a GPU host.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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
from infer_stack.leasing.models import DeploymentGroup, GroupState

STATE = {'hf_cache': '/cache/hf', 'ollama': '/cache/ollama'}
IMAGES = {
    'vllm': 'vllm/vllm-openai:test',
    'ollama': 'ollama/ollama:test',
    'litellm': 'ghcr.io/berriai/litellm:test',
}
PORTS = {'ollama': 11434}


def vllm(gid, *, hf='org/model', served=None, tp=1, max_len=32768, reclaim='keep-warm', t=0.0):
    # endpoint (public alias) is the group id; served_model_name is the upstream
    served_name = served or gid
    return DeploymentGroup(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': hf,
            'served_model_name': served_name,
            'runtime': {'tensor_parallel_size': tp, 'max_model_len': max_len},
            'reclaim': reclaim,
        },
        {gid: {'served_model_name': served_name}},
        GroupState.LIVE, t, t,
    )


def ollama(gid, *, tag='m:1b', t=0.0):
    return DeploymentGroup(
        gid, 'ck-' + gid, 'ollama', 'shared-compatible', {},
        {'engine': 'ollama', 'gpu_indices': [], 'settings': {'keep_alive': '2m'}},
        {gid: {'model': tag}}, GroupState.LIVE, t, t,
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
    group = vllm('grp-a', hf='Qwen/Q', served='qwen', tp=2)
    rc = render_compose(
        [group],
        {'grp-a': [0, 1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    # service name leads with the served model, suffixed by the group id
    name = vllm_service_name(group)
    assert name == 'vllm-qwen-grp-a'
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
    assert svc['labels']['infer-stack.group'] == 'grp-a'
    assert rc.services == {name: 'grp-a'}


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
    assert svc['environment']['CUDA_VISIBLE_DEVICES'] == '1'
    assert svc['ports'] == ['11434:11434']


def test_render_skips_unplaced_groups():
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
    # leads with the served model (slugified), suffixed by the full group id so
    # it is unique and correlates with `infer-stack leases`
    assert name == 'vllm-qwen2-5-0-5b-grp-098ed1'
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


def test_converge_removes_dropped_group(tmp_path):
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
    # no model groups running, but the compose project still has the gateway/UI
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


def test_placement_error_skips_group(tmp_path):
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
    assert set(backend.observe()) == set(g.id for g in out.groups)
    # two vLLM services + the LiteLLM front door
    compose = yaml.safe_load(backend.compose_file.read_text())
    group_services = [s for s in compose['services'] if s.startswith('vllm-')]
    assert len(group_services) == 2
    assert 'litellm' in compose['services']

    # release the lease: reranker is stop-policy -> torn down; qwen keep-warm stays
    qwen_gid = next(g.id for g in out.groups
                    if 'qwen-coder' in g.served)
    ctl.release(out.lease.id)
    assert backend.observe() == {qwen_gid}


def test_evict_tears_down_keep_warm_idle_group(tmp_path):
    catalog = Catalog.from_dict(CATALOG)
    ledger = Ledger(SqliteStore(':memory:'))
    backend = make_backend(tmp_path)
    ctl = Controller(ledger, backend)

    out = ctl.acquire('alice', catalog.resolve_names(['qwen-coder']))
    qwen_gid = out.groups[0].id
    ctl.release(out.lease.id)
    assert backend.observe() == {qwen_gid}        # keep-warm stays resident

    ev = ctl.evict(None)                           # force-evict idle groups
    assert qwen_gid in ev.evicted_group_ids
    assert backend.observe() == set()             # GPU freed: service gone
    # ledger records it as stopped, not idle
    _, groups = ledger.status()
    assert {g.state for g in groups} == {GroupState.STOPPED}


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
    # api_base host == the vLLM service name, so LiteLLM routes over the network
    assert entry['litellm_params']['api_base'] == (
        'http://vllm-qwen-served-grp-a:8000/v1'
    )


def test_litellm_config_hash_label_tracks_model_list(tmp_path):
    """The litellm service must change when its routing config changes.

    Regression: a second alias coalesced onto a live group rewrote the config
    file, but `docker compose up -d` left the old litellm container running
    (spec unchanged), so the new alias never became routable. Stamping the
    config hash onto a label makes converge recreate litellm on a config change.
    """
    from infer_stack.leasing.compose import CONFIG_HASH_LABEL

    def label(group):
        rc = render_compose(
            [group], {group.id: [0]}, images=IMAGES, ports=PORTS, state=STATE,
            litellm=True, litellm_port=14042, aux_dir=tmp_path,
        )
        return rc.compose['services']['litellm']['labels'][CONFIG_HASH_LABEL]

    one = vllm('grp-a', served='qwen-served')
    # same group serving two aliases (the coalesced case)
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

    def render(groups, assigns):
        return render_compose(
            groups, assigns, images=images, ports=PORTS, state=state,
            litellm=True, ui=True, ui_port=13000,
            litellm_master_key='sk-x', aux_dir=tmp_path,
        )

    one = render([vllm('grp-a', served='a')], {'grp-a': [0]})
    ow = one.compose['services']['open-webui']
    assert ow['ports'] == ['13000:8080']
    assert ow['environment']['OPENAI_API_BASE_URL'] == 'http://litellm:4000/v1'
    assert ow['environment']['OPENAI_API_KEY'] == 'sk-x'
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


def test_acquire_rolls_back_when_group_cannot_be_placed(tmp_path):
    """A serve whose group can't be placed must fail fast, not hang on ready.

    Regression: with the only GPU already held, `serve <other>` left the new
    group LIVE in the ledger (so `leases` showed a phantom "live" group) and
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
    qwen_gid = first.groups[0].id

    with pytest.raises(PlacementError) as ei:
        ctl.acquire('bob', catalog.resolve_names(['reranker']))
    assert ei.value.reasons                            # carries the planner reason

    leases, groups = ledger.status()
    # the rolled-back request leaves no second active lease, and the reranker
    # never lingers as a LIVE group with no container behind it
    assert [le.owner for le in leases if le.state == LeaseState.ACTIVE] == ['alice']
    live_served = {ep for g in groups
                   if g.state == GroupState.LIVE for ep in g.served}
    assert 'reranker' not in live_served
    assert backend.observe() == {qwen_gid}             # only qwen is running


def test_controller_acquire_render_only_stages(tmp_path):
    """`serve --render-only` records intent + writes on-disk state, no `up`."""
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
    assert backend.observe() == {out.groups[0].id}


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


def test_converge_bakes_master_key_into_litellm(tmp_path):
    be = make_backend(tmp_path)
    be.converge([vllm('a')])
    compose = yaml.safe_load(be.compose_file.read_text())
    baked = compose['services']['litellm']['environment']['LITELLM_MASTER_KEY']
    assert baked == be.master_key() and baked.startswith('sk-')


def test_envfile_carries_managed_api_key(tmp_path):
    from infer_stack.leasing.envfile import build_descriptor, render_env_file
    from infer_stack.leasing.models import Lease

    be = make_backend(tmp_path)
    info = be.access(['qwen-coder'])
    lease = Lease('sess-x', 'me', 'active', 0.0, None, None, 0.0,
                  endpoints=['qwen-coder'])
    g = DeploymentGroup('g', 'ck', 'vllm', 'shared-compatible', {}, {},
                        {'qwen-coder': {'served_model_name': 'qwen-coder'}},
                        GroupState.LIVE, 0.0, 0.0)
    d = build_descriptor(lease, [g], base_url=info['base_url'],
                         api_key_env=info['api_key_env'], api_key=info['api_key'],
                         request_names=info['request_names'])
    env = render_env_file(d)
    assert 'OPENAI_API_KEY=sk-' in env            # source-and-go: key is in the env-file
    assert f"OPENAI_BASE_URL={info['base_url']}" in env


def test_access_none_without_litellm(tmp_path):
    be = make_backend(tmp_path, litellm=False)
    assert be.access(['x']) is None


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


def _render(groups, *, litellm, tmp_path):
    plan = plan_placement(groups, simulate_inventory('2x48'))
    return render_compose(
        groups, plan.assignments,
        images=IMAGES, ports=PORTS, state=STATE,
        litellm=litellm, litellm_port=14042, aux_dir=tmp_path,
    )


@requires_compose
@pytest.mark.parametrize(
    'name, groups, litellm',
    [
        ('vllm-single', [vllm('grp-v', served='qwen')], True),
        ('vllm-tp2', [vllm('grp-v', served='qwen', tp=2)], True),
        ('vllm-and-ollama',
         [vllm('grp-v', served='qwen', t=0), ollama('grp-o', tag='m:1b', t=1)],
         True),
        ('no-litellm', [vllm('grp-v', served='qwen')], False),
    ],
)
def test_rendered_compose_passes_docker_schema(name, groups, litellm, tmp_path):
    rc = _render(groups, litellm=litellm, tmp_path=tmp_path)
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
