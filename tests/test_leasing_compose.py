"""Tests for the Compose backend, driven by a stateful fake docker seam.

These exercise render + converge + observe + teardown logic without real docker
or GPUs. The real docker/GPU path is validated separately on a GPU host.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import (
    Catalog,
    ComposeBackend,
    Controller,
    Ledger,
    SqliteStore,
    render_compose,
)
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


class FakeHttp:
    """Fake /v1/models that lists whatever aliases the LiteLLM config declares."""

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)

    def __call__(self, url):
        cfg = self.state_dir / 'litellm_config.yaml'
        names = []
        if cfg.exists():
            data = yaml.safe_load(cfg.read_text()) or {}
            names = [e['model_name'] for e in data.get('model_list', [])]
        return 200, {'data': [{'id': n} for n in names]}


# -- pure render -----------------------------------------------------------


def test_render_vllm_service():
    rc = render_compose(
        [vllm('grp-a', hf='Qwen/Q', served='qwen', tp=2)],
        {'grp-a': [0, 1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    svc = rc.compose['services']['vllm-grp-a']
    assert svc['image'] == 'vllm/vllm-openai:test'
    assert svc['command'][0] == 'Qwen/Q'
    assert '--tensor-parallel-size=2' in svc['command']
    assert '--served-model-name=qwen' in svc['command']
    assert svc['ports'] == ['18000:8000']
    devs = svc['deploy']['resources']['reservations']['devices'][0]
    assert devs['device_ids'] == ['0', '1']
    assert svc['labels']['infer-stack.group'] == 'grp-a'
    assert rc.services == {'vllm-grp-a': 'grp-a'}


def test_render_two_vllm_distinct_ports():
    rc = render_compose(
        [vllm('a', t=0), vllm('b', t=1)],
        {'a': [0], 'b': [1]},
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert rc.compose['services']['vllm-a']['ports'] == ['18000:8000']
    assert rc.compose['services']['vllm-b']['ports'] == ['18001:8000']


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
    rc = render_compose(
        [vllm('a'), vllm('b')], {'a': [0]},  # b not placed
        images=IMAGES, ports=PORTS, state=STATE,
    )
    assert set(rc.compose['services']) == {'vllm-a'}


# -- ComposeBackend converge/observe --------------------------------------


def make_backend(tmp_path, *, spec='4x80', **kw):
    return ComposeBackend(
        state_dir=tmp_path,
        inventory=simulate_inventory(spec),
        run=FakeDocker(),
        http_get=FakeHttp(tmp_path),
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
    assert entry['litellm_params']['api_base'] == 'http://vllm-grp-a:8000/v1'


def test_access_reports_litellm_base_url(tmp_path):
    be = make_backend(tmp_path)
    info = be.access(['qwen-coder', 'reranker'])
    assert info['base_url'] == 'http://127.0.0.1:14042/v1'
    assert info['api_key_env'] == 'LITELLM_MASTER_KEY'
    assert info['request_names'] == {'qwen-coder': 'qwen-coder',
                                     'reranker': 'reranker'}


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
