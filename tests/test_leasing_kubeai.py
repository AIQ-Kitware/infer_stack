"""Tests for the KubeAI leasing backend, driven by a stateful fake kubectl.

Mirrors tests/test_leasing_compose.py: render + converge + observe + probe
without a real cluster. The real kubectl/cluster path is validated separately
on a k3s host (docs/kubeai-backend.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from infer_stack.backends.kubeai import KubeaiBackend, render_models
from infer_stack.leasing import (
    Controller,
    EndpointRequest,
    Ledger,
    LeaseState,
    SqliteStore,
    vllm_structural,
)
from infer_stack.leasing.models import Deployment, DeploymentState


def vllm(gid, *, hf='org/model', served=None, tp=1, profile='rtx-4090',
         reclaim='keep-warm', protocol='chat', t=0.0, **runtime_extra):
    served_name = served or gid
    runtime = {'tensor_parallel_size': tp, 'max_model_len': 4096}
    if profile is not None:
        runtime['resource_profile'] = profile
    runtime.update(runtime_extra)
    return Deployment(
        gid, 'ck-' + gid, 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': hf,
            'served_model_name': served_name,
            'runtime': runtime,
            'reclaim': reclaim,
        },
        {gid: {'served_model_name': served_name, 'protocol': protocol}},
        DeploymentState.LIVE, t, t,
    )


def ollama(gid, *, t=0.0):
    return Deployment(
        gid, 'ck-' + gid, 'ollama', 'shared-compatible', {},
        {'engine': 'ollama', 'gpu_indices': [], 'settings': {}},
        {gid: {'model': 'm:1b'}}, DeploymentState.LIVE, t, t,
    )


class FakeKubectl:
    """Stateful kubectl stand-in: `apply` reflects the manifest file."""

    def __init__(self):
        self.applied: dict[str, dict] = {}  # model name -> CR doc
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        verb = args[3] if len(args) > 3 else ''
        if verb == 'apply':
            path = Path(args[args.index('-f') + 1])
            for doc in yaml.safe_load_all(path.read_text()):
                if doc:
                    self.applied[doc['metadata']['name']] = doc
            return ''
        if verb == 'get':
            items = list(self.applied.values())
            if '-l' in args:  # emulate the label selector
                key, _, value = args[args.index('-l') + 1].partition('=')
                items = [
                    doc for doc in items
                    if (doc.get('metadata', {}).get('labels') or {})
                    .get(key) == value
                ]
            return json.dumps({'items': items})
        if verb == 'delete':
            self.applied.pop(args[5], None)
            return ''
        return ''


class FakeHttp:
    """OpenAI-surface fake: /models lists applied names, POSTs generate."""

    def __init__(self, kubectl: FakeKubectl):
        self.kubectl = kubectl

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    def get(self, url, **kw):
        if url.endswith('/models'):
            data = [{'id': n} for n in self.kubectl.applied]
            return self._Resp(200, {'data': data})
        return self._Resp(404, {'detail': 'not found'})

    def post(self, url, **kw):
        model = (kw.get('json') or {}).get('model')
        if model not in self.kubectl.applied:
            return self._Resp(404, {'detail': f'{model} not found'})
        if 'chat/completions' in url:
            return self._Resp(
                200, {'choices': [{'message': {'content': 'ok'}}]}
            )
        return self._Resp(200, {'choices': [{'text': 'ok'}]})


def make_backend(tmp_path, **kw):
    kubectl = FakeKubectl()
    be = KubeaiBackend(
        state_dir=tmp_path,
        run=kubectl,
        http=FakeHttp(kubectl),
        **kw,
    )
    return be, kubectl


# -- pure render -----------------------------------------------------------


def test_render_model_doc_shape():
    rendered = render_models(
        [vllm('grp-a', hf='Qwen/Q', served='qwen', tp=2)],
        namespace='kubeai', default_resource_profile=None,
    )
    assert rendered.errors == []
    (doc,) = rendered.docs
    assert doc['metadata']['name'] == 'qwen'
    assert doc['metadata']['labels']['infer-stack/deployment'] == 'grp-a'
    assert doc['metadata']['labels']['infer-stack/managed'] == 'true'
    spec = doc['spec']
    assert spec['url'] == 'hf://Qwen/Q'
    assert spec['engine'] == 'VLLM'
    assert spec['resourceProfile'] == 'rtx-4090:2'  # tp=2 -> 2 GPUs
    assert spec['minReplicas'] == 1 and spec['maxReplicas'] == 1
    assert '--tensor-parallel-size=2' in spec['args']
    # the gateway request name and vLLM's served name must agree
    assert '--served-model-name=qwen' in spec['args']
    assert rendered.models == {'qwen': 'grp-a'}
    assert rendered.request_names == {'grp-a': 'qwen'}


def test_render_serving_knobs_reach_args():
    """The compat-key knobs flow through the same vllm_args path as compose."""
    dep = vllm('grp-k', served='q-awq', pipeline_parallel_size=2)
    dep.spec['quantization'] = 'awq'
    dep.spec['dtype'] = 'half'
    rendered = render_models(
        [dep], namespace='kubeai', default_resource_profile=None,
    )
    (doc,) = rendered.docs
    args = doc['spec']['args']
    assert '--quantization=awq' in args
    assert '--dtype=half' in args
    assert '--pipeline-parallel-size=2' in args
    assert doc['spec']['resourceProfile'] == 'rtx-4090:2'  # pp counts


def test_render_explicit_profile_count_wins():
    """A `profile:N` value is passed through, not re-suffixed."""
    rendered = render_models(
        [vllm('a', profile='l4:4')],
        namespace='kubeai', default_resource_profile=None,
    )
    assert rendered.docs[0]['spec']['resourceProfile'] == 'l4:4'


def test_render_missing_profile_is_unrenderable():
    """No resource profile anywhere -> loud error, never an invalid CR."""
    rendered = render_models(
        [vllm('a', profile=None)],
        namespace='kubeai', default_resource_profile=None,
    )
    assert rendered.docs == []
    assert rendered.unrenderable == {'a'}
    assert any('resource profile' in e for e in rendered.errors)
    # ...but the settings-level default fills it in
    rendered = render_models(
        [vllm('a', profile=None)],
        namespace='kubeai', default_resource_profile='default-gpu',
    )
    assert rendered.errors == []
    assert rendered.docs[0]['spec']['resourceProfile'] == 'default-gpu:1'


def test_render_ollama_is_unrenderable():
    rendered = render_models(
        [ollama('daemon')],
        namespace='kubeai', default_resource_profile='p',
    )
    assert rendered.unrenderable == {'daemon'}
    assert any('ollama' in e.lower() for e in rendered.errors)


def test_render_name_collision_oldest_wins():
    a = vllm('grp-old', served='qwen', t=0)
    b = vllm('grp-new', served='qwen', t=1)
    rendered = render_models(
        [a, b], namespace='kubeai', default_resource_profile=None,
    )
    assert rendered.models == {'qwen': 'grp-old'}
    assert rendered.unrenderable == {'grp-new'}
    assert any(e.startswith('grp-new') and 'grp-old' in e
               for e in rendered.errors)


# -- converge / apply / observe ---------------------------------------------


def test_converge_applies_and_observes(tmp_path):
    be, kubectl = make_backend(tmp_path)
    be.converge([vllm('grp-a', served='qwen')])
    assert be.models_file.exists()
    assert 'qwen' in kubectl.applied
    assert be.observe() == {'grp-a'}


def test_converge_render_only_defers_apply(tmp_path):
    be, kubectl = make_backend(tmp_path)
    be.converge([vllm('grp-a', served='qwen')], apply=False)
    assert be.models_file.exists()
    assert kubectl.applied == {}          # nothing hit the cluster
    assert be.observe() == set()
    be.apply()                            # the coalesced apply catches up
    assert be.observe() == {'grp-a'}


def test_converge_prunes_dropped_models(tmp_path):
    be, kubectl = make_backend(tmp_path)
    be.converge([vllm('a', served='aa', t=0), vllm('b', served='bb', t=1)])
    assert set(kubectl.applied) == {'aa', 'bb'}
    be.converge([vllm('a', served='aa', t=0)])
    assert set(kubectl.applied) == {'aa'}  # bb pruned
    be.converge([])
    assert kubectl.applied == {}


def test_converge_surfaces_collision_as_unplaced(tmp_path):
    be, _ = make_backend(tmp_path)
    be.converge([
        vllm('grp-old', served='qwen', t=0),
        vllm('grp-new', served='qwen', t=1),
    ])
    assert 'grp-new' in be.last_unplaced
    assert any('grp-new' in e for e in be.last_errors)
    assert be.observe() == {'grp-old'}


def test_prune_never_touches_unmanaged_models(tmp_path):
    """A hand-applied Model (no infer-stack labels) must survive converges —
    prune diffs only against the managed-label selector."""
    be, kubectl = make_backend(tmp_path)
    kubectl.applied['hand-rolled'] = {
        'metadata': {'name': 'hand-rolled', 'labels': {}}
    }
    be.converge([vllm('a', served='aa')])
    assert 'hand-rolled' in kubectl.applied   # untouched by the prune
    assert be.observe() == {'a'}              # and invisible to observe
    be.converge([])
    assert 'hand-rolled' in kubectl.applied   # even a full drain spares it


def test_observe_is_best_effort_on_kubectl_failure(tmp_path):
    def broken(args):
        raise RuntimeError('no cluster')

    be = KubeaiBackend(state_dir=tmp_path, run=broken, http=object())
    assert be.observe() == set()


# -- probe / access -----------------------------------------------------------


def test_probe_ready_requires_generation(tmp_path):
    be, kubectl = make_backend(tmp_path)
    dep = vllm('grp-a', served='qwen')
    assert not be.probe_ready(dep, 'grp-a').ready  # nothing applied yet
    be.converge([dep])
    assert be.probe_ready(dep, 'grp-a').ready


def test_probe_ready_completions_protocol(tmp_path):
    be, _ = make_backend(tmp_path)
    dep = vllm('grp-a', served='qwen', protocol='completions')
    be.converge([dep])
    assert be.probe_ready(dep, 'grp-a').ready


def test_access_maps_endpoints_to_model_names(tmp_path):
    be, _ = make_backend(tmp_path, base_url='http://10.0.0.5:8000/openai/v1/')
    be.converge([vllm('grp-a', served='qwen-32b')])
    info = be.access(['grp-a'])
    assert info['base_url'] == 'http://10.0.0.5:8000/openai/v1'
    assert info['api_key'] == 'EMPTY'          # unauthenticated gateway
    assert info['api_key_env'] is None
    assert info['request_names'] == {'grp-a': 'qwen-32b'}


# -- controller + ledger integration ------------------------------------------


def _req(endpoint, *, profile='rtx-4090', reclaim='stop'):
    return EndpointRequest(
        endpoint=endpoint,
        engine='vllm',
        structural=vllm_structural(model_ref=endpoint),
        capacity={'max_model_len': 4096},
        spec={
            'engine': 'vllm',
            'hf_model_id': f'org/{endpoint}',
            'served_model_name': endpoint,
            'runtime': {'resource_profile': profile},
            'reclaim': reclaim,
        },
        served={'served_model_name': endpoint, 'protocol': 'chat'},
    )


def make_controller(tmp_path):
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    be, kubectl = make_backend(tmp_path / 'state')
    return Controller(ledger, be), be, kubectl


def test_acquire_release_lifecycle(tmp_path):
    ctl, be, kubectl = make_controller(tmp_path)
    out = ctl.acquire('alice', [_req('qwen')], wait=True, timeout=10)
    assert out.wait is not None and out.wait.ready
    assert set(kubectl.applied) == {'qwen'}

    rel = ctl.release(out.lease.id)
    assert ctl.ledger.get_lease(out.lease.id).state == LeaseState.RELEASED
    # reclaim=stop -> the Model leaves the desired set and is pruned
    assert kubectl.applied == {}
    assert rel.reconcile is not None


def test_acquire_missing_profile_rolls_back(tmp_path):
    """An unrenderable deployment behaves exactly like a placement failure:
    the acquire fails loudly and the lease is rolled back."""
    from infer_stack.leasing.backend import PlacementError

    ctl, be, kubectl = make_controller(tmp_path)
    with pytest.raises(PlacementError, match='resource profile'):
        ctl.acquire('alice', [_req('qwen', profile=None)], wait=False)
    leases, deployments = ctl.ledger.status()
    assert [le.state for le in leases] == [LeaseState.RELEASED]
    assert kubectl.applied == {}


def test_keep_warm_stays_resident_after_release(tmp_path):
    ctl, be, kubectl = make_controller(tmp_path)
    out = ctl.acquire(
        'alice', [_req('qwen', reclaim='keep-warm')], wait=False,
    )
    ctl.release(out.lease.id)
    assert set(kubectl.applied) == {'qwen'}   # idle keep-warm stays up
    ctl.evict(None)
    assert kubectl.applied == {}              # evict frees the cluster