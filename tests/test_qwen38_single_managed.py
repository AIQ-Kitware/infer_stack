"""Offline integration contract for the patched vLLM single-3090 image.

No container, checkpoint download, CUDA device, or Docker daemon is needed.
"""

from pathlib import Path

import pytest

from infer_stack.backends.kubeai import render_models
from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Catalog, CatalogError, plan_placement, render_compose
from infer_stack.leasing.compose import vllm_service_name
from infer_stack.leasing.models import Deployment, DeploymentState
from infer_stack.leasing.placement import min_vram_per_gpu

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/qwen38-flash-next-3090.yaml'
IMAGES = {'vllm': 'vllm/vllm-openai:default', 'litellm': 'litellm:test'}
STATE = {'runtime': '/state/runtime', 'hf_cache': '/state/hf'}


def deployment():
    cat = Catalog.load(EXAMPLE)
    req = cat.resolve_endpoint('qwen38-flash-next-3090').to_request()
    return Deployment(
        'grp-q38', req.compat_key, req.engine, req.sharing, req.capacity,
        req.spec, {req.endpoint: req.served}, DeploymentState.LIVE, 0.0, 0.0,
    )


def test_example_renders_single_gpu_with_upstream_entrypoint():
    dep = deployment()
    plan = plan_placement([dep], simulate_inventory('2x24'))
    assert plan.ok, plan.errors
    assert plan.assignments == {dep.id: [1]}
    rendered = render_compose(
        [dep], plan.assignments, images=IMAGES, ports={}, state=STATE,
    )
    svc = rendered.compose['services'][vllm_service_name(dep)]
    assert svc['image'].startswith('ghcr.io/dominikbucko/qwen38-flash-next-3090@sha256:')
    assert svc['command'] == ['/model']
    assert svc['environment']['PORT'] == '8000'
    assert svc['environment']['MAX_MODEL_LEN'] == '135168'
    assert svc['environment']['SERVED_MODEL_NAME'] == 'Qwen3.8-Flash-Next'
    assert 'CUDA_VISIBLE_DEVICES' not in svc['environment']
    assert svc['deploy']['resources']['reservations']['devices'][0]['device_ids'] == ['1']
    assert svc['ipc'] == 'host'
    assert svc['cap_add'] == ['SYS_PTRACE']
    assert svc['ulimits'] == {'memlock': -1, 'stack': 67108864}
    assert sorted(svc['volumes']) == sorted([
        '/state/runtime/qwen38-flash-next-3090/model:/model',
        '/state/runtime/qwen38-flash-next-3090/humming:/root/.humming',
        '/state/runtime/qwen38-flash-next-3090/triton:/root/.triton',
    ])
    assert min_vram_per_gpu(dep) == 23


def test_offloaded_checkpoint_floor_is_not_used():
    dep = deployment()
    dep.spec['placement']['floor_vram_gib'] = 125  # an enriched floor from HF
    assert min_vram_per_gpu(dep) == 23  # offloaded, not GPU resident
    dep.spec['placement']['weight_floor'] = True
    assert min_vram_per_gpu(dep) == 125


def test_custom_compose_options_change_deployment_identity():
    data = Catalog.load(EXAMPLE).source
    original = Catalog.from_dict(data).resolve_endpoint('qwen38-flash-next-3090').to_request()
    data['endpoints']['qwen38-flash-next-3090']['runtime']['compose']['ulimits']['stack'] = 65536
    updated = Catalog.from_dict(data).resolve_endpoint('qwen38-flash-next-3090').to_request()
    assert original.compat_key != updated.compat_key


@pytest.mark.parametrize('compose,needle', [
    ({'privileged': True}, 'unknown runtime.compose fields'),
    ({'ipc': 'none'}, "only supports 'host'"),
    ({'cap_add': ['SYS_ADMIN']}, 'SYS_PTRACE'),
    ({'ulimits': {'memlock': True}}, 'positive integers'),
    ({'ulimits': {'stack': 0}}, 'positive integers'),
    ({'ulimits': {'rtprio': 99}}, 'memlock/stack'),
    (['ipc', 'host'], 'must be a mapping'),
])
def test_bad_compose_options_are_rejected(compose, needle):
    data = Catalog.load(EXAMPLE).source
    data['endpoints']['qwen38-flash-next-3090']['runtime']['compose'] = compose
    with pytest.raises(CatalogError, match=needle):
        Catalog.from_dict(data)


def test_kubeai_refuses_compose_specific_runtime():
    dep = deployment()
    result = render_models([dep], namespace='kubeai', default_resource_profile='gpu')
    assert result.docs == []
    assert dep.id in result.unrenderable
    assert any('compose' in e.lower() for e in result.errors)
