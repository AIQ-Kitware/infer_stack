"""
The oracle mock must deploy the way vLLM does.

Starting the mock out-of-band exercised the card and the endpoint but
skipped infer-stack's own acquire / converge / release path -- the very
machinery a dress rehearsal is meant to validate. Presenting vLLM's
command line lets an endpoint point at a mock image and be leased like any
other, so these tests pin the seam: what infer-stack renders is what the
mock can parse.

This is the answer-key mock (``catalog-mock-oracle.yaml``), not the
API-fidelity one; see that catalog's header for why both exist.
"""

from __future__ import annotations

import pathlib

import yaml

from infer_stack.leasing.compose import _vllm_service, default_state_paths
from infer_stack.leasing.models import Deployment, DeploymentState
from infer_stack.mockserver.vllm_serve import build_parser, config_from_args

CATALOG = (pathlib.Path(__file__).parent.parent
           / 'dev' / 'e2e_tests' / 'catalog-mock-oracle.yaml')


def _render(endpoint_name: str):
    """Render the compose service infer-stack would create for an endpoint."""
    catalog = yaml.safe_load(CATALOG.read_text())
    endpoint = catalog['endpoints'][endpoint_name]
    runtime = dict(endpoint.get('runtime') or {})
    hf_id = catalog['models'][endpoint['model']]['source'].split('://', 1)[-1]
    served = endpoint.get('public_name', endpoint_name)

    spec = {
        'engine': 'vllm', 'hf_model_id': hf_id, 'served_model_name': served,
        'revision': None, 'quantization': None, 'dtype': None,
        'runtime': runtime, 'reclaim': {'policy': 'stop'},
    }
    capacity = {}
    if runtime.get('max_model_len') is not None:
        capacity['max_model_len'] = runtime['max_model_len']
    deployment = Deployment(
        'd1', 'ck', 'vllm', 'shared-compatible', capacity, spec,
        {served: {'served_model_name': served}},
        DeploymentState.LIVE, 0.0, 0.0)
    return _vllm_service(deployment, [0], 8000,
                         {'vllm': 'vllm/vllm-openai:latest'},
                         default_state_paths())


def test_the_catalog_selects_the_mock_image():
    # runtime.image is part of the structural compat key, so a mock endpoint
    # is a distinct deployment from a real one serving the same model --
    # they must never coalesce onto one process.
    service = _render('mock-smol')
    assert service['image'] == 'aiq-mock-vllm:latest'


def test_the_mock_parses_exactly_what_infer_stack_renders():
    for endpoint_name in ('mock-smol', 'mock-qwen17', 'mock-sycophant'):
        command = [str(part) for part in _render(endpoint_name)['command']]
        args, unknown = build_parser().parse_known_args(command)
        assert not unknown, (endpoint_name, unknown)
        config = config_from_args(args, unknown)
        assert config['models'], endpoint_name
        # The served alias must survive, or clients asking for it 404.
        served = [block.get('served_model_name')
                  for block in config['models'].values()]
        assert any(served), endpoint_name


def test_runtime_knobs_reach_the_simulator():
    command = [str(part) for part in _render('mock-smol')['command']]
    args, unknown = build_parser().parse_known_args(command)
    config = config_from_args(args, unknown)
    block = config['models']['HuggingFaceTB/SmolLM2-135M-Instruct']
    assert block['max_model_len'] == 2048, 'enforced, not just advertised'
    assert block['served_model_name'] == 'mock-smol'
    assert block['ability'] == 0.6, 'extra_args reached the simulator'


def test_extra_args_can_switch_the_response_mode():
    command = [str(part) for part in _render('mock-sycophant')['command']]
    args, unknown = build_parser().parse_known_args(command)
    config = config_from_args(args, unknown)
    assert all(block.get('mode') == 'sycophant'
               for block in config['models'].values())


def test_gpu_flags_are_accepted_and_ignored():
    # Rejecting --gpu-memory-utilization would make the mock undeployable
    # for a reason unrelated to what it simulates.
    args, unknown = build_parser().parse_known_args([
        'some/model', '--host', '0.0.0.0', '--port', '8000',
        '--tensor-parallel-size=4', '--gpu-memory-utilization=0.85',
        '--dtype=half', '--enforce-eager', '--enable-prefix-caching',
        '--kv-cache-dtype=fp8',
    ])
    assert not unknown
    config = config_from_args(args, unknown)
    assert 'some/model' in config['models']


def test_a_served_endpoint_answers_under_its_alias():
    import json
    import urllib.request

    from infer_stack.mockserver import MockServer

    command = [str(part) for part in _render('mock-smol')['command']]
    args, unknown = build_parser().parse_known_args(command)
    config = config_from_args(args, unknown)

    with MockServer(config, port=0) as server:
        request = urllib.request.Request(
            server.url + '/v1/completions',
            data=json.dumps({'model': 'mock-smol',
                             'prompt': 'What is the capital of France?'}
                            ).encode(),
            headers={'Content-Type': 'application/json'})
        body = json.loads(urllib.request.urlopen(request, timeout=10).read())
    assert body['choices'][0]['text']
