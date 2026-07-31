"""
``runtime.simulator``: serving an endpoint from a vLLM *API* simulator.

These pin the three things that make a simulator endpoint work where a vLLM
endpoint would not: its own CLI, no GPU requirement, and no curl healthcheck.
Everything else about the endpoint -- leasing, gateway routing, the readiness
generation probe -- is deliberately unchanged, so it is covered by the
existing vLLM tests.
"""

import pytest

from infer_stack.leasing.compose import _vllm_service
from infer_stack.leasing.models import Deployment, DeploymentState
from infer_stack.leasing.placement import required_gpu_count
from infer_stack.profile_runtime import simulator_args

IMAGE = 'ghcr.io/llm-d/llm-d-inference-sim:v0.9.0'
IMAGES = {'vllm': 'vllm/vllm-openai:latest'}


def _deployment(runtime, *, hf='HuggingFaceTB/SmolLM2-135M-Instruct',
                served='mock-smol'):
    return Deployment(
        'dep', 'ck-dep', 'vllm', 'shared-compatible', {},
        {
            'engine': 'vllm',
            'hf_model_id': hf,
            'served_model_name': served,
            'runtime': runtime,
            'reclaim': 'stop',
        },
        {served: {'served_model_name': served, 'protocol': 'chat'}},
        DeploymentState.LIVE, 0.0, 0.0,
    )


def test_simulator_args_use_the_simulators_own_cli():
    args = simulator_args({
        'served_model_name': 'mock-smol',
        'max_model_len': 2048,
        'max_num_seqs': 8,
        'simulator': {'kind': 'llm-d-sim', 'mode': 'random', 'seed': 7},
    })
    assert args[:2] == ['--model', 'mock-smol']
    assert '--served-model-name=mock-smol' in args
    assert '--max-model-len=2048' in args
    assert ['--mode', 'random'] == args[args.index('--mode'):][:2]
    # vLLM-only flags must NOT appear: the simulator rejects unknown flags and
    # exits, so leaking one turns every acquire into a crash-loop.
    joined = ' '.join(args)
    for absent in ('--tensor-parallel-size', '--gpu-memory-utilization',
                   '--host', '--max-num-batched-tokens'):
        assert absent not in joined


def test_simulator_model_flag_avoids_the_hf_repo_id():
    """The repo id would send it looking for a tokenizer render service."""
    svc = _vllm_service(
        _deployment({'image': IMAGE, 'simulator': {'kind': 'llm-d-sim'}}),
        [], 18000, IMAGES, {},
    )
    assert '--model' in svc['command']
    model = svc['command'][svc['command'].index('--model') + 1]
    assert model == 'mock-smol'
    assert 'HuggingFaceTB' not in ' '.join(svc['command'])


def test_simulator_flags_pass_through_by_name():
    """Any simulator knob is reachable from the catalog without a code change."""
    args = simulator_args({
        'served_model_name': 'm',
        'simulator': {
            'kind': 'llm-d-sim',
            'failure_injection_rate': 20,
            'failure_types': ['rate_limit', 'server_error'],
            'enable_kvcache': True,
            'log_http': False,
        },
    })
    assert ['--failure-injection-rate', '20'] == \
        args[args.index('--failure-injection-rate'):][:2]
    assert ['--failure-types', 'rate_limit', 'server_error'] == \
        args[args.index('--failure-types'):][:3]
    assert '--enable-kvcache' in args
    assert '--log-http' not in args  # false => omitted, not `--log-http false`


def test_unknown_simulator_kind_is_refused():
    with pytest.raises(ValueError):
        simulator_args({
            'served_model_name': 'm',
            'simulator': {'kind': 'not-a-simulator'},
        })


def test_simulator_needs_no_gpu():
    """The whole point is running on a host that has none."""
    sim = _deployment({'image': IMAGE, 'simulator': {'kind': 'llm-d-sim'}})
    real = _deployment({'tensor_parallel_size': 2})
    assert required_gpu_count(sim) == 0
    assert required_gpu_count(real) == 2


def test_simulator_service_drops_caches_and_healthcheck():
    svc = _vllm_service(
        _deployment({'image': IMAGE, 'simulator': {'kind': 'llm-d-sim'}}),
        [], 18000, IMAGES, {},
    )
    assert svc['image'] == IMAGE
    # Distroless: no curl to run the check, and no writable /root to cache into.
    assert svc['healthcheck'] == {'disable': True}
    assert 'volumes' not in svc
    # Still a normal upstream in every other respect.
    assert svc['ports'] == ['18000:8000']
    assert 'deploy' not in svc


def test_absent_simulator_key_leaves_vllm_untouched():
    svc = _vllm_service(_deployment({'max_model_len': 4096}), [1], None,
                        IMAGES, {})
    assert svc['image'] == IMAGES['vllm']
    assert svc['command'][0] == 'HuggingFaceTB/SmolLM2-135M-Instruct'
    assert '--tensor-parallel-size=1' in svc['command']
    assert svc['healthcheck']['test'][0] == 'CMD'
    assert svc['volumes']
