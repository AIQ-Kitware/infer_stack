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

import pytest
import yaml

from infer_stack.leasing.catalog import Catalog
from infer_stack.leasing.compose import _vllm_service, default_state_paths
from infer_stack.leasing.models import Deployment, DeploymentState
from infer_stack.leasing.placement import required_gpu_count
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


def _local_fixture(container_path):
    """Map the in-image fixture path to the source-tree copy.

    The catalog names an absolute path inside the image, which is the point --
    no bind-mount to arrange. Tests run outside the image, so they read the
    same file from the tree it was COPYed from.
    """
    if not container_path:
        return container_path
    name = pathlib.PurePosixPath(str(container_path)).name
    local = (pathlib.Path(__file__).parent.parent
             / 'infer_stack' / 'mockserver' / 'data' / name)
    assert local.exists(), f'{container_path} has no source-tree copy at {local}'
    return str(local)


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
        args.mock_config = _local_fixture(args.mock_config)
        config = config_from_args(args, unknown)
        assert config['models'], endpoint_name
        # The served alias must survive, or clients asking for it 404.
        served = [block.get('served_model_name')
                  for block in config['models'].values()]
        assert any(served), endpoint_name


def test_runtime_knobs_reach_the_simulator():
    command = [str(part) for part in _render('mock-smol')['command']]
    args, unknown = build_parser().parse_known_args(command)
    args.mock_config = _local_fixture(args.mock_config)
    config = config_from_args(args, unknown)
    block = config['models']['HuggingFaceTB/SmolLM2-135M-Instruct']
    assert block['max_model_len'] == 2048, 'enforced, not just advertised'
    assert block['served_model_name'] == 'mock-smol'
    assert block['ability'] == 0.6, 'extra_args reached the simulator'


def test_extra_args_can_switch_the_response_mode():
    command = [str(part) for part in _render('mock-sycophant')['command']]
    args, unknown = build_parser().parse_known_args(command)
    args.mock_config = _local_fixture(args.mock_config)
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
    args.mock_config = _local_fixture(args.mock_config)
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


# ---------------------------------------------------------------------------
# The oracle seam: catalog -> placement -> rendered command -> answer key.
# ---------------------------------------------------------------------------

FIXTURE_IN_IMAGE = (
    '/opt/infer-stack/infer_stack/mockserver/data/oracle_questions.yaml'
)

#: Where that fixture lives in the source tree, for tests that load it. The
#: container path above is the same file after `COPY infer_stack ./infer_stack`.
FIXTURE_ON_DISK = (
    pathlib.Path(__file__).parent.parent
    / 'infer_stack' / 'mockserver' / 'data' / 'oracle_questions.yaml'
)


def _deployment_spec(endpoint_name: str) -> dict:
    """Resolve an oracle endpoint the way a real acquire does."""
    request = Catalog.load(CATALOG).resolve_endpoint(endpoint_name)
    return request.spec


def _as_deployment(spec: dict) -> Deployment:
    return Deployment(
        'd1', 'ck', spec['engine'], 'shared-compatible', {}, spec, {},
        DeploymentState.LIVE, 0.0, 0.0)


@pytest.mark.parametrize(
    'endpoint_name', ['mock-smol', 'mock-qwen17', 'mock-sycophant'])
def test_the_oracle_needs_no_gpu(endpoint_name):
    """It is a CPU server, so it must be placeable on a GPU-less host.

    `runtime.cpu_only` is what says so. It cannot be `runtime.simulator`:
    that also switches the command renderer, and this mock deliberately
    parses vLLM's own command line.
    """
    spec = _deployment_spec(endpoint_name)
    assert spec['runtime'].get('cpu_only') is True
    assert 'simulator' not in spec['runtime'], 'must stay vLLM-shaped'
    assert required_gpu_count(_as_deployment(spec)) == 0


def test_a_real_vllm_endpoint_still_needs_its_gpus():
    """The marker must not have relaxed the default for everything else."""
    spec = {
        'engine': 'vllm',
        'runtime': {'tensor_parallel_size': 2, 'pipeline_parallel_size': 2},
    }
    assert required_gpu_count(_as_deployment(spec)) == 4
    assert required_gpu_count(_as_deployment({'engine': 'vllm',
                                              'runtime': {}})) == 1


def test_gpu_memory_utilization_alone_does_not_mean_cpu_only():
    """A real deployment may set it low; only the explicit marker counts."""
    spec = {'engine': 'vllm', 'runtime': {'gpu_memory_utilization': 0.0}}
    assert required_gpu_count(_as_deployment(spec)) == 1


@pytest.mark.parametrize('endpoint_name,expected', [
    ('mock-smol', '--mock-ability=0.6'),
    ('mock-qwen17', '--mock-ability=0.55'),
    ('mock-sycophant', '--mock-mode=sycophant'),
])
def test_the_rendered_command_carries_both_the_profile_and_the_answer_key(
        endpoint_name, expected):
    """One shared corpus, per-endpoint behaviour: both must survive."""
    command = [str(part) for part in _render(endpoint_name)['command']]
    assert expected in command
    assert f'--mock-config={FIXTURE_IN_IMAGE}' in command
    # Still vLLM's own CLI shape, not the simulator renderer's.
    assert any(part.startswith('--served-model-name=') for part in command)
    assert any(part.startswith('--max-model-len=') for part in command)


def test_the_rendered_configuration_builds_an_answer_aware_simulator():
    """The whole point: the deployed server knows gold answers.

    Parsing the rendered command and loading the fixture it names must yield
    a simulator with a real corpus -- not one that can only invent
    `answer-...` strings for questions it has never seen.
    """
    from infer_stack.mockserver.server import build_simulator

    command = [str(part) for part in _render('mock-smol')['command']]
    args, unknown = build_parser().parse_known_args(command)
    # Point at the source-tree copy; in the image the two are the same file.
    args.mock_config = str(FIXTURE_ON_DISK)
    config = config_from_args(args, unknown)

    sim = build_simulator(config)
    assert sim.questions, 'no question corpus reached the simulator'
    assert sim.answer_key, 'no answer key reached the simulator'
    assert set(sim.answer_key) == set(sim.questions)
    assert sim.composition, 'compositional behaviour is part of the corpus'
    # The endpoint's own ability came from the catalog, not the fixture.
    assert config['models'][
        'HuggingFaceTB/SmolLM2-135M-Instruct']['ability'] == 0.6


def test_a_known_question_can_return_its_configured_gold_answer():
    """Answer-awareness, made deterministic.

    Ability one removes the draw, so this asserts the gold answer is wired
    through rather than asserting that the catalog's 0.6 happens to hit.
    """
    from infer_stack.mockserver.server import build_simulator

    fixture = yaml.safe_load(FIXTURE_ON_DISK.read_text())
    config = dict(fixture)
    config['models'] = {'oracle': {'ability': 1.0}}
    sim = build_simulator(config)

    question = fixture['questions']['cap-fr']
    result = sim.complete('oracle', [{'role': 'user', 'content': question}])
    assert result.latent_key == 'cap-fr'
    assert result.is_correct
    assert result.text == fixture['answer_key']['cap-fr'] == 'Paris'


def test_the_compose_healthcheck_is_runnable_in_the_mock_image():
    """Compose overrides the image's own healthcheck, so its command has to
    exist in the image. It shells out to curl, and python:3.11-slim has none
    unless the dockerfile installs it -- without that, a healthy container is
    marked unhealthy forever."""
    service = _render('mock-smol')
    healthcheck = service.get('healthcheck') or {}
    assert healthcheck.get('disable') is not True, (
        'the oracle is not a runtime.simulator, so it keeps a real healthcheck')

    test_cmd = [str(part) for part in healthcheck['test']]
    dockerfile = (pathlib.Path(__file__).parent.parent
                  / 'dockerfiles' / 'mock-vllm.dockerfile').read_text()
    binary = test_cmd[1] if test_cmd and test_cmd[0] == 'CMD' else test_cmd[0]
    assert f'install -y --no-install-recommends {binary}' in dockerfile, (
        f'compose healthchecks the mock with {binary!r}, '
        f'which the image must therefore contain')
