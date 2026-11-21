"""Exercise the public CLI and the descriptor delivered to a real leased child."""
import copy
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infer_stack.leasing import Catalog
from infer_stack.leasing.envfile import build_descriptor
from infer_stack.leasing.models import Deployment, DeploymentState, Lease
from infer_stack.leasing.provenance import endpoint_provenance


def source():
    return {
        'models': {'weights': {'source': 'hf://example/weights', 'revision': 'a' * 40}},
        'endpoints': {'local': {'model': 'weights', 'engine': 'vllm', 'runtime': {
            'image': 'example/vllm@sha256:' + 'b' * 64, 'max_model_len': 8192,
            'env': {'CUSTOM_TOKEN': 'private-fixture-value'},
        }}},
    }


def provenance(data):
    return endpoint_provenance(Catalog.from_dict(data).resolve_endpoint('local'))


@pytest.mark.parametrize('change', ['revision', 'quantization', 'image', 'max_model_len', 'extra_args'])
def test_model_and_launch_changes_invalidate_identity(change):
    data = source()
    before = provenance(data)
    assert before['immutable'] and not before['unknown_reasons']
    assert 'private-fixture-value' not in json.dumps(before)
    if change in {'revision', 'quantization'}:
        data['models']['weights'][change] = 'c' * 40 if change == 'revision' else 'awq'
    else:
        data['endpoints']['local']['runtime'][change] = {
            'image': 'example/vllm@sha256:' + 'd' * 64,
            'max_model_len': 16384, 'extra_args': ['--reasoning-parser', 'example'],
        }[change]
    assert provenance(data)['digest'] != before['digest']


def test_alias_and_placement_are_not_model_identity():
    data = source()
    before = provenance(data)
    data['endpoints']['local']['served_name'] = 'another-upstream-name'
    data['endpoints']['local']['placement'] = {'gpu_indices': [2]}
    data['endpoints']['local']['reclaim'] = 'keep-warm'
    assert provenance(data) == before


@pytest.mark.parametrize('change', ['tag', 'branch', 'command', 'mounts', 'override', 'external'])
def test_unpinned_or_unattested_inputs_remain_uncertain(change):
    data = source()
    runtime = data['endpoints']['local']['runtime']
    if change == 'tag':
        runtime['image'] = 'example/vllm:latest'
    elif change == 'branch':
        data['models']['weights']['revision'] = 'main'
    elif change == 'command':
        runtime['command'] = ['serve-model']
    elif change == 'mounts':
        runtime['mounts'] = {'/weights': 'weights'}
    elif change == 'override':
        runtime['extra_args'] = ['--revision=main']
    else:
        data['endpoints']['local'] = {'external': {'model': 'remote',
            'api_base': 'http://localhost:9000/v1', 'api_key_env': 'REMOTE_KEY'}}
    facts = provenance(data)
    assert not facts['immutable'] and facts['unknown_reasons']


def test_descriptor_projects_assigned_deployment_not_requested_capacity():
    request = Catalog.from_dict(source()).resolve_endpoint('local').to_request()
    lease = Lease('lease-id', 'test', 'active', 0, None, None, 0, endpoints=['local'])
    deployment = Deployment('deployment-id', request.compat_key, request.engine, request.sharing,
        request.capacity, copy.deepcopy(request.spec), {'local': request.served},
        DeploymentState.LIVE, 0, 0)
    def actual():
        return build_descriptor(lease, [deployment], base_url='http://localhost/v1')
    assert actual()['serving_provenance']['local'] == provenance(source())
    deployment.spec['runtime']['max_model_len'] = 16384
    assert actual()['serving_provenance']['local']['digest'] != provenance(source())['digest']


@pytest.mark.parametrize('child_status', [0, 7])
def test_real_cli_provenance_matches_leased_child_and_descriptor_is_cleaned(tmp_path, child_status):
    catalog = tmp_path / 'catalog.yaml'
    catalog.write_text(yaml.safe_dump(source()))
    env = dict(os.environ, INFER_STACK_BACKEND='null', INFER_STACK_CATALOG=str(catalog),
               XDG_CONFIG_HOME=str(tmp_path / 'config'), XDG_DATA_HOME=str(tmp_path / 'data'))
    for key in ('INFER_STACK_LEASE_ID', 'INFER_STACK_ENDPOINT_DESCRIPTOR'):
        env.pop(key, None)
    command = [sys.executable, '-m', 'infer_stack']
    scheduled = subprocess.run(command + ['catalog', 'endpoint', 'provenance', 'local'],
                               env=env, text=True, capture_output=True, check=True)
    expected = json.loads(scheduled.stdout)
    captured = tmp_path / 'captured.json'
    child = (
        'import json,os; from pathlib import Path; '
        'p=Path(os.environ["INFER_STACK_ENDPOINT_DESCRIPTOR"]); '
        f'Path({str(captured)!r}).write_text(json.dumps({{"path": str(p), '
        '"lease_id": os.environ["INFER_STACK_LEASE_ID"], "descriptor": json.loads(p.read_text())})); '
        f'raise SystemExit({child_status})'
    )
    ledger = tmp_path / 'ledger.db'
    process = subprocess.run(command + ['run', '--endpoint', 'local', '--ledger', str(ledger),
                                      '--', sys.executable, '-c', child],
                             env=env, text=True, capture_output=True)
    assert process.returncode == child_status, process.stdout + process.stderr
    value = json.loads(captured.read_text())
    assert value['descriptor']['serving_provenance']['local'] == expected
    assert value['descriptor']['lease_id'] == value['lease_id']
    assert 'private-fixture-value' not in captured.read_text()
    assert not Path(value['path']).exists()
    with sqlite3.connect(ledger) as db:
        assert db.execute('select state from leases').fetchall() == [('released',)]
