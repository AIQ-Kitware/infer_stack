"""External endpoints: an alias fulfilled by a server infer-stack does not run.

Campaign 2 (docs/planning/external-endpoints.md). An external endpoint is a
catalog definition with an ``external:`` target; it has no lease, deployment
or model entry, and it is reached through the front door like any other.
"""

from __future__ import annotations

import pytest
import yaml

from infer_stack.leasing import Catalog
from infer_stack.leasing.catalog import CatalogError
from infer_stack.leasing.endpoints import ExternalTarget

REMOTE = {'external': {'api_base': 'http://box:8000/v1', 'model': 'Qwen/Qwen3-32B',
                       'api_key_env': 'REMOTE_QWEN_KEY'}, 'protocol': 'chat'}


def catalog(**endpoints):
    return {'models': {'m': {'source': 'hf://org/m'}},
            'endpoints': {'local': {'engine': 'vllm', 'model': 'm'}, **endpoints},
            'bundles': {'pair': ['local', 'remote']} if 'remote' in endpoints else {}}


# -- 1. parses and round-trips semantically --------------------------------------


def test_an_external_endpoint_parses_and_round_trips():
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    resolved = cat.resolve_endpoint('remote')
    assert not resolved.managed
    assert resolved.target == ExternalTarget('http://box:8000/v1', 'Qwen/Qwen3-32B',
                                             'REMOTE_QWEN_KEY')
    again = Catalog.from_dict(yaml.safe_load(yaml.safe_dump(cat.source)))
    assert again.resolve_endpoint('remote').semantic_key() == resolved.semantic_key()
    assert 'remote' not in cat.models          # no managed model artifact


# -- 2. invalid mixtures fail clearly ----------------------------------------------


@pytest.mark.parametrize('extra, needle', [
    ({'runtime': {'max_model_len': 8}}, "'runtime' describes a runtime"),
    ({'engine': 'vllm'}, "'engine' describes a runtime"),
    ({'reclaim': 'stop'}, "'reclaim' describes a runtime"),
    ({'placement': {'gpu_indices': [0]}}, "'placement' describes a runtime"),
    ({'host': 'h'}, "'host' describes a runtime"),
])
def test_managed_only_keys_beside_external_are_refused(extra, needle):
    with pytest.raises(CatalogError, match=needle):
        Catalog.from_dict(catalog(remote={**REMOTE, **extra}))


@pytest.mark.parametrize('external, needle', [
    ({'api_base': 'box:8000', 'model': 'm'}, 'must be an http'),
    ({'api_base': 'http://box/v1'}, 'external.model is required'),
    ({'api_base': 'http://box/v1', 'model': 'm', 'api_key_env': 'not a name'},
     'environment variable name'),
    ({'api_base': 'http://box/v1', 'model': 'm', 'api_key_env': 'LITELLM_MASTER_KEY'},
     "infer-stack's own secrets"),
    ({'api_base': 'http://box/v1', 'model': 'm', 'api_key': 'sk-literal'},
     'unknown external key'),
])
def test_bad_external_blocks_are_refused(external, needle):
    with pytest.raises(CatalogError, match=needle):
        Catalog.from_dict(catalog(remote={'external': external}))


# -- acquire stays lease-only ------------------------------------------------------


def test_asking_for_a_lease_on_an_external_endpoint_points_at_access():
    cat = Catalog.from_dict(catalog(remote=REMOTE))
    with pytest.raises(CatalogError, match='does not require a lease.*infer-stack access remote'):
        cat.resolve_requests(['remote'])
    with pytest.raises(CatalogError, match='does not require a lease'):
        cat.resolve_requests(['pair'])       # a mixed bundle goes through access too
    assert [r.endpoint for r in cat.resolve_requests(['local'])] == ['local']


def test_the_cli_adds_an_external_endpoint_and_refuses_runtime_options(tmp_path, capsys):
    from infer_stack.cli.commands_catalog import EndpointAddCLI

    path = tmp_path / 'catalog.yaml'
    path.write_text('models: {}\nendpoints: {}\n')
    common = ['--catalog', str(path), 'qwen', '--external-api-base', 'http://box/v1',
              '--external-model', 'Q/Q']
    assert EndpointAddCLI.main(argv=[*common, '--external-api-key-env', 'QKEY']) == 0
    entry = yaml.safe_load(path.read_text())['endpoints']['qwen']
    assert entry == {'external': {'api_base': 'http://box/v1', 'model': 'Q/Q',
                                  'api_key_env': 'QKEY'}}
    for flags in (['--engine', 'vllm'], ['--gpu', '0'], ['--reclaim', 'stop']):
        with pytest.raises(SystemExit, match='runs nothing'):
            EndpointAddCLI.main(argv=[*common, '--force', *flags])
