"""Tests for the serving catalog parser and its handoff to the ledger."""

from __future__ import annotations

import pytest

from infer_stack.leasing import (
    Catalog,
    CatalogError,
    Ledger,
    Sharing,
    SqliteStore,
)

SAMPLE = {
    'models': {
        'qwen-coder-32b': {
            'source': 'hf://Qwen/Qwen2.5-Coder-32B-Instruct',
            'revision': 'main',
        },
        'llama-small': {'source': 'hf://meta-llama/Llama-3.2-3B-Instruct'},
    },
    'endpoints': {
        'qwen-coder': {
            'model': 'qwen-coder-32b',
            'engine': 'vllm',
            'runtime': {'tensor_parallel_size': 1, 'max_model_len': 32768},
            'sharing': {'mode': 'shared-compatible'},
            'reclaim': {'policy': 'keep-warm'},
        },
        # an alias of the same model + runtime: should coalesce with qwen-coder
        'qwen-coder-alias': {
            'model': 'qwen-coder-32b',
            'engine': 'vllm',
            'runtime': {'tensor_parallel_size': 1, 'max_model_len': 32768},
            'public_name': 'qwen-coder',
        },
        'draft-model': {'model': 'llama-small', 'engine': 'vllm'},
        'verifier-model': {
            'model': 'qwen-coder-32b',
            'engine': 'vllm',
            'runtime': {'tensor_parallel_size': 2},
        },
        'qwen-small': {
            'engine': 'ollama',
            'host': 'local-ollama',
            'model': 'qwen3.5:4b',
        },
        'smollm': {
            'engine': 'ollama',
            'host': 'local-ollama',
            'model': 'smollm2:135m',
        },
    },
    'runtime_hosts': {
        'local-ollama': {
            'engine': 'ollama',
            'placement': {'gpu_indices': [1]},
            'settings': {'keep_alive': '2m', 'max_loaded_models': 2},
            'storage': {'model_store': 'shared-ollama-store'},
        },
    },
    'bundles': {
        'draft-and-verify': ['draft-model', 'verifier-model'],
        'local-small-models': ['qwen-small', 'smollm'],
    },
}


@pytest.fixture
def catalog():
    return Catalog.from_dict(SAMPLE)


def test_resolve_vllm_endpoint(catalog):
    req = catalog.resolve_endpoint('qwen-coder')
    assert req.engine == 'vllm'
    assert req.capacity == {'max_model_len': 32768}
    assert req.sharing == Sharing.SHARED
    assert req.served['served_model_name'] == 'qwen-coder'
    assert req.spec['hf_model_id'] == 'Qwen/Qwen2.5-Coder-32B-Instruct'


def test_resolve_ollama_endpoint(catalog):
    req = catalog.resolve_endpoint('qwen-small')
    assert req.engine == 'ollama'
    assert req.host == 'local-ollama'
    assert req.served == {'model': 'qwen3.5:4b'}
    assert req.capacity == {}
    assert req.structural['gpu_indices'] == [1]


def test_alias_shares_compat_key(catalog):
    a = catalog.resolve_endpoint('qwen-coder')
    b = catalog.resolve_endpoint('qwen-coder-alias')
    # different endpoint names, same model+runtime -> same deployment identity
    assert a.endpoint != b.endpoint
    assert a.compat_key == b.compat_key


def test_runtime_difference_splits_compat_key(catalog):
    a = catalog.resolve_endpoint('qwen-coder')        # tp=1
    b = catalog.resolve_endpoint('verifier-model')    # tp=2, same model
    assert a.compat_key != b.compat_key


def test_sharing_override(catalog):
    req = catalog.resolve_endpoint('qwen-coder', sharing=Sharing.DEDICATED)
    assert req.sharing == Sharing.DEDICATED


def test_resolve_names_expands_bundles_and_dedups(catalog):
    reqs = catalog.resolve_names(['draft-and-verify', 'draft-model'])
    names = [r.endpoint for r in reqs]
    assert names == ['draft-model', 'verifier-model']   # dedup, order kept


@pytest.mark.parametrize(
    'mutation, needle',
    [
        ({'endpoints': {'x': {'engine': 'vllm', 'model': 'nope'}}}, 'unknown model'),
        ({'endpoints': {'x': {'engine': 'ollama', 'model': 't', 'host': 'no'}}}, 'unknown host'),
        ({'endpoints': {'x': {'engine': 'warp', 'model': 'm'}}, 'models': {'m': {'source': 's'}}}, 'unknown engine'),
        ({'endpoints': {'x': {'engine': 'vllm'}}, 'models': {}}, "needs a 'model'"),
        ({'bundles': {'b': ['ghost']}}, 'unknown endpoint'),
    ],
)
def test_validation_errors(mutation, needle):
    with pytest.raises(CatalogError) as exc:
        Catalog.from_dict(mutation)
    assert needle in str(exc.value)


def test_load_from_yaml(tmp_path):
    import yaml

    path = tmp_path / 'catalog.yaml'
    path.write_text(yaml.safe_dump(SAMPLE))
    catalog = Catalog.load(path)
    assert set(catalog.endpoints) >= {'qwen-coder', 'qwen-small'}


# -- integration with the ledger ------------------------------------------


def test_catalog_to_ledger_coalesces(catalog):
    ledger = Ledger(SqliteStore(':memory:'))
    a = ledger.acquire('alice', catalog.resolve_names(['qwen-coder']))
    b = ledger.acquire('bob', catalog.resolve_names(['qwen-coder-alias']))
    # alias resolves to the same deployment identity -> one group, demand 2
    assert a.groups[0].id == b.groups[0].id
    assert ledger.get_group(a.groups[0].id).demand == 2


def test_catalog_ollama_bundle_one_daemon(catalog):
    ledger = Ledger(SqliteStore(':memory:'))
    res = ledger.acquire('alice', catalog.resolve_names(['local-small-models']))
    # two tags, one daemon -> a single group serving both endpoints
    assert len(res.groups) == 1
    group = ledger.get_group(res.groups[0].id)
    assert set(group.served) == {'qwen-small', 'smollm'}


def test_catalog_bundle_distinct_models(catalog):
    ledger = Ledger(SqliteStore(':memory:'))
    res = ledger.acquire('alice', catalog.resolve_names(['draft-and-verify']))
    assert len({g.id for g in res.groups}) == 2
