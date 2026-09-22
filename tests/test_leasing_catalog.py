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


def test_resolve_model_name_points_at_its_endpoints(catalog):
    # passing a *model* name (a common slip) lists the endpoints that run it
    with pytest.raises(CatalogError) as exc:
        catalog.resolve_endpoint('qwen-coder-32b')
    msg = str(exc.value)
    assert 'is a model, not an endpoint' in msg
    # all three endpoints on that model are suggested
    for ep in ('qwen-coder', 'qwen-coder-alias', 'verifier-model'):
        assert ep in msg


def test_resolve_model_without_endpoints_suggests_adding_one():
    cat = Catalog.from_dict({'models': {'solo': {'source': 'hf://x/y'}},
                             'endpoints': {}})
    with pytest.raises(CatalogError) as exc:
        cat.resolve_endpoint('solo')
    assert 'no endpoints yet' in str(exc.value)
    assert 'catalog endpoint add --model solo' in str(exc.value)


def test_resolve_unknown_name_did_you_mean(catalog):
    with pytest.raises(CatalogError) as exc:
        catalog.resolve_endpoint('qwen-codr')       # typo of qwen-coder
    assert 'did you mean' in str(exc.value)
    assert 'qwen-coder' in str(exc.value)


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
    # alias resolves to the same deployment identity -> one deployment, demand 2
    assert a.deployments[0].id == b.deployments[0].id
    assert ledger.get_deployment(a.deployments[0].id).demand == 2


def test_catalog_ollama_bundle_one_daemon(catalog):
    ledger = Ledger(SqliteStore(':memory:'))
    res = ledger.acquire('alice', catalog.resolve_names(['local-small-models']))
    # two tags, one daemon -> a single deployment serving both endpoints
    assert len(res.deployments) == 1
    deployment = ledger.get_deployment(res.deployments[0].id)
    assert set(deployment.served) == {'qwen-small', 'smollm'}


def test_catalog_bundle_distinct_models(catalog):
    ledger = Ledger(SqliteStore(':memory:'))
    res = ledger.acquire('alice', catalog.resolve_names(['draft-and-verify']))
    assert len({g.id for g in res.deployments}) == 2


def test_resolve_vllm_carries_model_knobs_into_spec():
    """Regression: model-level revision/quantization/dtype went into the compat
    key but not the spec, so the renderer could never emit them."""
    cat = Catalog.from_dict({
        'models': {
            'q-awq': {
                'source': 'hf://Qwen/Q-AWQ',
                'revision': 'v1.2',
                'quantization': 'awq',
                'dtype': 'half',
            },
        },
        'endpoints': {
            'q': {'model': 'q-awq', 'engine': 'vllm'},
        },
    })
    req = cat.resolve_endpoint('q')
    assert req.spec['revision'] == 'v1.2'
    assert req.spec['quantization'] == 'awq'
    assert req.spec['dtype'] == 'half'
    # and they stay structural (distinct deployments per quantization)
    assert req.structural['quantization'] == 'awq'


def test_attention_backend_is_structural_and_splits_compat_key():
    """Two endpoints on the same model+runtime but different attention backends
    must be distinct deployments (the env var changes engine numerics), so their
    compat keys differ and neither coalesces onto the backend-less default."""
    cat = Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/model'}},
        'endpoints': {
            'default': {'model': 'm', 'engine': 'vllm'},
            'sdpa': {'model': 'm', 'engine': 'vllm',
                     'runtime': {'attention_backend': 'TORCH_SDPA'}},
            'flash': {'model': 'm', 'engine': 'vllm',
                      'runtime': {'attention_backend': 'FLASH_ATTN'}},
        },
    })
    default = cat.resolve_endpoint('default')
    sdpa = cat.resolve_endpoint('sdpa')
    flash = cat.resolve_endpoint('flash')
    # carried into the structural key...
    assert default.structural['attention_backend'] is None
    assert sdpa.structural['attention_backend'] == 'TORCH_SDPA'
    # ...and the runtime knob survives into the spec for rendering.
    assert sdpa.spec['runtime']['attention_backend'] == 'TORCH_SDPA'
    # ...so all three are distinct deployments (no coalescing).
    assert len({default.compat_key, sdpa.compat_key, flash.compat_key}) == 3


# ---------------------------------------------------------------------------
# VRAM-aware placement declarations (docs/planning/vram-aware-placement.md).
# ---------------------------------------------------------------------------


def _one_vllm(placement=None, engine='vllm'):
    ep = {'engine': engine, 'model': 'm'}
    if engine == 'ollama':
        ep = {'engine': 'ollama', 'model': 'tag', 'host': 'h'}
    if placement is not None:
        ep['placement'] = placement
    data = {
        'models': {'m': {'source': 'hf://org/m'}},
        'endpoints': {'e': ep},
        'runtime_hosts': {'h': {'engine': 'ollama'}},
    }
    return data


def test_placement_min_vram_reaches_resolved_spec():
    cat = Catalog.from_dict(_one_vllm({'min_vram_gib': 24}))
    req = cat.resolve_endpoint('e')
    assert req.spec['placement'] == {'min_vram_gib': 24}


def test_absent_placement_keeps_spec_byte_identical():
    # No declaration -> no 'placement' key at all, so existing catalogs
    # produce exactly the specs they produced before this feature.
    cat = Catalog.from_dict(_one_vllm())
    req = cat.resolve_endpoint('e')
    assert 'placement' not in req.spec


def test_min_vram_placement_is_not_structural():
    # Same model/runtime with different declarations still coalesces: the
    # requirement says where a deployment may LAND, not what process it is.
    a = Catalog.from_dict(_one_vllm({'min_vram_gib': 8})).resolve_endpoint('e')
    b = Catalog.from_dict(_one_vllm({'min_vram_gib': 24})).resolve_endpoint('e')
    assert a.compat_key == b.compat_key


def test_explicit_gpu_pin_reaches_spec_and_is_structural():
    auto = Catalog.from_dict(_one_vllm()).resolve_endpoint('e')
    pinned = Catalog.from_dict(
        _one_vllm({'gpu_indices': [1]})
    ).resolve_endpoint('e')
    assert pinned.spec['placement'] == {'gpu_indices': [1]}
    assert pinned.structural['gpu_indices'] == [1]
    assert auto.compat_key != pinned.compat_key
    # Auto placement deliberately omits the field so old compatibility hashes
    # do not change merely because the feature exists.
    assert 'gpu_indices' not in auto.structural


def _with_runtime(runtime):
    data = _one_vllm()
    data['endpoints']['e']['runtime'] = runtime
    return Catalog.from_dict(data).resolve_endpoint('e')


def test_every_launch_field_is_deployment_identity():
    """Two endpoints that launch differently must never share a process."""
    plain = _with_runtime({})
    keys = {plain.compat_key}
    for runtime in ({'command': ['single']},
                    {'command': ['single'], 'env': {'SPEC': 'mtp'}},
                    {'command': ['single'], 'env': {'SPEC': 'dflash2'}},
                    {'command': ['single'], 'mounts': {'/cache': 'x/cache'}},
                    {'extra_args': ['--reasoning-parser=qwen3']},
                    {'extra_args': ['--reasoning-parser=gemma4']}):
        keys.add(_with_runtime(runtime).compat_key)
    assert len(keys) == 7
    # An endpoint using none of them keeps the key it had before they existed.
    assert 'launch' not in plain.structural
    # Capacity is still not identity: a longer context can serve a shorter one.
    assert (_with_runtime({'command': ['c'], 'max_model_len': 8192}).compat_key
            == _with_runtime({'command': ['c'], 'max_model_len': 65536}).compat_key)


def test_a_legacy_serve_recipe_reads_as_the_generic_fields_it_meant():
    legacy = _with_runtime({'serve_recipe': 'hyperqwen-3090-single',
                            'enable_prefix_caching': True})
    runtime = legacy.spec['runtime']
    assert 'serve_recipe' not in runtime
    assert runtime['command'] == ['single']
    assert runtime['env']['MAX_LEN'] == '{max_model_len}'
    assert runtime['env']['PREFIX_CACHE'] == '1'
    assert runtime['mounts']['/cache'] == 'hyperqwen/qwen3.8-27b/cache'


@pytest.mark.parametrize('runtime,message', [
    ({'env': {'HF_TOKEN': 'x'}}, 'may not set HF_TOKEN'),
    ({'env': {'CUDA_VISIBLE_DEVICES': '1'}}, 'may not set CUDA_VISIBLE_DEVICES'),
    ({'env': {'BAD-NAME': '1'}}, 'not a valid environment variable name'),
    ({'env': {'A': [1]}}, 'must be a string, number or boolean'),
    ({'command': 'single'}, 'non-empty list'),
    ({'mounts': {'/c': '/etc'}}, 'subdirectory of the runtime data dir'),
    ({'mounts': {'/c': '../up'}}, 'subdirectory of the runtime data dir'),
    ({'mounts': {'c': 'x'}}, 'must be absolute'),
    ({'command': ['x'], 'extra_args': ['--seed=0']}, 'apply to the stock vLLM command'),
    ({'extra_args': ['--max-model-len=4096']}, 'repeats --max-model-len'),
    ({'extra_args': ['--served-model-name=other']}, 'repeats --served-model-name'),
])
def test_launch_fields_are_validated(runtime, message):
    data = _one_vllm()
    data['endpoints']['e']['runtime'] = runtime
    with pytest.raises(CatalogError, match=message):
        Catalog.from_dict(data)


def test_ordinary_extra_args_are_still_accepted():
    ep = _with_runtime({'extra_args': ['--reasoning-parser=qwen3', '--dtype=half']})
    assert ep.spec['runtime']['extra_args'] == ['--reasoning-parser=qwen3', '--dtype=half']


def test_unknown_vllm_serve_recipe_is_rejected():
    data = _one_vllm()
    data['endpoints']['e']['runtime'] = {'serve_recipe': 'typo-recipe'}
    with pytest.raises(CatalogError) as exc:
        Catalog.from_dict(data)
    assert 'unknown runtime.serve_recipe' in str(exc.value)


def test_gpu_pin_count_matches_runtime_parallelism():
    data = _one_vllm({'min_vram_gib': 24, 'gpu_indices': [0, 2]})
    data['endpoints']['e']['runtime'] = {'tensor_parallel_size': 2}
    req = Catalog.from_dict(data).resolve_endpoint('e')
    assert req.spec['placement'] == {
        'min_vram_gib': 24,
        'gpu_indices': [0, 2],
    }


@pytest.mark.parametrize(
    'placement, needle',
    [
        ({'min_vram_gib': -1}, 'positive number'),
        ({'min_vram_gib': 0}, 'positive number'),
        ({'min_vram_gib': 'lots'}, 'positive number'),
        ({'min_vram_gib': True}, 'positive number'),
        ({'min_vram_gb': 24}, 'unknown placement key'),   # the typo case
        ({'gpu_indices': []}, 'non-empty list'),
        ({'gpu_indices': [-1]}, 'non-negative integers'),
        ({'gpu_indices': [True]}, 'non-negative integers'),
        ({'gpu_indices': [1, 1]}, 'duplicates'),
    ],
)
def test_placement_validation_errors(placement, needle):
    with pytest.raises(CatalogError) as exc:
        Catalog.from_dict(_one_vllm(placement))
    assert needle in str(exc.value)


def test_placement_rejected_on_ollama_endpoints():
    with pytest.raises(CatalogError) as exc:
        Catalog.from_dict(_one_vllm({'min_vram_gib': 8}, engine='ollama'))
    assert 'only supported on vllm' in str(exc.value)


def test_gpu_pin_count_mismatch_is_rejected():
    data = _one_vllm({'gpu_indices': [0]})
    data['endpoints']['e']['runtime'] = {'tensor_parallel_size': 2}
    with pytest.raises(CatalogError) as exc:
        Catalog.from_dict(data)
    assert 'requires exactly 2' in str(exc.value)
