"""Tests for the suggestion pool + the pure ``inventory × pool → catalog`` join."""

from __future__ import annotations

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Catalog
from infer_stack.leasing.suggest import (
    builtin_pool,
    derive_runtime,
    fits_on,
    migrate_known_suggestion_aliases,
    suggest_catalog,
)


def _gpu(index, mem, name='GPU', display=False):
    return {'index': index, 'name': name, 'memory_gib': mem, 'display_active': display}


def test_builtin_pool_is_nonempty_and_real():
    pool = builtin_pool()
    assert pool, 'the shipped suggestion pool should not be empty'
    # the shipped Qwen/Gemma families are represented in the curated pool
    families = {m.family for m in pool.values()}
    assert {'qwen3.8', 'qwen3.5', 'qwen3.6', 'gemma4'} <= families
    # ...with the real Hugging Face ids, not slugs
    assert pool['qwen3.5-9b'].hf_model_id == 'Qwen/Qwen3.5-9B'
    # The generic name is intentionally not reused for a third-party quantized
    # derivative: it remains available for the official Qwen/Qwen3.8-27B.
    assert 'qwen3.8-27b' not in pool
    qwen38 = pool['qwen3.8-27b-dbirks-hyperqwen']
    assert qwen38.hf_model_id == 'dbirks/Qwen3.8-27B-W4A16-AutoRound'
    assert qwen38.family == 'qwen3.8'
    assert qwen38.memory_class_gib == 20
    assert qwen38.min_vram_gib_per_replica == 24
    assert qwen38.context_window == 262144
    assert qwen38.gpu_name_hints == []
    assert qwen38.requires_ampere is True
    assert qwen38.defaults['max_model_len'] == 65536
    assert qwen38.defaults['gpu_memory_utilization'] == 0.93
    assert qwen38.defaults['env']['CTX'] == 'fast'
    assert 'serve_recipe' not in qwen38.defaults       # generic fields only
    assert qwen38.defaults['command'] == ['single']
    assert set(qwen38.endpoint_variants) == {'long', 'huge'}
    assert qwen38.endpoint_variants['long']['runtime']['max_model_len'] == 150000
    assert qwen38.endpoint_variants['huge']['runtime']['max_model_len'] == 245760
    assert qwen38.defaults['image'] == 'ghcr.io/syv-ai/hyperqwen:sha-684e927'
    assert pool['gemma4-31b'].hf_model_id == 'google/gemma-4-31B-it'
    # the demo's models are reproducible from the pool
    assert {'smollm2-1.7b', 'qwen2.5-0.5b'} <= set(pool)


def test_derive_runtime_never_sizes_below_pool_default():
    # Regression: on a big GPU the bare footprint ratio for a small model is well
    # below the pool's hand-tuned gpu_memory_utilization (sized for its KV cache
    # at its context). Sizing below it starved the KV cache and vLLM OOM'd at
    # startup. The computed util must only *raise* the default.
    pool = builtin_pool()
    model = pool['smollm2-1.7b']
    rt = derive_runtime(model, [_gpu(0, 24)])
    assert rt['gpu_memory_utilization'] >= model.defaults['gpu_memory_utilization']
    # a smaller GPU may need a bigger slice — the footprint estimate can raise it
    rt_small = derive_runtime(model, [_gpu(0, 8)])
    assert rt_small['gpu_memory_utilization'] >= rt['gpu_memory_utilization']


def test_rtx_3090_suggests_the_current_gen_models_that_fit():
    # A single 24 GiB RTX 3090: the current-gen models that fit a 24 GiB card
    # should be suggested; the ones needing a bigger/second GPU should not.
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 24, name='NVIDIA GeForce RTX 3090')]}
    models = suggest_catalog(inv)['models']
    fits = {'qwen3.8-27b-dbirks-hyperqwen', 'qwen3.5-0.8b', 'qwen3.5-2b',
            'qwen3.5-4b', 'qwen3.5-9b',
            'qwen3.6-35b-a3b-fp8', 'gemma4-e2b', 'gemma4-e4b', 'gemma4-26b',
            'gemma4-31b'}
    too_big = {'qwen3.5-27b', 'qwen3.5-35b-a3b', 'qwen3.5-122b-a10b',
               'qwen3.6-35b-a3b'}  # 35b-a3b needs 2 GPUs even at 24 GiB each
    assert fits <= set(models)
    assert too_big.isdisjoint(models)


def test_rtx_3090_adds_explicit_hyperqwen_context_variants():
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 24, name='NVIDIA GeForce RTX 3090')]}
    out = suggest_catalog(inv)
    base = 'qwen3.8-27b-dbirks-hyperqwen'
    assert {base, f'{base}-long', f'{base}-huge'} <= set(out['endpoints'])
    # One model identity, three explicit ways to serve it.
    assert set(out['models']) & {f'{base}-long', f'{base}-huge'} == set()
    assert out['endpoints'][f'{base}-long']['model'] == base
    assert out['endpoints'][f'{base}-huge']['model'] == base

    fast = out['endpoints'][base]['runtime']
    long = out['endpoints'][f'{base}-long']['runtime']
    huge = out['endpoints'][f'{base}-huge']['runtime']
    assert (fast['max_model_len'], fast['env']['SPEC'], fast['env']['CTX']) == (
        65536, 'dflash2', 'fast')
    assert (long['max_model_len'], long['env']['SPEC'], long['env']['CTX']) == (
        150000, 'mtp', 'long')
    assert (huge['max_model_len'], huge['env']['SPEC'], huge['env']['CTX']) == (
        245760, 'dflash2', 'huge')
    # The variant inherits the generic launcher contract rather than repeating a
    # model-specific recipe in Python. The hardware gate becomes an exact pin so
    # a later best-fit placement cannot silently move the measured profile.
    for name in (f'{base}-long', f'{base}-huge'):
        ep = out['endpoints'][name]
        assert ep['runtime']['command'] == ['single']
        assert ep['runtime']['env']['MAX_LEN'] == '{max_model_len}'
        assert ep['runtime']['mounts']['/cache'] == 'hyperqwen/qwen3.8-27b/cache'
        assert ep['placement']['gpu_indices'] == [0]
        assert ep['reclaim']['policy'] == 'stop'


def test_hyperqwen_context_variants_are_not_injected_on_other_ampere_cards():
    # The base HyperQwen endpoint is portable to other supported >=24 GiB Ampere
    # cards, but the extra long/huge suggestions are intentionally tied to the
    # reference card whose profiles were measured. A user can still author the
    # same generic runtime data explicitly elsewhere.
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 48, name='NVIDIA A40')]}
    endpoints = suggest_catalog(inv)['endpoints']
    base = 'qwen3.8-27b-dbirks-hyperqwen'
    assert base in endpoints
    assert f'{base}-long' not in endpoints
    assert f'{base}-huge' not in endpoints


def test_fits_on_respects_vram_and_gpu_count():
    pool = builtin_pool()
    big = pool['qwen2.5-72b']            # needs 2 GPUs, 72 GiB each
    assert not fits_on(big, [_gpu(0, 80)])               # one GPU: no
    assert fits_on(big, [_gpu(0, 80), _gpu(1, 80)])      # two: yes
    assert not fits_on(big, [_gpu(0, 48), _gpu(1, 48)])  # too small per GPU


def test_suggested_catalog_roundtrips_through_catalog():
    out = suggest_catalog(simulate_inventory('2x48'))
    # The fragment is shaped like a catalog.yaml and parses/cross-refs cleanly.
    cat = Catalog.from_dict(out)
    assert cat.models and cat.endpoints
    for ep in cat.endpoints.values():
        assert ep.model in cat.models


def test_fit_filter_tracks_gpu_size():
    one_small = suggest_catalog(simulate_inventory('1x16'))['models']
    assert 'qwen2.5-7b' in one_small        # 16 GiB model fits a 16 GiB GPU
    assert 'qwen3.8-27b-dbirks-hyperqwen' not in one_small   # 24 GiB serving floor
    assert 'gpt-oss-20b' not in one_small    # 40 GiB model does not
    assert 'qwen2.5-72b' not in one_small    # needs two GPUs


def test_qwen38_27b_suggestion_uses_the_hyperqwen_profile_on_any_card_that_fits():
    inv = {'gpu_count': 2, 'gpus': [
        _gpu(0, 96, name='NVIDIA RTX PRO 6000 Blackwell Workstation Edition'),
        _gpu(3, 24, name='NVIDIA GeForce RTX 3090'),
    ]}
    out = suggest_catalog(inv)
    assert out['models']['qwen3.8-27b-dbirks-hyperqwen']['source'] == (
        'hf://dbirks/Qwen3.8-27B-W4A16-AutoRound'
    )
    ep = out['endpoints']['qwen3.8-27b-dbirks-hyperqwen']
    # Fit decides, not the card's name: no GPU pin, the placer chooses.
    assert ep['placement'] == {'min_vram_gib': 24}
    assert ep['runtime'] == {
        'max_model_len': 65536,
        'gpu_memory_utilization': 0.93,
        'enable_prefix_caching': True,
        'image': 'ghcr.io/syv-ai/hyperqwen:sha-684e927',
        'command': ['single'],
        'env': {'PORT': '{port}', 'SPEC': 'dflash2', 'CTX': 'fast',
                'PREFIX_CACHE': 1, 'MAX_LEN': '{max_model_len}',
                'GPU_UTIL': '{gpu_memory_utilization}',
                'EXTRA_ARGS': '--served-model-name={served_model_name}'},
        'mounts': {'/app/models': 'hyperqwen/qwen3.8-27b/models',
                   '/cache': 'hyperqwen/qwen3.8-27b/cache'},
    }


def test_qwen38_27b_profile_is_suggested_wherever_it_fits():
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 96, name='NVIDIA RTX PRO 6000 Blackwell')]}
    assert 'qwen3.8-27b-dbirks-hyperqwen' in suggest_catalog(inv)['models']


def test_qwen38_27b_profile_is_not_suggested_before_ampere():
    # 48 GiB is plenty, but a Turing card cannot run the recipe's image.
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 48, name='Quadro RTX 8000')]}
    assert 'qwen3.8-27b-dbirks-hyperqwen' not in suggest_catalog(inv)['models']


def test_old_dbirks_qwen38_suggestion_name_migrates_without_touching_official():
    old = {
        'models': {
            'qwen3.8-27b': {
                'source': 'hf://dbirks/Qwen3.8-27B-W4A16-AutoRound',
            },
        },
        'endpoints': {
            'qwen3.8-27b': {
                'engine': 'vllm',
                'model': 'qwen3.8-27b',
                'runtime': {'serve_recipe': 'hyperqwen-3090-single'},
                'placement': {'gpu_indices': [1]},
            },
        },
    }
    renamed = migrate_known_suggestion_aliases(old)
    assert renamed
    new = 'qwen3.8-27b-dbirks-hyperqwen'
    assert set(old['models']) == {new}
    assert set(old['endpoints']) == {new}
    assert old['endpoints'][new]['model'] == new
    assert old['endpoints'][new]['placement'] == {'gpu_indices': [1]}

    official = {
        'models': {'qwen3.8-27b': {'source': 'hf://Qwen/Qwen3.8-27B'}},
        'endpoints': {
            'qwen3.8-27b': {
                'engine': 'vllm',
                'model': 'qwen3.8-27b',
            },
        },
    }
    assert migrate_known_suggestion_aliases(official) == []
    assert 'qwen3.8-27b' in official['models']


def test_derive_runtime_clamps_len_and_sizes_utilization():
    pool = builtin_pool()
    # a tiny model on a big GPU should not greedily claim it (low utilization)...
    rt_small = derive_runtime(pool['smollm2-135m'], [_gpu(0, 48)])
    assert rt_small['gpu_memory_utilization'] <= 0.3
    assert rt_small['max_model_len'] <= pool['smollm2-135m'].context_window
    # ...and a snug model claims most of it.
    rt_big = derive_runtime(pool['qwen2.5-7b'], [_gpu(0, 16)])
    assert rt_big['gpu_memory_utilization'] >= 0.8


def test_pre_ampere_gpu_pins_fp16():
    pool = builtin_pool()
    turing = [_gpu(0, 48, name='Quadro RTX 8000')]
    ampere = [_gpu(0, 48, name='NVIDIA A40')]
    assert derive_runtime(pool['qwen2.5-7b'], turing).get('extra_args') == ['--dtype=half']
    assert 'extra_args' not in derive_runtime(pool['qwen2.5-7b'], ampere)


def test_tensor_parallel_for_multi_gpu_model():
    pool = builtin_pool()
    rt = derive_runtime(pool['qwen2.5-72b'], [_gpu(0, 80), _gpu(1, 80)])
    assert rt['tensor_parallel_size'] == 2


def test_largest_fitting_model_is_kept_warm():
    out = suggest_catalog(simulate_inventory('1x48'))
    warm = [n for n, e in out['endpoints'].items()
            if e['reclaim']['policy'] == 'keep-warm']
    assert len(warm) == 1
    others = [e['reclaim']['policy'] for n, e in out['endpoints'].items()
              if n not in warm]
    assert set(others) == {'stop'}


def test_display_gpu_reservation_shrinks_the_pool():
    inv = {'gpu_count': 2, 'gpus': [
        _gpu(0, 80, name='A100'),
        _gpu(1, 80, name='A100', display=True),
    ]}
    reserved = suggest_catalog(inv, reserve_display_gpu='auto')['models']
    used_all = suggest_catalog(inv, reserve_display_gpu=False)['models']
    assert 'qwen2.5-72b' not in reserved     # only 1 usable GPU -> no 2-GPU model
    assert 'qwen2.5-72b' in used_all         # both GPUs -> it fits


def test_empty_inventory_yields_empty_suggestion():
    out = suggest_catalog({'gpu_count': 0, 'gpus': []})
    assert out == {'models': {}, 'endpoints': {}}
