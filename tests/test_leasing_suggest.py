"""Tests for the suggestion pool + the pure ``inventory × pool → catalog`` join."""

from __future__ import annotations

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Catalog
from infer_stack.leasing.suggest import (
    builtin_pool,
    derive_runtime,
    fits_on,
    suggest_catalog,
)


def _gpu(index, mem, name='GPU', display=False):
    return {'index': index, 'name': name, 'memory_gib': mem, 'display_active': display}


def test_builtin_pool_is_nonempty_and_real():
    pool = builtin_pool()
    assert pool, 'the shipped suggestion pool should not be empty'
    # the current-generation families (real, released 2026) are carried over
    families = {m.family for m in pool.values()}
    assert {'qwen3.5', 'qwen3.6', 'gemma4'} <= families
    # ...with the real Hugging Face ids, not slugs
    assert pool['qwen3.5-9b'].hf_model_id == 'Qwen/Qwen3.5-9B'
    assert pool['gemma4-31b'].hf_model_id == 'google/gemma-4-31B-it'
    # the demo's models are reproducible from the pool
    assert {'smollm2-1.7b', 'qwen2.5-0.5b'} <= set(pool)


def test_rtx_3090_suggests_the_current_gen_models_that_fit():
    # A single 24 GiB RTX 3090: the current-gen models that fit a 24 GiB card
    # should be suggested; the ones needing a bigger/second GPU should not.
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 24, name='NVIDIA GeForce RTX 3090')]}
    models = suggest_catalog(inv)['models']
    fits = {'qwen3.5-0.8b', 'qwen3.5-2b', 'qwen3.5-4b', 'qwen3.5-9b',
            'qwen3.6-35b-a3b-fp8', 'gemma4-e2b', 'gemma4-e4b', 'gemma4-26b',
            'gemma4-31b'}
    too_big = {'qwen3.5-27b', 'qwen3.5-35b-a3b', 'qwen3.5-122b-a10b',
               'qwen3.6-35b-a3b'}  # 35b-a3b needs 2 GPUs even at 24 GiB each
    assert fits <= set(models)
    assert too_big.isdisjoint(models)


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
    assert 'gpt-oss-20b' not in one_small    # 40 GiB model does not
    assert 'qwen2.5-72b' not in one_small    # needs two GPUs


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
