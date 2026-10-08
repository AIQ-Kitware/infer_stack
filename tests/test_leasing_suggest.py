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


def _gpu(index, mem, name='GPU', display=False, compute_cap=None):
    gpu = {'index': index, 'name': name, 'memory_gib': mem, 'display_active': display}
    if compute_cap is not None:
        gpu['compute_cap'] = compute_cap
    return gpu


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
    assert qwen38.min_vram_gib_per_replica == 23.9
    assert qwen38.context_window == 262144
    assert qwen38.gpu_name_hints == []
    assert qwen38.requires_ampere is False
    assert qwen38.default_hardware == {'min_compute_cap': 8.0}
    assert qwen38.defaults['max_model_len'] == 65536
    assert qwen38.defaults['gpu_memory_utilization'] == 0.93
    assert qwen38.defaults['env']['CTX'] == 'fast'
    assert 'serve_recipe' not in qwen38.defaults       # generic fields only
    assert qwen38.defaults['command'] == ['single']
    assert set(qwen38.endpoint_variants) == {
        'long', 'huge', 'full', 'turing-fast-full', 'turing-full'
    }
    assert qwen38.endpoint_variants['long']['runtime']['max_model_len'] == 150000
    assert qwen38.endpoint_variants['huge']['runtime']['max_model_len'] == 245760
    assert qwen38.defaults['image'] == 'ghcr.io/syv-ai/hyperqwen:sha-53557bc'
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
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 24, name='NVIDIA GeForce RTX 3090', compute_cap=8.6)]}
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
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 24, name='NVIDIA GeForce RTX 3090', compute_cap=8.6)]}
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
    # The variants inherit the generic launcher contract rather than repeating
    # a model-specific recipe in Python.  A homogeneous host needs no arbitrary
    # exact pin: the min-VRAM placement is sufficient for this capability class.
    for name in (f'{base}-long', f'{base}-huge'):
        ep = out['endpoints'][name]
        assert ep['runtime']['command'] == ['single']
        assert ep['runtime']['env']['MAX_LEN'] == '{max_model_len}'
        assert ep['runtime']['mounts']['/cache'] == 'hyperqwen/qwen3.8-27b/cache'
        assert ep['placement'] == {'min_vram_gib': 23.9}
        assert ep['reclaim']['policy'] == 'stop'


def test_rtx_a6000_gets_fast_and_full_context_hyperqwen_profiles():
    # RTX A6000 is an sm86 Ampere card with 48 GiB.  It should follow the
    # roomy-Ampere policy rather than requiring an exact product-name gate.
    base = 'qwen3.8-27b-dbirks-hyperqwen'
    inv = {'gpu_count': 1, 'gpus': [
        _gpu(0, 47.99, name='NVIDIA RTX A6000', compute_cap=8.6),
    ]}
    endpoints = suggest_catalog(inv)['endpoints']

    assert {base, f'{base}-full'} <= set(endpoints)
    assert f'{base}-long' not in endpoints
    assert f'{base}-huge' not in endpoints

    full = endpoints[f'{base}-full']
    assert full['placement'] == {'min_vram_gib': 47.9}
    assert full['runtime']['max_model_len'] == 262144
    assert full['runtime']['command'] == ['batch']
    assert '--dtype=bfloat16' in full['runtime']['env']['EXTRA_ARGS']
    assert '--kv-cache-dtype=auto' in full['runtime']['env']['EXTRA_ARGS']


def test_hyperqwen_variants_follow_capability_and_vram_classes():
    base = 'qwen3.8-27b-dbirks-hyperqwen'

    # A non-3090 24 GiB Ampere card gets the same memory-tight context choices:
    # the class is what matters, not an exact product string.
    small = {'gpu_count': 1, 'gpus': [
        _gpu(0, 23.99, name='NVIDIA RTX A5000', compute_cap=8.6),
    ]}
    small_eps = suggest_catalog(small)['endpoints']
    assert {base, f'{base}-long', f'{base}-huge'} <= set(small_eps)
    assert f'{base}-full' not in small_eps

    # A roomy Ampere card keeps the speed-first baseline and gains an ordinary-
    # KV full-context profile; the compressed 24-GiB long/huge profiles are not
    # cluttering this class.
    roomy = {'gpu_count': 1, 'gpus': [
        _gpu(0, 48, name='NVIDIA A40', compute_cap=8.6),
    ]}
    roomy_eps = suggest_catalog(roomy)['endpoints']
    assert {base, f'{base}-full'} <= set(roomy_eps)
    assert f'{base}-long' not in roomy_eps
    assert f'{base}-huge' not in roomy_eps


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
        _gpu(0, 96, name='NVIDIA RTX PRO 6000 Blackwell Workstation Edition', compute_cap=12.0),
        _gpu(3, 24, name='NVIDIA GeForce RTX 3090', compute_cap=8.6),
    ]}
    out = suggest_catalog(inv)
    assert out['models']['qwen3.8-27b-dbirks-hyperqwen']['source'] == (
        'hf://dbirks/Qwen3.8-27B-W4A16-AutoRound'
    )
    ep = out['endpoints']['qwen3.8-27b-dbirks-hyperqwen']
    # Fit decides, not the card's name: no GPU pin, the placer chooses.
    assert ep['placement'] == {'min_vram_gib': 23.9}
    assert ep['runtime'] == {
        'max_model_len': 65536,
        'gpu_memory_utilization': 0.93,
        'enable_prefix_caching': True,
        'image': 'ghcr.io/syv-ai/hyperqwen:sha-53557bc',
        'command': ['single'],
        'env': {'PORT': '{port}', 'SPEC': 'dflash2', 'CTX': 'fast',
                'PREFIX_CACHE': 1, 'MAX_LEN': '{max_model_len}',
                'GPU_UTIL': '{gpu_memory_utilization}',
                'EXTRA_ARGS': '--served-model-name={served_model_name}'},
        'mounts': {'/app/models': 'hyperqwen/qwen3.8-27b/models',
                   '/cache': 'hyperqwen/qwen3.8-27b/cache'},
    }


def test_roomy_blackwell_gets_provisional_full_context_prefab():
    base = 'qwen3.8-27b-dbirks-hyperqwen'
    inv = {'gpu_count': 4, 'gpus': [
        _gpu(i, 96, name='NVIDIA RTX PRO 6000 Blackwell', compute_cap=12.0)
        for i in range(4)
    ]}
    ep = suggest_catalog(inv)['endpoints'][f'{base}-full']
    # All four GPUs are the same eligible class, so class gating need not turn
    # into an arbitrary exact GPU pin.
    assert ep['placement'] == {'min_vram_gib': 47.9}
    rt = ep['runtime']
    assert rt['max_model_len'] == 262144
    assert rt['command'] == ['batch']
    assert rt['max_num_seqs'] == 1
    assert rt['env']['MODEL'].endswith('AutoRound-fast')
    assert '--dtype=bfloat16' in rt['env']['EXTRA_ARGS']
    assert '--kv-cache-dtype=auto' in rt['env']['EXTRA_ARGS']
    assert 'SPEC' not in rt['env']
    assert 'CTX' not in rt['env']


def test_class_gated_profile_pins_only_on_heterogeneous_host():
    base = 'qwen3.8-27b-dbirks-hyperqwen'
    inv = {'gpu_count': 2, 'gpus': [
        _gpu(0, 48, name='Quadro RTX 8000', compute_cap=7.5),
        _gpu(1, 96, name='NVIDIA RTX PRO 6000 Blackwell', compute_cap=12.0),
    ]}
    out = suggest_catalog(inv)['endpoints']
    # min_vram alone cannot express either compute-capability class here, so the
    # generated catalog preserves the suggestion-time class choice exactly.
    assert out[base]['placement']['gpu_indices'] == [1]
    assert out[f'{base}-full']['placement']['gpu_indices'] == [1]
    assert out[f'{base}-turing-fast-full']['placement']['gpu_indices'] == [0]


def test_qwen38_27b_profile_is_suggested_wherever_it_fits():
    inv = {'gpu_count': 1, 'gpus': [_gpu(0, 96, name='NVIDIA RTX PRO 6000 Blackwell', compute_cap=12.0)]}
    assert 'qwen3.8-27b-dbirks-hyperqwen' in suggest_catalog(inv)['models']


def test_qwen38_high_vram_turing_gets_measured_full_context_profiles_only():
    base = 'qwen3.8-27b-dbirks-hyperqwen'
    inv = {'gpu_count': 1, 'gpus': [
        _gpu(0, 48, name='Quadro RTX 8000', compute_cap=7.5),
    ]}
    out = suggest_catalog(inv)
    assert base in out['models']
    assert base not in out['endpoints']  # speculative default is Ampere+
    assert {f'{base}-turing-fast-full', f'{base}-turing-full'} <= set(out['endpoints'])
    fast = out['endpoints'][f'{base}-turing-fast-full']
    assert fast['runtime']['max_model_len'] == 262144
    assert fast['runtime']['command'] == ['batch']
    assert fast['runtime']['env']['MODEL'].endswith('AutoRound-fast')
    assert '--dtype=half' in fast['runtime']['env']['EXTRA_ARGS']
    assert '--attention-backend=TRITON_ATTN' in fast['runtime']['env']['EXTRA_ARGS']
    assert fast['reclaim']['policy'] in {'keep-warm', 'stop'}

    # Turing with only 24 GiB has neither the Ampere speculative path nor enough
    # room for the measured ordinary-KV full-context recipe.
    small = {'gpu_count': 1, 'gpus': [
        _gpu(0, 24, name='Quadro RTX 6000', compute_cap=7.5),
    ]}
    assert base not in suggest_catalog(small)['models']

    # Older nvidia-smi builds may omit compute_cap.  The name fallback is
    # deliberately narrow enough to distinguish measured Turing from Pascal.
    turing_without_cap = {'gpu_count': 1, 'gpus': [
        _gpu(0, 48, name='Quadro RTX 8000'),
    ]}
    assert f'{base}-turing-fast-full' in suggest_catalog(
        turing_without_cap
    )['endpoints']
    pascal_without_cap = {'gpu_count': 1, 'gpus': [
        _gpu(0, 48, name='Tesla P40'),
    ]}
    assert base not in suggest_catalog(pascal_without_cap)['models']


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


def test_a_4x96_host_gets_the_former_recipes_as_variants():
    # Queue item 8a: the pre-leasing 4 x 96 GB recipes live on as gated
    # variants, so `catalog suggest` offers their tuning where it applies.
    inv = {'gpu_count': 4, 'gpus': [_gpu(i, 96) for i in range(4)]}
    out = suggest_catalog(inv)
    eps = out['endpoints']
    wanted = {'qwen3.5-122b-a10b-tp4-128k': (4, 131072),
              'qwen3.5-122b-a10b-fp8-tp4-262k': (4, 262144),
              'qwen3.6-35b-a3b-tp2-262k': (2, 262144)}
    for name, (tp, ctx) in wanted.items():
        rt = eps[name]['runtime']
        assert (rt['tensor_parallel_size'], rt['max_model_len']) == (tp, ctx)
        assert rt['shm_size'] == '16g'
        assert rt['extra_args'] == ['--language-model-only', '--reasoning-parser', 'qwen3']
    Catalog.from_dict(out)                          # a valid catalog as suggested

    small = {'gpu_count': 2, 'gpus': [_gpu(i, 48) for i in range(2)]}
    assert not set(wanted) & set(suggest_catalog(small)['endpoints'])
