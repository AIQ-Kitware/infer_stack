"""Tests for VRAM requirement sources: floor, measurement parse, overlay.

Phase 3 of docs/planning/vram-aware-placement.md.
"""

from __future__ import annotations

import json

from infer_stack.leasing.vram import (
    Measurements,
    derive_min_vram_gib,
    looks_like_cuda_oom,
    measurement_key,
    measurement_key_for_spec,
    parse_vllm_memory_profile,
    weight_floor_gib,
)

# A realistic v1-engine startup excerpt (vLLM ~0.8+ gpu_worker phrasing).
V1_LOG = """
INFO 07-17 01:02:03 [model_runner.py:1024] Model loading took 18.2600 GiB and 41.3 seconds
INFO 07-17 01:02:45 [gpu_worker.py:298] Memory profiling takes 11.20 seconds.
Total non KV cache memory: 20.19GiB; torch peak memory increase: 1.42GiB;
non-torch forward increase memory: 0.51GiB; weights memory: 18.26GiB.
INFO 07-17 01:02:46 [kv_cache_utils.py:634] GPU KV cache size: 1,193,984 tokens
"""

# The older v0 worker phrasing.
V0_LOG = (
    'INFO worker.py:256 Memory profiling results: '
    'total_gpu_memory=44.53GiB, initial_memory_usage=18.31GiB, '
    'peak_torch_memory=19.73GiB, non_torch_memory=0.62GiB, '
    'kv_cache_size=17.11GiB, gpu_memory_utilization=0.85. '
    'Also: model weights take 17.89GiB.'
)


def test_parse_v1_profile():
    profile = parse_vllm_memory_profile(V1_LOG)
    assert profile['weights_gib'] == 18.26
    assert profile['non_kv_total_gib'] == 20.19
    assert profile['activation_gib'] == 1.42
    assert profile['non_torch_gib'] == 0.51


def test_parse_v0_profile():
    profile = parse_vllm_memory_profile(V0_LOG)
    assert profile['weights_gib'] == 17.89
    assert profile['non_torch_gib'] == 0.62
    assert profile['activation_gib'] == 19.73
    assert 'non_kv_total_gib' not in profile


def test_parse_uses_last_serve_after_restart():
    # A crashed-then-restarted container logs two serves; the last one
    # describes the running process.
    two_serves = V1_LOG + '\n' + V1_LOG.replace('18.26', '4.53').replace(
        '20.19', '5.80'
    )
    profile = parse_vllm_memory_profile(two_serves)
    assert profile['weights_gib'] == 4.53
    assert profile['non_kv_total_gib'] == 5.80


def test_parse_without_weights_is_none():
    assert parse_vllm_memory_profile('nothing to see here') is None


def test_derive_prefers_engine_total():
    profile = {'weights_gib': 18.26, 'non_kv_total_gib': 20.19}
    # 20.19 * 1.05 + 2.0 = 23.1995 -> ceil to 23.2
    assert derive_min_vram_gib(profile) == 23.2


def test_derive_component_sum_fallback():
    profile = {'weights_gib': 4.53, 'non_torch_gib': 0.4,
               'activation_gib': 0.7}
    # (5.63) * 1.05 + 2.0 = 7.9115 -> 8.0
    assert derive_min_vram_gib(profile) == 8.0


def test_oom_detector():
    assert looks_like_cuda_oom('... torch.OutOfMemoryError: CUDA out of memory ...')
    assert looks_like_cuda_oom(
        'ValueError: Free memory on device (10.5/15.7 GiB) on startup '
        'is less than desired GPU memory utilization'
    )
    assert not looks_like_cuda_oom('error: connection refused')


def test_weight_floor_from_hub_cache(tmp_path):
    snap = (
        tmp_path / 'hub' / 'models--Qwen--Qwen3.5-2B-Base'
        / 'snapshots' / 'abc123'
    )
    snap.mkdir(parents=True)
    (snap / 'model-00001.safetensors').write_bytes(b'x' * (3 * 1024 ** 2))
    (snap / 'model-00002.safetensors').write_bytes(b'x' * (1 * 1024 ** 2))
    (snap / 'config.json').write_bytes(b'{}')   # not a weight file
    floor = weight_floor_gib('Qwen/Qwen3.5-2B-Base', tmp_path)
    assert floor == round(4 * 1024 ** 2 / 1024 ** 3, 2)


def test_weight_floor_takes_largest_snapshot_not_sum(tmp_path):
    base = tmp_path / 'hub' / 'models--org--m' / 'snapshots'
    for rev, mib in (('old', 2), ('new', 5)):
        d = base / rev
        d.mkdir(parents=True)
        (d / 'w.safetensors').write_bytes(b'x' * (mib * 1024 ** 2))
    floor = weight_floor_gib('org/m', tmp_path)
    # max(2,5)=5 MiB, NOT 7: two revisions must not double-count.
    assert floor == round(5 * 1024 ** 2 / 1024 ** 3, 2)


def test_weight_floor_absent_is_none(tmp_path):
    assert weight_floor_gib('org/never-downloaded', tmp_path) is None
    assert weight_floor_gib(None, tmp_path) is None
    assert weight_floor_gib('org/m', None) is None


def test_measurements_roundtrip(tmp_path):
    store = Measurements(tmp_path / 'measurements.json')
    key = measurement_key(model_ref='org/m', image='vllm:v1', dtype='float16',
                          max_model_len=4096)
    assert store.get_min_vram_gib(key) is None
    store.record(key, 8.0, endpoint='e')
    assert store.get_min_vram_gib(key) == 8.0
    # a different context length is a different measurement identity
    other = measurement_key(model_ref='org/m', image='vllm:v1',
                            dtype='float16', max_model_len=8192)
    assert store.get_min_vram_gib(other) is None


def test_measurements_corrupt_file_fails_open(tmp_path):
    path = tmp_path / 'measurements.json'
    path.write_text('{ not json !!!')
    store = Measurements(path)
    assert store.get_min_vram_gib('anything') is None
    store.record('k', 4.0)   # recovers by rewriting
    assert store.get_min_vram_gib('k') == 4.0
    assert json.loads(path.read_text())['k']['min_vram_gib'] == 4.0


def test_measurement_key_for_spec():
    spec = {
        'hf_model_id': 'Qwen/Qwen3.5-2B-Base',
        'dtype': 'float16',
        'runtime': {'image': 'vllm/vllm-openai:v0.25.1',
                    'max_model_len': 4096},
    }
    assert measurement_key_for_spec(spec) == (
        'Qwen/Qwen3.5-2B-Base|vllm/vllm-openai:v0.25.1|float16|4096'
    )
