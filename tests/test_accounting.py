"""Numerical coverage and durable accounting invariants without GPUs."""
from dataclasses import replace

import pytest

from infer_stack.accounting import UsageInterval, UsageStore, energy


def interval(**kw):
    return replace(UsageInterval('one', 0, 10, 'node', 'dep', 'replica', 1,
                                 gpu_uuids=('GPU-a',), source='synthetic',
                                 samples=((0, 100), (10, 300))), **kw)


def test_trapezoids_clip_adjacent_windows_without_double_counting():
    full = energy(interval())
    left = energy(interval(end=4))
    right = energy(interval(start=4))
    assert full['energy_joules'] == 2000
    assert full['energy_wh'] == pytest.approx(2000 / 3600)
    assert left['energy_joules'] + right['energy_joules'] == full['energy_joules']
    assert full['method'] == 'measured' and full['coverage'] == 1
    assert full['lease_attributed_energy'] is None


def test_gaps_remain_unknown_or_are_explicitly_estimated():
    partial = interval(end=50, samples=((0, 100), (10, 100), (50, 100)))
    estimate = energy(partial)
    assert estimate['energy_joules'] == 1000
    assert estimate['method'] == 'measured_partial'
    assert estimate['coverage'] == .2 and not estimate['is_complete']
    mixed = energy(replace(partial, proxy_watts=200, proxy_source='device class', proxy_version='v1'))
    assert mixed['method'] == 'mixed'
    assert mixed['measured_joules'] == 1000 and mixed['estimated_joules'] == 8000
    assert mixed['energy_joules'] == 9000


def test_unknown_is_distinct_from_measured_zero():
    unknown = energy(interval(samples=()))
    zero = energy(interval(samples=((0, 0), (10, 0))))
    assert unknown['energy_joules'] is None and unknown['method'] == 'resource_time_only'
    assert zero['energy_joules'] == 0 and zero['method'] == 'measured'


def test_proxy_recomputation_retains_raw_resource_time():
    raw = interval(samples=(), gpu_count=2, gpu_uuids=())
    first = energy(replace(raw, proxy_watts=100, proxy_source='class', proxy_version='1'))
    refined = energy(replace(raw, proxy_watts=150, proxy_source='measured class', proxy_version='2', proxy_method='observed_power_proxy'))
    assert first['gpu_seconds'] == refined['gpu_seconds'] == 20
    assert refined['energy_joules'] == first['energy_joules'] * 1.5
    assert refined['method'] == 'observed_power_proxy'


def test_append_is_idempotent_and_refuses_overlap_after_reopen(tmp_path):
    path = tmp_path / 'usage.db'
    store = UsageStore(path)
    try:
        assert store.append(interval())
        assert not store.append(interval())
        with pytest.raises(ValueError, match='different immutable facts'):
            store.append(interval(end=20))
        with pytest.raises(ValueError, match='double count'):
            store.append(interval(id='two', start=9, end=20, replica='other'))
        assert store.append(interval(id='adjacent', start=10, end=20))
    finally:
        store.close()
    store = UsageStore(path)
    try:
        assert len(store.records()) == 2
        assert sum(energy(r)['gpu_seconds'] for r in store.records()) == 20
    finally:
        store.close()


@pytest.mark.parametrize('kw', [{'samples': ((0, 100), (0, 200))},
                               {'samples': ((0, -1),)}, {'max_sample_gap': 0},
                               {'end': float('nan')}, {'gpu_count': 0},
                               {'gpu_uuids': ('GPU-a', 'GPU-a')}, {'source': ''},
                               {'request_count': -1}, {'proxy_watts': 100}])
def test_invalid_facts_cannot_be_integrated(kw):
    with pytest.raises(ValueError):
        energy(interval(**kw))


def test_usage_cli_empty_is_read_only(tmp_path, capsys):
    from infer_stack.cli.commands_usage import UsageCLI
    from infer_stack.paths import data_root
    UsageCLI.main(argv=False, json=True)
    assert capsys.readouterr().out.strip() == '[]'
    assert not (data_root() / 'usage').exists()
