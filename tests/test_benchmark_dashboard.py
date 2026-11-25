"""Report comparisons retain workload and physical-device identity."""
from infer_stack.benchmark import dashboard, report


def observation(i, endpoint='model', gpu='GPU-A', target=1024, phase='fresh', status='ok', **metrics):
    return {'id': str(i), 'kind': 'trial', 'run_id': 'r', 'endpoint': endpoint,
            'status': status, 'phase': phase, 'target_context': target,
            'host': {'hostname': '<host>'}, 'gpus': [{'uuid': gpu, 'name': 'RTX 3090',
                 'pci.bus_id': '00000000:03:00.0', 'pci_path': [{'bdf': '0000:00:1b.0',
                 'max_link_width': '2', 'max_link_speed': '8.0 GT/s PCIe'}]}],
            'metrics': metrics, 'reason': 'headroom' if status == 'skipped' else None,
            'settings': {'generation_tokens': 512, 'workload': 'test'},
            'telemetry': [{'uuid': gpu, 'pcie.link.gen.current': '3', 'pcie.link.width.current': '2'}]}


def test_dashboard_comparisons(tmp_path):
    rows = [observation(1, endpoint='<model A>', prompt_tokens=1024, completion_tokens=512,
                        ttft_seconds=2, stream_tps_estimate=100),
            observation(2, endpoint='<model A>', target=8192, status='skipped'),
            observation(3, endpoint='model B', prompt_tokens=1024, ttft_seconds=1, stream_tps_estimate=90),
            observation(4, endpoint='model B', phase='extended-prefix', prompt_tokens=1200,
                        cached_prompt_tokens=0, ttft_seconds=2, e2e_completion_tps=30),
            observation(5, endpoint='model B', gpu='GPU-B', prompt_tokens=1024, stream_tps_estimate=110),
            observation(6, endpoint='model B', target=32768, status='failed')]
    ready = {**rows[0], 'kind': 'endpoint', 'status': 'ready', 'acquire_seconds': 21,
             'configured_context': 8192}
    summary, ids = report.summarize(rows)
    groups = dashboard.comparison_groups(summary, ids)
    assert len(groups) == 2
    assert sorted(len(g[0]) for g in groups) == [1, 2]
    result = report.write_report([*rows, ready], tmp_path).read_text()
    assert 'Extended-prefix end-to-end throughput' in result
    assert 'Unsuccessful or skipped trial groups (2)' in result
    assert '6 trial observations: 4 successful, 1 failed, 1 skipped.' in result
    assert 'GPU-A' in result and 'GPU-B' in result and 'Gen3 x2' in result
    assert '&lt;model A&gt;' in result and '<model A>' not in result
    assert '21.0' in result and '8 Ki' in result
    assert '<summary>Full trial summary (6 groups)' in result
    assert 'PCI path maxima:' in result


def test_cache_fraction_and_chart_gap():
    assert dashboard.cache_percent({'prompt_tokens': 100, 'cached_tokens': 0}) == 0
    assert dashboard.cache_percent({'prompt_tokens': 100, 'cached_tokens': None}) is None
    assert dashboard.cache_percent({'prompt_tokens': 100, 'cached_tokens': 101}) is None
    records = [observation(1, target=1024, stream_tps_estimate=100),
               observation(2, target=8192, status='skipped'),
               observation(3, target=32768, stream_tps_estimate=60)]
    rows, ids = report.summarize(records)
    html = dashboard.chart(rows, list(ids), [1024, 8192, 32768], ids, title='Example', subtitle='Test',
                           phase='fresh', field='stream_tps', unit='tokens/s')
    assert html.count('<circle ') == 2
    assert '<polyline ' not in html


def test_duplicate_endpoint_configs():
    a = observation(1)
    b = {**observation(2), 'provenance': {'digest': 'different'}}
    rows, ids = report.summarize([a, b])
    keys, _, _ = dashboard.comparison_groups(rows, ids)[0]
    assert len(keys) == 2
    assert all(k[:8] in dashboard.endpoint_label(k, keys, ids) for k in keys)
    assert dashboard.endpoint_label(keys[0], keys, ids) != dashboard.endpoint_label(keys[1], keys, ids)


def test_live_empty_report(tmp_path):
    text = report.write_report([], tmp_path, state='Running', run_id='demo').read_text()
    assert 'Awaiting trial observations' in text
    assert 'http-equiv="refresh"' in text
    assert 'Run demo: Running' in text
