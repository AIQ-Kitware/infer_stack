"""High-value benchmark invariants; no GPU or network required."""
import json
from types import SimpleNamespace as NS

import pytest

from infer_stack.benchmark import hardware, ledger, report, workload


def test_prefix_workload():
    assert workload.prompt(32, 1).startswith(workload.prompt(16, 1))
    assert not workload.prompt(16, 2).startswith(workload.prompt(16, 1))


def test_stream_counts_and_reasoning(monkeypatch):
    events = [
        {'choices': [{'delta': {'role': 'assistant'}}]},
        {'choices': [{'delta': {'reasoning_content': 'abc'}}]},
        {'choices': [{'delta': {'content': 'def'}}]},
        {'choices': [{'delta': {}, 'finish_reason': 'length'}]},
        {'choices': [], 'usage': {'prompt_tokens': 123, 'completion_tokens': 8,
                                   'completion_tokens_details': {'reasoning_tokens': 4},
                                   'prompt_tokens_details': {'cached_tokens': 16}}},
    ]
    times = iter(range(100))
    monkeypatch.setattr(workload.time, 'monotonic', lambda: next(times))
    class Response:
        ok = True
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def iter_lines(self, **kwargs):
            for event in events:
                yield b'data: ' + json.dumps(event).encode()
            yield b'data: [DONE]'
    def post(url, **kwargs):
        assert url.endswith('/chat/completions')
        assert kwargs['json']['stream_options']['include_usage']
        assert kwargs['json']['reasoning_effort'] == 'high'
        return Response()
    monkeypatch.setattr(workload.requests, 'post', post)
    got = workload.request('http://localhost/v1', 'key', 'endpoint', 'prompt', effort='high')
    assert got['prompt_tokens'] == 123
    assert got['completion_tokens'] == 8
    assert got['reasoning_tokens'] == 4
    assert got['cached_prompt_tokens'] == 16
    assert got['ttft_seconds'] == 2
    assert got['stream_tps_estimate'] == 7


def test_pci_bridges(tmp_path):
    root = tmp_path / 'devices'
    root.mkdir()
    bridge = tmp_path / 'pci0000:00' / '0000:00:01.0'
    gpu = bridge / '0000:01:00.0'
    gpu.mkdir(parents=True)
    (root / gpu.name).symlink_to(gpu)
    (bridge / 'max_link_width').write_text('8\n')
    (gpu / 'max_link_width').write_text('16\n')
    got = hardware.pci_path('00000000:01:00.0', root=root)
    assert [(p['bdf'], p['max_link_width']) for p in got] == [(bridge.name, '8'), (gpu.name, '16')]


def observation(id, gpu_uuid='GPU-A', bus='00000000:01:00.0', image='sha256:a'):
    return {'schema': ledger.SCHEMA, 'id': id, 'run_id': 'run', 'kind': 'trial',
            'endpoint': 'test', 'status': 'ok', 'target_context': 1024, 'phase': 'fresh',
            'host': {'hostname': 'host'}, 'gpus': [{'uuid': gpu_uuid, 'name': 'RTX 3090', 'pci.bus_id': bus}],
            'runtime_image': {'image_id': image}, 'metrics': {'prompt_tokens': 1010, 'stream_tps_estimate': 100}}


def test_ledger_merge_and_hardware_groups(tmp_path):
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    first = observation('1')
    ledger.append(a, first)
    original = a.read_bytes()
    ledger.append(a, observation('2', 'GPU-B', '00000000:02:00.0'))
    assert a.read_bytes().startswith(original)
    ledger.append(b, first)
    ledger.append(b, observation('3', image='sha256:b'))
    got = ledger.read([a, b])
    assert len(got) == 3
    rows, ids = report.summarize(got)
    assert len(rows) == len(ids) == 3  # same GPU model is not the same physical device
    html = report.write_report(got, tmp_path / 'report')
    assert html.exists()
    assert 'GPU-B' in html.read_text()
    assert (html.parent / 'summary.csv').exists()
    ledger.append(b, observation('1', 'GPU-B'))
    with pytest.raises(ValueError, match='conflicting observation'):
        ledger.read([a, b])


def test_refuse_torn_tail(tmp_path):
    path = tmp_path / 'broken.jsonl'
    path.write_text('{')
    with pytest.raises(ValueError, match='incomplete last record'):
        ledger.append(path, observation('1'))
    assert path.read_text() == '{'


def test_lease_cleanup_and_serial_order(tmp_path, monkeypatch):
    pytest.importorskip('ubelt')
    from infer_stack.benchmark import runner
    calls = []
    class Controller:
        backend = NS(inventory={}, connection_info=lambda: True)
        ledger = NS(status=lambda: ([], []))
        def acquire(self, owner, requests, **kwargs):
            name = requests[0].endpoint
            calls.append(('acquire', name))
            dep = NS(id=name, assigned_gpus=[1], demand=1, served={name: {}}, engine='vllm', spec={'runtime': {'max_model_len': 8192}})
            return NS(lease=NS(id=name), deployments=[dep], reconcile=NS(publication_pending=False), released_on_timeout=False, wait=NS(ready=True))
        def release_leases(self, ids, *, evict):
            calls.append(('release', ids[0], evict))
            return NS(reconcile=NS(publication_pending=False))
    def descriptor(controller, lease, deployments, config):
        return {'base_url': 'http://localhost/v1', 'endpoints': {lease.id: lease.id}, 'serving_provenance': {lease.id: {'digest': lease.id}}}
    monkeypatch.setattr(hardware, 'gpu_facts', lambda *a: [])
    monkeypatch.setattr(hardware, 'image_facts', lambda *a: {})
    def request(url, key, model, text, **kwargs):
        if kwargs['max_tokens'] != 1 and model == 'a':
            raise RuntimeError('HTTP 500')
        return {'prompt_tokens': 25 * len(text.splitlines()), 'completion_tokens': 1}
    path = tmp_path / 'ledger.jsonl'
    for name in ('a', 'b'):
        endpoint = NS(alias=name, managed=True, protocol='chat', to_request=lambda: NS(endpoint=name, spec={'runtime': {'max_model_len': 8192}}))
        runner.benchmark_endpoint(Controller(), endpoint, NS(timeout=10), path, 'run', {}, descriptor_for=descriptor, contexts=[1024], request_fn=request)
    assert calls == [('acquire', 'a'), ('release', 'a', True), ('acquire', 'b'), ('release', 'b', True)]
    got = ledger.read([path])
    assert [r['status'] for r in got if r['kind'] == 'trial'] == ['failed', 'ok', 'ok', 'ok']
