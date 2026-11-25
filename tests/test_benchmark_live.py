"""Live reports must not interfere with durable observations or lease cleanup."""
from pathlib import Path

import pytest

from infer_stack.benchmark import ledger, report, runner


def test_live_report_after_durable_append(tmp_path, capsys):
    path = tmp_path / 'ledger.jsonl'
    publisher = report.LiveReport(tmp_path / 'run', 'run')
    html = publisher.start()
    assert 'Run run: Running' in html.read_text()
    assert 'http-equiv="refresh"' in html.read_text()
    assert '0 trial observations' in html.read_text()

    base = runner.record_base('run', 'example', {}, {})
    seen = []

    def publish(row):
        # A browser/report failure or interruption cannot lose a completed trial.
        assert ledger.read([path])[-1] == row
        seen.append(row)
        publisher.record(row)

    runner.emit(path, base, on_record=publish, kind='trial', status='ok',
                target_context=8192, phase='fresh', metrics={
                    'prompt_tokens': 8181, 'completion_tokens': 512,
                    'ttft_seconds': 1.234, 'stream_tps_estimate': 99.9,
                    'e2e_completion_tps': 70.0})
    text = capsys.readouterr().out
    assert 'prompt 8,181 tok' in text and 'stream 99.9 tok/s' in text
    assert 'cached - tok' in text
    assert '1 trial observations: 1 successful' in html.read_text()
    assert '8181' in (html.parent / 'summary.csv').read_text()

    runner.emit(path, base, on_record=publish, kind='trial', status='failed',
                target_context=8192, phase='exact-prefix', reason='HTTP 500')
    assert 'failed (HTTP 500)' in capsys.readouterr().out
    runner.emit(path, base, on_record=publish, kind='trial', status='skipped',
                target_context=163840, phase='fresh', reason='headroom')
    assert 'skipped (headroom)' in capsys.readouterr().out
    assert '3 trial observations: 1 successful, 1 failed, 1 skipped.' in html.read_text()
    assert len(seen) == 3
    snapshot = path.read_bytes()
    publisher.finish(ledger.read([path]), state='Interrupted')
    assert 'Run run: Interrupted' in html.read_text()
    assert 'http-equiv="refresh"' not in html.read_text()
    assert path.read_bytes() == snapshot


def test_atomic_report_write_preserves_previous_snapshot(tmp_path):
    path = tmp_path / 'report.html'
    path.write_text('previous')
    with pytest.raises(RuntimeError):
        with report._atomic_text(path) as f:
            f.write('incomplete')
            assert path.read_text() == 'previous'
            raise RuntimeError('render failed')
    assert path.read_text() == 'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_report_failure_is_advisory_and_recovers(tmp_path, monkeypatch, capsys):
    real_write = report.write_report
    publisher = report.LiveReport(tmp_path / 'run', 'run')
    base = runner.record_base('run', 'example', {}, {})
    path = tmp_path / 'ledger.jsonl'

    def fail(*args, **kwargs):
        raise OSError('no space')

    monkeypatch.setattr(report, 'write_report', fail)
    assert publisher.start() is None
    for _ in range(2):
        runner.emit(path, base, on_record=publisher.record, kind='endpoint', status='ready')
    assert len(ledger.read([path])) == 2
    assert len(publisher.records) == 2
    assert capsys.readouterr().err.count('Benchmark report update failed') == 1
    monkeypatch.setattr(report, 'write_report', real_write)
    runner.emit(path, base, on_record=publisher.record, kind='release', status='ok')
    html = publisher.directory / 'report.html'
    assert '3 records' in html.read_text()
    assert publisher.error is None


def test_final_report_only(tmp_path):
    publisher = report.LiveReport(tmp_path / 'run', 'run', enabled=False)
    assert publisher.start() is None
    publisher.record({'id': 'ignored'})
    assert not publisher.directory.exists()
    html = publisher.finish([], state='Finished')
    assert html == Path(publisher.directory) / 'report.html'
    assert 'Finished' in html.read_text()
    assert 'http-equiv="refresh"' not in html.read_text()
