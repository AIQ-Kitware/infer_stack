"""Real signals must unwind a run's lease lifecycle, independent of monitors."""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml


def states(path):
    if not path.exists():
        return []
    try:
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            return [r[0] for r in db.execute('SELECT state FROM leases')]
    except sqlite3.OperationalError:
        return []


def start_run(tmp_path, *, readiness=False):
    catalog = tmp_path / 'catalog.yaml'
    catalog.write_text(
        yaml.safe_dump(
            {
                'models': {'fixture': {'source': 'hf://org/fixture-model'}},
                'endpoints': {
                    'fixture': {'model': 'fixture', 'engine': 'vllm'}
                },
            }
        )
    )
    ledger = tmp_path / 'ledger.db'
    marker = tmp_path / 'child.pid'
    child = f'import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(120)'
    args = [
        '--backend',
        'null',
        '--endpoint',
        'fixture',
        '--catalog',
        str(catalog),
        '--ledger',
        str(ledger),
        '--interval',
        '0.1',
        '--timeout',
        '120',
        '--',
        sys.executable,
        '-c',
        child,
    ]
    setup = ''
    if readiness:
        setup = (
            'from infer_stack.leasing import MemoryBackend\n'
            'cl._make_backend = lambda *a, **kw: MemoryBackend(ready=False)\n'
            'wait_ready = cl.Controller.wait_ready\n'
            'def waiting(self, *a, **kw):\n'
            f'    from pathlib import Path; Path({str(tmp_path / "waiting")!r}).touch()\n'
            '    return wait_ready(self, *a, **kw)\n'
            'cl.Controller.wait_ready = waiting\n'
        )
    driver = (
        'from infer_stack.cli import commands_leasing as cl\n'
        + setup
        + f'raise SystemExit(cl.RunCLI.main(argv={args!r}))\n'
    )
    log = (tmp_path / 'run.log').open('w')
    proc = subprocess.Popen(
        [sys.executable, '-c', driver],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return proc, ledger, marker


def wait_until(proc, predicate, log):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if predicate():
            return
        if proc.poll() is not None:
            pytest.fail(log.read_text())
        time.sleep(0.05)
    pytest.fail('run did not reach the test phase: ' + log.read_text())


def stop_fixture(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


@pytest.mark.parametrize('sig', [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize('whole_group', [False, True])
def test_run_releases_and_stops_child_on_signal(tmp_path, sig, whole_group):
    proc, ledger, marker = start_run(tmp_path)
    try:
        wait_until(proc, marker.exists, tmp_path / 'run.log')
        assert states(ledger) == ['active']
        child_pid = int(marker.read_text())
        if whole_group:
            os.killpg(proc.pid, sig)
        else:
            proc.send_signal(sig)
        proc.wait(timeout=15)
        assert states(ledger) == ['released']
        assert proc.returncode == (
            143 if sig == signal.SIGTERM else -signal.SIGINT
        )
        assert not Path(f'/proc/{child_pid}').exists()
    finally:
        stop_fixture(proc)


def test_run_sigterm_during_readiness_releases(tmp_path):
    proc, ledger, marker = start_run(tmp_path, readiness=True)
    try:
        wait_until(proc, (tmp_path / 'waiting').exists, tmp_path / 'run.log')
        assert states(ledger) == ['active']
        assert not marker.exists()
        proc.terminate()
        proc.wait(timeout=15)
        assert states(ledger) == ['released']
        assert proc.returncode == 143
    finally:
        stop_fixture(proc)
