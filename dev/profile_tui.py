"""Measure how responsive the TUI's event loop is, and what blocks it.

Runs the real app headless against a real ComposeBackend (over the tests'
fake Docker, so the numbers are the TUI's own cost, not Docker's), with a
ledger holding a few leases and deployments. A 50 ms timer on the app's own
loop measures lag: how late it fires is how late a keypress would be
handled. Scenarios:

  idle      the dashboard refreshing on its normal timer
  docker    the docker pane open on the Logs tab (service list, log stream)
  flood     a log stream emitting lines as fast as it can (an engine loading)

    python dev/profile_tui.py [--profile]   # --profile: cProfile the UI thread
"""

from __future__ import annotations

import asyncio
import cProfile
import io
import pstats
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'tests'))

from test_leasing_admission import CAT, acquire, make  # noqa: E402

from infer_stack.tui import InferStackTUI  # noqa: E402

FLOOD_LINES = 20000


class FloodProc:
    """A log process that writes FLOOD_LINES lines as fast as it can."""

    def __init__(self, n):
        self.n = n

    @property
    def stdout(self):
        for i in range(self.n):
            yield f'vllm-one  | INFO 09-24 12:00:00 loader.py:1 loading shard {i}\n'

    def terminate(self):
        pass


async def lag_probe(seconds: float, period: float = 0.05) -> list[float]:
    lags = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        await asyncio.sleep(period)
        lags.append(time.perf_counter() - t0 - period)
    return lags


def summary(name: str, lags: list[float]) -> str:
    ms = sorted(x * 1000 for x in lags)
    p95 = ms[int(len(ms) * 0.95) - 1] if ms else 0
    return (f'{name:8s} samples={len(ms):4d}  median={statistics.median(ms):7.1f} ms  '
            f'p95={p95:7.1f} ms  max={max(ms):7.1f} ms')


def main() -> None:
    do_profile = '--profile' in sys.argv
    # --docker-latency=S: each docker call takes S seconds, like a loaded host.
    latency = next((float(a.split('=', 1)[1]) for a in sys.argv
                    if a.startswith('--docker-latency=')), 0.0)
    # --tty: run on a real terminal (not headless), so drawing is measured too.
    tty = '--tty' in sys.argv
    import threading
    tmp = Path(tempfile.mkdtemp())
    ledger, ctl, docker = make(tmp)
    for name in ('one', 'two'):
        out = acquire(ctl, name)
        if name == 'two':
            ctl.release(out.lease.id)             # an idle keep-warm resident
    flood = {'on': False}
    if latency:
        real_run = ctl.backend.run

        def slow_run(args, **kw):
            time.sleep(latency)
            return real_run(args, **kw)
        ctl.backend.run = slow_run

    def proc_factory(service):
        return FloodProc(FLOOD_LINES) if flood['on'] else FloodProc(0)

    results = []
    profiler = cProfile.Profile()

    async def scenario():
        app = InferStackTUI(ctl, CAT, interval=1.0, proc_factory=proc_factory)
        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause(1.0)
            if do_profile:
                profiler.enable()
            results.append(summary('idle', await lag_probe(8)))
            app.query_one('#docker').collapsed = False
            await pilot.pause(0.5)
            results.append(summary('docker', await lag_probe(8)))
            flood['on'] = True
            app._restart_logs(app._log_service)
            t0 = time.perf_counter()
            results.append(summary('flood', await lag_probe(8)))
            results.append(f'flood: {len(app._log_lines)} of {FLOOD_LINES} lines shown '
                           f'after {time.perf_counter() - t0:.1f}s')
            results.append(f'threads alive at end: {threading.active_count()}')
            if do_profile:
                profiler.disable()

    if tty:
        # The app on the real terminal, driven by its own timers.
        app = InferStackTUI(ctl, CAT, interval=1.0, proc_factory=proc_factory)

        async def drive():
            await asyncio.sleep(1.5)
            results.append(summary('idle', await lag_probe(8)))
            app.query_one('#docker').collapsed = False
            await asyncio.sleep(0.5)
            results.append(summary('docker', await lag_probe(8)))
            flood['on'] = True
            app._restart_logs(app._log_service)
            t0 = time.perf_counter()
            results.append(summary('flood', await lag_probe(8)))
            results.append(f'flood: {len(app._log_lines)} of {FLOOD_LINES} lines shown '
                           f'after {time.perf_counter() - t0:.1f}s')
            results.append(f'threads alive at end: {threading.active_count()}')
            app.exit()

        app.call_later(lambda: asyncio.ensure_future(drive()))
        app.run()
    else:
        asyncio.run(scenario())
    for line in results:
        print(line)
    if do_profile:
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats('cumulative').print_stats(35)
        print(out.getvalue())


if __name__ == '__main__':
    main()
