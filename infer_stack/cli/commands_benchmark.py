"""Benchmark managed endpoints using the ordinary local lease lifecycle."""
from __future__ import annotations

import fnmatch
import uuid
from pathlib import Path

import kwconf as kw

from ..paths import data_root
from .commands_leasing import (_LeasingCommonMixin, _collect_names, _descriptor_for,
                               _open_controller, _requests_catalog)
from .context import _apply_path_overrides
from .options import _PathOverridesMixin


def _default_results():
    return data_root() / 'benchmark' / 'observations.jsonl'


def _select(catalog, patterns):
    """Expand catalog bundles and shell-style globs in user order, once each."""
    expanded = []
    for pattern in patterns:
        if any(c in pattern for c in '*?['):
            matches = sorted(fnmatch.filter(catalog.endpoints, pattern))
            if not matches:
                raise ValueError(f'no catalog endpoint matches {pattern!r}; '
                                 'see `infer-stack catalog endpoint list`')
            expanded.extend(matches)
        else:
            expanded.extend(catalog.expand([pattern]))
    return [catalog.resolve_endpoint(name) for name in dict.fromkeys(expanded)]


class BenchmarkRunCLI(_LeasingCommonMixin):
    """Lease, benchmark and release selected endpoints serially on this host.

    Run `infer-stack catalog endpoint list` to discover available endpoint names.
    """

    __command__ = 'run'
    names = kw.Value([], nargs='*', position=1, type=str,
                     help='Endpoint aliases, bundles, or quoted glob patterns. '
                          'See `infer-stack catalog endpoint list` for available endpoints.')
    endpoints = kw.Value(None, type=str,
                         help='Optional comma-separated endpoint aliases or glob patterns. '
                              'See `infer-stack catalog endpoint list`.')
    catalog = kw.Value(None, type=str, help='Path to catalog.yaml.')
    base_url = kw.Value('http://127.0.0.1:14042/v1', type=str)
    api_key_env = kw.Value('LITELLM_MASTER_KEY', type=str)
    contexts = kw.Value('1024,8192,32768,65536,131072,163840', type=str,
                        help='Comma-separated target prompt token counts (approximate).')
    generation_tokens = kw.Value(512, type=int, help='Maximum generated tokens per trial.')
    repeats = kw.Value(1, type=int, help='Repetitions of each context and prefix workload.')
    timeout = kw.Value(1800, type=float, help='Seconds to wait for endpoint readiness.')
    request_timeout = kw.Value(300, type=float, help='Seconds per inference request.')
    effort = kw.Value(None, type=str, help='Optional explicit reasoning_effort; otherwise the endpoint default.')
    results = kw.Value(None, type=str, help='Append-only JSONL path; defaults to the infer-stack data directory.')
    output = kw.Value(None, type=str, help='Report directory; defaults to a unique per-run directory.')
    final_report_only = kw.Value(False, isflag=True, help='Write the report only when the sweep ends; still log each result.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..benchmark import hardware, ledger, report, runner
        from ..leasing import ComposeBackend
        from ._run_lifecycle import RunCancelled, cancellation_signals
        from .remote import server_url

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if server_url(config):
            raise SystemExit('benchmark run: execute on the local Compose inference host, not a remote control client')
        try:
            contexts = list(dict.fromkeys(int(x) for x in _collect_names(config.contexts)))
            names = _collect_names(config.names) + _collect_names(config.endpoints)
            if not names:
                raise ValueError('give endpoint aliases, bundles, or quoted glob patterns; '
                                 'see `infer-stack catalog endpoint list`')
            if not contexts or min(contexts) < 1 or config.generation_tokens < 1 or config.repeats < 1:
                raise ValueError('contexts, generation tokens and repeats must be positive')
            import math
            if any(not math.isfinite(t) or t <= 0 for t in (config.timeout, config.request_timeout)):
                raise ValueError('timeouts must be finite and positive')
            controller = _open_controller(config)
            if not isinstance(controller.backend, ComposeBackend):
                raise ValueError('benchmark run currently requires a local Compose backend')
            if not controller.backend.litellm:
                raise ValueError('benchmark run currently requires the LiteLLM gateway')
            catalog = _requests_catalog(controller, config)
            endpoints = _select(catalog, names)
        except (ValueError, TypeError) as exc:
            raise SystemExit(f'benchmark run: {exc}') from None
        path = Path(config.results).expanduser() if config.results else _default_results()
        run_id = uuid.uuid4().hex
        destination = Path(config.output).expanduser() if config.output else path.parent / 'reports' / run_id
        host = hardware.host_facts()
        failures = 0
        publisher = report.LiveReport(destination, run_id, enabled=not config.final_report_only)
        print(f'Benchmark run {run_id}\nAppend-only ledger: {path}\nReport directory: {destination}', flush=True)
        state = 'Aborted'
        try:
            with cancellation_signals():
                live_path = publisher.start()
                if live_path is not None:
                    print(f'Live report: {live_path}', flush=True)
                for endpoint in endpoints:
                    failures += runner.benchmark_endpoint(
                        controller, endpoint, config, path, run_id, host, descriptor_for=_descriptor_for,
                        contexts=contexts, repeats=config.repeats, generation_tokens=config.generation_tokens,
                        request_timeout=config.request_timeout, effort=config.effort,
                        on_record=publisher.record)
                state = 'Finished with failures' if failures else 'Finished'
        except (KeyboardInterrupt, RunCancelled):
            state = 'Interrupted'
            raise
        finally:
            # Re-read the durable source of truth even after cancellation.
            # Report errors must not mask the original run/lease exception.
            try:
                records = [r for r in ledger.read([path]) if r['run_id'] == run_id] if path.exists() else []
            except (OSError, ValueError, KeyError) as exc:
                import sys
                print(f'Unable to read final benchmark ledger: {exc}', file=sys.stderr, flush=True)
                failures += 1
            else:
                final_path = publisher.finish(records, state=state)
                if final_path is not None:
                    print(f'Report: {final_path}', flush=True)
                else:
                    failures += 1
        return 1 if failures else 0


class BenchmarkReportCLI(_PathOverridesMixin):
    """Render a report offline from one or more benchmark ledgers."""

    __command__ = 'report'
    files = kw.Value([], nargs='*', position=1, type=str, help='JSONL ledgers to combine; defaults to this host ledger.')
    output = kw.Value(None, type=str, help='Report directory; defaults to benchmark/reports/combined.')
    run_id = kw.Value(None, type=str, help='Optional run ID filter.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..benchmark import ledger, report
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        paths = [Path(p).expanduser() for p in config.files] or [_default_results()]
        destination = Path(config.output).expanduser() if config.output else _default_results().parent / 'reports' / 'combined'
        try:
            records = ledger.read(paths)
            if config.run_id:
                records = [r for r in records if r['run_id'] == config.run_id]
            print(f'Report: {report.write_report(records, destination)} ({len(records)} unique observations)')
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f'benchmark report: {exc}') from None
        return 0


class BenchmarkModalCLI(kw.ModalCLI):
    """Measure inference configurations and compare historical observations."""

    __command__ = 'benchmark'
    run = BenchmarkRunCLI
    report = BenchmarkReportCLI
