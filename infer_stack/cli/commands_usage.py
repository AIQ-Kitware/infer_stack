"""Inspect durable accelerator accounting; import collector facts explicitly."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import kwconf as kw

from ..accounting import UsageInterval, UsageStore, energy
from ..paths import data_root
from .context import _apply_path_overrides
from .options import _PathOverridesMixin
from .remote import RemoteOptions


class UsageCLI(RemoteOptions, _PathOverridesMixin):
    """Show resource time/energy provenance, or import immutable collector records."""

    __command__ = 'usage'
    import_file = kw.Value(None, type=str, help='Explicitly append collector JSON records (list of UsageInterval mappings). Identical retries are safe.')
    deployment = kw.Value(None, type=str, help='Filter deployment ID.')
    node = kw.Value(None, type=str, help='Filter node identity.')
    json = kw.Value(False, isflag=True, help='Include raw facts plus energy method/coverage.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        from . import remote
        if remote.server_url(config):
            from urllib.parse import urlencode
            if config.import_file:
                raise SystemExit('Usage imports run on the controller host; remote usage is read-only')
            query = {field: getattr(config, field) for field in ['deployment', 'node'] if getattr(config, field)}
            results = remote.client(config).request('GET', '/v1/usage' + ('?' + urlencode(query) if query else ''))
            _print_results(results, machine=config.json)
            return 0
        path = data_root() / 'usage' / 'records.sqlite3'
        store = None
        try:
            if path.exists() or config.import_file:
                store = UsageStore(path, readonly=not config.import_file)
                if config.import_file:
                    rows = json.loads(Path(config.import_file).expanduser().read_text())
                    if not isinstance(rows, list):
                        raise ValueError('Collector JSON must be a list of interval records')
                    records = [UsageInterval.from_dict(row) for row in rows]
                    for record in records:
                        store.append(record)
                records = store.records()
            else:
                records = []
            records = [r for r in records if (not config.deployment or r.deployment == config.deployment)
                       and (not config.node or r.node == config.node)]
            results = [{'facts': asdict(r), 'energy': energy(r)} for r in records]
            _print_results(results, machine=config.json)
            return 0
        except (OSError, ValueError, TypeError) as ex:
            raise SystemExit(f'usage: {ex}') from ex
        finally:
            if store is not None:
                store.close()


def _print_results(results, *, machine=False):
    if machine:
        print(json.dumps(results, indent=2))
    elif not results:
        print('No usage observations recorded.')
    else:
        for row in results:
            estimate, facts = row['energy'], row['facts']
            amount = f"{estimate['energy_wh']:.3g} Wh" if estimate['energy_wh'] is not None else 'energy unavailable'
            print(f"{facts['deployment']} / {facts['replica']} on {facts['node']}: {estimate['gpu_hours']:.4g} GPU-hours; {amount}; {estimate['method']}; coverage {estimate['coverage']:.1%}")
