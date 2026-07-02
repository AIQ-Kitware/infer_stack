"""Day-2 ops on the running leasing stack, plus a holistic ``status`` command.

The ``stack`` wrappers (``logs`` / ``ps`` / ``up`` / ``down`` / …) target the
leasing Compose deployment — the project rendered under the data dir by
``acquire`` / ``apply``. ``status`` is the one-glance overview: where
everything lives, the active backend, and a leasing summary (active leases / live
deployments), with pointers to dig deeper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import scriptconfig as scfg

from ..paths import config_root, data_root, get_setting, settings_path
from .context import _apply_path_overrides
from .options import _PathOverridesMixin

# ---------------------------------------------------------------------------
# leasing compose project helpers (the target of the day-2 wrappers)
# ---------------------------------------------------------------------------


def _leasing_compose_file() -> Path:
    from ..leasing.compose import COMPOSE_FILENAME

    return data_root() / 'leasing' / 'compose' / COMPOSE_FILENAME


def _day2_compose_base(config, command: str) -> list[str]:
    """``docker compose -p <proj> -f <leasing-file>`` base for the wrappers.

    Targets the leasing Compose deployment (no ``config.yaml`` needed). Raises a
    helpful error when nothing has been deployed yet — the file is written by
    ``acquire`` / ``apply``.
    """
    from ..leasing.compose import LEASING_PROJECT

    _apply_path_overrides(config)
    compose_file = _leasing_compose_file()
    if not compose_file.exists():
        hint = ''
        if (get_setting('backend') or '') == 'kubeai':
            hint = (
                ' Note: the configured backend is kubeai — these verbs manage '
                'the docker compose stack only; inspect the cluster with '
                '`infer-stack doctor` / `kubectl -n <namespace> get models`.'
            )
        raise SystemExit(
            f'nothing deployed yet (no {compose_file}). '
            f'Bring a model up first, e.g. `infer-stack acquire <endpoint>`.'
            f'{hint}'
        )
    return [
        'docker', 'compose', '-p', LEASING_PROJECT, '-f', str(compose_file)
    ]


# ---------------------------------------------------------------------------
# status — holistic overview
# ---------------------------------------------------------------------------


def _catalog_summary(config) -> dict[str, Any] | None:
    raw = getattr(config, 'catalog', None) or (config_root() / 'catalog.yaml')
    path = Path(raw).expanduser()
    info: dict[str, Any] = {'path': str(path), 'exists': path.exists()}
    if path.exists():
        try:
            from ..leasing import Catalog

            cat = Catalog.load(path)
            info['models'] = len(cat.models)
            info['endpoints'] = len(cat.endpoints)
        except Exception:  # noqa: BLE001 - status must never crash
            info['models'] = info['endpoints'] = None
    return info


def _leasing_status() -> dict[str, Any]:
    """Read-only ledger snapshot for ``status`` (never mutates)."""
    from ..leasing import (
        DeploymentState,
        LeaseState,
        Ledger,
        SqliteStore,
        default_ledger_path,
    )

    path = default_ledger_path()
    out: dict[str, Any] = {'path': str(path), 'exists': path.exists(),
                           'leases': [], 'deployments': [], 'summary': None}
    if not path.exists():
        return out
    try:
        leases, deployments = Ledger(SqliteStore(str(path))).status()
    except Exception:  # noqa: BLE001
        return out
    active = sum(1 for le in leases if le.state == LeaseState.ACTIVE)
    live = sum(1 for d in deployments if d.state == DeploymentState.LIVE)
    out['leases'] = [
        (le.id, le.owner, str(le.state), ','.join(le.deployment_ids) or '-')
        for le in leases
    ]
    out['deployments'] = [
        (d.id, str(d.state), ','.join(sorted(d.served)) or '-', d.demand)
        for d in deployments
    ]
    out['summary'] = (active, len(leases), live, len(deployments))
    return out


def _gather_status(config) -> dict[str, Any]:
    compose_file = _leasing_compose_file()
    return {
        'backend': str(get_setting('backend') or 'null'),
        'data_dir': str(data_root()),
        'config_dir': str(config_root()),
        'configured': settings_path().exists(),
        'settings': {'path': str(settings_path()),
                     'exists': settings_path().exists()},
        'catalog': _catalog_summary(config),
        'compose': {'path': str(compose_file), 'exists': compose_file.exists()},
        'leasing': _leasing_status(),
    }


_DIG_DEEPER = (
    ('infer-stack leases', 'full lease + deployment tables'),
    ('infer-stack tui', 'live dashboard (opt-in: infer-stack[tui])'),
    ('infer-stack stack ps', 'running containers'),
    ('infer-stack logs -f', 'tail service logs'),
    ('infer-stack catalog show', 'what you can serve'),
)

_GETTING_STARTED = (
    ('infer-stack config init', 'storage + default backend'),
    ('infer-stack catalog suggest --apply', 'a catalog sized to your GPUs'),
    ('infer-stack acquire <endpoint>', 'bring a model up'),
)


def _print_status_plain(d: dict[str, Any]) -> None:
    print('infer-stack status')
    print(f'  backend:     {d["backend"]}')
    print(f'  data dir:    {d["data_dir"]}')
    print(f'  config dir:  {d["config_dir"]}')
    cat = d['catalog']
    if cat and cat['exists']:
        counts = ''
        if cat.get('models') is not None:
            counts = f'  ({cat["models"]} models, {cat["endpoints"]} endpoints)'
        print(f'  catalog:     {cat["path"]}{counts}')
    else:
        print('  catalog:     (none — infer-stack catalog init)')
    print(f'  settings:    {d["settings"]["path"]}'
          f'{"" if d["configured"] else "  (run infer-stack config init)"}')
    lz = d['leasing']
    print(f'  ledger:      {lz["path"]}{"" if lz["exists"] else "  (none yet)"}')
    print(f'  compose:     {d["compose"]["path"]}'
          f'{"" if d["compose"]["exists"] else "  (not rendered yet)"}')
    if lz['summary']:
        active, total_l, live, total_d = lz['summary']
        print()
        print(f'leasing: {active} active / {total_l} lease(s), '
              f'{live} live / {total_d} deployment(s)  (infer-stack leases)')
    if not d['configured']:
        print()
        for cmd, comment in _GETTING_STARTED:
            print(f'  {cmd:<38}{comment}')


def _print_status_rich(d: dict[str, Any], console) -> None:
    from rich.table import Table
    from rich.text import Text

    console.print(Text('infer-stack status', style='bold'))
    table = Table(box=None, show_header=False, pad_edge=False,
                  padding=(0, 2, 0, 0))
    table.add_column(style='bold', justify='left', no_wrap=True)
    table.add_column(overflow='fold')

    table.add_row('backend', Text(d['backend'], style='bold cyan'))
    table.add_row('data dir', Text(d['data_dir'], style='cyan'))
    table.add_row('config dir', Text(d['config_dir'], style='cyan'))
    cat = d['catalog']
    if cat and cat['exists']:
        val = Text(cat['path'], style='cyan')
        if cat.get('models') is not None:
            val.append(f'  ({cat["models"]} models · {cat["endpoints"]} endpoints)',
                       style='dim')
        table.add_row('catalog', val)
    else:
        table.add_row('catalog', Text('(none — infer-stack catalog init)',
                                       style='yellow'))
    settings = Text(d['settings']['path'], style='cyan')
    if not d['configured']:
        settings = Text('(run infer-stack config init)', style='yellow')
    table.add_row('settings', settings)
    lz = d['leasing']
    ledger = Text(lz['path'], style='cyan')
    if not lz['exists']:
        ledger.append('  (none yet)', style='dim')
    table.add_row('ledger', ledger)
    compose = Text(d['compose']['path'], style='cyan')
    if not d['compose']['exists']:
        compose.append('  (not rendered yet)', style='dim')
    table.add_row('compose', compose)
    console.print(table)

    if lz['summary']:
        active, total_l, live, total_d = lz['summary']
        line = Text('leasing  ', style='bold')
        line.append(f'{active} active', style='green' if active else 'dim')
        line.append(f' / {total_l} lease(s)   ')
        line.append(f'{live} live', style='green' if live else 'dim')
        line.append(f' / {total_d} deployment(s)')
        console.print()
        console.print(line)

    console.print()
    if d['configured']:
        console.print(Text('dig deeper', style='bold'))
        rows = _DIG_DEEPER
    else:
        console.print(Text('getting started', style='bold'))
        rows = _GETTING_STARTED
    for cmd, comment in rows:
        line = Text('  ')
        line.append(cmd.ljust(30), style='cyan')
        line.append(comment, style='dim')
        console.print(line)


class StatusCLI(_PathOverridesMixin):
    """Holistic overview: where things live, the backend, and a leasing summary
    (active leases / live deployments), with pointers to dig deeper."""

    __command__ = 'status'
    catalog = scfg.Value(
        None, type=str, help='Catalog path (default: config dir).'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        data = _gather_status(config)
        from rich.console import Console

        console = Console()
        if console.is_terminal:
            _print_status_rich(data, console)
        else:
            _print_status_plain(data)
        return 0


# ---------------------------------------------------------------------------
# stack — docker compose day-2-ops wrappers over the leasing project
# ---------------------------------------------------------------------------


class _ComposeWrapperBase(_PathOverridesMixin):
    """Common fields for ``docker compose <subcmd>`` wrappers over the leasing
    Compose deployment."""

    services = scfg.Value(
        None,
        nargs='*',
        position=1,
        help='Optional service names to filter (empty = all).',
    )


class LogsCLI(_ComposeWrapperBase):
    """Tail leasing Compose service logs without typing the full docker path."""

    __command__ = 'logs'

    follow = scfg.Value(
        False, isflag=True, short_alias=['f'],
        help='Stream logs (docker compose logs -f).',
    )
    tail = scfg.Value(
        None, type=str,
        help="Tail the last N lines (default: all). Pass a number or 'all'.",
    )
    timestamps = scfg.Value(False, isflag=True)
    no_color = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'logs') + ['logs']
        if config.follow:
            cmd.append('--follow')
        if config.tail is not None:
            cmd.extend(['--tail', str(config.tail)])
        if config.no_color:
            cmd.append('--no-color')
        if config.timestamps:
            cmd.append('--timestamps')
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class PsCLI(_ComposeWrapperBase):
    """``docker compose ps`` for the leasing deployment."""

    __command__ = 'ps'

    all = scfg.Value(
        False, isflag=True, short_alias=['a'], help='Include stopped containers.'
    )
    services_only = scfg.Value(
        False, isflag=True,
        help='Print only service names (passes --services to docker compose).',
    )
    quiet = scfg.Value(
        False, isflag=True, short_alias=['q'], help='Print only container IDs.'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'ps') + ['ps']
        if config.all:
            cmd.append('--all')
        if config.services_only:
            cmd.append('--services')
        if config.quiet:
            cmd.append('--quiet')
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class RestartCLI(_ComposeWrapperBase):
    """``docker compose restart [services...]``."""

    timeout = scfg.Value(None, type=int, help='Stop timeout in seconds.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'restart') + ['restart']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class PullCLI(_ComposeWrapperBase):
    """``docker compose pull [services...]``."""

    quiet = scfg.Value(False, isflag=True, short_alias=['q'])
    ignore_pull_failures = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'pull') + ['pull']
        if config.quiet:
            cmd.append('--quiet')
        if config.ignore_pull_failures:
            cmd.append('--ignore-pull-failures')
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class StartCLI(_ComposeWrapperBase):
    """``docker compose start [services...]``."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'start') + ['start']
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class StopCLI(_ComposeWrapperBase):
    """``docker compose stop [services...]``."""

    timeout = scfg.Value(None, type=int)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'stop') + ['stop']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class StackDownCLI(_ComposeWrapperBase):
    """``docker compose down`` the leasing deployment.

    Tears the whole project down. Leasing's reconcile manages teardown
    automatically on release; this is the manual escape hatch.
    """

    volumes = scfg.Value(
        False, isflag=True, help='Also remove named volumes (--volumes).'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'down') + ['down', '--remove-orphans']
        if config.volumes:
            cmd.append('--volumes')
        return int(subprocess.run(cmd).returncode)


class StackUpCLI(_ComposeWrapperBase):
    """``docker compose up -d`` exactly what is on disk — the raw escape hatch.

    Brings up the on-disk compose file as-is, without touching the ledger or
    re-rendering. Prefer ``infer-stack apply``, which re-renders from intent
    first. Reach for ``stack up`` only to run a *hand-edited* compose file
    verbatim.
    """

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _day2_compose_base(config, 'up') + ['up', '-d', '--remove-orphans']
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd).returncode)


class StackModalCLI(scfg.ModalCLI):
    """Day-2 ops on the running leasing deployment."""

    __command__ = 'stack'

    up = StackUpCLI
    logs = LogsCLI
    ps = PsCLI
    restart = RestartCLI
    pull = PullCLI
    start = StartCLI
    stop = StopCLI
    down = StackDownCLI


class DoctorCLI(_PathOverridesMixin):
    """Preflight the configured backend: is everything acquire needs in place?

    Runs the backend's cheap dependency-ordered checks (for kubeai: cluster
    reachable -> KubeAI CRD installed -> namespace exists -> gateway
    answering) and prints a checklist. Exits nonzero if any check fails, so
    scripts can gate on it. Backends without a preflight (null/compose) report
    that there is nothing to check.
    """

    __command__ = 'doctor'

    backend = scfg.Value(
        None, type=str,
        help='Backend to check (default: the configured `backend` setting).',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        from .commands_leasing import _make_backend

        backend = _make_backend(config)
        doctor = getattr(backend, 'doctor', None)
        name = type(backend).__name__
        if doctor is None:
            print(f'{name}: no preflight checks defined — nothing to verify.')
            return 0
        failed = 0
        for check, ok, detail in doctor():
            mark = 'ok  ' if ok else 'FAIL'
            line = f'[{mark}] {check}'
            if detail:
                line += f' — {detail}'
            print(line)
            failed += 0 if ok else 1
        if failed:
            print(f'{failed} check(s) failed')
            return 1
        print('all checks passed')
        return 0
