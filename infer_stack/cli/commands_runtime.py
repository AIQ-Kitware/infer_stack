"""Day-2 ops on the running leasing stack, plus a holistic ``status`` command.

The ``stack`` wrappers (``logs`` / ``ps`` / ``up`` / ``down`` / …) target the
leasing Compose deployment — the project rendered under the data dir by
``acquire`` / ``apply``. ``status`` is the one-glance overview: where
everything lives, the active backend, and a leasing summary (active leases / live
deployments), with pointers to dig deeper.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import kwconf as kw

from ..log_filter import compact_litellm_tracebacks
from ..paths import config_root, data_root, get_setting, settings_path
from .commands_leasing import ApplyCLI
from .context import _apply_path_overrides
from .options import _PathOverridesMixin

# ---------------------------------------------------------------------------
# the backend behind the day-2 verbs
# ---------------------------------------------------------------------------


def _docker_env() -> dict[str, str]:
    from ..leasing.compose import docker_environment

    return docker_environment()


def _day2_backend(config):
    """The configured backend, built as the leasing verbs build it."""
    from .commands_leasing import _make_backend

    _apply_path_overrides(config)
    return _make_backend(config)


def _served_by_deployment() -> dict[str, list[str]]:
    """Deployment id -> the endpoint aliases it serves (read-only ledger)."""
    from ..leasing import Ledger, SqliteStore, default_ledger_path

    path = default_ledger_path()
    if not path.exists():
        return {}
    try:
        _, deployments = Ledger(SqliteStore(str(path))).status(virtual_expiry=True)
    except Exception:  # noqa: BLE001 - a view never fails on the ledger
        return {}
    return {d.id: sorted(d.served) for d in deployments}


def _instances(backend) -> list:
    """What the backend runs, or a clean exit when it cannot be read."""
    from ..leasing.residency import ResidencyUnknown

    try:
        return list(backend.instances())
    except ResidencyUnknown as ex:
        raise SystemExit(f'cannot read what is running: {ex}')


def _compose_argv(config) -> list[str]:
    """``docker compose ...`` for the Compose project on this host.

    That is the stack itself on the compose backend, and the gateway on the
    kubeai backend. Exits when there is none, or nothing is rendered yet.
    """
    backend = _day2_backend(config)
    project = getattr(backend, 'compose_project', lambda: None)()
    if project is None:
        raise SystemExit(
            f'the {_backend_name(backend)} backend has no Compose project on this '
            'host; use `infer-stack ps` and `infer-stack logs`')
    if not project.compose_file.exists():
        raise SystemExit(
            f'nothing rendered yet (no {project.compose_file}); bring a model up '
            'first, e.g. `infer-stack acquire <endpoint>`')
    return project.compose_argv()


def _backend_name(backend) -> str:
    name = type(backend).__name__
    return {'ComposeBackend': 'compose', 'KubeaiBackend': 'kubeai',
            'NullBackend': 'null (dry-run)'}.get(name, name)


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
        ledger = Ledger(SqliteStore(str(path)))
        leases, deployments = ledger.status(virtual_expiry=True)
        out['pending'] = ledger.publication_pending() is not None
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
    out['live_deployments'] = [d for d in deployments if d.state == DeploymentState.LIVE]
    return out


def _served_models(deployments, backend=None, *,
                   pending: bool = False) -> list[tuple[str, str, str, str]]:
    """``(endpoint, model, engine, health)`` rows for what is serving.

    Reads the ledger for what *should* be up and the backend's strict
    residency for what is, so `status` answers "what can I send a request to
    right now" without cross-reading `leases` against `ps`.
    """
    from ..leasing import DeploymentState
    from ..leasing.residency import ResidencyUnknown

    live = [d for d in deployments if d.state == DeploymentState.LIVE]
    if not live:
        return []
    residency = None
    if backend is not None:
        try:
            residency = backend.residency()
        except (ResidencyUnknown, Exception):  # noqa: BLE001 - status never fails here
            residency = None

    rows: list[tuple[str, str, str, str]] = []
    for d in live:
        if residency is None:
            health = 'unverified'
        elif residency.resident(d.id) is not None:
            # Warm; a health check that has not passed yet means still loading.
            health = ('starting' if residency.resident(d.id).health == 'starting'
                      else 'up')
        elif residency.containers(d.id):
            # There, but not warm: starting, crashed, or ambiguous.
            health = residency.containers(d.id)[0].state
        elif pending:
            # Recorded, not applied yet: an apply is running, or a failed one
            # left the change for `infer-stack apply`.
            health = 'pending'
        else:
            # The ledger says live, nothing exists, and no change is pending.
            health = 'STALE'
        for endpoint in sorted(d.served):
            payload = d.served.get(endpoint) or {}
            model = (payload.get('hf_model_id')
                     or payload.get('model')
                     or d.spec.get('hf_model_id') or '-')
            rows.append((endpoint, model, d.engine, health))
    return rows


def _gather_status(config) -> dict[str, Any]:
    try:
        backend = _day2_backend(config)
    except (SystemExit, Exception):  # noqa: BLE001 - status never fails on this
        backend = None
    rendered = getattr(backend, 'rendered_file', None)
    leasing = _leasing_status()
    if leasing.get('live_deployments'):
        leasing['served'] = _served_models(leasing.pop('live_deployments'), backend,
                                           pending=bool(leasing.get('pending')))
    else:
        leasing.pop('live_deployments', None)
    return {
        'backend': str(get_setting('backend') or 'null'),
        'data_dir': str(data_root()),
        'config_dir': str(config_root()),
        'configured': settings_path().exists(),
        'settings': {'path': str(settings_path()),
                     'exists': settings_path().exists()},
        'catalog': _catalog_summary(config),
        'rendered': {'path': str(rendered) if rendered else None,
                     'exists': bool(rendered and rendered.exists())},
        'leasing': leasing,
    }


_DIG_DEEPER = (
    ('infer-stack leases', 'full lease + deployment tables'),
    ('infer-stack tui', 'live dashboard (opt-in: infer-stack[tui])'),
    ('infer-stack ps', 'what is running (containers or pods)'),
    ('infer-stack logs -f <endpoint>', 'follow an engine\'s log'),
    ('infer-stack catalog show', 'what you can serve'),
)

_GETTING_STARTED = (
    ('infer-stack config init', 'storage + default backend'),
    ('infer-stack catalog suggest --apply', 'a catalog sized to your GPUs'),
    ('infer-stack acquire <endpoint>', 'bring a model up'),
)


def _served_lines(served: list[tuple[str, str, str, str]]) -> list[str]:
    """Plain-text 'serving now' block; empty when nothing is up."""
    if not served:
        return []
    w_ep = max(8, *(len(r[0]) for r in served))
    w_mo = max(5, *(len(r[1]) for r in served))
    out = ['', 'serving now',
           f'  {"endpoint".ljust(w_ep)}  {"model".ljust(w_mo)}  health']
    for endpoint, model, _engine, health in served:
        out.append(f'  {endpoint.ljust(w_ep)}  {model.ljust(w_mo)}  {health}')
    if any(r[3] == 'STALE' for r in served):
        out.append('  STALE = the ledger records this live but nothing is '
                   'running for it; `infer-stack apply` or `gc`')
    if any(r[3] == 'pending' for r in served):
        out.append('  pending = recorded but not applied yet: an apply is running, '
                   'or `infer-stack apply` finishes it')
    if any(r[3] == 'starting' for r in served):
        out.append('  starting = up, but not ready yet (loading the model); '
                   '`infer-stack wait <endpoint>`')
    if any(r[3] == 'unverified' for r in served):
        out.append('  unverified = the runtime could not be read to confirm; '
                   'the ledger says live')
    return out


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
    if d['rendered']['path']:
        print(f'  rendered:    {d["rendered"]["path"]}'
              f'{"" if d["rendered"]["exists"] else "  (not rendered yet)"}')
    if lz['summary']:
        active, total_l, live, total_d = lz['summary']
        print()
        print(f'leasing: {active} active / {total_l} lease(s), '
              f'{live} live / {total_d} deployment(s)  (infer-stack leases)')
    for line in _served_lines(lz.get('served') or []):
        print(line)
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
    if d['rendered']['path']:
        rendered = Text(d['rendered']['path'], style='cyan')
        if not d['rendered']['exists']:
            rendered.append('  (not rendered yet)', style='dim')
        table.add_row('rendered', rendered)
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

    served = lz.get('served') or []
    if served:
        served_table = Table(box=None, pad_edge=False, padding=(0, 2, 0, 0))
        served_table.add_column('endpoint', style='bold cyan', no_wrap=True)
        served_table.add_column('model', overflow='fold')
        served_table.add_column('engine', style='dim', no_wrap=True)
        served_table.add_column('health', no_wrap=True)
        styles = {'up': 'green', 'starting': 'yellow', 'pending': 'yellow',
                  'STALE': 'red', 'unverified': 'yellow'}
        for endpoint, model, engine, health in served:
            served_table.add_row(
                endpoint, model, engine,
                Text(health, style=styles.get(health, 'dim')),
            )
        console.print()
        console.print(Text('serving now', style='bold'))
        console.print(served_table)
        if any(r[3] == 'STALE' for r in served):
            console.print(Text(
                '  STALE = recorded live but nothing is running for it; '
                '`infer-stack apply` or `gc`', style='dim'))

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
    """Holistic overview: where things live, the backend, and what is serving.

    A leasing summary (active leases, live deployments) with each endpoint's
    health, and pointers to dig deeper."""

    __command__ = 'status'
    catalog = kw.Value(
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
# ps / logs / stack: what the backend runs, on either backend
# ---------------------------------------------------------------------------


class _InstancesBase(_PathOverridesMixin):
    """Options shared by the verbs that read the backend's instances."""

    services = kw.Value(
        None,
        nargs='*',
        position=1,
        help='Which instances: a service or pod name, a container id (prefix), '
        'a deployment id, or an endpoint alias. Empty = all.',
    )
    backend = kw.Value(
        None, type=str,
        help='Backend to read (default: the configured `backend` setting).',
    )


#: States `ps` hides unless --all: finished, or on the way out.
_PS_HIDDEN = frozenset({'exited', 'dead', 'removing'})


def _ps_rows(instances, served) -> list[dict[str, Any]]:
    rows = []
    for inst in instances:
        rows.append({
            'name': inst.name,
            'id': inst.id,
            'deployment': inst.deployment_id or None,
            'serves': served.get(inst.deployment_id, []) if inst.deployment_id else [],
            'state': inst.state,
            'status': inst.status,
            'restarts': inst.restarts,
            'gpus': list(inst.gpus),
            'started': inst.started or None,
            'ports': inst.ports or None,
            'runtime': inst.runtime,
        })
    return rows


def _print_ps(rows) -> None:
    from ..leasing.instances import local_time

    def cell(row):
        serves = ', '.join(row['serves']) or ('-' if row['deployment'] else '(front door)')
        gpus = ','.join(map(str, row['gpus'])) or '-'
        ident = row['id'][:12] if row['runtime'] == 'docker' else '-'
        return (row['name'], row['status'], serves, gpus,
                local_time(row['started'] or '') or '-', ident, row['ports'] or '-')

    head = ('NAME', 'STATUS', 'SERVES', 'GPUS', 'STARTED', 'ID', 'PORTS')
    table = [head, *(cell(r) for r in rows)]
    widths = [max(len(str(r[i])) for r in table) for i in range(len(head))]
    for r in table:
        print('  '.join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip())


class PsCLI(_InstancesBase):
    """What the backend is running: engine containers or pods, and the gateway.

    One shape on every backend. An engine row names the endpoints it serves;
    the gateway, UI and proxy show as the front door. Reads the same strict
    residency the controller decides with, so a row here is what admission
    sees.
    """

    __command__ = 'ps'

    all = kw.Value(
        False, isflag=True, short_alias=['a'],
        help='Include exited and dying instances.',
    )
    services_only = kw.Value(
        False, isflag=True, help='Print only instance names.',
    )
    quiet = kw.Value(
        False, isflag=True, short_alias=['q'],
        help='Print only ids (container ids, or pod names).',
    )
    json = kw.Value(False, isflag=True, help='Print the rows as JSON.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        import json

        from ..leasing.instances import UnknownTarget, resolve

        config = cls.cli(argv=argv, data=kwargs)
        backend = _day2_backend(config)
        instances = _instances(backend)
        served = _served_by_deployment()
        if config.services:
            try:
                instances = resolve(instances, config.services, served)
            except UnknownTarget as ex:
                raise SystemExit(str(ex))
        if not config.all:
            instances = [i for i in instances if i.state not in _PS_HIDDEN]
        rows = _ps_rows(instances, served)
        if config.json:
            print(json.dumps(rows, indent=2))
        elif config.quiet:
            for row in rows:
                print(row['id'])
        elif config.services_only:
            for row in rows:
                print(row['name'])
        elif not rows:
            print(f'nothing running ({_backend_name(backend)} backend); '
                  'bring a model up with `infer-stack acquire <endpoint>`')
        else:
            _print_ps(rows)
        return 0


#: Prefix colors for `logs` on a terminal, cycled per instance name.
_LOG_COLORS = ('36', '33', '32', '35', '34', '96', '93', '92', '95', '94')


def _colorize(lines, *, enabled: bool):
    """Color each ``name  | `` prefix, one color per name, like Compose did.

    Disabled (a pipe, a file, ``--no-color``), the engines' own color codes
    go too: they are noise in a file and break a grep.
    """
    if not enabled:
        from ..log_filter import _ANSI_ESCAPE_RE

        for line in lines:
            yield _ANSI_ESCAPE_RE.sub('', line)
        return
    colors: dict[str, str] = {}
    for line in lines:
        name, sep, rest = line.partition('  | ')
        if not sep:
            yield line
            continue
        code = colors.setdefault(name, _LOG_COLORS[len(colors) % len(_LOG_COLORS)])
        yield f'\x1b[{code}m{name}  |\x1b[0m {rest}'


class LogsCLI(_InstancesBase):
    """Engine and gateway logs, by instance, deployment or endpoint alias.

    ``infer-stack logs qwen -f`` follows whatever serves the ``qwen``
    endpoint, container or pod; with no names, every instance. Following
    picks up instances that start later and follows a restarted one again.
    """

    __command__ = 'logs'

    follow = kw.Value(
        False, isflag=True, short_alias=['f'],
        help='Keep streaming, including instances that start later.',
    )
    tail = kw.Value(
        None, type=str,
        help="Only the last N lines of each (default: all). A number or 'all'.",
    )
    timestamps = kw.Value(False, isflag=True)
    # A positive flag: kwconf reads a leading `no-` as negation, so a flag
    # named `no_color` silently ignored `--no-color`.
    color = kw.Value(True, isflag=True,
                     help='Color the name prefixes on a terminal (`--no-color` to not).')
    raw = kw.Value(
        False,
        isflag=True,
        help='Show raw followed logs without known LiteLLM traceback compaction.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.instances import (
            LogFollower,
            UnknownTarget,
            history_argv,
            resolve,
            runtime_env,
        )

        config = cls.cli(argv=argv, data=kwargs)
        backend = _day2_backend(config)
        served = _served_by_deployment()
        names = list(config.services or [])

        def pick(instances):
            return resolve(instances, names, served) if names else instances

        try:
            chosen = pick(_instances(backend))
        except UnknownTarget as ex:
            raise SystemExit(str(ex))
        color = sys.stdout.isatty() and bool(config.color)
        if config.follow:
            def listing():
                try:
                    return pick(backend.instances())
                except UnknownTarget:
                    return []           # the named one is between restarts
            follower = LogFollower(listing, history=config.tail or 'all',
                                   timestamps=bool(config.timestamps))
            lines = follower.stdout
            if sys.stdout.isatty() and not config.raw:
                lines = compact_litellm_tracebacks(lines)
            try:
                for line in _colorize(lines, enabled=color):
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                return 130
            finally:
                follower.terminate()
            return 0
        if not chosen:
            print(f'nothing running ({_backend_name(backend)} backend)')
            return 0
        prefix = len(chosen) > 1
        status = 0
        for inst in chosen:
            argv_ = history_argv(inst, tail=config.tail,
                                 timestamps=bool(config.timestamps))
            if argv_ is None:
                print(f'{inst.name}: the {_backend_name(backend)} backend keeps no logs')
                continue
            proc = subprocess.run(argv_, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, env=runtime_env(inst))
            text = proc.stdout.decode('utf-8', 'replace')
            lines = [f'{inst.name}  | {ln}\n' if prefix else f'{ln}\n'
                     for ln in text.splitlines()]
            for line in _colorize(lines, enabled=color):
                sys.stdout.write(line)
            status = status or proc.returncode
        return int(status)


class _ComposeWrapperBase(_PathOverridesMixin):
    """``docker compose <subcmd>`` over the Compose project on this host.

    The stack itself on the compose backend; the gateway on kubeai.
    """

    services = kw.Value(
        None,
        nargs='*',
        position=1,
        help='Optional service names to filter (empty = all).',
    )
    backend = kw.Value(
        None, type=str,
        help='Backend (default: the configured `backend` setting).',
    )


class RestartCLI(_ComposeWrapperBase):
    """``docker compose restart [services...]`` (the Compose project on this host)."""

    timeout = kw.Value(None, type=int, help='Stop timeout in seconds.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _compose_argv(config) + ['restart']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd, env=_docker_env()).returncode)


class PullCLI(_ComposeWrapperBase):
    """``docker compose pull [services...]`` (the Compose project on this host)."""

    quiet = kw.Value(False, isflag=True, short_alias=['q'])
    ignore_pull_failures = kw.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _compose_argv(config) + ['pull']
        if config.quiet:
            cmd.append('--quiet')
        if config.ignore_pull_failures:
            cmd.append('--ignore-pull-failures')
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd, env=_docker_env()).returncode)


class StartCLI(_ComposeWrapperBase):
    """``docker compose start [services...]`` (the Compose project on this host)."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _compose_argv(config) + ['start']
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd, env=_docker_env()).returncode)


class StopCLI(_ComposeWrapperBase):
    """``docker compose stop [services...]`` (the Compose project on this host)."""

    timeout = kw.Value(None, type=int)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _compose_argv(config) + ['stop']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        return int(subprocess.run(cmd, env=_docker_env()).returncode)


class StackComposeCLI(_PathOverridesMixin):
    """Run any ``docker compose`` command on the Compose project on this host.

    The raw escape hatch: the stack itself on the compose backend, the gateway
    on kubeai. It bypasses the ledger and renders nothing, e.g.
    ``infer-stack stack compose -- up -d litellm``.
    """

    __command__ = 'compose'

    args = kw.Value(None, nargs='*', position=1,
                    help='Arguments for docker compose (put them after --).')
    backend = kw.Value(
        None, type=str,
        help='Backend (default: the configured `backend` setting).',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        cmd = _compose_argv(config) + list(config.args or [])
        return int(subprocess.run(cmd, env=_docker_env()).returncode)


class StackDownCLI(_PathOverridesMixin):
    """Stop everything the backend runs, bypassing the ledger.

    The manual escape hatch: releases no lease, so a later publish brings
    leased models back. Compose: ``docker compose down``; kubeai: deletes every
    managed Model, then the gateway. ``--volumes`` (Compose only) also removes
    named volumes.
    """

    __command__ = 'down'

    volumes = kw.Value(
        False, isflag=True, help='Also remove named volumes (Compose only).'
    )
    backend = kw.Value(
        None, type=str,
        help='Backend (default: the configured `backend` setting).',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        if config.volumes:
            cmd = _compose_argv(config) + ['down', '--remove-orphans', '--volumes']
            return int(subprocess.run(cmd, env=_docker_env()).returncode)
        backend = _day2_backend(config)
        down = getattr(backend, 'down', None)
        if down is None:
            print(f'the {_backend_name(backend)} backend runs nothing to bring down')
            return 0
        down()
        return 0


class StackUpCLI(ApplyCLI):
    """``apply``: bring up what the ledger says should run (both backends).

    The raw form, ``docker compose up`` of the file on disk, is
    ``infer-stack stack compose -- up -d``.
    """

    __command__ = 'up'


class StackModalCLI(kw.ModalCLI):
    """Day-2 ops on what the backend runs.

    ``up`` is ``apply`` and ``down`` stops everything, on either backend. The
    ``docker compose`` verbs (restart, pull, start, stop, and ``compose`` for
    anything else) act on the Compose project on this host: the stack itself
    on compose, the gateway on kubeai.
    """

    __command__ = 'stack'

    up = StackUpCLI
    logs = LogsCLI
    ps = PsCLI
    down = StackDownCLI
    compose = StackComposeCLI
    restart = RestartCLI
    pull = PullCLI
    start = StartCLI
    stop = StopCLI


class DoctorCLI(_PathOverridesMixin):
    """Preflight the configured backend: is everything acquire needs in place?

    Runs the backend's cheap dependency-ordered checks (for kubeai: cluster
    reachable -> KubeAI CRD installed -> namespace exists -> KubeAI's API
    answering) and prints a checklist. Exits nonzero if any check fails, so
    scripts can gate on it. Backends without a preflight (null/compose) report
    that there is nothing to check.

    ``--gpu`` adds host GPU checks: is any card busy with nothing allocated,
    and who holds the device nodes. ``--sudo`` lets the holder scan run as
    root, which is the only way it means anything -- unprivileged it sees only
    your own processes, and every containerd shim and Kubernetes pod is owned
    by root. Without it the holder check reports *not checked* rather than a
    clean bill of health.

    It never resets or kills anything. The same symptom with a different holder
    means something else entirely, so the fix is always the operator's call.
    """

    __command__ = 'doctor'

    backend = kw.Value(
        None, type=str,
        help='Backend to check (default: the configured `backend` setting).',
    )
    gpu = kw.Value(False, isflag=True,
                     help='Also check host GPUs (implied by --sudo).')
    sudo = kw.Value(False, isflag=True,
                      help='Run the GPU holder scan as root. Without it that '
                           'check reports "not checked", never "clear".')

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
            if not (config['gpu'] or config['sudo']):
                return 0
            doctor = list  # GPU checks still apply; the backend just has none
        failed = 0
        checks = list(doctor())
        if config['gpu'] or config['sudo']:
            from ..gpu_doctor import gpu_checks
            checks += list(gpu_checks(use_sudo=bool(config['sudo'])))
        for check, ok, detail in checks:
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
