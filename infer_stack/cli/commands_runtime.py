from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

import scriptconfig as scfg

from ..config import load_yaml, normalized_state
from ..docker_utils import (
    DockerCommandError,
    check_docker_compose_version,
    compose_down,
    compose_up,
    docker_rm_dirs,
)
from ..env_utils import parse_env_file
from ..kubeai_ops import CommandError, deploy_rendered_artifacts
from ..kubeai_ops import print_status as kubeai_print_status
from .commands_profile import RenderCLI
from .compose import (
    _compose_base_cmd,
    _compose_up_with_router_recreate,
    _kubeai_stub,
)
from .context import (
    _apply_path_overrides,
    _as_mapping,
    backend_name,
    config_for_runtime,
    config_path,
    config_root,
    generated_dir,
    has_runtime_overrides,
    kubeai_generated_dir,
    plan_path,
    render_is_stale,
    runtime_env_path,
)
from .options import (
    _BackendOverrideMixin,
    _ClusterOverridesMixin,
    _ComposeOverrideMixin,
    _PathOverridesMixin,
    _PlanOverridesCLI,
    _PortOverridesMixin,
)


def _maybe_rerender(config: Any, cfg: dict[str, Any]) -> None:
    """Re-run RenderCLI if runtime overrides changed or rendered outputs are stale.

    Both ``up`` and ``deploy`` need this so the rendered artifacts always
    match the current config + overrides before any container action.
    """
    if has_runtime_overrides(config) or render_is_stale(cfg):
        overrides = _as_mapping(config)
        RenderCLI.main(
            argv=False,
            profile=overrides.get('profile'),
            backend=overrides.get('backend'),
            compose_cmd=overrides.get('compose_cmd'),
            litellm_port=overrides.get('litellm_port'),
            open_webui_port=overrides.get('open_webui_port'),
            postgres_port=overrides.get('postgres_port'),
            namespace=overrides.get('namespace'),
            ingress_host=overrides.get('ingress_host'),
            ingress_enabled=overrides.get('ingress_enabled'),
            allow_unsupported=bool(overrides.get('allow_unsupported')),
            simulate_hardware=overrides.get('simulate_hardware'),
            allowed_gpus=overrides.get('allowed_gpus'),
            yes=bool(overrides.get('yes')),
        )


# ---------------------------------------------------------------------------
# Runtime commands
# ---------------------------------------------------------------------------


class UpCLI(_PlanOverridesCLI):
    """Bring the rendered compose stack up. Re-renders first if anything changed."""

    detach = scfg.Value(
        False,
        isflag=True,
        short_alias=['d'],
        help='Run in background instead of attaching to logs.',
    )
    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=['y'],
        help='If `up` triggers a re-render, apply changes without prompting.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            raise SystemExit(
                '`up` only supports the compose backend. Use `deploy` for kubeai.'
            )
        compose_cmd = cfg['runtime']['compose_cmd']
        check_docker_compose_version(compose_cmd)
        _maybe_rerender(config, cfg)
        _compose_up_with_router_recreate(cfg, detach=bool(config.detach))
        return 0


class DownCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
):
    """Bring the rendered compose stack down (does not touch volumes)."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            raise SystemExit('`down` only supports the compose backend.')
        # Re-render before down so a stale/invalid compose file from an older
        # package version does not strand containers.  Compose's
        # --remove-orphans still removes services from the previous profile
        # when the project name / generated directory is unchanged.
        _maybe_rerender(config, cfg)
        compose_cmd = cfg['runtime']['compose_cmd']
        check_docker_compose_version(compose_cmd)
        try:
            compose_down(
                compose_cmd,
                generated_dir(cfg) / 'docker-compose.yml',
                runtime_env_path(cfg),
            )
        except DockerCommandError as ex:
            raise SystemExit(
                f'compose down failed after re-rendering the current stack: {ex}'
            ) from ex
        return 0


class PurgeCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Stop the stack and delete all Docker-written state directories.

    Uses a temporary Alpine container to remove directories that Docker wrote
    as root, avoiding ``Permission denied`` errors from plain ``rm -rf``.
    """

    yes = scfg.Value(
        False, isflag=True, short_alias=['y'], help='Skip confirmation prompt.'
    )
    delete_cache = scfg.Value(
        False,
        isflag=True,
        help='Also delete hf-cache and vllm-cache (model weights). By default those are preserved.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config, allow_missing=True)

        state = normalized_state(cfg.get('state', {}))
        always_delete = [
            'postgres_litellm',
            'postgres_open_webui',
            'open_webui',
            'ollama',
            'runtime',
        ]
        model_dirs = ['hf_cache', 'vllm_cache']
        keys = (
            always_delete + model_dirs if config.delete_cache else always_delete
        )
        dirs_to_delete = [
            Path(state[k]) for k in keys if Path(state[k]).exists()
        ]

        if not dirs_to_delete:
            print('Nothing to purge — state directories do not exist.')
            return 0

        if not config.yes:
            print(
                'The following directories will be deleted (via Docker to handle root-owned files):'
            )
            for d in dirs_to_delete:
                print(f'  {d}')
            answer = input('Proceed? [y/N] ').strip().lower()
            if answer not in {'y', 'yes'}:
                print('Aborted.')
                return 1

        compose_file = generated_dir(cfg) / 'docker-compose.yml'
        if compose_file.exists() and backend_name(cfg) == 'compose':
            try:
                compose_down(
                    cfg['runtime']['compose_cmd'],
                    compose_file,
                    runtime_env_path(cfg),
                )
            except Exception as ex:
                print(
                    f'Warning: compose down failed (containers may already be stopped): {ex}'
                )

        compose_cmd = cfg.get('runtime', {}).get(
            'compose_cmd', 'docker compose'
        )
        docker_cmd = compose_cmd.split()[0]
        docker_rm_dirs(dirs_to_delete, docker_cmd=docker_cmd)
        print('Purge complete.')
        return 0


class DeployCLI(_PlanOverridesCLI):
    """Apply the rendered stack to its backend (kubeai apply / compose up)."""

    detach = scfg.Value(False, isflag=True, short_alias=['d'])
    yes = scfg.Value(
        False,
        isflag=True,
        short_alias=['y'],
        help='If `deploy` triggers a re-render, apply changes without prompting.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        _maybe_rerender(config, cfg)
        if backend_name(cfg) == 'kubeai':
            plan = load_yaml(plan_path(cfg))
            try:
                deploy_rendered_artifacts(plan['deployment'])
            except CommandError as ex:
                namespace = cfg.get('cluster', {}).get('namespace', 'kubeai')
                raise SystemExit(
                    f'Failed to deploy to namespace {namespace!r}. Confirm '
                    f'`infer-stack setup --backend kubeai --namespace {namespace}` '
                    'matches the namespace where the KubeAI Helm release is installed.\n'
                    f'Original error: {ex}'
                ) from ex
            return 0
        compose_up(
            cfg['runtime']['compose_cmd'],
            generated_dir(cfg) / 'docker-compose.yml',
            generated_dir(cfg) / '.env',
            detach=bool(config.detach),
            remove_orphans=True,
        )
        return 0


class EnvCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
):
    """Inspect the rendered .env file (path, single value, or eval-friendly export)."""

    key = scfg.Value(
        None,
        type=str,
        position=1,
        help="Print only this variable's value. Empty = all.",
    )
    export = scfg.Value(
        False,
        isflag=True,
        help='Print `export KEY=value` lines suitable for `eval`.',
    )
    path = scfg.Value(
        False,
        isflag=True,
        help='Print only the absolute path to .env (default if no flags).',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            raise SystemExit('`env` only applies to the compose backend.')
        env_file = runtime_env_path(cfg)
        if not env_file.exists():
            raise SystemExit(
                f'No .env at {env_file}. Run `infer-stack render` first.'
            )
        # Default with no flags and no key: print the path so users can do
        #   `source $(infer-stack env)` (with `set -a` if they want export semantics).
        if config.key is None and not config.export:
            print(env_file)
            return 0
        env = parse_env_file(env_file)
        if config.key:
            if config.key not in env:
                raise SystemExit(f'{config.key!r} not found in {env_file}')
            print(env[config.key])
            return 0
        # --export: emit eval-friendly export lines.
        for k, v in env.items():
            print(f'export {k}={shlex.quote(v)}')
        return 0


def _enabled_components(deployment: dict[str, Any]) -> list[str]:
    """Names of the components the resolved plan turned on."""
    out: list[str] = []
    providers = deployment.get('providers', {}) or {}
    if (providers.get('ollama') or {}).get('enabled'):
        out.append('ollama')
    vllm = providers.get('vllm') or {}
    if vllm.get('enabled'):
        runtimes = vllm.get('runtimes') or {}
        out.append(f'vllm({len(runtimes)})' if runtimes else 'vllm')
    if ((deployment.get('gateways', {}) or {}).get('litellm') or {}).get(
        'enabled'
    ):
        out.append('litellm')
    frontends = deployment.get('frontends', {}) or {}
    if (frontends.get('open_webui') or {}).get('enabled'):
        out.append('open_webui')
    if (frontends.get('reverse_proxy') or {}).get('enabled'):
        out.append('reverse_proxy')
    return out


def _access_endpoints(access: dict[str, Any]) -> list[tuple[str, str]]:
    """(name, base_url) pairs from the resolved access map."""
    out: list[tuple[str, str]] = []
    for name, entry in (access or {}).items():
        if isinstance(entry, dict) and entry.get('base_url'):
            out.append((name, str(entry['base_url'])))
    return out


def _leasing_configured() -> bool:
    """Has the user set up the leasing model (catalog / durable settings)?

    The leasing world doesn't use ``config.yaml``; it uses ``catalog.yaml`` and
    ``settings.yaml``. Treat either (or any ledger state, reported separately) as
    "set up" so ``status`` doesn't nag a leasing user to run the legacy setup.
    """
    from ..paths import settings_path

    return (config_root() / 'catalog.yaml').exists() or settings_path().exists()


def _status_data(
    cfg: dict[str, Any], *, initialized: bool, leasing_mode: bool
) -> dict[str, Any]:
    """Gather cheap, no-network status context once (shared by both renderers).

    Everything here comes from config.yaml / catalog.yaml / settings.yaml, the
    on-disk artifacts, and the already-resolved plan.yaml — so it never touches
    Docker, the network, or hardware detection. ``initialized`` is about the
    *legacy* ``config.yaml`` (profiles); the leasing model is reported above it.
    """
    from ..paths import data_root, settings_path

    backend = backend_name(cfg)
    d: dict[str, Any] = {
        'backend': backend,
        'data_dir': str(data_root()),
        'config_dir': str(config_root()),
        'catalog': None,
        'settings': None,
        'initialized': initialized,
        'config_path': str(config_path()),
        'leasing_mode': leasing_mode,
        'active_profile': None,
        'generated_dir': None,
        'rendered': None,        # 'yes' | 'yes-stale' | 'no' | None
        'description': None,
        'validation': None,      # (text, 'ok'|'warn'|'error')
        'components': None,
        'endpoints': [],
    }

    cat_file = config_root() / 'catalog.yaml'
    if cat_file.exists():
        counts = None
        try:
            from ..leasing import Catalog

            cat = Catalog.load(cat_file)
            counts = (
                f'{len(cat.models)} model(s), '
                f'{len(cat.endpoints)} endpoint(s)'
            )
        except Exception:
            pass
        d['catalog'] = {'path': str(cat_file), 'counts': counts}
    if settings_path().exists():
        d['settings'] = str(settings_path())

    if not initialized:
        return d

    d['active_profile'] = cfg.get('active_profile') or '<unset>'
    out_dir = (
        kubeai_generated_dir(cfg) if backend == 'kubeai' else generated_dir(cfg)
    )
    d['generated_dir'] = str(out_dir)
    rendered_marker = out_dir / (
        'models.yaml' if backend == 'kubeai' else 'docker-compose.yml'
    )
    if rendered_marker.exists():
        d['rendered'] = 'yes-stale' if render_is_stale(cfg) else 'yes'
    else:
        d['rendered'] = 'no'

    # The resolved view comes straight from plan.yaml (cheap file read; no
    # hardware probe or re-resolution).
    plan_file = plan_path(cfg)
    if not plan_file.exists():
        return d
    try:
        plan = load_yaml(plan_file) or {}
    except Exception:
        return d
    deployment = plan.get('deployment', {}) or {}
    validated = plan.get('validated', {}) or {}

    d['description'] = (deployment.get('serving_profile', {}) or {}).get(
        'description'
    )
    if validated:
        errors = validated.get('errors') or []
        warnings = validated.get('warnings') or []
        if errors:
            vtext = f'{len(errors)} error(s)' + (
                f', {len(warnings)} warning(s)' if warnings else ''
            )
            d['validation'] = (vtext, 'error')
        elif warnings:
            d['validation'] = (f'ok, {len(warnings)} warning(s)', 'warn')
        else:
            d['validation'] = ('ok', 'ok')
    d['components'] = _enabled_components(deployment) or None
    d['endpoints'] = _access_endpoints(deployment.get('access', {}) or {})
    return d


_GETTING_STARTED = (
    ('  Nothing set up yet. For the leasing model:', None),
    ('    infer-stack config init', '# backend + data dir'),
    ('    infer-stack catalog init', '# then `catalog model add` …'),
    ('    infer-stack serve <endpoint>', None),
    ('  (Pre-leasing profiles live under `infer-stack legacy setup`.)', None),
)
# Column where the `#` comments align (longest command + a 2-space gutter).
_GS_WIDTH = max(len(t) for t, c in _GETTING_STARTED if c) + 2


def _print_status_plain(d: dict[str, Any]) -> None:
    """Plain key/value summary (used when stdout is not a terminal)."""
    print('infer-stack status')
    print()
    print(f'  backend:        {d["backend"]}')
    print(f'  data dir:       {d["data_dir"]}')
    print(f'  config dir:     {d["config_dir"]}')
    if d['catalog']:
        counts = d['catalog']['counts']
        suffix = f'  ({counts})' if counts else ''
        print(f'  catalog:        {d["catalog"]["path"]}{suffix}')
    if d['settings']:
        print(f'  settings:       {d["settings"]}')
    print(
        f'  legacy config:  {"yes" if d["initialized"] else "no"}  '
        f'({d["config_path"]})'
    )

    if not d['initialized']:
        if not d['leasing_mode']:
            print()
            for line, comment in _GETTING_STARTED:
                print(f'{line:<{_GS_WIDTH}}{comment}' if comment else line)
        return

    print(f'  active profile: {d["active_profile"]}')
    print(f'  generated dir:  {d["generated_dir"]}')
    if d['rendered'] == 'yes':
        print('  rendered:       yes')
    elif d['rendered'] == 'yes-stale':
        print('  rendered:       yes  (stale — run `infer-stack render`)')
    elif d['rendered'] == 'no':
        print('  rendered:       no  (run `infer-stack render`)')
    if d['description']:
        print(f'  description:    {d["description"]}')
    if d['validation']:
        print(f'  validation:     {d["validation"][0]}')
    if d['components']:
        print(f'  components:     {", ".join(d["components"])}')
    if d['endpoints']:
        print('  endpoints:')
        for name, url in d['endpoints']:
            print(f'    {name}: {url}')


def _print_status_rich(d: dict[str, Any], counts, console) -> None:
    """Styled status summary for an interactive terminal."""
    from rich.table import Table
    from rich.text import Text

    def cell(value, style=None):
        # Append (don't set a base style) so column padding stays unstyled.
        t = Text()
        t.append(value, style=style)
        return t

    console.print(Text('infer-stack status', style='bold'))
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style='bold', justify='left', no_wrap=True)
    table.add_column(overflow='fold')

    table.add_row('backend', cell(d['backend'], 'bold cyan'))
    table.add_row('data dir', cell(d['data_dir'], 'cyan'))
    table.add_row('config dir', cell(d['config_dir'], 'cyan'))
    if d['catalog']:
        val = cell(d['catalog']['path'], 'cyan')
        if d['catalog']['counts']:
            val.append(f'  ({d["catalog"]["counts"]})', style='dim')
        table.add_row('catalog', val)
    if d['settings']:
        table.add_row('settings', cell(d['settings'], 'cyan'))

    legacy = cell('yes', 'green') if d['initialized'] else cell('no', 'yellow')
    legacy.append(f'  ({d["config_path"]})', style='dim')
    table.add_row('legacy config', legacy)

    if d['initialized']:
        table.add_row('active profile', cell(d['active_profile'], 'magenta'))
        table.add_row('generated dir', cell(d['generated_dir'], 'cyan'))
        if d['rendered'] == 'yes':
            table.add_row('rendered', cell('yes', 'green'))
        elif d['rendered'] == 'yes-stale':
            r = cell('yes', 'green')
            r.append('  (stale — run `infer-stack render`)', style='yellow')
            table.add_row('rendered', r)
        elif d['rendered'] == 'no':
            r = cell('no', 'yellow')
            r.append('  (run `infer-stack render`)', style='dim')
            table.add_row('rendered', r)
        if d['description']:
            table.add_row('description', cell(d['description']))
        if d['validation']:
            vtext, sev = d['validation']
            style = {'ok': 'green', 'warn': 'yellow', 'error': 'red'}[sev]
            table.add_row('validation', cell(vtext, style))
        if d['components']:
            table.add_row('components', cell(', '.join(d['components']), 'cyan'))
    console.print(table)

    if d['initialized'] and d['endpoints']:
        console.print(Text('  endpoints', style='bold'))
        for name, url in d['endpoints']:
            line = Text('    ')
            line.append(name, style='cyan')
            line.append(f': {url}')
            console.print(line)

    if counts is not None:
        active, live = counts
        line = Text('leasing  ', style='bold')
        line.append(
            f'{active} active lease(s)', style='green' if active else 'dim'
        )
        line.append(' · ')
        line.append(f'{live} live group(s)', style='green' if live else 'dim')
        line.append('   → infer-stack leases', style='dim')
        console.print()
        console.print(line)

    if not d['initialized'] and not d['leasing_mode']:
        console.print()
        for text, comment in _GETTING_STARTED:
            line = Text()
            is_cmd = text.lstrip().startswith('infer-stack')
            if comment:
                line.append(text.ljust(_GS_WIDTH), style='cyan' if is_cmd else None)
                line.append(comment, style='dim')
            else:
                line.append(text, style='cyan' if is_cmd else 'dim')
            console.print(line)


def _leasing_counts():
    """``(active_leases, live_groups)`` when a non-empty ledger exists, else None.

    Leases live in their own ledger (``infer-stack leases``), not in the
    active-profile deployment ``status`` otherwise reports on — this keeps the
    status command relevant after the leasing redesign.
    """
    from ..leasing import (
        GroupState,
        LeaseState,
        Ledger,
        SqliteStore,
        default_ledger_path,
    )

    path = default_ledger_path()
    if not path.exists():
        return None
    leases, groups = Ledger(SqliteStore(str(path))).status()
    if not leases and not groups:
        return None
    active = sum(1 for le in leases if le.state == LeaseState.ACTIVE)
    live = sum(1 for g in groups if g.state == GroupState.LIVE)
    return (active, live)


def _print_leasing_plain(counts) -> None:
    if counts is None:
        return
    active, live = counts
    print()
    print(
        f'leasing: {active} active lease(s), {live} live group(s) '
        '(see `infer-stack leases`)'
    )


class StatusCLI(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
    _PortOverridesMixin,
    _ClusterOverridesMixin,
):
    """Show stack status: backend + where config/catalog/settings live, the
    leasing summary (active leases / live groups; see ``infer-stack leases``),
    and — for a legacy profile user — the active profile, whether it has been
    rendered, the resolved components/endpoints, and the live container/cluster
    state. The leasing model needs no ``config.yaml`` (reported as ``legacy
    config``)."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        initialized = config_path().exists()
        cfg = config_for_runtime(config, allow_missing=True)
        data = _status_data(
            cfg, initialized=initialized, leasing_mode=_leasing_configured()
        )
        counts = _leasing_counts()
        from rich.console import Console

        console = Console()
        if console.is_terminal:
            _print_status_rich(data, counts, console)
        else:
            _print_status_plain(data)
            _print_leasing_plain(counts)
        if not initialized:
            return 0

        if backend_name(cfg) == 'kubeai':
            namespace = cfg.get('cluster', {}).get('namespace', 'kubeai')
            print()
            print(f'cluster resources (namespace {namespace!r}):')
            try:
                kubeai_print_status(namespace)
            except CommandError as ex:
                raise SystemExit(
                    f'Failed to query KubeAI resources in namespace {namespace!r}. Confirm '
                    f'`infer-stack legacy setup --backend kubeai --namespace {namespace}` '
                    'matches the namespace where the KubeAI Helm release is installed.\n'
                    f'Original error: {ex}'
                ) from ex
            return 0

        compose_file = generated_dir(cfg) / 'docker-compose.yml'
        if not compose_file.exists():
            return 0
        compose_cmd = cfg['runtime']['compose_cmd']
        check_docker_compose_version(compose_cmd)
        print()
        print('containers (docker compose ps):')
        proc = subprocess.run(_compose_base_cmd(cfg) + ['ps'])
        return int(proc.returncode)


# ---------------------------------------------------------------------------
# Compose day-2-ops wrappers (raise NotImplementedError on kubeai)
# ---------------------------------------------------------------------------


def _day2_compose_base(config, command: str) -> list[str]:
    """``docker compose -p <proj> -f <file>`` base for the day-2 wrappers.

    Prefer the leasing Compose deployment when present — it is the current model
    and needs no ``config.yaml`` — otherwise the legacy rendered stack (which
    requires ``setup`` + ``render``). Raises the kubeai stub for a legacy kubeai
    config, since compose day-2 ops don't apply there.
    """
    from ..leasing.compose import COMPOSE_FILENAME, LEASING_PROJECT
    from ..paths import data_root

    leasing_file = data_root() / 'leasing' / 'compose' / COMPOSE_FILENAME
    if leasing_file.exists():
        return [
            'docker', 'compose', '-p', LEASING_PROJECT, '-f', str(leasing_file)
        ]
    cfg = config_for_runtime(config)
    if backend_name(cfg) != 'compose':
        _kubeai_stub(command)
    return _compose_base_cmd(cfg)


class _ComposeWrapperBase(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Common fields for ``docker compose <subcmd>`` wrappers.

    These target the leasing Compose deployment when one exists (no
    ``config.yaml`` needed), else the legacy rendered stack.
    """

    services = scfg.Value(
        None,
        nargs='*',
        position=1,
        help='Optional service names to filter (empty = all).',
    )


class LogsCLI(_ComposeWrapperBase):
    """Tail rendered Compose service logs without typing the full docker compose path."""

    follow = scfg.Value(
        False,
        isflag=True,
        short_alias=['f'],
        help='Stream logs (docker compose logs -f).',
    )
    tail = scfg.Value(
        None,
        type=str,
        help="Tail the last N lines (default: all). Pass a number or 'all'.",
    )
    timestamps = scfg.Value(False, isflag=True)
    no_color = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
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
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class PsCLI(_ComposeWrapperBase):
    """``docker compose ps`` for the rendered stack."""

    all = scfg.Value(
        False,
        isflag=True,
        short_alias=['a'],
        help='Include stopped containers.',
    )
    services_only = scfg.Value(
        False,
        isflag=True,
        help='Print only service names (passes --services to docker compose).',
    )
    quiet = scfg.Value(
        False, isflag=True, short_alias=['q'], help='Print only container IDs.'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cmd = _day2_compose_base(config, 'ps') + ['ps']
        if config.all:
            cmd.append('--all')
        if config.services_only:
            cmd.append('--services')
        if config.quiet:
            cmd.append('--quiet')
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class RestartCLI(_ComposeWrapperBase):
    """``docker compose restart [services...]``."""

    timeout = scfg.Value(None, type=int, help='Stop timeout in seconds.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cmd = _day2_compose_base(config, 'restart') + ['restart']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class PullCLI(_ComposeWrapperBase):
    """``docker compose pull [services...]``."""

    quiet = scfg.Value(False, isflag=True, short_alias=['q'])
    ignore_pull_failures = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cmd = _day2_compose_base(config, 'pull') + ['pull']
        if config.quiet:
            cmd.append('--quiet')
        if config.ignore_pull_failures:
            cmd.append('--ignore-pull-failures')
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StartCLI(_ComposeWrapperBase):
    """``docker compose start [services...]``."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cmd = _day2_compose_base(config, 'start') + ['start']
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StopCLI(_ComposeWrapperBase):
    """``docker compose stop [services...]``."""

    timeout = scfg.Value(None, type=int)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cmd = _day2_compose_base(config, 'stop') + ['stop']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StackDownCLI(_ComposeWrapperBase):
    """``docker compose down`` the deployment (leasing stack when present).

    Tears the whole project down. Leasing's reconcile manages teardown
    automatically on release; this is the manual escape hatch (cf. cleanup).
    """

    volumes = scfg.Value(
        False, isflag=True, help='Also remove named volumes (--volumes).'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cmd = _day2_compose_base(config, 'down') + ['down', '--remove-orphans']
        if config.volumes:
            cmd.append('--volumes')
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class StackModalCLI(scfg.ModalCLI):
    """Day-2 ops on the running deployment (leasing stack when present)."""

    __command__ = 'stack'

    logs = LogsCLI
    ps = PsCLI
    restart = RestartCLI
    pull = PullCLI
    start = StartCLI
    stop = StopCLI
    down = StackDownCLI


class OllamaPullCLI(
    _PathOverridesMixin, _BackendOverrideMixin, _ComposeOverrideMixin
):
    """Pull an Ollama model into the rendered Ollama model store."""

    __command__ = 'ollama-pull'

    model = scfg.Value(
        None,
        type=str,
        position=1,
        help='Ollama model tag to pull, e.g. smollm2:135m.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.model:
            raise SystemExit(
                'ollama-pull: missing required model tag, e.g. `infer-stack ollama-pull smollm2:135m`'
            )
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            raise SystemExit('`ollama-pull` only supports the compose backend.')
        cmd = _compose_base_cmd(cfg) + [
            'exec',
            'ollama',
            'ollama',
            'pull',
            str(config.model),
        ]
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class OllamaListCLI(
    _PathOverridesMixin, _BackendOverrideMixin, _ComposeOverrideMixin
):
    """List models installed in the rendered Ollama service."""

    __command__ = 'ollama-list'

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            raise SystemExit('`ollama-list` only supports the compose backend.')
        cmd = _compose_base_cmd(cfg) + ['exec', 'ollama', 'ollama', 'list']
        proc = subprocess.run(cmd)
        return int(proc.returncode)


class OllamaPsCLI(
    _PathOverridesMixin, _BackendOverrideMixin, _ComposeOverrideMixin
):
    """Show loaded Ollama models for the rendered Ollama service."""

    __command__ = 'ollama-ps'

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            raise SystemExit('`ollama-ps` only supports the compose backend.')
        cmd = _compose_base_cmd(cfg) + ['exec', 'ollama', 'ollama', 'ps']
        proc = subprocess.run(cmd)
        return int(proc.returncode)
