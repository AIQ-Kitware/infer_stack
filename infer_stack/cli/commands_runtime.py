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


def _print_status_summary(cfg: dict[str, Any], *, initialized: bool) -> None:
    """Print cheap, no-network context about the current stack.

    Everything here comes from config.yaml, on-disk artifact existence, and the
    already-resolved plan.yaml, so it never touches Docker, the network, or
    hardware detection.
    """
    backend = backend_name(cfg)
    print('infer-stack status')
    print()
    print(
        f'  initialized:    {"yes" if initialized else "no"}  ({config_path()})'
    )
    print(f'  backend:        {backend}')
    print(f'  active profile: {cfg.get("active_profile") or "<unset>"}')
    print(f'  config dir:     {config_root()}')

    out_dir = (
        kubeai_generated_dir(cfg) if backend == 'kubeai' else generated_dir(cfg)
    )
    rendered_marker = out_dir / (
        'models.yaml' if backend == 'kubeai' else 'docker-compose.yml'
    )
    print(f'  generated dir:  {out_dir}')

    if not initialized:
        print()
        print(
            '  Not initialized — run '
            '`infer-stack setup --backend compose --profile <profile>` '
            'to create config.yaml.'
        )
        return

    if rendered_marker.exists():
        stale = render_is_stale(cfg)
        print(
            f'  rendered:       yes{"  (stale — run `infer-stack render`)" if stale else ""}'
        )
    else:
        print('  rendered:       no  (run `infer-stack render`)')

    # The resolved view comes straight from plan.yaml (cheap file read; no
    # hardware probe or re-resolution).
    plan_file = plan_path(cfg)
    if not plan_file.exists():
        return
    try:
        plan = load_yaml(plan_file) or {}
    except Exception:
        return
    deployment = plan.get('deployment', {}) or {}
    validated = plan.get('validated', {}) or {}

    description = (deployment.get('serving_profile', {}) or {}).get(
        'description'
    )
    if description:
        print(f'  description:    {description}')

    if validated:
        errors = validated.get('errors') or []
        warnings = validated.get('warnings') or []
        if errors:
            vstate = f'{len(errors)} error(s)' + (
                f', {len(warnings)} warning(s)' if warnings else ''
            )
        elif warnings:
            vstate = f'ok, {len(warnings)} warning(s)'
        else:
            vstate = 'ok'
        print(f'  validation:     {vstate}')

    components = _enabled_components(deployment)
    if components:
        print(f'  components:     {", ".join(components)}')

    endpoints = _access_endpoints(deployment.get('access', {}) or {})
    if endpoints:
        print('  endpoints:')
        for name, url in endpoints:
            print(f'    {name}: {url}')


def _print_leasing_summary() -> None:
    """One-line pointer to the leasing model's state, when a ledger exists.

    Keeps the legacy ``status`` relevant after the leasing redesign: leases live
    in their own ledger (``infer-stack leases``), not in the active-profile
    deployment this command otherwise reports on.
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
        return
    leases, groups = Ledger(SqliteStore(str(path))).status()
    if not leases and not groups:
        return
    active = sum(1 for le in leases if le.state == LeaseState.ACTIVE)
    live = sum(1 for g in groups if g.state == GroupState.LIVE)
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
    """Show stack status: where config/artifacts live, the active profile,
    whether it has been rendered, the resolved components/endpoints, and the
    live container/cluster state. Leases (the newer model) are summarized too;
    see ``infer-stack leases`` for detail."""

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        initialized = config_path().exists()
        cfg = config_for_runtime(config, allow_missing=True)
        _print_status_summary(cfg, initialized=initialized)
        _print_leasing_summary()
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
                    f'`infer-stack setup --backend kubeai --namespace {namespace}` '
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


class _ComposeWrapperBase(
    _PathOverridesMixin,
    _BackendOverrideMixin,
    _ComposeOverrideMixin,
):
    """Common fields for ``docker compose <subcmd>`` wrappers."""

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
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            _kubeai_stub('logs')
        cmd = _compose_base_cmd(cfg) + ['logs']
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
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            _kubeai_stub('ps')
        cmd = _compose_base_cmd(cfg) + ['ps']
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
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            _kubeai_stub('restart')
        cmd = _compose_base_cmd(cfg) + ['restart']
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
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            _kubeai_stub('pull')
        cmd = _compose_base_cmd(cfg) + ['pull']
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
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            _kubeai_stub('start')
        cmd = _compose_base_cmd(cfg) + ['start']
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
        cfg = config_for_runtime(config)
        if backend_name(cfg) != 'compose':
            _kubeai_stub('stop')
        cmd = _compose_base_cmd(cfg) + ['stop']
        if config.timeout is not None:
            cmd.extend(['--timeout', str(config.timeout)])
        cmd.extend(config.services or [])
        proc = subprocess.run(cmd)
        return int(proc.returncode)


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
