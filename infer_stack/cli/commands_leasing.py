"""Leasing subcommands: the acquire/release/run surface for the new model.

These verbs drive the leasing controller (ledger + backend) instead of the
legacy "active profile". They are what a kwdagger pipeline node uses to request
the models it needs, block until ready, and release after:

    infer-stack acquire qwen-coder reranker --ttl 2h --env-file is.env
    infer-stack run --endpoint qwen-coder -- python my_node.py
    infer-stack release --env-file is.env
    infer-stack serve qwen-coder        # standing service (no TTL)
    infer-stack leases                  # status of leases + deployment groups

Until the Compose/KubeAI backends land, the default ``--backend null`` is a
dry-run: the ledger does all the real bookkeeping (coalescing, demand, TTL) but
nothing is actually served. This lets the whole surface — including env-file
emission and the ``run`` wrapper — be exercised end to end without docker.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import scriptconfig as scfg

from ..env_utils import parse_env_file, write_env_file
from ..leasing import (
    Catalog,
    CatalogError,
    ComposeBackend,
    Controller,
    GroupState,
    LeaseState,
    Ledger,
    NullBackend,
    Sharing,
    SqliteStore,
    default_ledger_path,
)
from ..leasing.envfile import (
    build_descriptor,
    descriptor_env,
    read_session_id,
    render_env_file,
)
from ..paths import config_root, data_root
from .context import _apply_path_overrides
from .options import _AllowedGpusMixin, _DisplayGpuMixin, _PathOverridesMixin

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_owner() -> str:
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return 'unknown'


def _parse_duration(text) -> float | None:
    """Parse ``2h`` / ``30m`` / ``90s`` / ``1d`` / bare seconds; None = infinite."""
    if text is None:
        return None
    text = str(text).strip().lower()
    if text in ('', 'none', 'inf', 'infinite', '0'):
        return None
    units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _collect_names(value) -> list[str]:
    """Normalize positional/flag names: accept a list or comma-separated str."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        for part in str(item).split(','):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _parse_gpus(text) -> list[int] | None:
    if text in (None, ''):
        return None
    return [int(p) for p in str(text).split(',') if p.strip() != '']


def _coerce_bool(value, default: bool) -> bool:
    """Interpret a setting/flag as a bool (None -> default)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _resolve_ui(config) -> bool:
    """Open WebUI on/off: explicit --ui/--no-ui wins, else `config set ui`,
    else on by default."""
    from ..paths import get_setting

    flag = getattr(config, 'ui', None)
    if flag is not None:
        return bool(flag)
    return _coerce_bool(get_setting('ui'), True)


def _resolve_reverse_proxy(config) -> tuple[bool, int, str | None]:
    """Resolve the reverse-proxy front door: (enabled, port, byo_config_path).

    The ``reverse_proxy`` setting may be a bool (enable with defaults) or a
    ``{enabled, port, config_path}`` block (via `config edit`); the
    ``--reverse-proxy`` flag overrides the enabled bit.
    """
    from ..paths import get_setting

    raw = get_setting('reverse_proxy')
    block = raw if isinstance(raw, dict) else {}
    flag = getattr(config, 'reverse_proxy', None)
    if flag is not None:
        enabled = bool(flag)
    elif isinstance(raw, dict):
        enabled = _coerce_bool(block.get('enabled'), False)
    else:
        enabled = _coerce_bool(raw, False)
    port = int(block.get('port') or 80)
    return enabled, port, block.get('config_path')


def _resolve_skip_display(config) -> bool:
    """Whether to skip display-attached GPUs during placement.

    Off by default — placement uses every GPU, so a single-GPU host (where the
    only GPU drives the display) works without ceremony. Opt in with the
    ``--skip-display-gpus`` flag (wins) or `config set skip_display_gpus true`.
    """
    from ..paths import get_setting

    flag = getattr(config, 'skip_display_gpus', None)
    if flag is not None:
        return bool(flag)
    return _coerce_bool(get_setting('skip_display_gpus'), False)


def _resolve_assume_yes(config, *, interactive: bool) -> bool:
    """Whether to apply compose changes without the diff prompt.

    Only the additive verbs (acquire/serve) prompt, and only on a real terminal
    without ``--yes``. Everything else (release/leases/run/non-TTY) auto-applies.
    """
    import sys

    if not interactive:
        return True
    if getattr(config, 'yes', False):
        return True
    return not sys.stdout.isatty()


def _make_backend(config, *, interactive: bool = False):
    from ..paths import get_setting

    # Resolve the backend: explicit --backend wins, else the persisted default
    # (`config set backend compose`), else the dry-run null backend.
    name = getattr(config, 'backend', None) or get_setting('backend') or 'null'
    if name in (None, '', 'null', 'dry-run'):
        return NullBackend()
    if name == 'compose':
        from ..hardware import detect_inventory

        rp_enabled, rp_port, rp_config = _resolve_reverse_proxy(config)
        return ComposeBackend(
            state_dir=data_root() / 'leasing' / 'compose',
            inventory=detect_inventory(),
            allowed_gpus=_parse_gpus(getattr(config, 'allowed_gpus', None)),
            skip_display=_resolve_skip_display(config),
            ui=_resolve_ui(config),
            reverse_proxy=rp_enabled,
            reverse_proxy_port=rp_port,
            reverse_proxy_config=rp_config,
            require_generation=bool(getattr(config, 'require_generation', False)),
            assume_yes=_resolve_assume_yes(config, interactive=interactive),
        )
    raise SystemExit(
        f'backend {name!r} is not implemented in the leasing CLI yet '
        '(kubeai is a later stage). Use --backend null or compose.'
    )


def _open_controller(config, *, interactive: bool = False) -> Controller:
    from .._log import configure_logging

    configure_logging()  # narrate the leasing verbs (stderr); off the help path
    _apply_path_overrides(config)
    ledger_path = config.ledger or str(default_ledger_path())
    ledger = Ledger(SqliteStore(ledger_path))
    return Controller(ledger, _make_backend(config, interactive=interactive))


def _load_catalog(config) -> Catalog:
    raw = config.catalog or (config_root() / 'catalog.yaml')
    path = Path(raw).expanduser()
    if not path.exists():
        raise SystemExit(
            f'catalog not found: {path} (pass --catalog or create catalog.yaml)'
        )
    try:
        return Catalog.load(path)
    except CatalogError as ex:
        raise SystemExit(f'invalid catalog {path}: {ex}')


def _resolve(catalog, names, *, sharing=None):
    try:
        return catalog.resolve_names(names, sharing=sharing)
    except CatalogError as ex:
        raise SystemExit(str(ex))


def _resolve_session(config) -> str | None:
    sid = getattr(config, 'session', None)
    if not sid and getattr(config, 'env_file', None):
        sid = read_session_id(config.env_file)
    return sid


def _descriptor_for(controller, lease, groups, config):
    """Build the descriptor, preferring backend-supplied access (real base_url)."""
    base_url = config.base_url
    api_key_env = config.api_key_env
    api_key = None
    request_names = None
    access = getattr(controller.backend, 'access', None)
    info = access(list(lease.endpoints)) if access else None
    if info:
        base_url = info.get('base_url', base_url)
        api_key_env = info.get('api_key_env', api_key_env)
        api_key = info.get('api_key')
        request_names = info.get('request_names')
    return build_descriptor(
        lease,
        groups,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        request_names=request_names,
    )


def _compose_file_path(controller) -> str | None:
    path = getattr(controller.backend, 'compose_file', None)
    return str(path) if path else None


def _gpu_where(gpus) -> str:
    """Human label for a group's GPU assignment (or its absence)."""
    if gpus is None:
        return 'unplaced'
    if not gpus:
        return 'cpu'
    return 'GPU ' + ','.join(str(i) for i in gpus)


def _emit_staged(config, controller, outcome) -> int:
    """Output for a ``--no-apply`` acquire: what was written + what would run."""
    descriptor = _descriptor_for(
        controller, outcome.lease, outcome.groups, config
    )
    if config.env_file:
        Path(config.env_file).expanduser().write_text(render_env_file(descriptor))
    assignments = outcome.reconcile.assignments
    if config.json:
        print(json.dumps({
            'session_id': outcome.lease.id,
            'owner': outcome.lease.owner,
            'applied': False,
            'descriptor': descriptor,
            'compose_file': _compose_file_path(controller),
            'placement': [
                {'group': g.id, 'served': sorted(g.served),
                 'gpus': assignments.get(g.id)}
                for g in outcome.groups
            ],
        }, indent=2))
        return 0
    print(f'staged {outcome.lease.id} (owner={outcome.lease.owner}) — not applied')
    for g in outcome.groups:
        eps = ', '.join(sorted(g.served)) or g.id
        print(f'  {eps}: {_gpu_where(assignments.get(g.id))}  ({g.id})')
    path = _compose_file_path(controller)
    if path:
        print(f'  compose: {path}')
    print('  apply:   infer-stack apply           # bring the staged set up')
    if path:
        # The file carries `name: infer-stack`, so plain docker works too.
        print(f'  ...or:   docker compose -f {path} up -d')
    print(f'  discard: infer-stack release {outcome.lease.id}')
    if config.env_file:
        print(f'  env-file: {config.env_file}')
    return 0


def _emit_acquire(config, controller, outcome) -> int:
    if not outcome.applied:
        return _emit_staged(config, controller, outcome)
    descriptor = _descriptor_for(
        controller, outcome.lease, outcome.groups, config
    )
    if config.env_file:
        Path(config.env_file).expanduser().write_text(
            render_env_file(descriptor)
        )
    not_ready = outcome.wait is not None and not outcome.wait.ready
    if config.json:
        print(
            json.dumps(
                {
                    'session_id': outcome.lease.id,
                    'owner': outcome.lease.owner,
                    'descriptor': descriptor,
                    'realized': outcome.reconcile.realized,
                    'ready': None
                    if outcome.wait is None
                    else outcome.wait.ready,
                    'pending': []
                    if outcome.wait is None
                    else outcome.wait.pending,
                },
                indent=2,
            )
        )
    else:
        print(f'acquired {outcome.lease.id} (owner={outcome.lease.owner})')
        for endpoint, model in descriptor['endpoints'].items():
            print(f'  endpoint {endpoint} -> {model}')
        if outcome.wait is not None:
            print(f'  ready: {outcome.wait.ready}')
            for gid, endpoint in outcome.wait.pending:
                print(f'    pending: {endpoint} ({gid})')
        access = getattr(controller.backend, 'access', None)
        info = access(list(outcome.lease.endpoints)) if access else None
        if info and info.get('ui_url'):
            print(f'  open webui: {info["ui_url"]}')
        if info and info.get('proxy_url'):
            print(f'  front door: {info["proxy_url"]}  (UI: /   API: /v1)')
        if config.env_file:
            print(f'  env-file: {config.env_file}')
    return 2 if not_ready else 0


def _do_acquire(config, *, owner: str, ttl_seconds: float | None) -> int:
    from .._log import logger
    from ..leasing.backend import ConvergeAborted, PlacementError

    render_only = not bool(getattr(config, 'apply', True))
    # Staging (--no-apply) is non-destructive (no docker up), so don't gate it
    # behind the diff prompt — the rendered file *is* the thing you asked to see.
    controller = _open_controller(config, interactive=not render_only)
    catalog = _load_catalog(config)
    names = _collect_names(config.names)
    if not names:
        raise SystemExit('give at least one endpoint or bundle name')
    sharing = Sharing.DEDICATED if getattr(config, 'dedicated', False) else None
    requests = _resolve(catalog, names, sharing=sharing)
    logger.info('Acquiring {} for {}', ', '.join(names), owner)
    if render_only:
        logger.info('Render-only: staging the compose project without applying')
    elif config.wait:
        logger.info(
            'Will wait up to {:.0f}s for readiness (poll {:.0f}s)',
            float(config.timeout), float(config.interval),
        )
    try:
        outcome = controller.acquire(
            owner,
            requests,
            ttl_seconds=ttl_seconds,
            wait=bool(config.wait),
            timeout=float(config.timeout),
            interval=float(config.interval),
            apply=not render_only,
        )
    except ConvergeAborted:
        raise SystemExit('aborted: compose changes not applied (no lease kept)')
    except PlacementError as ex:
        lines = ['could not place every requested endpoint (no lease kept):']
        lines += [f'  {r}' for r in ex.reasons] or [
            f'  {", ".join(ex.group_ids)}'
        ]
        lines.append(
            '  free a GPU first — `infer-stack leases` to see what holds them, '
            'then `infer-stack release`/`evict`. (Every GPU, including any '
            'display-attached one, is used unless you set --skip-display-gpus.)'
        )
        raise SystemExit('\n'.join(lines))
    return _emit_acquire(config, controller, outcome)


# ---------------------------------------------------------------------------
# shared flag mixins
# ---------------------------------------------------------------------------


class _LeasingCommonMixin(_PathOverridesMixin, _AllowedGpusMixin, _DisplayGpuMixin):
    backend = scfg.Value(
        None,
        choices=['null', 'compose', 'kubeai'],
        help='Serving backend. "null" (dry-run) and "compose" are implemented. '
        'Defaults to `config set backend …`, else "null".',
    )
    ledger = scfg.Value(
        None, type=str, help='Path to the lease ledger sqlite db.'
    )
    require_generation = scfg.Value(
        False,
        isflag=True,
        help='Readiness requires a real generation, not just model listing '
        '(compose backend; Ollama always generates to warm the tag).',
    )
    ui = scfg.Value(
        None,
        isflag=True,
        help='Render a managed Open WebUI in front of the gateway (compose '
        'backend). On by default; use --no-ui to skip. Overrides '
        '`config set ui …`.',
    )
    reverse_proxy = scfg.Value(
        None,
        isflag=True,
        alias=['reverse-proxy'],
        help='Front the gateway + UI with a single-port HTTP reverse proxy, so '
        'you hit one origin (UI at /, API at /v1) — off by default, no TLS/auth '
        '(localhost / trusted networks only). Port + bring-your-own nginx.conf '
        'live in the `reverse_proxy` setting (`config set` / `config edit`).',
    )


class _ApprovalMixin(_LeasingCommonMixin):
    """A verb that converges the compose project and so gates on a diff.

    Any change to the on-disk compose project is shown and confirmed on a
    terminal; ``--yes`` (or a non-TTY) applies without prompting.
    """

    yes = scfg.Value(
        False, isflag=True, alias=['y'],
        help='Apply compose changes without showing the diff / prompting '
        '(compose backend). Implied when stdout is not a terminal.',
    )


class _AcquireFlagsMixin(_LeasingCommonMixin):
    catalog = scfg.Value(None, type=str, help='Path to catalog.yaml.')
    base_url = scfg.Value(
        'http://127.0.0.1:14042/v1',
        type=str,
        help='Base URL written into the endpoint descriptor (dry-run placeholder).',
    )
    api_key_env = scfg.Value(
        'LITELLM_MASTER_KEY',
        type=str,
        help='Name of the env var holding the API key (kept out of artifacts).',
    )
    wait = scfg.Value(
        True, isflag=True, help='Block until ready (use --no-wait to skip).'
    )
    apply = scfg.Value(
        True,
        isflag=True,
        help='Apply the render (docker compose up). Use --no-apply to *stage* '
        'only: declare the lease and write the on-disk compose project + '
        'placement WITHOUT starting it, then `infer-stack apply` to bring it up '
        '(compose backend). --no-apply implies no readiness wait and no diff '
        'prompt; `release` discards a staged lease.',
    )
    timeout = scfg.Value(600, type=float, help='Readiness wait timeout (s).')
    interval = scfg.Value(5, type=float, help='Readiness poll interval (s).')
    env_file = scfg.Value(
        None, type=str, help='Write the sourceable endpoint env-file here.'
    )
    yes = scfg.Value(
        False, isflag=True, alias=['y'],
        help='Apply compose changes without showing the diff / prompting '
        '(compose backend). Implied when stdout is not a terminal.',
    )
    json = scfg.Value(False, isflag=True, help='Emit JSON instead of text.')


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


class AcquireCLI(_AcquireFlagsMixin):
    """Acquire a lease on one or more endpoints/bundles and block until ready.

    Like ``serve`` but with a soft ``--ttl`` (default infinite): the lease
    protects its models while held, and once it expires (or is released) nothing
    else needs them they can be reclaimed. This is the programmatic seam — write
    a sourceable endpoint env-file with ``--env-file`` and release by that file
    or session id when done. (For a one-shot "acquire, run a command, release",
    use ``infer-stack run`` instead.)
    """

    __command__ = 'acquire'

    names = scfg.Value(
        [], nargs='*', position=1, type=str, help='Endpoint or bundle names.'
    )
    ttl = scfg.Value(
        None, type=str, help='Soft TTL (e.g. 2h, 30m); default infinite.'
    )
    owner = scfg.Value(None, type=str, help='Lease owner (default: $USER).')
    dedicated = scfg.Value(
        False,
        isflag=True,
        help='Force a dedicated deployment instead of coalescing.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _do_acquire(
            config,
            owner=config.owner or _default_owner(),
            ttl_seconds=_parse_duration(config.ttl),
        )


class ServeCLI(_AcquireFlagsMixin):
    """Stand up endpoints as a standing service (an infinite ``manual`` lease).

    ``serve NAME…`` is the everyday verb. It acquires an infinite (no-TTL) lease
    on each endpoint or bundle, renders the compose project (the LiteLLM gateway
    + one container per model + a managed Open WebUI), brings it up, and blocks
    until every endpoint is ready. Run it again with more names to add models
    side by side — the gateway and UI stay put. Placement, ``docker compose``,
    and readiness are narrated on stderr.

    The work is render (write the on-disk compose project) then apply (``docker
    compose up``). ``--no-apply`` does just the render so you can see what
    would run before pulling the trigger (then ``infer-stack apply``).
    ``--no-wait`` applies but returns immediately so several models load in
    parallel (``wait`` for them later). ``--no-ui`` skips Open WebUI. On a
    terminal you are shown the compose diff and asked before applying; ``--yes``
    skips that prompt (and it is skipped automatically off a TTY).
    """

    __command__ = 'serve'
    __epilog__ = """
    Examples:
        # serve one model: render + up + wait until it can generate
        infer-stack serve qwen05-1 --require-generation

        # stage only: write the compose project but don't start it, then apply
        infer-stack serve qwen05-1 --no-apply
        infer-stack apply

        # fan out: start several without blocking, then wait together
        infer-stack serve smol17b-1 --no-wait --yes
        infer-stack serve qwen15-1  --no-wait --yes
        infer-stack wait  smol17b-1 qwen15-1 --require-generation

        # see what is actually running and on which GPUs
        infer-stack leases
    """

    names = scfg.Value(
        [], nargs='*', position=1, type=str, help='Endpoint or bundle names.'
    )
    owner = scfg.Value('manual', type=str, help='Lease owner.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _do_acquire(config, owner=config.owner, ttl_seconds=None)


class RenderCLI(_LeasingCommonMixin):
    """Write the on-disk compose project for the current desired set — no up.

    Lease-free and idempotent: ``render`` re-materializes the manifest (compose
    file + gateway config + GPU placement) from whatever is *already* declared,
    WITHOUT starting anything — to inspect what would run, or refresh a file you
    touched by hand. It creates no lease; to stage a *new* endpoint use ``serve
    --no-apply`` (which declares it too). Apply with ``infer-stack apply``.
    """

    __command__ = 'render'

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=False)
        rec = controller.reconcile(apply=False)
        path = _compose_file_path(controller)
        if config.json:
            print(json.dumps({
                'applied': False,
                'compose_file': path,
                'placement': rec.assignments,
                'unplaced': rec.unplaced,
            }, indent=2))
            return 0
        print(f'rendered desired set -> {path or "(backend has no on-disk project)"}')
        for gid, gpus in sorted(rec.assignments.items()):
            print(f'  {gid}: {_gpu_where(gpus)}')
        for err in rec.placement_errors:
            print(f'  ! {err}')
        print('  apply:   infer-stack apply')
        return 0


class ApplyCLI(_ApprovalMixin):
    """Bring the current desired set up to match the ledger (render + up).

    Lease-free: ``apply`` creates no lease — it converges whatever is already
    declared (by ``serve``/``acquire``) onto the backend. It is the *trigger*
    for a staged ``serve --no-apply`` and the re-sync button after a manual edit
    or a backend hiccup; idempotent (a second apply with nothing changed is a
    no-op). On a terminal it shows the compose diff and asks (``--yes`` skips).
    The rendered file carries ``name: infer-stack``, so ``docker compose -f
    <file> up -d`` is an exact equivalent. (``infer-stack stack up`` is the
    lower-level "run exactly what is on disk" hatch; ``apply`` re-renders from
    intent first.)
    """

    __command__ = 'apply'

    wait = scfg.Value(
        False, isflag=True, help='Also block until ready after bringing it up.'
    )
    timeout = scfg.Value(600, type=float, help='Readiness wait timeout (s).')
    interval = scfg.Value(5, type=float, help='Readiness poll interval (s).')
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)
        try:
            rec = controller.reconcile(apply=True)
        except ConvergeAborted:
            raise SystemExit('aborted: compose changes not applied')
        result = None
        if config.wait:
            controller.ledger.sweep()
            _, groups = controller.ledger.status()
            live = [g for g in groups if g.state == GroupState.LIVE]
            if live:
                result = controller.wait_ready(
                    live,
                    timeout=float(config.timeout),
                    interval=float(config.interval),
                )
        if config.json:
            print(json.dumps({
                'applied': True,
                'realized': rec.realized,
                'torn_down': rec.torn_down,
                'placement': rec.assignments,
                'unplaced': rec.unplaced,
                'ready': None if result is None else result.ready,
            }, indent=2))
            return 0 if (result is None or result.ready) else 2
        print(
            f'applied: {len(rec.realized)} started, {len(rec.torn_down)} stopped'
        )
        for gid in rec.realized:
            print(f'  + {gid}')
        for gid in rec.torn_down:
            print(f'  - {gid}')
        for err in rec.placement_errors:
            print(f'  ! {err}')
        if result is not None and not result.ready:
            print('  not ready (timed out)')
            for gid, ep in result.pending:
                print(f'    pending: {ep} ({gid})')
            return 2
        return 0


def _declined_exit() -> SystemExit:
    return SystemExit(
        'declined: the ledger change stands but docker was not touched — '
        'run `infer-stack apply` to apply it.'
    )


class ReleaseCLI(_ApprovalMixin):
    """Release a lease; deployments idle/teardown per their reclaim policy.

    On a terminal the resulting compose change is shown and confirmed before
    docker is touched (``--yes`` skips); declining leaves the lease released in
    the ledger but the containers running — ``infer-stack apply`` then applies
    it. All the lease's changes converge in a single step, so you are asked at
    most once.
    """

    __command__ = 'release'

    session = scfg.Value(
        None, position=1, type=str, help='Session id (or use --env-file).'
    )
    env_file = scfg.Value(
        None, type=str, help='Read the session id from this env-file.'
    )
    all = scfg.Value(
        False, isflag=True,
        help='Release every active lease (the whole stack idles/tears down).',
    )
    evict = scfg.Value(
        False, isflag=True,
        help='Also evict (tear down) the released group(s) now, even if their '
        'reclaim policy is keep-warm — frees the GPU immediately.',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)

        if config.all:
            if config.session or config.env_file:
                raise SystemExit('release: --all takes no session/--env-file')
            controller.ledger.sweep()
            leases, _ = controller.ledger.status()
            released = [le.id for le in leases if le.state == LeaseState.ACTIVE]
            for sid in released:
                controller.ledger.release(sid)
            evicted: list[str] = []
            if config.evict:
                evicted = controller.ledger.evict_idle(None)  # every idle group
        else:
            sid = _resolve_session(config)
            if not sid:
                raise SystemExit(
                    'release: give a session id, --env-file, or --all'
                )
            rel = controller.ledger.release(sid)
            released = [sid]
            evicted = []
            if config.evict:
                evicted = controller.ledger.evict_idle(rel.idled_group_ids)

        # One converge for the whole command -> at most one diff prompt.
        try:
            rec = controller.reconcile()
        except ConvergeAborted:
            raise _declined_exit()
        return _emit_release(config, released, sorted(rec.torn_down), evicted)


def _emit_release(config, released, torn_down, evicted) -> int:
    if config.json:
        print(json.dumps({
            'released': released,
            'torn_down': torn_down,
            'evicted': evicted,
        }, indent=2))
        return 0
    if not released:
        print('no active leases to release')
    else:
        print(f'released {len(released)} lease(s)')
        for sid in released:
            print(f'  {sid}')
    for gid in torn_down:
        print(f'  torn down: {gid}')
    return 0


def _resolve_idle_targets(controller, names: list[str]) -> tuple[list[str], list[str]]:
    """Map endpoint aliases / group ids to currently-idle group ids.

    Returns ``(target_group_ids, unmatched_names)``.
    """
    controller.ledger.sweep()
    _, groups = controller.ledger.status()
    idle = [g for g in groups if g.state == GroupState.IDLE]
    wanted = set(names)
    targets, matched = [], set()
    for g in idle:
        hit = wanted & ({g.id} | set(g.served))
        if hit:
            targets.append(g.id)
            matched |= hit
    return targets, sorted(wanted - matched)


class EvictCLI(_ApprovalMixin):
    """Force-evict released (idle) models now, freeing their GPUs.

    A released ``keep-warm`` model stays resident (idle) to avoid cold-start
    thrash — handy, but it holds a GPU. ``evict`` tears such groups down now,
    overriding keep-warm. Target by served endpoint alias or group id, or
    ``--all`` for every idle group. (Live models — those with an active lease —
    are never evicted; release them first.) On a terminal the teardown is shown
    and confirmed before docker is touched (``--yes`` skips).
    """

    __command__ = 'evict'

    names = scfg.Value(
        [], nargs='*', position=1, type=str,
        help='Endpoint alias or group id to evict.',
    )
    all = scfg.Value(False, isflag=True, help='Evict every idle group.')
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)
        names = _collect_names(config.names)
        if not names and not config.all:
            raise SystemExit('evict: give an endpoint/group name or --all')
        try:
            if config.all:
                outcome = controller.evict(None)
            else:
                targets, missing = _resolve_idle_targets(controller, names)
                if missing:
                    print(f'no idle group for: {", ".join(missing)}')
                if not targets:
                    if config.json:
                        print(json.dumps(
                            {'evicted': [], 'torn_down': []}, indent=2))
                    else:
                        print('nothing to evict')
                    return 0
                outcome = controller.evict(targets)
        except ConvergeAborted:
            raise _declined_exit()
        if config.json:
            print(json.dumps({
                'evicted': outcome.evicted_group_ids,
                'torn_down': outcome.reconcile.torn_down,
            }, indent=2))
        elif not outcome.evicted_group_ids:
            print('nothing to evict')
        else:
            print(f'evicted {len(outcome.evicted_group_ids)} group(s)')
            for gid in outcome.evicted_group_ids:
                print(f'  {gid}')
        return 0


class WaitCLI(_LeasingCommonMixin):
    """Block until served endpoints are ready — the companion to ``serve
    --no-wait``.

    Fan out, then wait: ``serve --no-wait smol17b-1`` + ``serve --no-wait
    smol135-1`` kick both deployments off in parallel (each converges and starts
    its container without blocking), then ``wait smol17b-1 smol135-1`` blocks
    until they can actually serve. With no names it waits for every live group.

    ``--require-generation`` makes "ready" mean a real generated token (not just
    a model that is listed) — the same readiness *criterion* the acquire/serve
    verbs take; this command is the *blocking* half, distinct from it.
    """

    __command__ = 'wait'

    names = scfg.Value(
        [], nargs='*', position=1, type=str,
        help='Endpoint names to wait for (default: every live group).',
    )
    timeout = scfg.Value(600, type=float, help='Overall wait timeout (s).')
    interval = scfg.Value(5, type=float, help='Readiness poll interval (s).')
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        controller.ledger.sweep()
        _, groups = controller.ledger.status()
        live = [g for g in groups if g.state == GroupState.LIVE]
        names = _collect_names(config.names)
        if names:
            wanted = set(names)
            served = {ep for g in live for ep in g.served}
            missing = sorted(wanted - served)
            if missing:
                raise SystemExit(
                    f'not served by any live group: {", ".join(missing)} '
                    '(serve it first, or check `infer-stack leases`)'
                )
            targets = [g for g in live if wanted & set(g.served)]
            endpoints = wanted
        else:
            targets, endpoints = live, None
        if not targets:
            print('nothing to wait for (no live groups)')
            return 0
        result = controller.wait_ready(
            targets,
            endpoints=endpoints,
            timeout=float(config.timeout),
            interval=float(config.interval),
        )
        if config.json:
            print(json.dumps({
                'ready': result.ready,
                'pending': [
                    {'group': gid, 'endpoint': ep}
                    for gid, ep in result.pending
                ],
            }, indent=2))
        elif result.ready:
            print('ready')
        else:
            print('not ready (timed out)')
            for gid, ep in result.pending:
                print(f'  pending: {ep} ({gid})')
        return 0 if result.ready else 2


class TuiCLI(_LeasingCommonMixin):
    """Launch the Textual TUI: a live monitor of the stack with controls to
    serve / release / evict models.

    Mostly a monitor — the lease + group tables (desired state vs running, GPUs)
    refresh live — with key-bound controls: ``s`` serve, ``d`` release, ``a``
    release-all, ``e`` evict, ``r`` refresh, ``q`` quit. Opt-in extra: install
    textual with ``pip install "infer-stack[tui]"``.
    """

    __command__ = 'tui'

    catalog = scfg.Value(None, type=str, help='Path to catalog.yaml.')
    interval = scfg.Value(3.0, type=float, help='Auto-refresh interval (s).')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        try:
            from ..tui import run_tui
        except ImportError as ex:
            raise SystemExit(
                'the TUI needs the optional `textual` dependency — install it '
                'with `pip install "infer-stack[tui]"` (or `pip install textual`). '
                f'[{ex}]'
            )
        controller = _open_controller(config)
        catalog = _load_catalog(config)
        return run_tui(controller, catalog, interval=float(config.interval))


class RenewCLI(_LeasingCommonMixin):
    """Extend (or make infinite) a lease's protection window."""

    __command__ = 'renew'

    session = scfg.Value(None, position=1, type=str, help='Session id.')
    env_file = scfg.Value(None, type=str)
    ttl = scfg.Value(None, type=str, help='New soft TTL (e.g. 2h); empty=infinite.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        sid = _resolve_session(config)
        if not sid:
            raise SystemExit('renew: give a session id or --env-file')
        lease = controller.ledger.renew(
            sid, ttl_seconds=_parse_duration(config.ttl)
        )
        if lease is None:
            raise SystemExit(f'renew: no such lease {sid}')
        print(f'renewed {sid}')
        return 0


class RunCLI(_LeasingCommonMixin):
    """Acquire endpoints, run a command with the endpoint env, then release.

    Everything after ``--`` is the command. The lease is always released on
    exit; the TTL is the backstop if the process is hard-killed.
    """

    __command__ = 'run'

    catalog = scfg.Value(None, type=str, help='Path to catalog.yaml.')
    endpoint = scfg.Value(
        None,
        type=str,
        alias=['endpoints'],
        help='Comma-separated endpoint or bundle names.',
    )
    base_url = scfg.Value('http://127.0.0.1:14042/v1', type=str)
    api_key_env = scfg.Value('LITELLM_MASTER_KEY', type=str)
    owner = scfg.Value(None, type=str)
    ttl = scfg.Value('2h', type=str, help='Soft TTL backstop (default 2h).')
    timeout = scfg.Value(600, type=float)
    interval = scfg.Value(5, type=float)
    command = scfg.Value(
        [], nargs='*', position=1, type=str, help='Command to run (after --).'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        catalog = _load_catalog(config)
        names = _collect_names(config.endpoint)
        command = list(config.command or [])
        if not names:
            raise SystemExit('run: --endpoint is required')
        if not command:
            raise SystemExit('run: give a command after --')
        requests = _resolve(catalog, names)
        outcome = controller.acquire(
            config.owner or _default_owner(),
            requests,
            ttl_seconds=_parse_duration(config.ttl),
            wait=True,
            timeout=float(config.timeout),
            interval=float(config.interval),
        )
        if outcome.wait is not None and not outcome.wait.ready:
            controller.release(outcome.lease.id)
            raise SystemExit(
                f'run: endpoints not ready: {outcome.wait.pending}'
            )
        descriptor = _descriptor_for(
            controller, outcome.lease, outcome.groups, config
        )
        env = dict(os.environ)
        env.update(descriptor_env(descriptor))
        try:
            proc = subprocess.run(command, env=env)
            return int(proc.returncode)
        finally:
            controller.release(outcome.lease.id)


def _lease_ttl(le) -> str:
    return 'inf' if le.expires_at is None else f'@{le.expires_at:.0f}'


def _placement_view(controller):
    """Best-effort (observed running gids, group id -> GPU indices).

    Reads what the backend *actually* has running (``observe``) and where each
    desired group is/would be placed (``plan``). Both are guarded so a dry-run
    (``NullBackend``) or a docker-less host degrades to "unknown" rather than
    erroring — ``leases`` must always render.
    """
    backend = controller.backend
    observed: set[str] = set()
    try:
        observed = set(backend.observe())
    except Exception:  # noqa: BLE001 - status must never crash
        pass
    assignments: dict[str, list[int]] = {}
    plan = getattr(backend, 'plan', None)
    if plan is not None:
        try:
            assignments = dict(plan(controller.desired_groups()).assignments)
        except Exception:  # noqa: BLE001
            pass
    return observed, assignments


def _running_label(gid, observed) -> str:
    return 'running' if gid in observed else '—'


def _gpu_label(gid, observed, assignments) -> str:
    """Where a group is (running) or is slated to be (desired, not yet up)."""
    gpus = assignments.get(gid)
    if gpus is None:
        return '-'
    where = ','.join(str(i) for i in gpus) if gpus else 'cpu'
    return where if gid in observed else f'→{where}'  # → = slated, not yet up


def _print_leases_plain(leases, groups, observed, assignments) -> None:
    print('leases:')
    if not leases:
        print('  (none)')
    for le in leases:
        print(
            f'  {le.id}  owner={le.owner}  state={le.state}  '
            f'ttl={_lease_ttl(le)}  endpoints={",".join(le.endpoints) or "-"}'
        )
    print('groups:')
    if not groups:
        print('  (none)')
    for g in groups:
        print(
            f'  {g.id}  {g.engine}  state={g.state}  '
            f'running={_running_label(g.id, observed)}  '
            f'gpus={_gpu_label(g.id, observed, assignments)}  '
            f'demand={g.demand}  served={",".join(sorted(g.served)) or "-"}'
        )


def _print_leases_rich(leases, groups, observed, assignments, console) -> None:
    from rich.table import Table
    from rich.text import Text

    def state_style(state) -> str:
        s = str(state).lower()
        if 'active' in s or 'live' in s:
            return 'green'
        if 'stop' in s or 'expir' in s or 'idle' in s:
            return 'yellow'
        return 'dim'

    console.print(Text('leases', style='bold'))
    if not leases:
        console.print('  [dim](none)[/dim]')
    else:
        lt = Table(box=None, pad_edge=False, padding=(0, 2, 0, 0),
                   header_style='dim')
        lt.add_column('id', style='cyan', no_wrap=True)
        lt.add_column('owner')
        lt.add_column('state')
        lt.add_column('ttl', style='dim')
        lt.add_column('endpoints', style='magenta', overflow='fold')
        for le in leases:
            lt.add_row(
                le.id, le.owner,
                Text(str(le.state), style=state_style(le.state)),
                _lease_ttl(le), ','.join(le.endpoints) or '-',
            )
        console.print(lt)

    console.print(Text('groups', style='bold'))
    if not groups:
        console.print('  [dim](none)[/dim]')
    else:
        gt = Table(box=None, pad_edge=False, padding=(0, 2, 0, 0),
                   header_style='dim')
        gt.add_column('id', style='cyan', no_wrap=True)
        gt.add_column('engine')
        gt.add_column('state')           # desired (ledger)
        gt.add_column('running')         # actual (backend.observe)
        gt.add_column('gpus')            # on / →slated
        gt.add_column('demand', justify='right', style='dim')
        gt.add_column('served', style='magenta', overflow='fold')
        for g in groups:
            running = g.id in observed
            gpus = _gpu_label(g.id, observed, assignments)
            gt.add_row(
                g.id, g.engine,
                Text(str(g.state), style=state_style(g.state)),
                Text('running' if running else '—',
                     style='green' if running else 'dim'),
                Text(gpus, style='cyan' if running else 'yellow'),
                str(g.demand), ','.join(sorted(g.served)) or '-',
            )
        console.print(gt)


class LeasesCLI(_LeasingCommonMixin):
    """Show current leases and deployment groups (the leasing-model status).

    Two tables. **leases** are who asked for what (id, owner, state, ttl,
    endpoints). **groups** are the actual deployments behind them — one group is
    one model in one container (it may serve several endpoint aliases). For each
    group:

    \b
      state    what the ledger *wants* — live / idle / stopped
      running  what the backend *actually* has up right now
      gpus     the GPU indices it is on; a ``→`` prefix means slated (desired
               but not yet started), ``-`` means no placement info
      demand   how many active leases are protecting it
      served   the endpoint aliases routed to it

    A row that is ``state=live`` but ``running=—`` is desired-but-not-up: either
    still starting, staged via ``serve --no-apply`` (run ``apply`` to bring it
    up), or unplaceable (no free GPU — see ``serve``'s error / ``evict``).
    """

    __command__ = 'leases'
    __epilog__ = """
    Examples:
        infer-stack leases          # the two tables (rich on a terminal)
        infer-stack leases --json   # JSON (adds running + gpus per group)
    """

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        controller.ledger.sweep()  # materialize TTL expiry for an accurate view
        leases, groups = controller.ledger.status()
        observed, assignments = _placement_view(controller)
        if config.json:
            print(
                json.dumps(
                    {
                        'leases': [
                            {
                                'id': le.id,
                                'owner': le.owner,
                                'state': le.state,
                                'endpoints': le.endpoints,
                                'expires_at': le.expires_at,
                            }
                            for le in leases
                        ],
                        'groups': [
                            {
                                'id': g.id,
                                'engine': g.engine,
                                'state': g.state,  # desired (ledger)
                                'running': g.id in observed,  # actual (backend)
                                'gpus': assignments.get(g.id),
                                'demand': g.demand,
                                'served': sorted(g.served),
                            }
                            for g in groups
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        from rich.console import Console

        console = Console()
        if console.is_terminal:
            _print_leases_rich(leases, groups, observed, assignments, console)
        else:
            _print_leases_plain(leases, groups, observed, assignments)
        return 0


def _secret_env_path() -> Path:
    """The managed compose secrets file (.env that docker compose auto-loads)."""
    return data_root() / 'leasing' / 'compose' / '.env'


def _front_door(config) -> tuple[str, str | None]:
    """Resolve the front-door base_url + master key for a smoke test.

    Reads them straight from the managed state (front-door port + ``.env``) so
    `test` is cheap and doesn't need GPU detection or a backend object. An
    explicit ``--base-url`` overrides the derived URL.
    """
    from ..config import DEFAULT_PORTS

    base_url = getattr(config, 'base_url', None)
    if not base_url:
        port = int(getattr(config, 'port', None) or DEFAULT_PORTS['litellm'])
        base_url = f'http://127.0.0.1:{port}/v1'
    key = None
    env_path = _secret_env_path()
    if env_path.exists():
        key = parse_env_file(env_path).get('LITELLM_MASTER_KEY')
    return base_url.rstrip('/'), key


class TestCLI(_PathOverridesMixin):
    """Smoke-test a served endpoint through the front door (a real generation).

    The concise alternative to hand-rolling ``curl``: sends one chat completion
    to the endpoint *alias* via the LiteLLM gateway, then prints latency and the
    reply (or an actionable error). Exit code is non-zero on failure, so it is
    usable in scripts/CI.
    """

    __command__ = 'test'

    name = scfg.Value(
        None, position=1, type=str, help='Endpoint alias to test (e.g. chat).'
    )
    prompt = scfg.Value(
        'Reply with the single word: ready.', type=str, help='Prompt to send.'
    )
    max_tokens = scfg.Value(32, type=int)
    timeout = scfg.Value(60, type=float, help='Request timeout (s).')
    base_url = scfg.Value(
        None, type=str, help='Override the gateway base URL (…/v1).'
    )
    port = scfg.Value(
        None, type=int, help='Override the gateway port (default: 14042).'
    )
    json = scfg.Value(False, isflag=True, help='Emit JSON instead of text.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        import time

        import requests

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.name:
            raise SystemExit('test: give an endpoint alias (e.g. `test chat`)')
        base_url, key = _front_door(config)
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = f'Bearer {key}'
        payload = {
            'model': config.name,
            'messages': [{'role': 'user', 'content': config.prompt}],
            'max_tokens': int(config.max_tokens),
        }
        t0 = time.monotonic()
        try:
            resp = requests.post(
                f'{base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=float(config.timeout),
            )
        except requests.exceptions.RequestException as ex:
            return _test_fail(config, base_url, f'not reachable: {ex}')
        dt = time.monotonic() - t0
        if resp.status_code >= 400:
            body = (resp.text or '').strip()[:300]
            return _test_fail(
                config, base_url, f'HTTP {resp.status_code}: {body}'
            )
        try:
            reply = resp.json()['choices'][0]['message']['content']
        except (ValueError, KeyError, IndexError) as ex:
            return _test_fail(config, base_url, f'unexpected response: {ex}')
        reply = (reply or '').strip()
        if config.json:
            print(json.dumps(
                {'endpoint': config.name, 'ok': True,
                 'seconds': round(dt, 3), 'reply': reply}, indent=2))
        else:
            print(f'{config.name}: ok ({dt:.2f}s) {reply!r}')
        return 0


def _test_fail(config, base_url: str, reason: str) -> int:
    if config.json:
        print(json.dumps(
            {'endpoint': config.name, 'ok': False,
             'base_url': base_url, 'reason': reason}, indent=2))
    else:
        print(f'{config.name}: FAILED via {base_url} — {reason}')
        print('  is it served?  infer-stack leases   |   '
              'infer-stack serve ' + str(config.name))
    return 1


class EnvCLI(_PathOverridesMixin):
    """The managed env-file: print its path, read a value, or set one.

    infer-stack keeps managed secrets — the LiteLLM master key, ``HF_TOKEN``,
    … — in a ``.env`` that docker compose auto-loads. Anyone who can read it
    already has the secrets, so there's nothing to hide behind a separate
    ``secret`` verb; one ``env`` does it all:

    \b
      infer-stack env                       # the .env path (source it to load)
      infer-stack env LITELLM_MASTER_KEY    # print one value
      infer-stack env HF_TOKEN=hf_…         # set one value (merges, before serve)
      infer-stack env --export              # every entry as `export KEY=value`

    The argument is a KEY to read, or ``KEY=VALUE`` to write (writes merge
    non-destructively, so the managed LiteLLM key is preserved).
    """

    __command__ = 'env'

    arg = scfg.Value(
        None, position=1, type=str,
        help='KEY to read its value, or KEY=VALUE to set it. Empty = path.',
    )
    export = scfg.Value(
        False, isflag=True, help='Print every entry as `export KEY=value`.'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        env_path = _secret_env_path()

        # Write: `env KEY=VALUE`
        if config.arg and '=' in config.arg:
            key, _, value = config.arg.partition('=')
            key = key.strip()
            if not key:
                raise SystemExit('env: empty key in KEY=VALUE')
            write_env_file(env_path, {key: value})
            print(f'set {key} ({env_path})')
            return 0

        # Path first and foremost (it may not exist yet — that's fine).
        if not (config.arg or config.export):
            print(env_path)
            return 0

        # Read: `env KEY` / `env --export`
        if not env_path.exists():
            raise SystemExit(
                f'no managed env-file at {env_path}; run an `acquire`/`serve` '
                'with --backend compose first (or `infer-stack env KEY=VALUE`)'
            )
        env = parse_env_file(env_path)
        if config.arg:
            if config.arg not in env:
                raise SystemExit(f'{config.arg!r} not found in {env_path}')
            print(env[config.arg])
            return 0
        for name, value in env.items():
            print(f'export {name}={shlex.quote(value)}')
        return 0
