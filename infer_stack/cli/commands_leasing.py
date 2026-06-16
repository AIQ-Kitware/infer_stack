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
import subprocess
from pathlib import Path

import scriptconfig as scfg

from ..leasing import (
    Catalog,
    CatalogError,
    Controller,
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
from ..paths import config_root
from .context import _apply_path_overrides
from .options import _PathOverridesMixin

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


def _make_backend(name):
    if name in (None, '', 'null', 'dry-run'):
        return NullBackend()
    raise SystemExit(
        f'backend {name!r} is not implemented in the leasing CLI yet; the '
        'Compose/KubeAI backends land in a later stage. Use --backend null '
        '(dry-run) for now.'
    )


def _open_controller(config) -> Controller:
    _apply_path_overrides(config)
    ledger_path = config.ledger or str(default_ledger_path())
    ledger = Ledger(SqliteStore(ledger_path))
    return Controller(ledger, _make_backend(config.backend))


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


def _emit_acquire(config, outcome) -> int:
    descriptor = build_descriptor(
        outcome.lease,
        outcome.groups,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
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
        if config.env_file:
            print(f'  env-file: {config.env_file}')
    return 2 if not_ready else 0


def _do_acquire(config, *, owner: str, ttl_seconds: float | None) -> int:
    controller = _open_controller(config)
    catalog = _load_catalog(config)
    names = _collect_names(config.names)
    if not names:
        raise SystemExit('give at least one endpoint or bundle name')
    sharing = Sharing.DEDICATED if getattr(config, 'dedicated', False) else None
    requests = _resolve(catalog, names, sharing=sharing)
    outcome = controller.acquire(
        owner,
        requests,
        ttl_seconds=ttl_seconds,
        wait=bool(config.wait),
        timeout=float(config.timeout),
        interval=float(config.interval),
    )
    return _emit_acquire(config, outcome)


# ---------------------------------------------------------------------------
# shared flag mixins
# ---------------------------------------------------------------------------


class _LeasingCommonMixin(_PathOverridesMixin):
    backend = scfg.Value(
        'null',
        choices=['null', 'compose', 'kubeai'],
        help='Serving backend. Only "null" (dry-run) is implemented so far.',
    )
    ledger = scfg.Value(
        None, type=str, help='Path to the lease ledger sqlite db.'
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
    timeout = scfg.Value(600, type=float, help='Readiness wait timeout (s).')
    interval = scfg.Value(5, type=float, help='Readiness poll interval (s).')
    env_file = scfg.Value(
        None, type=str, help='Write the sourceable endpoint env-file here.'
    )
    json = scfg.Value(False, isflag=True, help='Emit JSON instead of text.')


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


class AcquireCLI(_AcquireFlagsMixin):
    """Acquire a lease on one or more endpoints/bundles and block until ready."""

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
    """Stand up endpoints as a standing service (an infinite ``manual`` lease)."""

    __command__ = 'serve'

    names = scfg.Value(
        [], nargs='*', position=1, type=str, help='Endpoint or bundle names.'
    )
    owner = scfg.Value('manual', type=str, help='Lease owner.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _do_acquire(config, owner=config.owner, ttl_seconds=None)


class ReleaseCLI(_LeasingCommonMixin):
    """Release a lease; deployments idle/teardown per their reclaim policy."""

    __command__ = 'release'

    session = scfg.Value(
        None, position=1, type=str, help='Session id (or use --env-file).'
    )
    env_file = scfg.Value(
        None, type=str, help='Read the session id from this env-file.'
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        sid = _resolve_session(config)
        if not sid:
            raise SystemExit('release: give a session id or --env-file')
        outcome = controller.release(sid)
        if config.json:
            print(
                json.dumps(
                    {
                        'released': sid,
                        'idled': outcome.idled_group_ids,
                        'torn_down': outcome.reconcile.torn_down,
                    },
                    indent=2,
                )
            )
        else:
            print(f'released {sid}')
            for gid in outcome.reconcile.torn_down:
                print(f'  torn down: {gid}')
        return 0


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
        descriptor = build_descriptor(
            outcome.lease,
            outcome.groups,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
        )
        env = dict(os.environ)
        env.update(descriptor_env(descriptor))
        try:
            proc = subprocess.run(command, env=env)
            return int(proc.returncode)
        finally:
            controller.release(outcome.lease.id)


class LeasesCLI(_LeasingCommonMixin):
    """Show current leases and deployment groups (the leasing-model status)."""

    __command__ = 'leases'

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        controller.ledger.sweep()  # materialize TTL expiry for an accurate view
        leases, groups = controller.ledger.status()
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
                                'state': g.state,
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
        print('leases:')
        if not leases:
            print('  (none)')
        for le in leases:
            ttl = 'inf' if le.expires_at is None else f'@{le.expires_at:.0f}'
            print(
                f'  {le.id}  owner={le.owner}  state={le.state}  '
                f'ttl={ttl}  endpoints={",".join(le.endpoints) or "-"}'
            )
        print('groups:')
        if not groups:
            print('  (none)')
        for g in groups:
            print(
                f'  {g.id}  {g.engine}  state={g.state}  demand={g.demand}  '
                f'served={",".join(sorted(g.served)) or "-"}'
            )
        return 0
