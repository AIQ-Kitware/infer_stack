"""Leasing subcommands: the acquire/release/run surface for the new model.

These verbs drive the leasing controller (ledger + backend) instead of the
legacy "active profile". They are what a kwdagger pipeline node uses to request
the models it needs, block until ready, and release after:

    infer-stack acquire qwen-coder reranker --ttl 2h --env-file is.env
    infer-stack run --endpoint qwen-coder -- python my_node.py
    infer-stack release --env-file is.env
    infer-stack acquire qwen-coder      # standing service (no --ttl)
    infer-stack leases                  # status of leases + deployment deployments

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
import sys
from typing import Any
from pathlib import Path

import scriptconfig as scfg

from ..env_utils import parse_env_file, write_env_file
from ..leasing import (
    Catalog,
    CatalogError,
    ComposeBackend,
    Controller,
    DeploymentState,
    Ledger,
    LeaseState,
    NullBackend,
    Sharing,
    SqliteStore,
    default_ledger_path,
    is_reservation,
    reservation_request,
)
from ..leasing.envfile import (
    build_descriptor,
    descriptor_env,
    read_lease_id,
    render_env_file,
)
from ..paths import config_root, data_root
from .context import _apply_path_overrides
from .options import _AllowedGpusMixin, _DisplayGpuMixin, _PathOverridesMixin

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


# Managed .env values that participate in rendered service behaviour.  Changing
# one while a lease is active would make a later unrelated publication recreate
# serving containers with a new fingerprint.  Treat them as configuration-time
# state; client-only values such as OPENAI_BASE_URL may still change freely.
_SERVICE_ENV_KEYS = frozenset({
    'HF_TOKEN', 'LITELLM_MASTER_KEY', 'LITELLM_DB_PASSWORD',
})


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


def _resolve_litellm(config) -> bool:
    """LiteLLM gateway on/off: explicit --litellm/--no-litellm wins, else
    `config set litellm`, else on by default.

    With the gateway off there is no single OpenAI ``base_url`` fronting every
    alias; Open WebUI (if on) then talks to the rendered upstreams directly —
    handy for a lean ``ollama + Open WebUI`` stack that manages its own models.
    """
    from ..paths import get_setting

    flag = getattr(config, 'litellm', None)
    if flag is not None:
        return bool(flag)
    return _coerce_bool(get_setting('litellm'), True)


def _resolve_dynamic_routing(config) -> bool:
    """Dynamic LiteLLM routing (admin API + Postgres) on/off.

    Off by default (the static-superset gateway). When on, each deployment gets
    its own upstream container and the gateway's routes are managed live via the
    admin API against a Postgres model store — so same-model ``--dedicated``
    deployments land on distinct GPUs and the gateway is never recreated (no
    blip). Enable with ``--dynamic-routing`` or ``config set dynamic_routing
    true``. Because every verb (acquire/release/gc/apply) must agree on the mode
    to render consistent service names, the persisted setting is the primary
    switch; the flag is a per-invocation override.
    """
    from ..paths import get_setting

    flag = getattr(config, 'dynamic_routing', None)
    if flag is not None:
        return bool(flag)
    return _coerce_bool(get_setting('dynamic_routing'), False)


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

    Only the additive verb (``acquire``) prompts, and only on a real terminal
    without ``--yes``. Everything else (release/leases/run/non-TTY) auto-applies.

    Both streams are checked, because the prompt writes to one and reads from
    the other. Testing stdout alone meant that under ``pytest -s`` -- where
    stdout is the terminal -- acquire decided to prompt and then blocked
    forever on a stdin nothing was going to type into.
    """
    import sys

    if not interactive:
        return True
    if getattr(config, 'yes', False):
        return True
    answerable = sys.stdin is not None and sys.stdin.isatty()
    return not (answerable and sys.stdout.isatty())


def _make_backend(config, *, interactive: bool = False):
    from ..paths import get_setting

    # Resolve the backend: explicit --backend wins, then the environment, then
    # the persisted default (`config set backend compose`), else the dry-run
    # null backend.
    #
    # The env rung exists for the same reason as INFER_STACK_CATALOG: a DAG
    # whose nodes lease their own endpoints generates `infer-stack run`
    # commands, and those commands must not have to name the machine's
    # backend -- that would make the pipeline machine-specific. Exporting it
    # once, outside the DAG, also avoids silently falling through to the null
    # backend and producing a run where nothing was ever served.
    name = (
        getattr(config, 'backend', None)
        or os.environ.get('INFER_STACK_BACKEND', '').strip()
        or get_setting('backend')
        or 'null'
    )
    if name in (None, '', 'null', 'dry-run'):
        return NullBackend()
    if name == 'compose':
        rp_enabled, rp_port, rp_config = _resolve_reverse_proxy(config)
        # Best-effort catalog so the LiteLLM gateway gets a static superset route
        # table (one route per catalog endpoint) and is never recreated as models
        # come/go. Loaded on EVERY converge — including release/gc, which don't
        # take a catalog arg — so no-blip holds across acquire AND release; for
        # that the catalog must be discoverable (default path or always --catalog).
        try:
            catalog = _load_catalog(config)
        except SystemExit:
            catalog = None  # no catalog -> legacy per-deployment gateway config
        return ComposeBackend(
            state_dir=data_root() / 'leasing' / 'compose',
            # None => the backend detects lazily on first placement, so no verb
            # (and especially not the TUI's first frame) blocks on the
            # nvidia-smi subprocess at construction time.
            inventory=None,
            allowed_gpus=_parse_gpus(getattr(config, 'allowed_gpus', None)),
            skip_display=_resolve_skip_display(config),
            litellm=_resolve_litellm(config),
            ui=_resolve_ui(config),
            reverse_proxy=rp_enabled,
            reverse_proxy_port=rp_port,
            reverse_proxy_config=rp_config,
            require_generation=bool(getattr(config, 'require_generation', False)),
            assume_yes=_resolve_assume_yes(config, interactive=interactive),
            catalog=catalog,
            dynamic_routing=_resolve_dynamic_routing(config),
        )
    if name == 'kubeai':
        from ..backends.kubeai import KubeaiBackend

        gateway = None
        if _resolve_litellm(config):
            # The same LiteLLM front door as the compose backend, fronting the
            # cluster: one base_url, the managed key, and endpoint aliases as
            # request names. Its own state dir and compose project, so it can
            # never touch a compose stack's containers on the same host.
            gateway = ComposeBackend(
                state_dir=data_root() / 'leasing' / 'kubeai-gateway',
                inventory={'gpu_count': 0, 'gpus': []},
                project='infer-stack-gateway',
                litellm=True,
                ui=False,
                assume_yes=_resolve_assume_yes(config, interactive=interactive),
            )
        backend = KubeaiBackend(
            state_dir=data_root() / 'leasing' / 'kubeai',
            namespace=get_setting('kubeai_namespace') or 'kubeai',
            base_url=get_setting('kubeai_base_url') or None,
            default_resource_profile=get_setting('kubeai_resource_profile')
            or None,
            assume_yes=_resolve_assume_yes(config, interactive=interactive),
            gateway=gateway,
            gateway_upstream=get_setting('kubeai_gateway_upstream') or None,
        )
        try:
            backend.catalog = _load_catalog(config)   # frozen into the profile
        except SystemExit:
            backend.catalog = None
        return backend
    raise SystemExit(
        f'backend {name!r} is not implemented in the leasing CLI. '
        'Use --backend null, compose, or kubeai.'
    )


def _open_controller(config, *, interactive: bool = False) -> Controller:
    from .._log import configure_logging

    configure_logging()  # narrate the leasing verbs (stderr); off the help path
    _apply_path_overrides(config)
    ledger_path = config.ledger or str(default_ledger_path())
    ledger = Ledger(SqliteStore(ledger_path))
    return Controller(ledger, _make_backend(config, interactive=interactive))


def _catalog_path(config) -> Path:
    """Resolve the catalog: ``--catalog``, then the env, then the default.

    The env fallback exists for generated job scripts. A pipeline that leases
    its own endpoints per node cannot hard-code a catalog path without
    becoming a different pipeline for a rehearsal than for a real run --
    which defeats the purpose of rehearsing. Exporting
    ``INFER_STACK_CATALOG`` once, outside the DAG, lets the same generated
    command serve both.

    ``--catalog`` still wins, so nothing that passes it explicitly changes.
    """
    # `catalog` is declared only on the verbs that take a --catalog arg
    # (acquire/run/tui); converge verbs that don't (release/evict/gc) still load
    # it for no-blip, so fall back when the field is absent rather than raising
    # AttributeError off a bare `config.catalog`.
    raw = (
        getattr(config, 'catalog', None)
        or os.environ.get('INFER_STACK_CATALOG', '').strip()
        or (config_root() / 'catalog.yaml')
    )
    return Path(raw).expanduser()


def _load_catalog(config) -> Catalog:
    path = _catalog_path(config)
    if not path.exists():
        raise SystemExit(
            f'catalog not found: {path} (pass --catalog or create catalog.yaml)'
        )
    try:
        return Catalog.load(path)
    except CatalogError as ex:
        raise SystemExit(f'invalid catalog {path}: {ex}')


def _load_catalog_for_tui(config) -> tuple[Catalog, Path]:
    """Catalog + its path for the TUI, tolerating a missing/empty file.

    The TUI is the place a brand-new user lands, so an absent catalog is not a
    hard error here: it loads an empty catalog (the dashboard then shows the
    empty-state with a Suggest button) and returns the path it would write to.
    """
    path = _catalog_path(config)
    if not path.exists():
        return Catalog.from_dict({'models': {}, 'endpoints': {}}), path
    try:
        return Catalog.load(path), path
    except CatalogError as ex:
        raise SystemExit(f'invalid catalog {path}: {ex}')


def _requests_catalog(controller, config):
    """The user catalog endpoint names resolve against.

    The catalog on disk is the authoritative configuration surface.  The
    controller's persisted profile is only an internal recovery snapshot and
    must not become a second catalog the user has to manage.  When a current
    catalog exists, resolve the requested endpoint from it; acquire will merge
    compatible additions into the recovery snapshot under the publication lock.

    A published union is only the fallback for advanced runbooks that invoke a
    lease command without any local/default catalog at all.
    """
    from ..leasing.profile import CatalogUnion

    published = getattr(controller.backend, 'catalog', None)
    explicit = (
        getattr(config, 'catalog', None)
        or os.environ.get('INFER_STACK_CATALOG', '').strip()
    )
    path = _catalog_path(config)
    if explicit or path.exists():
        return _load_catalog(config)
    if isinstance(published, CatalogUnion):
        return published
    return _load_catalog(config)


def _resolve(catalog, names, *, sharing=None):
    try:
        return catalog.resolve_names(names, sharing=sharing)
    except CatalogError as ex:
        raise SystemExit(str(ex))


def _resolve_lease(config) -> str | None:
    sid = getattr(config, 'lease', None)
    if not sid and getattr(config, 'env_file', None):
        sid = read_lease_id(config.env_file)
    return sid


def _descriptor_for(controller, lease, deployments, config, *, assignments=None):
    """Build the descriptor, preferring backend-supplied access (real base_url).

    For a GPU *reservation* the env-file's only useful payload is the reserved
    GPU index(es): fold the placement assignment into ``cuda_visible_devices`` so
    a consumer that ``source``s the env-file runs its own process on exactly the
    GPU infer-stack held for it.
    """
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
    cuda_visible_devices = None
    if assignments:
        reserved_gpus: list[int] = []
        for deployment in deployments:
            if is_reservation(deployment):
                reserved_gpus.extend(assignments.get(deployment.id, []))
        if reserved_gpus:
            cuda_visible_devices = ','.join(str(i) for i in reserved_gpus)
    return build_descriptor(
        lease,
        deployments,
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
        request_names=request_names,
        cuda_visible_devices=cuda_visible_devices,
    )


def _public_descriptor(descriptor: dict) -> dict:
    """The descriptor as printed on stdout: real key material redacted.

    ``--json`` output lands in job logs that get collected, rsynced, and
    shared — the key must not travel with them. Consumers get the key from the
    env-file (the delivery mechanism, which keeps the real value) or via
    ``infer-stack env $api_key_env``. The 'EMPTY' placeholder (keyless direct
    upstreams) is not a secret and passes through untouched.
    """
    key = descriptor.get('api_key')
    if not key or key == 'EMPTY':
        return descriptor
    public = dict(descriptor)
    public['api_key'] = (
        f'<redacted — source the env-file or run'
        f' `infer-stack env {descriptor.get("api_key_env", "")}`>'
    )
    return public


def _compose_file_path(controller) -> str | None:
    path = getattr(controller.backend, 'compose_file', None)
    return str(path) if path else None


def _gpu_where(gpus) -> str:
    """Human label for a deployment's GPU assignment (or its absence)."""
    if gpus is None:
        return 'unplaced'
    if not gpus:
        return 'cpu'
    return 'GPU ' + ','.join(str(i) for i in gpus)


def _emit_staged(config, controller, outcome) -> int:
    """Output for a ``--no-apply`` acquire: what was written + what would run."""
    assignments = outcome.reconcile.assignments
    descriptor = _descriptor_for(
        controller, outcome.lease, outcome.deployments, config,
        assignments=assignments,
    )
    if config.env_file:
        Path(config.env_file).expanduser().write_text(render_env_file(descriptor))
    if config.json:
        print(json.dumps({
            'lease_id': outcome.lease.id,
            'owner': outcome.lease.owner,
            'applied': False,
            'descriptor': _public_descriptor(descriptor),
            'compose_file': _compose_file_path(controller),
            'placement': [
                {'deployment': g.id, 'served': sorted(g.served),
                 'gpus': assignments.get(g.id)}
                for g in outcome.deployments
            ],
        }, indent=2))
        return 0
    print(f'staged {outcome.lease.id} (owner={outcome.lease.owner}) — not applied')
    for g in outcome.deployments:
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


def _oom_hints(controller, outcome) -> list[str]:
    """Guided-failure diagnosis for a not-ready acquire (vram-aware-placement §3).

    When a pending endpoint's engine log shows CUDA OOM, the failure is a
    *diagnosed misdeclaration*, not a generic crash: report the GPU it OOM'd
    on, the declared requirement (or its absence), and the exact command
    that computes the right number. Fail-open — no logs, no hints.
    """
    from ..leasing.placement import declared_min_vram
    from ..leasing.vram import looks_like_cuda_oom

    backend = controller.backend
    logs_fn = getattr(backend, 'deployment_logs', None)
    if logs_fn is None or outcome.wait is None or outcome.wait.ready:
        return []
    by_id = {g.id: g for g in outcome.deployments}
    try:
        memory = {
            g['index']: g.get('memory_gib')
            for g in getattr(backend, 'inventory', {}).get('gpus', [])
        }
    except Exception:
        memory = {}
    assignments = dict(getattr(backend, 'last_assignments', {}) or {})
    hints: list[str] = []
    for gid, endpoint in outcome.wait.pending:
        deployment = by_id.get(gid)
        if deployment is None:
            continue
        text = logs_fn(deployment)
        if not text or not looks_like_cuda_oom(text):
            continue
        gpus = assignments.get(gid, [])
        where = (
            ', '.join(f'gpu{i}={memory.get(i)}GiB' for i in gpus)
            if gpus else 'its GPU'
        )
        declared = declared_min_vram(deployment)
        declared_txt = (
            f'declared min_vram_gib={declared:g} looks too low'
            if declared
            else 'no min_vram_gib is declared for it'
        )
        hints.append(
            f'{endpoint}: engine log shows CUDA OUT-OF-MEMORY on {where} — '
            f'{declared_txt}. Compute the real requirement with: '
            f'infer-stack measure {endpoint} --record'
        )
    return hints


def _emit_acquire(config, controller, outcome) -> int:
    if not outcome.applied:
        return _emit_staged(config, controller, outcome)
    descriptor = _descriptor_for(
        controller, outcome.lease, outcome.deployments, config,
        assignments=outcome.reconcile.assignments,
    )
    # A readiness timeout means the controller already released the lease, so
    # there is no standing endpoint to point a sourceable env-file at.
    if config.env_file and not outcome.released_on_timeout:
        Path(config.env_file).expanduser().write_text(
            render_env_file(descriptor)
        )
    not_ready = outcome.wait is not None and not outcome.wait.ready
    if config.json:
        print(
            json.dumps(
                {
                    'lease_id': outcome.lease.id,
                    'owner': outcome.lease.owner,
                    'descriptor': _public_descriptor(descriptor),
                    'realized': outcome.reconcile.realized,
                    'ready': None
                    if outcome.wait is None
                    else outcome.wait.ready,
                    'pending': []
                    if outcome.wait is None
                    else outcome.wait.pending,
                    'failures': []
                    if outcome.wait is None
                    else outcome.wait.failures,
                    'released_on_timeout': outcome.released_on_timeout,
                },
                indent=2,
            )
        )
    elif outcome.released_on_timeout:
        # The readiness wait ended without readiness; the controller released
        # the lease so it doesn't pin a GPU. Report the teardown, not a phantom
        # "acquired". A crash-looping engine ends the wait early and says why.
        failures = outcome.wait.failures if outcome.wait else []
        if failures:
            print(f'engine cannot start — lease {outcome.lease.id} released')
        else:
            print(
                f'not ready within {config.timeout:.0f}s — '
                f'lease {outcome.lease.id} released'
            )
        for gid, endpoint, detail in failures:
            print(f'  {endpoint} ({gid}): {detail}')
        for gid, endpoint in outcome.wait.pending:
            if any(gid == g and endpoint == ep for g, ep, _ in failures):
                continue
            print(f'  pending: {endpoint} ({gid})')
        for hint in _oom_hints(controller, outcome):
            print(f'  {hint}')
        print('  (use --no-wait to hold a lease while a slow model loads)')
    else:
        print(f'acquired {outcome.lease.id} (owner={outcome.lease.owner})')
        for endpoint, model in descriptor['endpoints'].items():
            print(f'  endpoint {endpoint} -> {model}')
        if outcome.wait is not None:
            print(f'  ready: {outcome.wait.ready}')
            for gid, endpoint in outcome.wait.pending:
                print(f'    pending: {endpoint} ({gid})')
            for hint in _oom_hints(controller, outcome):
                print(f'    {hint}')
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
    from ..leasing.profile import ProfileMismatch

    render_only = not bool(getattr(config, 'apply', True))
    # Staging (--no-apply) is non-destructive (no docker up), so don't gate it
    # behind the diff prompt — the rendered file *is* the thing you asked to see.
    controller = _open_controller(config, interactive=not render_only)
    reserve_gpus = int(getattr(config, 'reserve_gpus', 0) or 0)
    if reserve_gpus > 0:
        # Reservation mode: hold N available GPUs, serve nothing. No catalog, no
        # endpoint names — just a count that first-fits free GPUs.
        if _collect_names(config.names):
            raise SystemExit(
                '--reserve-gpus takes no endpoint names (it reserves GPUs, not '
                'a served model)'
            )
        requests = [reservation_request(reserve_gpus)]
        names = [f'{reserve_gpus} gpu(s)']
    else:
        catalog = _requests_catalog(controller, config)
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
            wait_for_placement=bool(getattr(config, 'queue', False)),
        )
    except ConvergeAborted:
        raise SystemExit('aborted: compose changes not applied (no lease kept)')
    except ProfileMismatch as ex:
        raise SystemExit(f'acquire: {ex}')
    except PlacementError as ex:
        lines = ['could not place every requested endpoint (no lease kept):']
        lines += [f'  {r}' for r in ex.reasons] or [
            f'  {", ".join(ex.deployment_ids)}'
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
        help='Serving backend: "null" (dry-run), "compose" (single-host '
        'docker), or "kubeai" (cluster; see docs/kubeai-backend.md). '
        'Defaults to `config set backend …`, else "null".',
    )
    ledger = scfg.Value(
        None, type=str, help='Path to the lease ledger sqlite db.'
    )
    require_generation = scfg.Value(
        False,
        isflag=True,
        help='Deprecated/no-op: readiness now ALWAYS verifies a real generation '
        '(a listed alias or a running container is not proof the model serves). '
        'Accepted for compatibility.',
    )
    litellm = scfg.Value(
        None,
        isflag=True,
        help='Render the LiteLLM gateway — one OpenAI base_url fronting every '
        'endpoint alias (compose backend). On by default; use --no-litellm for '
        'a lean stack where Open WebUI talks to the upstreams (e.g. an Ollama '
        'daemon) directly. Overrides `config set litellm …`.',
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
    dynamic_routing = scfg.Value(
        None,
        isflag=True,
        alias=['dynamic-routing'],
        help='Manage the LiteLLM gateway routes LIVE via its admin API against a '
        'Postgres model store, instead of a static config file. Gives each '
        'deployment its own upstream (so same-model --dedicated deployments land '
        'on distinct GPUs) with no gateway recreation/blip. Off by default; the '
        'mode must be consistent across verbs, so prefer `config set '
        'dynamic_routing true`. Overrides that setting per-invocation.',
    )


class _ApprovalMixin(_LeasingCommonMixin):
    """A verb that converges the compose project and so gates on a diff.

    Any change to the on-disk compose project is shown and confirmed on a
    terminal; ``--yes`` (or a non-TTY) applies without prompting.
    """

    catalog = scfg.Value(
        None, type=str,
        help='Path to catalog.yaml. release/gc/evict reconcile the gateway too, '
        'so pass the same catalog as acquire to keep the static superset route '
        'table (no gateway blip); omitted, it falls back to the default-path '
        'catalog, else legacy per-deployment routing.',
    )
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
    queue = scfg.Value(
        False, isflag=True,
        help='Admission queue: if every GPU is busy, WAIT for one to free '
        '(up to --timeout) instead of failing fast. Each retry sweeps the '
        'ledger, so a crashed job\'s TTL-expired lease is reclaimed while '
        'waiting. Intended for batch/pipeline fan-out; interactive use '
        'defaults off (fail fast with a clear "no GPU" error).',
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
    """Acquire a lease on one or more endpoints/bundles, bring them up, wait.

    ``acquire NAME…`` is the everyday verb. It takes a lease on each endpoint or
    bundle, renders the compose project (the LiteLLM gateway + one container per
    model + a managed Open WebUI), brings it up, and blocks until every endpoint
    is ready. Run it again with more names to add models side by side — the
    gateway and UI stay put. Placement, ``docker compose``, and readiness are
    narrated on stderr.

    With no ``--ttl`` the lease is infinite — a standing service you tear down
    explicitly (``release`` / ``evict``). Pass ``--ttl`` (e.g. ``2h``, ``30m``)
    for a soft, time-boxed reservation: the lease protects its models while
    held, and once it expires (or is released) and nothing else needs them they
    can be reclaimed. This is also the programmatic seam — write a sourceable
    endpoint env-file with ``--env-file`` and release by that file or lease id
    when done. (For a one-shot "acquire, run a command, release", use
    ``infer-stack run`` instead.)

    The work is render (write the on-disk compose project) then apply (``docker
    compose up``). ``--no-apply`` does just the render so you can see what
    would run before pulling the trigger (then ``infer-stack apply``).
    ``--no-wait`` applies but returns immediately so several models load in
    parallel (``wait`` for them later). ``--no-ui`` skips Open WebUI. On a
    terminal you are shown the compose diff and asked before applying; ``--yes``
    skips that prompt (and it is skipped automatically off a TTY).
    """

    __command__ = 'acquire'
    __epilog__ = """
    Examples:
        # stand up one model: render + up + wait until it can generate
        infer-stack acquire qwen05-1 --require-generation

        # a time-boxed reservation with a sourceable env-file, released by id
        infer-stack acquire qwen-coder --ttl 2h --env-file is.env

        # stage only: write the compose project but don't start it, then apply
        infer-stack acquire qwen05-1 --no-apply
        infer-stack apply

        # fan out: start several without blocking, then wait together
        infer-stack acquire smol17b-1 --no-wait --yes
        infer-stack acquire qwen15-1  --no-wait --yes
        infer-stack wait    smol17b-1 qwen15-1 --require-generation

        # see what is actually running and on which GPUs
        infer-stack leases
    """

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
    reserve_gpus = scfg.Value(
        0,
        type=int,
        alias=['reserve-gpus'],
        help='Reserve N *available* GPUs without launching any server (N is a '
        'count, not an index — infer-stack first-fits free GPUs). The lease holds '
        'the GPU(s) out of the placement pool so concurrent served runs skip '
        'them, and reports the chosen index(es) via the env-file\'s '
        'CUDA_VISIBLE_DEVICES so you can run your own process (e.g. HELM\'s '
        'in-process HuggingFaceClient) on exactly those GPUs. Takes no endpoint '
        'names; use --ttl/--queue/--env-file as usual, release by lease id or '
        'env-file.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        return _do_acquire(
            config,
            owner=config.owner or _default_owner(),
            ttl_seconds=_parse_duration(config.ttl),
        )


class RenderCLI(_LeasingCommonMixin):
    """Write the on-disk compose project for the current desired set — no up.

    Lease-free and idempotent: ``render`` re-materializes the manifest (compose
    file + gateway config + GPU placement) from whatever is *already* declared,
    WITHOUT starting anything — to inspect what would run, or refresh a file you
    touched by hand. It creates no lease; to stage a *new* endpoint use ``acquire
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
    declared (by ``acquire``) onto the backend. It is the *trigger*
    for a staged ``acquire --no-apply`` and the re-sync button after a manual edit
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
            # apply_now() always runs the docker up and publishes anything
            # pending -- this is the "re-sync after a manual edit / backend
            # hiccup" button, so it must heal drift.
            rec = controller.apply_now()
        except ConvergeAborted:
            raise SystemExit('aborted: compose changes not applied')
        result = None
        if config.wait:
            _, deployments = controller.ledger.status(virtual_expiry=True)
            live = [g for g in deployments if g.state == DeploymentState.LIVE]
            if live:
                result = controller.wait_ready(
                    live,
                    timeout=float(config.timeout),
                    interval=float(config.interval),
                )
        # Exit 3: the apply ran but did not fully take effect (e.g. gateway
        # routes did not verify); the change stays pending.
        pending_exit = 3 if rec.publication_pending else None
        if config.json:
            print(json.dumps({
                'applied': not rec.publication_pending,
                'publication_pending': rec.publication_pending,
                'realized': rec.realized,
                'torn_down': rec.torn_down,
                'placement': rec.assignments,
                'unplaced': rec.unplaced,
                'ready': None if result is None else result.ready,
            }, indent=2))
            if pending_exit:
                return pending_exit
            return 0 if (result is None or result.ready) else 2
        print(
            f'applied: {len(rec.realized)} started, {len(rec.torn_down)} stopped'
        )
        if rec.publication_pending:
            print('  ! the apply did not fully take effect; the change is still '
                  'pending -- retry `infer-stack apply`')
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
            return pending_exit or 2
        return pending_exit or 0


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

    lease = scfg.Value(
        None, position=1, type=str, help='Lease id (or use --env-file).'
    )
    env_file = scfg.Value(
        None, type=str, help='Read the lease id from this env-file.'
    )
    all = scfg.Value(
        False, isflag=True,
        help='Release every active lease (the whole stack idles/tears down).',
    )
    evict = scfg.Value(
        False, isflag=True,
        help='Also evict (tear down) the released deployment(s) now, even if their '
        'reclaim policy is keep-warm — frees the GPU immediately.',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)

        if config.all:
            if config.lease or config.env_file:
                raise SystemExit('release: --all takes no lease/--env-file')
            targets = None
        else:
            sid = _resolve_lease(config)
            if not sid:
                raise SystemExit(
                    'release: give a lease id, --env-file, or --all'
                )
            targets = [sid]

        # One publication for the whole command -> at most one diff prompt.
        try:
            out = controller.release_leases(targets, evict=bool(config.evict))
        except ConvergeAborted:
            raise _declined_exit()
        if out.missing_lease_ids:
            raise SystemExit(f'release: no such lease: {out.missing_lease_ids[0]}')
        # A single named lease is reported even when it was already released
        # (idempotent release; a cleanup trap may fire twice).
        released = targets if targets is not None else out.released_lease_ids
        rec = out.reconcile
        return _emit_release(
            config, released, sorted(rec.torn_down) if rec else [],
            out.evicted_deployment_ids,
            pending=bool(rec and rec.publication_pending),
        )


def _emit_release(config, released, torn_down, evicted, *, pending=False) -> int:
    if config.json:
        print(json.dumps({
            'released': released,
            'torn_down': torn_down,
            'evicted': evicted,
            'publication_pending': pending,
        }, indent=2))
        return 3 if pending else 0
    if not released:
        print('no active leases to release')
    else:
        print(f'released {len(released)} lease(s)')
        for sid in released:
            print(f'  {sid}')
    for gid in torn_down:
        print(f'  torn down: {gid}')
    if pending:
        print('  ! the apply did not fully take effect; the change is still '
              'pending -- retry `infer-stack apply`')
        return 3
    return 0


def _resolve_idle_targets(controller, names: list[str]) -> tuple[list[str], list[str]]:
    """Map endpoint aliases / deployment ids to currently-idle deployment ids.

    Returns ``(target_deployment_ids, unmatched_names)``.
    """
    _, deployments = controller.ledger.status(virtual_expiry=True)
    idle = [g for g in deployments if g.state == DeploymentState.IDLE]
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
    thrash — handy, but it holds a GPU. ``evict`` tears such deployments down now,
    overriding keep-warm. Target by served endpoint alias or deployment id, or
    ``--all`` for every idle deployment. (Live models — those with an active lease —
    are never evicted; release them first.) On a terminal the teardown is shown
    and confirmed before docker is touched (``--yes`` skips).
    """

    __command__ = 'evict'

    names = scfg.Value(
        [], nargs='*', position=1, type=str,
        help='Endpoint alias or deployment id to evict.',
    )
    all = scfg.Value(False, isflag=True, help='Evict every idle deployment.')
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)
        names = _collect_names(config.names)
        if not names and not config.all:
            raise SystemExit('evict: give an endpoint/deployment name or --all')
        missing: list[str] = []
        try:
            if config.all:
                outcome = controller.evict(None)
            else:
                targets, missing = _resolve_idle_targets(controller, names)
                if missing:
                    # Diagnostic, not payload: stdout must stay pure JSON
                    # under --json (and stay grep-able human output without).
                    print(
                        f'no idle deployment for: {", ".join(missing)}',
                        file=sys.stderr,
                    )
                if not targets:
                    if config.json:
                        print(json.dumps(
                            {'evicted': [], 'torn_down': [],
                             'missing': missing}, indent=2))
                    else:
                        print('nothing to evict')
                    return 0
                outcome = controller.evict(targets)
        except ConvergeAborted:
            raise _declined_exit()
        if config.json:
            print(json.dumps({
                'evicted': outcome.evicted_deployment_ids,
                'torn_down': outcome.reconcile.torn_down,
                'missing': missing,
            }, indent=2))
        elif not outcome.evicted_deployment_ids:
            print('nothing to evict')
        else:
            print(f'evicted {len(outcome.evicted_deployment_ids)} deployment(s)')
            for gid in outcome.evicted_deployment_ids:
                print(f'  {gid}')
        return 0


class GcCLI(_ApprovalMixin):
    """Reclaim leaked leases and free their GPUs — sweep TTL-expired leases, converge.

    A job that is hard-killed (SIGKILL / OOM / reboot) never runs its ``release``,
    so its lease lingers until its TTL elapses. ``gc`` sweeps those expired leases
    and reconciles, tearing down any ``stop``-policy deployment left with no demand
    and freeing its GPU. Run it periodically (cron) or as a final pipeline step; a
    blocking ``acquire`` (``--queue``) already does this implicitly while it waits.
    ``--evict`` additionally tears down idle *keep-warm* deployments (like ``evict
    --all``). On a terminal the teardown is shown and confirmed (``--yes`` skips).
    """

    __command__ = 'gc'

    evict = scfg.Value(
        False, isflag=True,
        help='Also tear down idle keep-warm deployments (like `evict --all`), '
        'not just leaked/expired demand.',
    )
    orphans = scfg.Value(
        False, isflag=True,
        help='Instead: remove containers in the project that infer-stack does not '
        'manage (listed and confirmed first; --yes skips the prompt).',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)
        if config.orphans:
            if not isinstance(controller.backend, ComposeBackend):
                raise SystemExit('gc --orphans needs the compose backend')

            def confirm(found):
                print(f'gc --orphans: {len(found)} unmanaged container(s):')
                for c in found:
                    print(f'  {c.container_id[:12]}  {c.service or "?"}  {c.state}')
                if config.yes or not sys.stdin.isatty():
                    return bool(config.yes)
                return input('remove them? [y/N] ').strip().lower() in {'y', 'yes'}

            removed = controller.remove_orphans(confirm)
            if config.json:
                print(json.dumps({'removed': [c.container_id for c in removed]}, indent=2))
            else:
                print(f'gc --orphans: removed {len(removed)} container(s)')
            return 0
        try:
            outcome = controller.gc(evict_idle=bool(config.evict))
        except ConvergeAborted:
            raise _declined_exit()
        if config.json:
            print(json.dumps({
                'expired_leases': outcome.expired_lease_ids,
                'idled': outcome.idled_deployment_ids,
                'evicted': outcome.evicted_deployment_ids,
                'torn_down': outcome.reconcile.torn_down,
            }, indent=2))
        else:
            n_exp = len(outcome.expired_lease_ids)
            torn = outcome.reconcile.torn_down
            if not n_exp and not torn and not outcome.evicted_deployment_ids:
                print('gc: nothing to reclaim')
            else:
                print(
                    f'gc: reclaimed {n_exp} expired lease(s), '
                    f'tore down {len(torn)} deployment(s)'
                )
                for gid in torn:
                    print(f'  {gid}')
        return 0


class CleanCLI(_LeasingCommonMixin):
    """Bring the stack to a clean slate: no leases, and nothing holding a GPU.

    Like ``git clean``: by default it only SHOWS what it would do; ``-f`` does
    it. Everything the ledger knows is released and torn down -- every active
    lease, whoever owns it, and every deployment, keep-warm included -- and
    containers in the project that infer-stack does not manage are removed. The
    gateway stays up. Use it when you know nothing else on the host needs its
    models, for example before a batch run whose scheduler cannot see GPUs held
    outside it.

    It composes existing verbs: ``release --all --evict`` (which also evicts
    idle deployments that no longer have a lease) and ``gc --orphans``.
    """

    __command__ = 'clean'
    __epilog__ = """
    Examples:
        infer-stack clean           # dry run: what would be released and torn down
        infer-stack clean -f        # do it
        infer-stack clean -f --no-orphans   # leave unmanaged containers alone
    """

    force = scfg.Value(
        False, isflag=True, short_alias=['f'],
        help='Actually release and tear down. Without it, clean only reports.',
    )
    orphans = scfg.Value(
        True, isflag=True,
        help='Also remove containers in the project that infer-stack does not '
        'manage (--no-orphans keeps them).',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted

        config = cls.cli(argv=argv, data=kwargs)
        # -f IS the consent, so the converge does not ask a second time.
        controller = _open_controller(config, interactive=False)
        leases, deployments = controller.ledger.status(virtual_expiry=True)
        active = [le for le in leases if le.state == LeaseState.ACTIVE]
        held = [g for g in deployments
                if g.state in (DeploymentState.LIVE, DeploymentState.IDLE)]
        observed, assignments = _placement_view(controller)
        can_orphan = bool(config.orphans) and isinstance(
            controller.backend, ComposeBackend)

        found: list = []

        def _list_only(orphans):
            found.extend(orphans)
            return False

        if not config.force:
            if can_orphan:
                controller.remove_orphans(_list_only)
            plan = {
                'dry_run': True,
                'leases': [{'id': le.id, 'owner': le.owner,
                            'endpoints': le.endpoints} for le in active],
                'deployments': [{'id': g.id, 'state': g.state,
                                 'running': g.id in observed,
                                 'gpus': assignments.get(g.id),
                                 'served': sorted(g.served)} for g in held],
                'orphans': [{'id': c.container_id, 'service': c.service}
                            for c in found],
            }
            if config.json:
                print(json.dumps(plan, indent=2))
                return 0
            if not (active or held or found):
                print('clean: already clean (no leases, no deployments, no orphans)')
                return 0
            print('clean: would release and tear down (dry run; -f to do it)')
            for le in active:
                print(f'  release   {le.id}  owner={le.owner}  '
                      f'{",".join(le.endpoints)}')
            for g in held:
                gpus = assignments.get(g.id)
                print(f'  tear down {g.id}  {g.state}'
                      f'{" running" if g.id in observed else ""}'
                      f'  gpus={gpus if gpus is not None else "-"}'
                      f'  {",".join(sorted(g.served))}')
            for c in found:
                print(f'  remove    {c.container_id[:12]}  {c.service or "?"}  '
                      '(unmanaged)')
            return 0

        try:
            out = controller.release_leases(None, evict=True)
        except ConvergeAborted:
            raise _declined_exit()
        removed = controller.remove_orphans(lambda _: True) if can_orphan else []
        torn = sorted(out.reconcile.torn_down) if out.reconcile else []
        pending = bool(out.reconcile and out.reconcile.publication_pending)
        if config.json:
            print(json.dumps({
                'released': out.released_lease_ids,
                'evicted': out.evicted_deployment_ids,
                'torn_down': torn,
                'orphans_removed': [c.container_id for c in removed],
                'publication_pending': pending,
            }, indent=2))
            return 3 if pending else 0
        print(f'clean: released {len(out.released_lease_ids)} lease(s), '
              f'evicted {len(out.evicted_deployment_ids)} deployment(s), '
              f'removed {len(removed)} unmanaged container(s)')
        for gid in torn:
            print(f'  torn down: {gid}')
        if pending:
            print('  ! the apply did not fully take effect; the change is still '
                  'pending -- retry `infer-stack apply`')
            return 3
        return 0


class WaitCLI(_LeasingCommonMixin):
    """Block until served endpoints are ready — the companion to ``acquire
    --no-wait``.

    Fan out, then wait: ``acquire --no-wait smol17b-1`` + ``acquire --no-wait
    smol135-1`` kick both deployments off in parallel (each converges and starts
    its container without blocking), then ``wait smol17b-1 smol135-1`` blocks
    until they can actually serve. With no names it waits for every live deployment.

    ``--require-generation`` makes "ready" mean a real generated token (not just
    a model that is listed) — the same readiness *criterion* the ``acquire``
    verb takes; this command is the *blocking* half, distinct from it.
    """

    __command__ = 'wait'

    names = scfg.Value(
        [], nargs='*', position=1, type=str,
        help='Endpoint names to wait for (default: every live deployment).',
    )
    timeout = scfg.Value(600, type=float, help='Overall wait timeout (s).')
    interval = scfg.Value(5, type=float, help='Readiness poll interval (s).')
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        _, deployments = controller.ledger.status(virtual_expiry=True)
        live = [g for g in deployments if g.state == DeploymentState.LIVE]
        names = _collect_names(config.names)
        if names:
            wanted = set(names)
            served = {ep for g in live for ep in g.served}
            missing = sorted(wanted - served)
            if missing:
                raise SystemExit(
                    f'not served by any live deployment: {", ".join(missing)} '
                    '(acquire it first, or check `infer-stack leases`)'
                )
            targets = [g for g in live if wanted & set(g.served)]
            endpoints = wanted
        else:
            targets, endpoints = live, None
        if not targets:
            print('nothing to wait for (no live deployments)')
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
                    {'deployment': gid, 'endpoint': ep}
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


class MeasureCLI(_LeasingCommonMixin):
    """Measure an endpoint's real per-GPU VRAM requirement from the engine's
    own memory-profiling log (docs/planning/vram-aware-placement.md §3).

    Parses vLLM's profiling breakdown (weights + non-torch + activation peak)
    — deliberately NOT ``nvidia-smi memory.used``, which only reflects the
    ``gpu_memory_utilization`` preallocation on whatever card the model
    landed on — then adds a KV budget (our serving choice, not a model
    property) and a small safety margin, and reports a paste-ready
    ``placement.min_vram_gib`` value.

    If the endpoint is already live it is measured in place; otherwise it is
    acquired once (normal placement/queueing applies), measured, and
    released. ``--record`` writes the measurements overlay, which plan-time
    enrichment consults automatically for endpoints that declare nothing;
    promoting the number into ``catalog.yaml`` stays an explicit operator
    edit (the printed line is paste-ready).
    """

    __command__ = 'measure'

    endpoint = scfg.Value(
        None, position=1, required=True, type=str,
        help='Catalog endpoint to measure.',
    )
    record = scfg.Value(
        False, isflag=True,
        help='Record the result into the measurements overlay '
        '(consulted automatically at plan time when the catalog declares '
        'nothing for this endpoint).',
    )
    kv_gib = scfg.Value(
        2.0, type=float,
        help='KV-cache budget (GiB) added on top of the non-KV profile. '
        'A serving choice (max_model_len / max_num_seqs), not a model fact.',
    )
    margin = scfg.Value(
        0.05, type=float,
        help='Safety-margin fraction over the non-KV profile '
        '(allocator fragmentation, engine drift).',
    )
    timeout = scfg.Value(
        900, type=float,
        help='Readiness timeout when the endpoint must be brought up first (s).',
    )
    catalog = scfg.Value(
        None, type=str,
        help='Catalog path (default: <config-root>/catalog.yaml).',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.vram import (
            derive_min_vram_gib,
            measurement_key_for_spec,
            parse_vllm_memory_profile,
            weight_floor_gib,
        )

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        backend = controller.backend
        logs_fn = getattr(backend, 'deployment_logs', None)
        if logs_fn is None:
            raise SystemExit(
                'measure needs the compose backend '
                '(the engine container log is the measurement source).'
            )
        catalog = _requests_catalog(controller, config)
        name = config.endpoint
        request = catalog.resolve_names([name])[0]
        if request.engine != 'vllm':
            raise SystemExit(
                f'measure supports vllm endpoints only '
                f'({name} is {request.engine}).'
            )

        _, deployments = controller.ledger.status(virtual_expiry=True)
        live = [
            g for g in deployments
            if g.state == DeploymentState.LIVE
            and g.compat_key == request.compat_key
        ]
        lease_id = None
        if live:
            deployment = live[0]
        else:
            print(
                f'{name} is not live — acquiring it once to measure '
                f'(released afterwards)…'
            )
            outcome = controller.acquire(
                'measure',
                [request],
                ttl_seconds=3600.0,
                wait=True,
                timeout=float(config.timeout),
                wait_for_placement=True,
            )
            not_ready = outcome.released_on_timeout or (
                outcome.wait is not None and not outcome.wait.ready
            )
            if not_ready:
                for hint in _oom_hints(controller, outcome):
                    print(f'  {hint}')
                for _gid, _ep, why in (outcome.wait.failures if outcome.wait else []):
                    print(f'  {why}')
                raise SystemExit(
                    f'{name} never became ready — cannot measure '
                    f'(see `infer-stack logs` for the engine output).'
                )
            lease_id = outcome.lease.id
            by_key = {g.compat_key: g for g in outcome.deployments}
            deployment = by_key.get(request.compat_key, outcome.deployments[0])

        try:
            text = logs_fn(deployment, tail=4000) or ''
            profile = parse_vllm_memory_profile(text)
            if profile is None:
                raise SystemExit(
                    f'no memory-profiling lines found in {name}\'s engine log '
                    f'— the container may have restarted past its startup '
                    f'output, or this vLLM build logs an unknown format. '
                    f'Try right after a fresh serve.'
                )
            value = derive_min_vram_gib(
                profile,
                kv_budget_gib=float(config.kv_gib),
                margin_fraction=float(config.margin),
            )
            key = measurement_key_for_spec(deployment.spec)
            floor = weight_floor_gib(
                deployment.spec.get('hf_model_id'),
                getattr(backend, 'state', {}).get('hf_cache'),
            )
            recorded_to = None
            if config.record:
                store = getattr(backend, 'measurements', None)
                if store is None:
                    raise SystemExit(
                        '--record needs the compose backend measurements '
                        'overlay.'
                    )
                store.record(
                    key, value, endpoint=name, profile=profile,
                    kv_budget_gib=float(config.kv_gib),
                    margin_fraction=float(config.margin),
                )
                recorded_to = str(store.path)
            if config.json:
                print(json.dumps({
                    'endpoint': name,
                    'min_vram_gib': value,
                    'profile': profile,
                    'floor_vram_gib': floor,
                    'measurement_key': key,
                    'recorded_to': recorded_to,
                }, indent=2))
            else:
                print(f'measured {name}:')
                parts = ', '.join(
                    f'{k.removesuffix("_gib").replace("_", " ")}='
                    f'{v:g}GiB'
                    for k, v in sorted(profile.items())
                )
                print(f'  engine profile: {parts}')
                print(
                    f'  derived min_vram_gib: {value:g}   '
                    f'(non-KV profile × {1 + float(config.margin):g} '
                    f'+ {float(config.kv_gib):g} KV budget)'
                )
                if floor:
                    print(f'  weight-bytes floor: {floor:g} GiB (local HF cache)')
                if recorded_to:
                    print(f'  recorded to overlay: {recorded_to}')
                print(
                    f'  catalog promotion (explicit edit): '
                    f'endpoints.{name}.placement.min_vram_gib: {value:g}'
                )
        finally:
            if lease_id is not None:
                controller.release(lease_id)
        return 0


class TuiCLI(_LeasingCommonMixin):
    """Launch the Textual TUI: a live monitor of the stack with controls to
    serve / release / evict models.

    Mostly a monitor — the lease + deployment tables (desired state vs running, GPUs)
    refresh live — with key-bound controls: ``s`` serve, ``d`` release, ``a``
    release-all, ``e`` evict, ``r`` refresh, ``q`` quit. Opt-in extra: install
    textual with ``pip install "infer-stack[tui]"``.
    """

    __command__ = 'tui'

    catalog = scfg.Value(None, type=str, help='Path to catalog.yaml.')
    interval = scfg.Value(3.0, type=float, help='Auto-refresh interval (s).')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..paths import settings_path

        config = cls.cli(argv=argv, data=kwargs)
        try:
            from ..tui import run_tui
        except ImportError as ex:
            raise SystemExit(
                'the TUI needs the optional `textual` dependency — install it '
                'with `pip install "infer-stack[tui]"` (or `pip install textual`). '
                f'[{ex}]'
            )
        # Require a one-time `config init` first: the TUI assumes a configured
        # world (backend, data dir, …). Without it the dashboard would launch
        # against defaults the user never chose.
        _apply_path_overrides(config)
        if not settings_path().exists():
            raise SystemExit(
                'no settings found — run `infer-stack config init` once before '
                f'launching the TUI (expected {settings_path()}).'
            )
        controller = _open_controller(config)
        catalog, catalog_path = _load_catalog_for_tui(config)
        return run_tui(
            controller, catalog,
            interval=float(config.interval), catalog_path=str(catalog_path),
        )


class RenewCLI(_LeasingCommonMixin):
    """Extend (or make infinite) a lease's protection window."""

    __command__ = 'renew'

    lease = scfg.Value(None, position=1, type=str, help='Lease id.')
    env_file = scfg.Value(None, type=str)
    ttl = scfg.Value(None, type=str, help='New soft TTL (e.g. 2h); empty=infinite.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        sid = _resolve_lease(config)
        if not sid:
            raise SystemExit('renew: give a lease id or --env-file')
        # Through the controller: a renew can revive an idle deployment, which
        # is a desired-state change and must be serialised with applies.
        from ..leasing.backend import PlacementError

        try:
            outcome = controller.renew(sid, ttl_seconds=_parse_duration(config.ttl))
        except PlacementError as ex:
            raise SystemExit(
                f'renew: {sid} needs its idle deployment(s) back, and they cannot be '
                f'admitted now: {ex}'
            )
        if outcome.lease is None:
            raise SystemExit(
                f'renew: no active lease {sid} (unknown, released, or already '
                'expired — re-acquire instead)'
            )
        print(f'renewed {sid}')
        if outcome.revived_deployment_ids:
            print(f'  revived: {", ".join(outcome.revived_deployment_ids)}')
        if outcome.reconcile is not None and outcome.reconcile.publication_pending:
            print('  ! the apply did not fully take effect; the change is still '
                  'pending -- retry `infer-stack apply`')
            return 3
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
    queue = scfg.Value(
        False, isflag=True,
        help='Admission queue: wait (up to --timeout) for a GPU to free '
        'instead of failing fast when the fleet is full. Recommended for '
        'pipeline fan-out, where many jobs contend for a few GPUs.',
    )
    command = scfg.Value(
        [], nargs='*', position=1, type=str, help='Command to run (after --).'
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        from ..leasing.profile import ProfileMismatch

        controller = _open_controller(config)
        catalog = _requests_catalog(controller, config)
        names = _collect_names(config.endpoint)
        command = list(config.command or [])
        if not names:
            raise SystemExit('run: --endpoint is required')
        if not command:
            raise SystemExit('run: give a command after --')
        requests = _resolve(catalog, names)
        try:
            outcome = controller.acquire(
                config.owner or _default_owner(),
                requests,
                ttl_seconds=_parse_duration(config.ttl),
                wait=True,
                timeout=float(config.timeout),
                interval=float(config.interval),
                wait_for_placement=bool(getattr(config, 'queue', False)),
            )
        except ProfileMismatch as ex:
            raise SystemExit(f'run: {ex}')
        if outcome.wait is not None and not outcome.wait.ready:
            # The controller already released the lease on timeout
            # (released_on_timeout); just surface why we're not running.
            if outcome.wait.failures:
                detail = '; '.join(
                    f'{ep} ({gid}): {why}'
                    for gid, ep, why in outcome.wait.failures
                )
                raise SystemExit(f'run: engine cannot start: {detail}')
            raise SystemExit(
                f'run: endpoints not ready: {outcome.wait.pending}'
            )
        descriptor = _descriptor_for(
            controller, outcome.lease, outcome.deployments, config
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
    """Best-effort (observed running gids, deployment id -> GPU indices).

    Reads what the backend *actually* has running (``observe``) and where each
    desired deployment is/would be placed (``plan``). Both are guarded so a dry-run
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
    if controller._admission_mode():
        # Committed allocations, and idle residents' physical GPUs; the
        # legacy planner view would show placements admission would not make.
        _, deployments = controller.ledger.status(virtual_expiry=True)
        for g in deployments:
            if g.assigned_gpus is not None:
                assignments[g.id] = list(g.assigned_gpus)
        try:
            residency = backend.residency()
            for g in deployments:
                c = residency.resident(g.id)
                if g.id not in assignments and c is not None:
                    assignments[g.id] = list(c.gpus)
        except Exception:  # noqa: BLE001
            pass
        return observed, assignments
    plan = getattr(backend, 'plan', None)
    if plan is not None:
        try:
            assignments = dict(plan(controller.desired_deployments()).assignments)
        except Exception:  # noqa: BLE001
            pass
    return observed, assignments


def _running_label(gid, observed) -> str:
    return 'running' if gid in observed else '—'


def _gpu_label(gid, observed, assignments) -> str:
    """Where a deployment is (running) or is slated to be (desired, not yet up)."""
    gpus = assignments.get(gid)
    if gpus is None:
        return '-'
    where = ','.join(str(i) for i in gpus) if gpus else 'cpu'
    return where if gid in observed else f'→{where}'  # → = slated, not yet up


def _print_leases_plain(leases, deployments, observed, assignments) -> None:
    print('leases:')
    if not leases:
        print('  (none)')
    for le in leases:
        print(
            f'  {le.id}  owner={le.owner}  state={le.state}  '
            f'ttl={_lease_ttl(le)}  endpoints={",".join(le.endpoints) or "-"}'
        )
    print('deployments:')
    if not deployments:
        print('  (none)')
    for g in deployments:
        print(
            f'  {g.id}  {g.engine}  state={g.state}  '
            f'running={_running_label(g.id, observed)}  '
            f'gpus={_gpu_label(g.id, observed, assignments)}  '
            f'demand={g.demand}  served={",".join(sorted(g.served)) or "-"}'
        )


def _print_leases_rich(leases, deployments, observed, assignments, console) -> None:
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

    console.print(Text('deployments', style='bold'))
    if not deployments:
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
        for g in deployments:
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
    """Show current leases and deployment deployments (the leasing-model status).

    Two tables. **leases** are who asked for what (id, owner, state, ttl,
    endpoints). **deployments** are the actual deployments behind them — one deployment is
    one model in one container (it may serve several endpoint aliases). For each
    deployment:

    \b
      state    what the ledger *wants* — live / idle / stopped
      running  what the backend *actually* has up right now
      gpus     the GPU indices it is on; a ``→`` prefix means slated (desired
               but not yet started), ``-`` means no placement info
      demand   how many active leases are protecting it
      served   the endpoint aliases routed to it

    A row that is ``state=live`` but ``running=—`` is desired-but-not-up: either
    still starting, staged via ``acquire --no-apply`` (run ``apply`` to bring it
    up), or unplaceable (no free GPU — see ``acquire``'s error / ``evict``).
    """

    __command__ = 'leases'
    __epilog__ = """
    Examples:
        infer-stack leases          # the two tables (rich on a terminal)
        infer-stack leases --json   # JSON (adds running + gpus per deployment)
    """

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        leases, deployments = controller.ledger.status(virtual_expiry=True)
        observed, assignments = _placement_view(controller)
        health = controller.observe_state()
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
                        'deployments': [
                            {
                                'id': g.id,
                                'engine': g.engine,
                                'state': g.state,  # desired (ledger)
                                'running': g.id in observed,  # actual (backend)
                                'gpus': assignments.get(g.id),
                                'demand': g.demand,
                                'served': sorted(g.served),
                            }
                            for g in deployments
                        ],
                        'health': health,
                    },
                    indent=2,
                )
            )
            return 0
        from rich.console import Console

        console = Console()
        if console.is_terminal:
            _print_leases_rich(leases, deployments, observed, assignments, console)
        else:
            _print_leases_plain(leases, deployments, observed, assignments)
        _print_health(health)
        return 0


def _print_health(health: dict) -> None:
    """The conditions worth an operator's attention, one line each."""
    lines = []
    marker = health.get('publication_pending')
    if marker:
        kind = 'apply requested' if marker.get('apply_requested') else 'staged'
        extra = ', interrupted' if marker.get('interrupted') else ''
        extra += ', approval guard' if marker.get('approved_digest') else ''
        lines.append(f'pending change: {kind}{extra} (`infer-stack apply` publishes it)')
    if health.get('residency_error'):
        lines.append(f'residency UNKNOWN: {health["residency_error"]}')
    for row in health.get('deployments') or []:
        if row['condition'] in {'unknown', 'ambiguous', 'degraded', 'displaced',
                                'unresolved', 'not-running'}:
            lines.append(f'{row["condition"].upper()}: {row["id"]} ({row["state"]})')
    for orphan in health.get('orphans') or []:
        lines.append(f'ORPHAN: {orphan["id"][:12]} {orphan["service"]} '
                     '(`infer-stack gc --orphans`)')
    if health.get('profile_drift'):
        lines.append('settings differ from the active recovery snapshot: '
                     + ', '.join(health['profile_drift']))
    for lease_id in health.get('expired_unswept') or []:
        lines.append(f'expired (not yet reclaimed): {lease_id}')
    if lines:
        print('health:')
        for line in lines:
            print(f'  {line}')


def _postgres_initialised() -> bool:
    """Has the gateway's Postgres already created its data directory?"""
    from ..config import default_state_paths

    path = Path(default_state_paths()['postgres_litellm'])
    return path.is_dir() and any(path.iterdir())


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


def _front_door_env(stored: dict[str, str]) -> dict[str, str]:
    """The front-door values ``env`` answers without them being stored.

    A base URL is not a secret and has nothing to be read *out* of: it is
    derived from the configured front-door port. ``_front_door`` is the one
    function that derives it, and reusing it here is the whole point -- if
    ``env`` computed its own, a script built from ``env`` could point at a door
    ``infer-stack test`` never knocked on, and the two would disagree silently.

    ``stored`` is the managed ``.env``, which wins: writing ``OPENAI_BASE_URL``
    is how you aim a script at a gateway that is not the local front door.
    THE PORT IS THEN READ BACK OFF THAT URL rather than re-derived, because a
    port that disagrees with the URL beside it is worse than no port at all --
    an override to ``:8443`` used to leave ``LITELLM_PORT`` reporting the
    default, which is exactly the hardcoded-mismatch this command exists to
    stop. A port explicitly written to the file still wins over both.

    ``LITELLM_PORT`` is omitted when the effective URL names no port: behind a
    proxy on 80/443 there is nothing to report, and inventing a number is a lie
    a script would then bake in.
    """
    from urllib.parse import urlparse

    base_url = stored.get('OPENAI_BASE_URL')
    if not base_url:
        base_url, _ = _front_door(None)
    entries = {'OPENAI_BASE_URL': base_url}
    port = urlparse(base_url).port
    if port is not None:
        entries['LITELLM_PORT'] = str(port)
    return entries


class TestCLI(_PathOverridesMixin):
    """Smoke-test a served endpoint through the front door (a real generation).

    The concise alternative to hand-rolling ``curl``: sends one chat completion
    to the endpoint *alias* via the LiteLLM gateway, then prints latency and the
    reply (or an actionable error). Exit code is non-zero on failure, so it is
    usable in scripts/CI.
    """

    __command__ = 'test'

    catalog = scfg.Value(
        None, type=str,
        help='Catalog path, used only to look up the endpoint protocol.',
    )
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
    protocol = scfg.Value(
        None, choices=['chat', 'completions'],
        help="Which surface to hit. Default: the endpoint's declared "
             '`protocol` from the catalog, falling back to chat.',
    )

    @classmethod
    def main(cls, argv=True, **kwargs):
        import time

        import requests

        config = cls.cli(argv=argv, data=kwargs)
        _apply_path_overrides(config)
        if not config.name:
            raise SystemExit('test: give an endpoint alias (e.g. `test chat`)')

        # Follow the endpoint's declared protocol, like the readiness probe
        # does. Hardcoding chat meant a completions-only endpoint -- a base
        # model, or anything served for a client that renders its own prompts
        # -- reported FAILED however healthy it was, because the server
        # correctly 404s a surface it does not serve.
        protocol = config.protocol or _endpoint_protocol(config, config.name)

        base_url, key = _front_door(config)
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = f'Bearer {key}'
        if protocol == 'completions':
            url = f'{base_url}/completions'
            payload = {
                'model': config.name,
                'prompt': config.prompt,
                'max_tokens': int(config.max_tokens),
            }
        else:
            url = f'{base_url}/chat/completions'
            payload = {
                'model': config.name,
                'messages': [{'role': 'user', 'content': config.prompt}],
                'max_tokens': int(config.max_tokens),
            }
        t0 = time.monotonic()
        try:
            resp = requests.post(
                url,
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
            choice = resp.json()['choices'][0]
            reply = (choice.get('text') if protocol == 'completions'
                     else choice['message']['content'])
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


def _endpoint_protocol(config, name: str) -> str:
    """The endpoint's declared protocol, or 'chat' when it cannot be resolved.

    Best-effort on purpose: ``test`` must still work against an endpoint that
    is live but absent from the catalog (an ad-hoc acquire), so a missing or
    unreadable catalog falls back to the default rather than failing.

    Resolution goes through :func:`_catalog_path` rather than repeating it.
    The copy here omitted ``INFER_STACK_CATALOG``, so leasing and ``test``
    could read different catalogs on the same machine and ``test`` would probe
    the wrong API surface for an endpoint it had resolved from the other one.
    """
    try:
        from ..leasing import Catalog
        path = _catalog_path(config)
        if not path.exists():
            return 'chat'
        ep = Catalog.load(path).endpoints.get(name)
        return getattr(ep, 'protocol', None) or 'chat'
    except Exception:  # noqa: BLE001 - a diagnostic must not fail on lookup
        return 'chat'


def _test_fail(config, base_url: str, reason: str) -> int:
    if config.json:
        print(json.dumps(
            {'endpoint': config.name, 'ok': False,
             'base_url': base_url, 'reason': reason}, indent=2))
    else:
        print(f'{config.name}: FAILED via {base_url} — {reason}')
        print('  is it served?  infer-stack leases   |   '
              'infer-stack acquire ' + str(config.name))
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
      infer-stack env OPENAI_BASE_URL       # …including the front door
      infer-stack env HF_TOKEN=hf_…         # set one value (merges, before acquire)
      infer-stack env --export              # every entry as `export KEY=value`

    The argument is a KEY to read, or ``KEY=VALUE`` to write (writes merge
    non-destructively, so the managed LiteLLM key is preserved).

    Two keys are DERIVED rather than stored -- ``OPENAI_BASE_URL`` and
    ``LITELLM_PORT`` -- so that everything a client needs comes from one verb
    and a script never has to hardcode a host and port next to a key it looked
    up properly::

        export OPENAI_BASE_URL=$(infer-stack env OPENAI_BASE_URL)
        export OPENAI_API_KEY=$(infer-stack env LITELLM_MASTER_KEY)

    They answer before any ``acquire``, because a URL needs no secret to exist.
    Writing one (``env OPENAI_BASE_URL=…``) pins it: a value in the file always
    wins over the derived one, which is how you point a script at a gateway
    that is not the local front door.
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
            # Under the controller's publication lock: a render/apply in progress
            # must read one consistent .env (its fingerprints hash the values
            # Compose will interpolate), and concurrent writers must not lose
            # each other's keys.
            ledger = Ledger(SqliteStore(str(default_ledger_path())))
            controller = Controller(ledger, NullBackend())
            with controller._global_lock():
                if key in _SERVICE_ENV_KEYS:
                    leases, _ = ledger.status(virtual_expiry=True)
                    active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
                    if active:
                        raise SystemExit(
                            f'env: {key} affects running services and cannot change '
                            f'while {len(active)} lease(s) are active; release them '
                            'first'
                        )
                if key == 'LITELLM_DB_PASSWORD' and _postgres_initialised():
                    raise SystemExit(
                        'env: Postgres stored LITELLM_DB_PASSWORD when its data '
                        'directory was created; changing the .env only would lock '
                        'the gateway out of its database'
                    )
                if key == 'LITELLM_MASTER_KEY':
                    from ..leasing.gateway import set_master_key
                    try:
                        # Pins the salt first, so DB-stored routes stay readable.
                        set_master_key(env_path, value)
                    except ValueError as ex:
                        raise SystemExit(f'env: {ex}')
                else:
                    write_env_file(env_path, {key: value})
            print(f'set {key} ({env_path})')
            return 0

        # Path first and foremost (it may not exist yet — that's fine).
        if not (config.arg or config.export):
            print(env_path)
            return 0

        # Read: `env KEY` / `env --export`. A stored value always beats the
        # derived one -- writing the key is how you override the front door.
        env = parse_env_file(env_path) if env_path.exists() else {}
        derived = _front_door_env(env)
        if config.arg:
            if config.arg in env:
                print(env[config.arg])
                return 0
            if config.arg in derived:
                print(derived[config.arg])
                return 0
            if not env_path.exists():
                raise SystemExit(
                    f'no managed env-file at {env_path}; run an `acquire` '
                    'with --backend compose first (or `infer-stack env KEY=VALUE`)'
                )
            raise SystemExit(f'{config.arg!r} not found in {env_path}')
        if not env_path.exists():
            # The URL still stands on its own; say what is missing on stderr so
            # `eval "$(infer-stack env --export)"` keeps working regardless.
            print(
                f'no managed env-file at {env_path} yet: no secrets to export',
                file=sys.stderr,
            )
        for name, value in {**derived, **env}.items():
            print(f'export {name}={shlex.quote(value)}')
        return 0


# ---------------------------------------------------------------------------
# `routes` — inspect / seed / prune the LiteLLM route registry (static mode)
# ---------------------------------------------------------------------------


def _require_compose_backend(controller):
    """The controller's ComposeBackend, or a SystemExit for other backends.

    The route registry is a compose-backend concept (it feeds the static-superset
    LiteLLM gateway); ``--backend null``/``kubeai`` have no registry to touch."""
    backend = controller.backend
    if not isinstance(backend, ComposeBackend):
        raise SystemExit(
            'the `routes` commands require the compose backend '
            '(set `--backend compose` or `config set backend compose`)'
        )
    return backend


def _live_endpoints(controller) -> set[str]:
    """Endpoint aliases served by a currently-live deployment (for annotation)."""
    _, deployments = controller.ledger.status(virtual_expiry=True)
    return {
        ep
        for g in deployments
        if g.state == DeploymentState.LIVE
        for ep in g.served
    }


class RoutesListCLI(_LeasingCommonMixin):
    """Print the accumulated LiteLLM route registry (static-superset mode).

    One row per persisted route: its alias, engine, served-model/tag, the
    upstream compose service it derives, and whether a live deployment is
    currently backing it. Routes with no live backer still list (that is the
    point — a released endpoint stays routable/testable); their upstream simply
    errors until something serves it.
    """

    __command__ = 'list'

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.compose import (
            OLLAMA_CONTAINER_PORT,
            VLLM_CONTAINER_PORT,
            ollama_service_name_for,
            vllm_service_name_for,
        )

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        backend = _require_compose_backend(controller)
        registry = backend.gateway._load_route_registry()
        entries = registry.get('entries', {})
        live = _live_endpoints(controller)

        rows = []
        for name in sorted(entries):
            row = entries[name]
            engine = row.get('engine')
            if engine == 'vllm':
                served = row.get('served') or name
                upstream = (
                    f'http://{vllm_service_name_for(served)}:'
                    f'{VLLM_CONTAINER_PORT}/v1'
                )
                target = served
            elif engine == 'ollama':
                target = row.get('model') or name
                host = row.get('host') or name
                upstream = (
                    f'http://{ollama_service_name_for(host)}:'
                    f'{OLLAMA_CONTAINER_PORT}'
                )
            else:
                target, upstream = '?', '?'
            rows.append({
                'name': name,
                'engine': engine,
                'target': target,
                'upstream': upstream,
                'live': name in live,
            })

        if config.json:
            print(json.dumps(
                {'version': registry.get('version'), 'routes': rows}, indent=2
            ))
            return 0
        if not rows:
            print('route registry is empty (no converge has run yet)')
            return 0
        print(f'{len(rows)} route(s) in the registry:')
        for r in rows:
            flag = 'live' if r['live'] else '   -'
            print(
                f'  [{flag}] {r["name"]:<28} {r["engine"]:<7} '
                f'{r["target"]:<24} -> {r["upstream"]}'
            )
        return 0


class RoutesPruneCLI(_ApprovalMixin):
    """Forget stale routes: rewrite the registry to *invoking catalog ∪ live*,
    then converge (one accepted gateway recreate).

    The registry is append-only by design (that is what keeps the gateway config
    byte-stable), so pruning is the explicit, operator-driven "forget" verb —
    automatic pruning is deliberately excluded because any catalog-keyed rule
    reintroduces the cross-catalog alternation churn this whole mechanism exists
    to avoid.

    A prune run from the WRONG ``INFER_STACK_CONFIG_DIR`` would silently drop
    every other runbook's routes, so the exact drop list is shown and confirmed
    first (``--yes`` / a non-terminal skips the prompt).
    """

    __command__ = 'prune'

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..diff_prompt import confirm_writes
        from ..leasing.backend import ConvergeAborted
        from ..leasing.gateway import (
            LITELLM_REGISTRY_VERSION,
            _dump_route_registry,
            _registry_incoming_from_catalog,
            _registry_incoming_from_deployments,
        )

        config = cls.cli(argv=argv, data=kwargs)
        # interactive=False so reconcile auto-applies the compose diff; the
        # meaningful gate (which routes get dropped) is confirmed here instead.
        controller = _open_controller(config, interactive=False)
        backend = _require_compose_backend(controller)

        def prune_plan() -> tuple[dict, dict, list[str]]:
            desired = controller.desired_deployments()
            plan = backend.plan(desired)
            keep: dict = {}
            if backend.catalog is not None:
                keep.update(_registry_incoming_from_catalog(backend.catalog))
            keep.update(_registry_incoming_from_deployments(desired, plan.assignments))
            current = backend.gateway._load_route_registry().get('entries', {})
            return current, keep, sorted(set(current) - set(keep))

        # Preview outside the lock (the prompt must not hold it); the change
        # itself is recomputed under the lock below.
        _, keep, dropped = prune_plan()
        if not dropped:
            print('routes prune: nothing to drop (registry already minimal)')
            return 0

        pruned = {'version': LITELLM_REGISTRY_VERSION, 'entries': keep}
        skip_prompt = bool(config.yes) or not sys.stdout.isatty()
        if not skip_prompt:
            print('routes prune will DROP these routes:')
            for name in dropped:
                print(f'  - {name}')
            ok = confirm_writes(
                {backend.gateway._registry_file: _dump_route_registry(pruned)},
                assume_yes=False,
                title='infer-stack routes prune',
            )
            if not ok:
                raise SystemExit('aborted: registry not pruned')

        confirmed = set(dropped)

        def change():
            # Under the lock: drop only routes that were confirmed AND are still
            # unneeded now; anything that became needed meanwhile is kept. (No
            # sweep: an expired-but-unswept deployment only keeps its routes.)
            current, _, still = prune_plan()
            drop = sorted(confirmed & set(still))
            entries = {k: v for k, v in current.items() if k not in drop}
            with backend._converge_lock():
                backend._atomic_write(
                    backend.gateway._registry_file,
                    _dump_route_registry(
                        {'version': LITELLM_REGISTRY_VERSION, 'entries': entries}),
                )
            return drop, sorted(entries)

        try:
            (dropped, kept), rec = controller.publish_change(change)
        except ConvergeAborted:
            raise SystemExit('aborted: compose changes not applied')

        if config.json:
            print(json.dumps({'dropped': dropped, 'kept': kept,
                              'publication_pending': rec.publication_pending}, indent=2))
        else:
            print(f'routes prune: dropped {len(dropped)} route(s), '
                  f'kept {len(kept)}')
        return 3 if rec.publication_pending else 0


class RoutesSeedCLI(_ApprovalMixin):
    """Merge extra catalog files' routes into the registry, then converge.

    The operational key to blip-free concurrency: seed *all* the overlapping
    runbooks' catalogs once while the stack is idle, and then no converge from
    any of them ever recreates the gateway (the registry is already the full
    union). Works before ``stack up`` too — the follow-up reconcile brings the
    standing gateway up with the complete route table, which is exactly what
    pre-seeding is for.

    Unlike a normal converge (which only ever merges the *invoking* process's own
    catalog), this folds in sibling catalogs you name explicitly.
    """

    __command__ = 'seed'

    catalogs = scfg.Value(
        None, nargs='+', position=1, type=str,
        help='One or more catalog.yaml files whose endpoints to merge into the '
        'route registry.',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted
        from ..leasing.compose import _registry_incoming_from_catalog

        config = cls.cli(argv=argv, data=kwargs)
        paths = _collect_names(config.catalogs)
        if not paths:
            raise SystemExit('routes seed: name at least one catalog.yaml file')
        # interactive=False: the reconcile auto-applies (seeding is additive, so
        # there is no destructive gate to confirm).
        controller = _open_controller(config, interactive=False)
        backend = _require_compose_backend(controller)

        incoming: dict = {}
        for raw in paths:
            path = Path(raw).expanduser()
            if not path.exists():
                raise SystemExit(f'catalog not found: {path}')
            try:
                cat = Catalog.load(path)
            except CatalogError as ex:
                raise SystemExit(f'invalid catalog {path}: {ex}')
            incoming.update(_registry_incoming_from_catalog(cat))
        if not incoming:
            raise SystemExit(
                'routes seed: the named catalog(s) resolved no routable endpoints'
            )

        def change():
            before = set(backend.gateway._load_route_registry().get('entries', {}))
            backend.merge_route_registry(incoming)
            return sorted(set(incoming) - before)

        try:
            added, rec = controller.publish_change(change)
        except ConvergeAborted:
            raise SystemExit('aborted: compose changes not applied')

        if config.json:
            print(json.dumps(
                {'merged': sorted(incoming), 'added': added,
                 'publication_pending': rec.publication_pending}, indent=2
            ))
        else:
            print(
                f'routes seed: merged {len(incoming)} route(s) '
                f'({len(added)} new): {", ".join(added) or "(all already present)"}'
            )
        return 3 if rec.publication_pending else 0


class ConfigPublishCLI(_ApprovalMixin):
    """Explicitly pre-seed/preview the internal recovery profile.

    This is an advanced operation, not part of the normal
    ``config init -> catalog suggest --apply -> acquire`` workflow. Acquire
    advances the recovery snapshot automatically from current user config.

    Use explicit publication when several independent runbooks should be
    pre-seeded as one catalog union before any of them acquires, or when an
    operator deliberately wants a quiescent preview/pre-pull of a future
    profile. Endpoints are merged, identical definitions deduplicated, and a
    name defined differently in two catalogs is refused.

    Examples:
        infer-stack config publish a.yaml b.yaml --yes
        infer-stack config publish --catalog a.yaml --ui --yes
    """

    __command__ = 'publish'

    catalogs = scfg.Value(
        [], nargs='*', position=1, type=str,
        help='Catalog files to publish as one union (default: --catalog, or the '
        'default-path catalog).',
    )
    pull = scfg.Value(
        True, isflag=True,
        help='Pre-pull every image the profile references (default; --no-pull skips).',
    )
    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted
        from ..leasing.profile import CatalogUnion, ProfileMismatch

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)
        render_profile = getattr(controller.backend, 'render_profile', None)
        if render_profile is None:
            raise SystemExit('config publish: this backend has no render profile')
        # The profile this invocation resolves to, before the backend was
        # switched to the published one (see Controller._sync_profile).
        profile = controller._invocation_profile or render_profile()
        paths = _collect_names(config.catalogs)
        if paths:
            sources = []
            for raw in paths:
                path = Path(raw).expanduser()
                if not path.exists():
                    raise SystemExit(f'catalog not found: {path}')
                try:
                    sources.append(Catalog.load(path).source or {})
                except CatalogError as ex:
                    raise SystemExit(f'invalid catalog {path}: {ex}')
            profile = {**profile, 'catalogs': sources}
        try:
            if profile.get('catalogs'):
                CatalogUnion.from_sources(profile['catalogs'])   # conflicts
            pull = getattr(controller.backend, 'pull_images', None)
            if config.pull and pull is not None and profile.get('backend') == 'compose':
                # Outside the lock, before anything is published: a steady-state
                # apply must never wait on a registry, and a missing image must
                # refuse the publication rather than break the next apply. The
                # image set comes from the CANDIDATE profile and its catalogs.
                from ..leasing.compose import profile_images

                try:
                    pull(profile_images(profile))
                except Exception as ex:  # noqa: BLE001
                    raise SystemExit(f'config publish: image pull failed, nothing published: {ex}')
            rec = controller.publish_profile(profile)
        except (ProfileMismatch, CatalogError) as ex:
            raise SystemExit(f'config publish: {ex}')
        except ConvergeAborted:
            raise SystemExit('aborted: profile not published')
        n = len(profile.get('catalogs') or [])
        if config.json:
            print(json.dumps({'published': True, 'catalogs': n,
                              'publication_pending': rec.publication_pending}, indent=2))
        else:
            print(f'published the render profile ({profile.get("backend")}, {n} catalog(s))')
            if rec.publication_pending:
                print('  ! the apply did not fully take effect; retry `infer-stack apply`')
        return 3 if rec.publication_pending else 0


class NetworkMigrateCLI(_ApprovalMixin):
    """Move the stack to stable per-service addresses on a fixed subnet.

    Every service gets a static address that no other service will ever
    receive, which prevents the gateway from routing one model's traffic to a
    container that inherited another's IP. Recreates every container once
    (including the gateway), so it is refused while leases are active unless
    ``--force``. A subnet overlapping an existing Docker network or host route
    is rejected.
    """

    __command__ = 'migrate'

    subnet = scfg.Value(None, type=str, help='IPv4 subnet, e.g. 172.30.0.0/24 (required).')
    force = scfg.Value(False, isflag=True, help='Migrate even with active leases.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted
        from ..leasing.profile import ProfileMismatch

        config = cls.cli(argv=argv, data=kwargs)
        if not config.subnet:
            raise SystemExit('network migrate: --subnet is required')
        controller = _open_controller(config, interactive=True)
        if not isinstance(controller.backend, ComposeBackend):
            raise SystemExit('network migrate needs the compose backend')
        try:
            rec = controller.network_migrate(config.subnet, force=bool(config.force))
        except ProfileMismatch as ex:
            raise SystemExit(f'network migrate: {ex}')
        except ConvergeAborted:
            raise SystemExit('aborted: network not migrated')
        table = controller.ledger.service_addresses()
        print(f'network migrate: {len(table)} service address(es) on {config.subnet}')
        for service, ip in sorted(table.items()):
            print(f'  {ip:<15} {service}')
        return 3 if rec.publication_pending else 0


class NetworkCheckCLI(_LeasingCommonMixin):
    """Probe every model upstream by name from inside the gateway's network.

    Distinguishes a routing fault (the name answers with ANOTHER model: the
    stale-address misroute) from an upstream that is simply not ready.
    """

    __command__ = 'check'

    json = scfg.Value(False, isflag=True)

    @classmethod
    def main(cls, argv=True, **kwargs):
        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config)
        check = getattr(controller.backend, 'upstream_check', None)
        if check is None:
            raise SystemExit('network check needs the compose backend')
        result = check()
        if config.json:
            print(json.dumps(result, indent=2))
        else:
            for service, row in result.items():
                print(f'  {row["status"]:<14} {service} (expects {row["expected"]})')
        return 4 if any(r['status'] == 'routing-fault' for r in result.values()) else 0


class SecretsRotateCLI(_ApprovalMixin):
    """Replace the LiteLLM master key and restart the gateway with it.

    Refused while leases are active unless ``--force``: their holders use the
    old key. Anything that copied the key must fetch it again
    (``infer-stack env LITELLM_MASTER_KEY``). Stored routes stay readable:
    the first rotation pins ``LITELLM_SALT_KEY`` to the old key, which is what
    LiteLLM encrypted them with.
    """

    __command__ = 'rotate'

    force = scfg.Value(False, isflag=True, help='Rotate even with active leases.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from ..leasing.backend import ConvergeAborted
        from ..leasing.profile import ProfileMismatch

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=True)
        # Any: rotate_gateway_key below refuses a backend without a gateway,
        # so past it these gateway methods exist.
        backend: Any = controller.backend
        old = backend.master_key() if isinstance(backend, ComposeBackend) else None
        try:
            rec = controller.rotate_gateway_key(force=bool(config.force))
        except ProfileMismatch as ex:
            raise SystemExit(f'secrets rotate: {ex}')
        except ConvergeAborted:
            raise SystemExit('aborted: the key was not changed')
        print('secrets rotate: LITELLM_MASTER_KEY replaced')
        if rec.publication_pending:
            print('  the gateway has not restarted yet; run `infer-stack apply`')
            return 3
        new = backend.master_key()
        accepted = backend.gateway_accepts(new, wait=60.0)
        if accepted is None:
            print('  gateway not running: it will use the new key when it starts')
        elif not accepted:
            raise SystemExit('secrets rotate: the gateway rejects the new key')
        elif backend.gateway_accepts(old) is not False:
            raise SystemExit('secrets rotate: could not confirm the gateway '
                             'rejects the OLD key')
        else:
            print('  gateway: new key accepted, old key rejected')
        if getattr(backend, 'ui', False):
            print('  Open WebUI may keep the old key in its own settings: '
                  'update it under Admin > Settings > Connections')
        return 0


class SecretsModalCLI(scfg.ModalCLI):
    """Manage the gateway's secrets."""

    __command__ = 'secrets'

    rotate = SecretsRotateCLI


class NetworkModalCLI(scfg.ModalCLI):
    """Stable per-service addressing (migrate) and the upstream routing check."""

    __command__ = 'network'

    migrate = NetworkMigrateCLI
    check = NetworkCheckCLI


class RoutesModalCLI(scfg.ModalCLI):
    """Inspect + manage the LiteLLM route registry (static-superset mode).

    The registry accumulates every catalog's and every live deployment's routes
    across the runbooks that share one stack, so a cross-catalog converge cannot
    strip another's live routes and the gateway config stays byte-stable. See
    docs/litellm-gateway-routing.md.
    """

    __command__ = 'routes'

    list = RoutesListCLI
    seed = RoutesSeedCLI
    prune = RoutesPruneCLI
