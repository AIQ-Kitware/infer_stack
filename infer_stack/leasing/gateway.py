"""The front door: the LiteLLM gateway, its routes and keys, and the UI and
proxy beside it.

Engines (vLLM, Ollama) are the backend's business; everything a client
talks to is here. A backend hands the gateway where its models are (route
rows), and the gateway renders its Compose services and config, keeps the
route registry, reconciles dynamic routes through LiteLLM's admin API, and
manages the master key. The compose backend runs it beside its engines; the
kubeai backend runs it alone, in front of a cluster.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config import PINNED_IMAGES
from ..config import DEFAULT_PORTS
from ..env_utils import ensure_secret, parse_env_file, write_env_file
from .backend import ConvergeScaffold
from .models import Deployment, served_name
from .naming import (
    OLLAMA_CONTAINER_PORT,
    VLLM_CONTAINER_PORT,
    ollama_service_name,
    ollama_service_name_for,
    vllm_service_name,
    vllm_service_name_for,
)
from .residency import ENGINE_LABEL

LITELLM_CONTAINER_PORT = 4000
LITELLM_CONFIG_FILENAME = 'litellm_config.yaml'
LITELLM_SERVICE = 'litellm'
API_KEY_ENV = 'LITELLM_MASTER_KEY'
# The key LiteLLM encrypts credentials stored in its database with. Unset, it
# uses the master key -- so rotating the master key would make every
# DB-stored route undecryptable. `set_master_key` pins it to the pre-rotation
# master key the first time the key changes; it must never change after that.
SALT_KEY_ENV = 'LITELLM_SALT_KEY'

# Dynamic-routing (admin-API) extras. When dynamic routing is on, the gateway's
# route table is managed live via LiteLLM's admin API against a Postgres-backed
# model store, instead of a static config file. See render_compose +
# ComposeBackend._reconcile_routes and docs/litellm-gateway-routing.md.
LITELLM_ROUTES_FILENAME = 'litellm_routes.json'  # rendered desired route set
# Append-only route registry for static-superset mode: accumulates the semantic
# route inputs (served name / engine / host) of every catalog *and* every live
# deployment ever merged, across all runbooks sharing this state dir. The gateway
# `model_list` is rendered from the whole registry, so a converge under one
# runbook's catalog can no longer strip another's still-live routes, and once
# every catalog has been merged once the rendered config is byte-stable (the
# gateway is never recreated). See docs/litellm-gateway-routing.md and
# ComposeBackend._update_route_registry.
LITELLM_REGISTRY_FILENAME = 'litellm_registry.json'
LITELLM_REGISTRY_VERSION = 1
# A route-registry row for a server this project does not run:
# ``{'engine': UPSTREAM_ROUTE, 'served': <its model name>, 'api_base': <url>}``.
UPSTREAM_ROUTE = 'upstream'
POSTGRES_SERVICE = 'postgres-litellm'
POSTGRES_CONTAINER_PORT = 5432
POSTGRES_DB_NAME = 'litellm'
POSTGRES_DB_USER = 'litellm'
DB_PASSWORD_ENV = 'LITELLM_DB_PASSWORD'  # managed secret in the sidecar .env
WEBUI_SECRET_ENV = 'WEBUI_SECRET_KEY'   # Open WebUI's session key, likewise
# Marks a LiteLLM route as infer-stack-managed, so reconcile only ever deletes
# routes it created (never a model added by hand through the UI/admin API).
ROUTE_ID_PREFIX = 'isr-'

#: Budget for reconciling dynamic routes against the gateway: listing, POSTs
#: and verification together. This default is the bootstrap budget (a fresh
#: gateway waits on Postgres health and runs DB migrations); it preserves the
#: previous 90 x 2 s listing retry.
ROUTE_RECONCILE_BOOTSTRAP_S = 180.0
#: Budget when the gateway was already running before this apply (steady state).
#: Short, because the controller holds its host-wide lock while applying; on
#: expiry the change stays pending and the next applying operation retries.
ROUTE_RECONCILE_STEADY_S = 20.0


def _vllm_route_entry(
    model_name: str, served: str, api_base: str
) -> dict[str, Any]:
    """One LiteLLM ``model_list`` entry routing ``model_name`` to a vLLM upstream.

    Shared by every render path (legacy per-deployment, catalog-superset, and
    the route registry) so a registry-rendered entry can never drift from what
    the catalog/deployment paths produce for the same endpoint."""
    return {
        'model_name': model_name,
        'litellm_params': {
            'model': f'openai/{served}',
            'api_base': api_base,
            'api_key': 'EMPTY',
        },
    }


def _ollama_route_entry(
    model_name: str, tag: str, api_base: str
) -> dict[str, Any]:
    """One LiteLLM ``model_list`` entry routing ``model_name`` to an Ollama tag
    (see :func:`_vllm_route_entry` for why this is factored out)."""
    return {
        'model_name': model_name,
        'litellm_params': {
            'model': f'ollama/{tag}',
            'api_base': api_base,
        },
    }


def _litellm_model_list(
    deployments: list[Deployment], assignments: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """One LiteLLM ``model_list`` entry per served endpoint alias."""
    entries: list[dict[str, Any]] = []
    for deployment in sorted(deployments, key=lambda g: (g.created_at, g.id)):
        if deployment.id not in assignments:
            continue
        if deployment.engine == 'vllm':
            served = served_name(deployment)
            api_base = f'http://{vllm_service_name(deployment)}:8000/v1'
            for endpoint in sorted(deployment.served):
                entries.append(_vllm_route_entry(endpoint, served, api_base))
        elif deployment.engine == 'ollama':
            api_base = f'http://{ollama_service_name(deployment)}:{OLLAMA_CONTAINER_PORT}'
            for endpoint, payload in sorted(deployment.served.items()):
                tag = payload.get('model', endpoint)
                entries.append(_ollama_route_entry(endpoint, tag, api_base))
    return entries


def _litellm_model_list_from_catalog(catalog: Any) -> list[dict[str, Any]]:
    """A *static superset* ``model_list``: one route per catalog endpoint.

    Unlike :func:`_litellm_model_list` (which routes only the currently-placed
    deployments), this routes *every* catalog endpoint to its deterministic
    upstream host (:func:`vllm_service_name_for` / :func:`ollama_service_name_for`).
    The resulting config therefore depends only on the catalog, not on which
    models happen to be up — so acquiring/releasing a model leaves the gateway's
    config (and its container) untouched (no blip). A route whose upstream is not
    currently running simply errors/cools-down until it comes up; the
    ``router_settings`` below make that warmup self-healing. ``/v1/models`` lists
    the whole catalog (some upstreams down) rather than only the live set.

    This static-superset path is the default. Its one limitation — it cannot give
    same-model ``--dedicated`` deployments distinct upstreams, and cannot route
    non-catalog acquires without a config change — is addressed by the opt-in
    *dynamic routing* mode (``dynamic_routing=True``), which manages routes live
    via LiteLLM's admin API against a Postgres model store (see
    :func:`_litellm_routes`, :meth:`ComposeBackend._reconcile_routes`, and
    ``docs/litellm-gateway-routing.md``). The two are mutually exclusive per
    converge; this function is used only when dynamic routing is off.
    """
    entries: list[dict[str, Any]] = []
    for name in sorted(getattr(catalog, 'endpoints', {})):
        try:
            req = catalog.resolve_endpoint(name)
        except Exception:  # noqa: BLE001 - a bad endpoint must not break the gateway
            continue
        if req.engine == 'vllm':
            served = req.served.get('served_model_name') or name
            api_base = (
                f'http://{vllm_service_name_for(served)}:{VLLM_CONTAINER_PORT}/v1'
            )
            entries.append(_vllm_route_entry(name, served, api_base))
        elif req.engine == 'ollama':
            host = req.spec.get('host') or req.host
            tag = req.served.get('model') or name
            api_base = (
                f'http://{ollama_service_name_for(host)}:{OLLAMA_CONTAINER_PORT}'
            )
            entries.append(_ollama_route_entry(name, tag, api_base))
    return entries


# -- Route registry (static-superset persistence) --------------------------
#
# The registry stores *semantic* route inputs (served name / engine / host),
# never rendered LiteLLM entries — render derives entries through the same
# helpers the catalog/deployment paths use (:func:`_litellm_model_list_from_registry`),
# so a future renderer change propagates to old registry rows automatically.
# All functions here are pure; the backend owns the file I/O and locking.


def _registry_incoming_from_catalog(catalog: Any) -> dict[str, dict[str, Any]]:
    """Semantic route rows for every resolvable endpoint of ``catalog``.

    Mirrors :func:`_litellm_model_list_from_catalog`'s iteration (unresolvable
    endpoints skipped) but emits registry rows keyed by endpoint name. A vLLM
    row carries only ``served`` (the upstream host is re-derived at render via
    :func:`vllm_service_name_for`); an Ollama row carries ``model`` (tag) +
    ``host``."""
    incoming: dict[str, dict[str, Any]] = {}
    for name in sorted(getattr(catalog, 'endpoints', {})):
        try:
            req = catalog.resolve_endpoint(name)
        except Exception:  # noqa: BLE001 - a bad endpoint must not break the gateway
            continue
        if req.engine == 'vllm':
            served = req.served.get('served_model_name') or name
            incoming[name] = {'engine': 'vllm', 'served': served}
        elif req.engine == 'ollama':
            host = req.spec.get('host') or req.host
            tag = req.served.get('model') or name
            incoming[name] = {'engine': 'ollama', 'model': tag, 'host': host}
    return incoming


def _registry_incoming_from_deployments(
    deployments: list[Deployment], assignments: dict[str, list[int]]
) -> dict[str, dict[str, Any]]:
    """Semantic route rows for every *placed* deployment in ``assignments``.

    ``deployments`` is the full ``desired`` set (which spans all runbooks via
    the shared ledger), so this keeps non-catalog / dedicated acquires routable
    and — because the registry persists — routable past release. One row per key
    of ``deployment.served`` (a coalesced deployment can back several endpoint
    aliases). Only ``vllm``/``ollama`` engines contribute; ``RESERVED_ENGINE``
    and unknown engines render no service, so they contribute no row — exactly
    as :func:`render_compose`'s service loop skips them.

    The vLLM ``served`` uses the same fallback chain as :func:`vllm_service_name`
    (``spec['served_model_name'] or sorted(served)[0] or id``), so a
    catalog-listed endpoint acquired live reduces to the identical row a catalog
    merge produces — live-vs-released status never moves the rendered bytes."""
    incoming: dict[str, dict[str, Any]] = {}
    for deployment in deployments:
        if deployment.id not in assignments:
            continue
        if deployment.engine == 'vllm':
            served = served_name(deployment)
            for endpoint in sorted(deployment.served):
                incoming[endpoint] = {'engine': 'vllm', 'served': served}
        elif deployment.engine == 'ollama':
            host = deployment.spec.get('host') or deployment.id
            for endpoint, payload in sorted(deployment.served.items()):
                tag = payload.get('model', endpoint)
                incoming[endpoint] = {
                    'engine': 'ollama',
                    'model': tag,
                    'host': host,
                }
    return incoming


def _litellm_model_list_from_registry(
    registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Render the gateway ``model_list`` from the whole accumulated registry.

    Iterates ``sorted(entries)`` (determinism, §8) and derives each upstream
    ``api_base`` through the live naming helpers, so the registry never becomes
    a rendered-config parse surface."""
    entries: list[dict[str, Any]] = []
    rows = registry.get('entries', {}) if isinstance(registry, dict) else {}
    for name in sorted(rows):
        entry = registry_route_entry(name, rows[name])
        if entry is not None:
            entries.append(entry)
    return entries


def registry_route_entry(name: str, row: Any) -> dict[str, Any] | None:
    """The LiteLLM entry one registry row renders to, or ``None`` if it cannot.

    The one derivation of a row's upstream: the render uses it, and so does
    ``routes list``. Upstreams come from the live naming helpers, so the
    registry never becomes a rendered-config parse surface.
    """
    if not isinstance(row, dict):
        return None
    engine = row.get('engine')
    if engine == 'vllm':
        served = row.get('served') or name
        api_base = f'http://{vllm_service_name_for(served)}:{VLLM_CONTAINER_PORT}/v1'
        return _vllm_route_entry(name, served, api_base)
    if engine == 'ollama':
        tag = row.get('model') or name
        host = row.get('host') or name
        api_base = f'http://{ollama_service_name_for(host)}:{OLLAMA_CONTAINER_PORT}'
        return _ollama_route_entry(name, tag, api_base)
    if engine == UPSTREAM_ROUTE and row.get('api_base'):
        # An OpenAI-compatible server this project does not run (a KubeAI
        # cluster's gateway): the row carries its address and the name it
        # serves the model under.
        return _vllm_route_entry(name, row.get('served') or name, str(row['api_base']))
    return None


def upstream_route(deployment_id: str, endpoint: str, served: str,
                   api_base: str) -> dict[str, Any]:
    """A dynamic route to a server this project does not run (a KubeAI Model).

    The same entry shape, and the same deterministic id, as a Compose engine's
    dynamic route (:func:`_litellm_routes`).
    """
    entry = _vllm_route_entry(endpoint, served, api_base)
    entry['model_info'] = {'id': _route_id(deployment_id, endpoint)}
    return entry


def _merge_route_registry(
    existing: dict[str, Any], incoming: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    """Merge ``incoming`` semantic rows into ``existing`` (append-only).

    Idempotent (merging identical rows is a no-op) and additive (never removes a
    row). On a conflict — same key, different row — *incoming wins* and a warning
    naming both definitions is emitted; the changed definition changes the
    rendered bytes, which is the one justified recreate. The existing ``version``
    is preserved (an unknown version merged under is not silently rewritten to
    the current schema; see :meth:`ComposeBackend._load_route_registry`)."""
    version = LITELLM_REGISTRY_VERSION
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(existing, dict):
        version = existing.get('version', LITELLM_REGISTRY_VERSION)
        prior = existing.get('entries')
        if isinstance(prior, dict):
            entries = {k: v for k, v in prior.items()}
    warnings: list[str] = []
    for name in sorted(incoming):
        row = incoming[name]
        if name in entries and entries[name] != row:
            warnings.append(
                f"route {name!r} redefined: {entries[name]} -> {row} "
                '(incoming wins; gateway will be recreated once)'
            )
        entries[name] = row
    return {'version': version, 'entries': entries}, warnings


def _seed_registry_from_litellm_config(
    config_text: str,
) -> tuple[dict[str, Any], list[str]]:
    """One-shot upgrade seed: recover registry rows from a rendered
    ``litellm_config.yaml`` so the first post-upgrade converge does not strip
    the other runbooks' routes.

    Single-format, migration-time only (no cross-version promise): ``openai/<served>``
    inverts *exactly* to ``{engine: vllm, served}``. Ollama rows are skipped with
    a warning — the host survives only as a non-invertible ``dns_slug`` inside
    ``api_base`` — and re-enter the registry at the next converge that has them
    in its catalog or live set. Anything else unparseable is likewise skipped."""
    entries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    try:
        data = yaml.safe_load(config_text) or {}
    except Exception:  # noqa: BLE001 - a torn file must not brick seeding
        return {'version': LITELLM_REGISTRY_VERSION, 'entries': {}}, [
            'seed: litellm_config.yaml is unparseable; starting an empty registry'
        ]
    for entry in data.get('model_list', []) or []:
        name = entry.get('model_name')
        model = (entry.get('litellm_params') or {}).get('model', '')
        if not name:
            continue
        if isinstance(model, str) and model.startswith('openai/'):
            entries[name] = {'engine': 'vllm', 'served': model[len('openai/'):]}
        elif isinstance(model, str) and model.startswith('ollama/'):
            warnings.append(
                f"seed: skipping Ollama route {name!r} (host not recoverable "
                'from the rendered api_base; it re-enters at its next converge)'
            )
        else:
            warnings.append(f'seed: skipping unparseable route {name!r}')
    return {'version': LITELLM_REGISTRY_VERSION, 'entries': entries}, warnings


def _dump_route_registry(registry: dict[str, Any]) -> str:
    """Canonical, byte-stable serialization (§3): sorted keys + trailing
    newline. A nondeterministic dump would manufacture phantom hash changes."""
    return json.dumps(registry, sort_keys=True, indent=2) + '\n'


def _route_id(deployment_id: str, endpoint: str) -> str:
    """Deterministic LiteLLM model id for one (deployment, endpoint) route.

    Stable across converges, so route reconcile (:meth:`ComposeBackend.
    _reconcile_routes`) can identify one logical route across renders.  A route
    whose id disappears is deleted by exactly this id; a route whose id remains
    but whose observable routing semantics drifted is replaced under the same id.
    The ``isr-`` prefix marks it
    infer-stack-managed so reconcile never deletes a model someone added by hand.
    """
    digest = hashlib.sha256(f'{deployment_id}|{endpoint}'.encode()).hexdigest()
    return f'{ROUTE_ID_PREFIX}{digest[:32]}'


def _litellm_routes(
    deployments: list[Deployment], assignments: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """Desired LiteLLM route set for the *live* deployments (dynamic routing).

    One entry per (placed deployment, served endpoint), addressing the
    deployment's **own** unique upstream service (:func:`vllm_service_name` with
    ``unique=True``). Several dedicated deployments of the same model therefore
    yield several entries that share one public ``model_name`` but point at
    distinct upstreams — LiteLLM load-balances the alias across them, so each
    runs on its own GPU while clients still ask for the single name. Each entry
    carries a deterministic ``model_info.id`` (:func:`_route_id`) so applying the
    set via the admin API is an idempotent diff, not fire-and-forget calls.
    """
    entries: list[dict[str, Any]] = []
    for deployment in sorted(deployments, key=lambda g: (g.created_at, g.id)):
        if deployment.id not in assignments:
            continue
        if deployment.engine == 'vllm':
            served = served_name(deployment)
            api_base = (
                f'http://{vllm_service_name(deployment, unique=True)}'
                f':{VLLM_CONTAINER_PORT}/v1'
            )
            for endpoint in sorted(deployment.served):
                entries.append(upstream_route(deployment.id, endpoint, served, api_base))
        elif deployment.engine == 'ollama':
            api_base = (
                f'http://{ollama_service_name(deployment)}:{OLLAMA_CONTAINER_PORT}'
            )
            for endpoint, payload in sorted(deployment.served.items()):
                tag = payload.get('model', endpoint)
                entries.append(
                    {
                        'model_name': endpoint,
                        'litellm_params': {
                            'model': f'ollama/{tag}',
                            'api_base': api_base,
                        },
                        'model_info': {'id': _route_id(deployment.id, endpoint)},
                    }
                )
    return entries


CONFIG_HASH_LABEL = 'infer-stack.config-hash'


def _postgres_service(
    images: dict[str, str], state: dict[str, str]
) -> dict[str, Any]:
    """Postgres backing LiteLLM's runtime model store (dynamic routing only).

    LiteLLM's admin API (``/model/new`` / ``/model/delete``) only functions with
    ``STORE_MODEL_IN_DB=true`` + a database, so dynamic routing needs a DB. This
    is an **internal** service (no published host port); LiteLLM reaches it on
    the compose network at ``postgres-litellm:5432``. The password is the managed
    :data:`DB_PASSWORD_ENV` secret in the sidecar ``.env`` (interpolated by
    ``docker compose --env-file``), so it never appears literally in the YAML.
    The healthcheck lets the litellm service ``depends_on`` it (condition:
    service_healthy) so the gateway only starts once the DB can accept queries.
    """
    data_path = state.get('postgres_litellm') or str(
        Path(next(iter(state.values()), '.')).parent / 'postgres-litellm'
    )
    return {
        'image': images.get('postgres', PINNED_IMAGES['postgres']),
        'environment': {
            'POSTGRES_USER': POSTGRES_DB_USER,
            'POSTGRES_PASSWORD': '${' + DB_PASSWORD_ENV + '}',
            'POSTGRES_DB': POSTGRES_DB_NAME,
        },
        'volumes': [f'{data_path}:/var/lib/postgresql/data'],
        'restart': 'unless-stopped',
        'labels': {ENGINE_LABEL: 'postgres'},
        'healthcheck': {
            'test': [
                'CMD-SHELL',
                f'pg_isready -U {POSTGRES_DB_USER} -d {POSTGRES_DB_NAME}',
            ],
            'interval': '5s',
            'timeout': '5s',
            'retries': 30,
            'start_period': '30s',
        },
    }


def _litellm_service(
    service_names: list[str],
    host_port: int,
    images: dict[str, str],
    aux_dir: str,
    master_key: str | None = None,
    config_hash: str | None = None,
    *,
    dynamic_routing: bool = False,
    salt_key: bool = False,
) -> dict[str, Any]:
    # Reference the managed key via ${...} rather than baking the literal secret
    # into the compose YAML. Its value lives in the sidecar .env next to the
    # compose file (written by master_key()), which `docker compose --env-file`
    # loads for interpolation — so the container and the readiness probe (which
    # reads the same .env) still agree regardless of the caller's shell env.
    key_value = (
        '${' + API_KEY_ENV + '}'
        if master_key is not None
        else '${' + API_KEY_ENV + ':-sk-local}'
    )
    environment = {API_KEY_ENV: key_value}
    if salt_key:
        # Only when the .env has one: LiteLLM treats an EMPTY salt as a key,
        # so a `${...:-}` default would silently change the encryption key.
        environment[SALT_KEY_ENV] = '${' + SALT_KEY_ENV + '}'
    if dynamic_routing:
        # DB-backed runtime model store so the admin API (/model/new,
        # /model/delete) works; the gateway then never needs recreating to learn
        # a route. Both are read from the env by LiteLLM. The password is
        # interpolated from the sidecar .env, so no secret lands in the YAML.
        environment['DATABASE_URL'] = (
            f'postgresql://{POSTGRES_DB_USER}:${{{DB_PASSWORD_ENV}}}'
            f'@{POSTGRES_SERVICE}:{POSTGRES_CONTAINER_PORT}/{POSTGRES_DB_NAME}'
        )
        environment['STORE_MODEL_IN_DB'] = 'True'
    labels = {ENGINE_LABEL: 'litellm'}
    if config_hash is not None:
        # LiteLLM reads its routing config once at startup; the file is bind-
        # mounted, so a config change alone does NOT change this service's spec
        # and `docker compose up -d` would leave the old container (and old
        # routes) running. Stamping the config hash onto a label makes the spec
        # change exactly when the config does, so converge recreates LiteLLM and
        # it picks up new/removed aliases. Without this, coalescing a second
        # alias onto a live deployment never becomes routable (readiness times out).
        labels[CONFIG_HASH_LABEL] = config_hash
    service: dict[str, Any] = {
        'image': images['litellm'],
        'command': [
            '--config',
            '/etc/litellm/config.yaml',
            '--port',
            str(LITELLM_CONTAINER_PORT),
        ],
        'ports': [f'{host_port}:{LITELLM_CONTAINER_PORT}'],
        'volumes': [f'{aux_dir}/{LITELLM_CONFIG_FILENAME}:/etc/litellm/config.yaml:ro'],
        'environment': environment,
        'restart': 'unless-stopped',
        'labels': labels,
    }
    if dynamic_routing:
        # Wait for the DB to accept queries before the gateway boots; do NOT add
        # per-model depends_on (that would churn the spec, i.e. blip, on every
        # model change). The route table is filled in afterward via the API.
        service['depends_on'] = {
            POSTGRES_SERVICE: {'condition': 'service_healthy'}
        }
    elif service_names:
        # Only wait on upstreams when there are any (zero models -> empty gateway).
        service['depends_on'] = sorted(service_names)
    return service


OPEN_WEBUI_SERVICE = 'open-webui'
OPEN_WEBUI_CONTAINER_PORT = 8080

NGINX_SERVICE = 'reverse-proxy'
NGINX_CONTAINER_PORT = 80
NGINX_CONFIG_FILENAME = 'nginx.conf'


def _open_webui_service(
    host_port: int,
    images: dict[str, str],
    state: dict[str, str],
    master_key: str | None,
    *,
    openai_urls: list[str] | None = None,
    ollama_urls: list[str] | None = None,
    depends_on: list[str] | None = None,
    run_as: str | None = None,
) -> dict[str, Any]:
    """A managed Open WebUI pointed at whatever front door is available.

    ``run_as`` (``uid:gid``, see :func:`open_webui_run_as`) runs it as the
    owner of its data directory, so what it writes there stays that owner's;
    None keeps the image's default user (root).


    Open WebUI holds two independent kinds of connection, wired here from the
    rendered services:

    * **OpenAI** (``openai_urls``) — the chat/completions front door. This is the
      LiteLLM gateway when it is enabled (so every declared endpoint alias is
      reachable at one URL); with LiteLLM off it falls back to the rendered
      upstreams' own ``/v1`` (a single vLLM/Ollama service, or several joined as
      ``OPENAI_API_BASE_URLS``). With nothing to point at, the OpenAI API is
      disabled rather than left dangling.
    * **Ollama** (``ollama_urls``) — the *native* Ollama API of any rendered
      Ollama daemon. This is what lets you pull/run/delete models from the UI
      and have the daemon load them on demand, independent of LiteLLM — i.e. a
      true drop-in for a hand-run ``ollama`` + Open WebUI stack.

    The spec is kept as independent of which models are live as it can be: the
    LiteLLM URL is fixed, and the Ollama daemon's service name is its stable
    structural id, so adding/removing other models does not rewrite this service
    and ``docker compose up -d`` leaves the UI running (the legacy "the UI never
    blinks" behavior). Chat history persists under the data dir.
    """
    # Reference the managed key via ${...} (resolved from the sidecar .env, see
    # _litellm_service) instead of inlining the secret into the compose YAML.
    key_value = (
        '${' + API_KEY_ENV + '}'
        if master_key is not None
        else '${' + API_KEY_ENV + ':-sk-local}'
    )
    data_path = state.get('open_webui') or str(
        Path(next(iter(state.values()), '.')).parent / 'open-webui'
    )
    openai_urls = list(openai_urls or [])
    ollama_urls = list(ollama_urls or [])
    env: dict[str, str] = {
        # Single-user workstation default; the port shouldn't be exposed
        # publicly. Tracked as a knob in dev/leasing-followups.md.
        'WEBUI_AUTH': 'False',
        # A managed secret (the .env, like the master key): without it the
        # image writes a generated one into its own directory, which a
        # non-root user cannot, and a recreate would sign everyone out.
        WEBUI_SECRET_ENV: '${' + WEBUI_SECRET_ENV + '}',
    }
    if run_as:
        # Its bundled static assets are copied at startup; into the data
        # directory, which this user can write, not the image's own.
        env['STATIC_DIR'] = '/app/backend/data/static'
        env['HOME'] = '/tmp'
    if openai_urls:
        env['ENABLE_OPENAI_API'] = 'True'
        if len(openai_urls) == 1:
            env['OPENAI_API_BASE_URL'] = openai_urls[0]
        else:
            env['OPENAI_API_BASE_URLS'] = ';'.join(openai_urls)
        env['OPENAI_API_KEY'] = key_value
    else:
        env['ENABLE_OPENAI_API'] = 'False'
    if ollama_urls:
        env['ENABLE_OLLAMA_API'] = 'True'
        if len(ollama_urls) == 1:
            env['OLLAMA_BASE_URL'] = ollama_urls[0]
        else:
            env['OLLAMA_BASE_URLS'] = ';'.join(ollama_urls)
    else:
        env['ENABLE_OLLAMA_API'] = 'False'
    service: dict[str, Any] = {
        'image': images['open_webui'],
        'ports': [f'{host_port}:{OPEN_WEBUI_CONTAINER_PORT}'],
        'environment': env,
        'volumes': [f'{data_path}:/app/backend/data'],
        'restart': 'unless-stopped',
        'labels': {ENGINE_LABEL: 'open-webui'},
    }
    if run_as:
        service['user'] = run_as
    if depends_on:
        service['depends_on'] = sorted(depends_on)
    return service


def open_webui_run_as(data_path: str | Path) -> tuple[str | None, str]:
    """``(uid:gid or None, why)``: who Open WebUI should run as.

    Its data directory's owner, so a data root stays removable by whoever
    owns it (as root, the container left files only root could delete). A
    directory not made yet will be made by this process, so its user. None,
    the image default (root), when the directory is root's, or when it or
    anything directly in it belongs to someone else: a data directory written
    by a root container before, whose files Open WebUI could no longer open.

    >>> import os, tempfile
    >>> d = tempfile.mkdtemp()
    >>> open_webui_run_as(os.path.join(d, 'open-webui'))[0] == f'{os.getuid()}:{os.stat(d).st_gid}'
    True
    """
    import os

    path = Path(data_path)
    if not path.exists():
        parent = next((p for p in path.parents if p.exists()), Path('/'))
        return f'{os.getuid()}:{parent.stat().st_gid}', 'new directory'
    owner = path.stat()
    if owner.st_uid == 0:
        return None, f'{path} is root\'s'
    try:
        foreign = [p.name for p in path.iterdir() if p.lstat().st_uid != owner.st_uid]
    except OSError:
        foreign = ['?']
    if foreign:
        return None, (f'{path} holds files another user owns ({", ".join(foreign[:3])}); '
                      f'`sudo chown -R {owner.st_uid}:{owner.st_gid} {path}` lets Open '
                      'WebUI run as its owner')
    return f'{owner.st_uid}:{owner.st_gid}', 'the directory\'s owner'


def _nginx_conf(*, litellm: bool, ui: bool) -> str:
    """A minimal HTTP reverse-proxy conf: one origin, path-routed.

    ``/v1/`` -> the LiteLLM gateway (the OpenAI API), ``/`` -> Open WebUI (or the
    gateway when there's no UI). Plain HTTP — no TLS, no auth — so the value is
    "one port, nothing to remember", not security. The ``map`` is valid here
    because a ``conf.d/*.conf`` file is included in nginx's ``http`` context.
    """
    api = f'http://{LITELLM_SERVICE}:{LITELLM_CONTAINER_PORT}'
    locations = ''
    if litellm:
        locations += (
            '    location /v1/ {\n'
            f'        proxy_pass {api}/v1/;\n'
            '        proxy_set_header Host $host;\n'
            '        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n'
            '        proxy_set_header X-Forwarded-Proto $scheme;\n'
            '        proxy_read_timeout 600s;\n'
            '    }\n'
        )
    # `/` serves the UI when present, else the gateway (so hitting the host root
    # still lands somewhere useful). Upgrade headers keep Open WebUI's websockets
    # working; client_max_body_size 0 allows large uploads.
    if ui:
        root = f'http://{OPEN_WEBUI_SERVICE}:{OPEN_WEBUI_CONTAINER_PORT}'
    elif litellm:
        root = api
    else:
        root = ''
    if root:
        locations += (
            '    location / {\n'
            f'        proxy_pass {root};\n'
            '        proxy_http_version 1.1;\n'
            '        proxy_set_header Upgrade $http_upgrade;\n'
            '        proxy_set_header Connection $connection_upgrade;\n'
            '        proxy_set_header Host $host;\n'
            '        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n'
            '        proxy_set_header X-Forwarded-Proto $scheme;\n'
            '    }\n'
        )
    return (
        'map $http_upgrade $connection_upgrade {\n'
        '    default upgrade;\n'
        "    ''      close;\n"
        '}\n\n'
        'server {\n'
        f'    listen {NGINX_CONTAINER_PORT};\n'
        '    server_name _;\n'
        '    client_max_body_size 0;\n'
        f'{locations}'
        '}\n'
    )


def _nginx_service(
    host_port: int,
    images: dict[str, str],
    *,
    aux_dir: str,
    depends_on: list[str],
    config_path: str | None = None,
    config_hash: str | None = None,
) -> dict[str, Any]:
    # BYO config (config_path) is mounted verbatim; otherwise the generated
    # nginx.conf in the state dir is used.
    mount = config_path or f'{aux_dir}/{NGINX_CONFIG_FILENAME}'
    labels = {ENGINE_LABEL: 'nginx'}
    if config_hash is not None:
        # Same trick as LiteLLM: the conf is bind-mounted, so stamp its hash on a
        # label to force a recreate when the routing changes.
        labels[CONFIG_HASH_LABEL] = config_hash
    service: dict[str, Any] = {
        'image': images['nginx'],
        'ports': [f'{host_port}:{NGINX_CONTAINER_PORT}'],
        'volumes': [f'{mount}:/etc/nginx/conf.d/default.conf:ro'],
        'restart': 'unless-stopped',
        'labels': labels,
    }
    if depends_on:
        service['depends_on'] = sorted(depends_on)
    return service


def set_master_key(env_path: Path, key: str) -> None:
    """Replace the LiteLLM master key in ``env_path`` without losing DB routes.

    The first time the key changes, the old one is pinned as
    ``LITELLM_SALT_KEY`` -- the value LiteLLM has been encrypting stored
    credentials with. After that the salt stays put and only the key moves.
    """
    if not key.startswith('sk-'):
        # master_key() would silently replace it on the next render.
        raise ValueError(f'{API_KEY_ENV} must start with "sk-" (LiteLLM rejects others)')
    existing = parse_env_file(env_path)
    values = {API_KEY_ENV: key}
    old = existing.get(API_KEY_ENV, '').strip()
    if old and old != key and not existing.get(SALT_KEY_ENV, '').strip():
        values[SALT_KEY_ENV] = old
    write_env_file(env_path, values)


@dataclass
class FrontDoor:
    """The rendered front door: its Compose services and config files."""

    services: dict[str, Any]
    litellm_config: str | None
    nginx_config: str | None
    litellm_routes: list[dict[str, Any]] | None


def render_front_door(
    deployments: list[Deployment],
    assignments: dict[str, list[int]],
    *,
    engine_services: list[str],
    vllm_v1_urls: list[str],
    ollama_native_urls: list[str],
    images: dict[str, str],
    state: dict[str, str],
    litellm: bool,
    litellm_port: int,
    litellm_master_key: str | None,
    litellm_salt_key: bool,
    ui: bool,
    ui_port: int,
    reverse_proxy: bool,
    reverse_proxy_port: int,
    reverse_proxy_config: str | None,
    aux_dir: str | Path | None,
    catalog: Any,
    route_registry: dict[str, Any] | None,
    dynamic_routing: bool,
    upstream_routes: list[dict[str, Any]] | None = None,
    ui_run_as: str | None = None,
) -> FrontDoor:
    """Render the gateway, its database, Open WebUI and the reverse proxy.

    The engines are the caller's: it passes the services it rendered
    (``engine_services``, only for the legacy per-model ``depends_on``) and
    the in-network URLs a UI with no gateway can talk to directly.
    ``upstream_routes`` are dynamic routes to servers this project does not
    run (see :func:`upstream_route`).
    """
    services: dict[str, Any] = {}
    litellm_config = None
    litellm_routes = None
    # The front door (gateway + UI) is rendered whenever it's enabled, even with
    # zero models — it's a standing entry point, not a per-model service. So
    # releasing/evicting every model leaves an empty gateway (and an empty Open
    # WebUI picker) up instead of tearing the whole stack down; only an explicit
    # `stack down` removes it. With no models the model_list is simply empty.
    if litellm:
        # Three route-table strategies, in order of preference:
        #  * DYNAMIC ROUTING: the rendered config is a STATIC base (empty
        #    model_list); the real routes live in Postgres and are applied to the
        #    running gateway via the admin API (see _reconcile_routes). The config
        #    hash never changes as models come/go, so the gateway is never
        #    recreated — no blip, and per-deployment routing works (so same-model
        #    --dedicated deployments each get their own upstream).
        #  * ROUTE REGISTRY (static-superset default from ComposeBackend): render
        #    from the whole accumulated registry (every catalog + live deployment
        #    ever merged, across all runbooks). Byte-stable once seeded, so the
        #    gateway is never recreated and a cross-catalog converge can no longer
        #    strip another runbook's routes. The backend loads/merges/writes the
        #    registry and passes the merged dict in; this function stays pure.
        #  * STATIC SUPERSET (catalog): one route per catalog endpoint to a
        #    deterministic host; config depends only on the catalog, so the
        #    gateway is not recreated as models come/go (no blip) but same-model
        #    dedicated collapses to one upstream. Unreachable from ComposeBackend
        #    once the registry is wired; kept for direct callers/tests.
        #  * LEGACY (no catalog): route only the placed deployments; churns the
        #    config (and recreates the gateway) on every model change.
        if dynamic_routing:
            entries: list[dict[str, Any]] = []
            litellm_routes = (_litellm_routes(deployments, assignments)
                              + list(upstream_routes or []))
            litellm_depends: list[str] = []
        elif route_registry is not None:
            entries = _litellm_model_list_from_registry(route_registry)
            litellm_depends = []  # no per-model depends_on -> no churn
        elif catalog is not None:
            entries = _litellm_model_list_from_catalog(catalog)
            litellm_depends = []  # no per-model depends_on -> no churn
        else:
            entries = _litellm_model_list(deployments, assignments)
            litellm_depends = list(engine_services)
        litellm_config = yaml.safe_dump(
            {
                'model_list': entries,
                'general_settings': {
                    'master_key': f'os.environ/{API_KEY_ENV}'
                },
                # An upstream vLLM/Ollama is unreachable only briefly, while it
                # loads its model (LiteLLM does not wait for upstream health to
                # start). Retry transient connection errors and don't park a
                # model in a long cooldown, so the warmup window is self-healing
                # instead of surfacing as client 500s ("Connection error.
                # Received Model Deployment=…").
                'router_settings': {
                    'num_retries': 3,
                    'timeout': 600,
                    'cooldown_time': 5,
                    'allowed_fails': 100,
                },
            },
            sort_keys=False,
        )
        config_hash = hashlib.sha256(
            litellm_config.encode('utf-8')
        ).hexdigest()[:12]
        if dynamic_routing:
            services[POSTGRES_SERVICE] = _postgres_service(images, state)
        services[LITELLM_SERVICE] = _litellm_service(
            litellm_depends,
            litellm_port,
            images,
            str(aux_dir or '.'),
            master_key=litellm_master_key,
            config_hash=config_hash,
            dynamic_routing=dynamic_routing,
            salt_key=litellm_salt_key,
        )

    # Open WebUI is its own standing front door, rendered whenever ``ui`` is set
    # — it does NOT require LiteLLM. Its OpenAI connection prefers the gateway
    # (one URL covers every alias) and falls back to the rendered vLLM upstreams'
    # own /v1 when there is no gateway. Its native Ollama connection always
    # points straight at any Ollama daemon, so you can pull/run models from the
    # UI and have the daemon load them on demand — a true drop-in for a
    # hand-run ollama + Open WebUI stack. depends_on lists only LiteLLM (the one
    # service guaranteed present alongside the UI); the per-model upstreams come
    # and go, so the UI tolerates them being absent rather than hard-depending.
    # With a gateway the UI is a standing front door (renders even at zero
    # models). Without one it is only meaningful pointed at a live upstream, so
    # render it only when there is something to connect to — otherwise an empty
    # desired set has nothing to run and converge tears the project down.
    if ui and (litellm or vllm_v1_urls or ollama_native_urls):
        if litellm:
            openai_urls = [f'http://{LITELLM_SERVICE}:{LITELLM_CONTAINER_PORT}/v1']
            ui_depends = [LITELLM_SERVICE]
        else:
            openai_urls = list(vllm_v1_urls)
            ui_depends = []
        services[OPEN_WEBUI_SERVICE] = _open_webui_service(
            ui_port,
            images,
            state,
            litellm_master_key,
            openai_urls=openai_urls,
            ollama_urls=ollama_native_urls,
            depends_on=ui_depends,
            run_as=ui_run_as,
        )

    # Optional single-port HTTP reverse proxy fronting the gateway (+ UI). Needs
    # the gateway, so it's only rendered alongside litellm.
    nginx_config = None
    if reverse_proxy and litellm:
        depends = [LITELLM_SERVICE] + ([OPEN_WEBUI_SERVICE] if ui else [])
        if reverse_proxy_config:
            services[NGINX_SERVICE] = _nginx_service(
                reverse_proxy_port, images, aux_dir=str(aux_dir or '.'),
                depends_on=depends, config_path=reverse_proxy_config,
            )
        else:
            nginx_config = _nginx_conf(litellm=litellm, ui=ui)
            services[NGINX_SERVICE] = _nginx_service(
                reverse_proxy_port, images, aux_dir=str(aux_dir or '.'),
                depends_on=depends,
                config_hash=hashlib.sha256(
                    nginx_config.encode('utf-8')
                ).hexdigest()[:12],
            )

    return FrontDoor(services=services, litellm_config=litellm_config,
                     nginx_config=nginx_config, litellm_routes=litellm_routes)


class Gateway(ConvergeScaffold):
    """The front door's state: settings, the managed keys, the route registry.

    Owned by the backend that runs it (``ComposeBackend.gateway``). The
    backend supplies route rows for what it serves; this merges, persists and
    renders them, reconciles dynamic routes through LiteLLM's admin API, and
    keeps the ``.env`` secrets. It shares the backend's state directory (and
    so its converge lock), because the gateway's files live beside the
    compose project that runs it.
    """

    def __init__(
        self,
        state_dir: str | Path,
        *,
        ports: dict[str, int],
        litellm: bool = True,
        ui: bool = True,
        reverse_proxy: bool = False,
        reverse_proxy_port: int = 80,
        dynamic_routing: bool = False,
        http: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        base_url: str | Callable[[], str] | None = None,
    ):
        self.state_dir = Path(state_dir)
        # Where clients reach the gateway (no ``/v1``). None: this host, on the
        # published port. The in-cluster gateway passes a callable, so the
        # node address is looked up only when something asks.
        self.base_url: str | Callable[[], str] | None = base_url
        self.ports = ports
        self.litellm = litellm
        self.ui = ui
        self.reverse_proxy = reverse_proxy
        self.reverse_proxy_port = reverse_proxy_port
        self.dynamic_routing = dynamic_routing
        # None: `requests`, imported on first use. It costs ~75 ms, and a TUI
        # launch or a ledger-only command never makes a request.
        self._http = http
        self._sleep = sleep
        self._clock = clock
        self.assume_yes = True

    @property
    def http(self) -> Any:
        if self._http is None:
            import requests

            self._http = requests
        return self._http

    @http.setter
    def http(self, value: Any) -> None:
        self._http = value

    @property
    def litellm_port(self) -> int:
        return self.ports.get('litellm', DEFAULT_PORTS['litellm'])

    @property
    def ui_port(self) -> int:
        return self.ports.get('open_webui', DEFAULT_PORTS['open_webui'])

    @property
    def _env_path(self) -> Path:
        return self.state_dir / '.env'

    @property
    def _routes_file(self) -> Path:
        return self.state_dir / LITELLM_ROUTES_FILENAME

    @property
    def _registry_file(self) -> Path:
        return self.state_dir / LITELLM_REGISTRY_FILENAME

    def master_key(self) -> str:
        """The managed LiteLLM master key.

        infer-stack manages this secret in the state dir's ``.env``: reused if
        already present (you may pin your own ``sk-`` key there), otherwise
        generated and persisted. The caller doesn't need to invent or export it
        — it is baked into the LiteLLM service, used by the readiness probe, and
        shipped in the env-file descriptor (``infer-stack env KEY`` prints it).
        """
        existing = parse_env_file(self._env_path)
        key = ensure_secret(existing, API_KEY_ENV, prefix='sk-')
        if key != existing.get(API_KEY_ENV):
            write_env_file(self._env_path, {API_KEY_ENV: key})
        return key

    def rotate_master_key(self) -> dict[str, str | None]:
        """Write a fresh master key to the ``.env``; return the values it replaced.

        Only the file: the gateway and Open WebUI pick it up when the next apply
        recreates them (their fingerprints hash the key). The caller holds the
        publication lock and passes the return value to
        :meth:`restore_env` if that apply does not happen.
        """
        before = parse_env_file(self._env_path)
        set_master_key(self._env_path, ensure_secret({}, API_KEY_ENV, prefix='sk-'))
        return {k: before.get(k) for k in (API_KEY_ENV, SALT_KEY_ENV)}

    def restore_env(self, values: dict[str, str | None]) -> None:
        """Put back what :meth:`rotate_master_key` replaced."""
        from ..env_utils import remove_env_keys

        write_env_file(self._env_path, {k: v for k, v in values.items() if v is not None})
        remove_env_keys(self._env_path, [k for k, v in values.items() if v is None])

    def gateway_accepts(self, key: str, *, wait: float = 0.0) -> bool | None:
        """Does the gateway accept ``key``? ``None`` if it never answered.

        Polls ``/v1/models`` for up to ``wait`` seconds while the gateway is
        unreachable or still starting. A definite no is 401/403, or the 400
        LiteLLM actually answers a wrong key with (measured on the pinned image).
        """
        deadline = self._clock() + wait
        while True:
            try:
                resp = self.http.get(
                    f'{self._gateway_base()}/v1/models',
                    headers={'Authorization': f'Bearer {key}'}, timeout=10.0,
                )
                status = getattr(resp, 'status_code', 0)
            except Exception:  # noqa: BLE001 - not up yet
                status = 0
            if status == 200:
                return True
            if status in (400, 401, 403):
                return False
            if self._clock() >= deadline:
                return None
            self._sleep(2.0)

    def webui_secret(self) -> str:
        """Open WebUI's managed session key (the .env), made on first use."""
        existing = parse_env_file(self._env_path)
        key = ensure_secret(existing, WEBUI_SECRET_ENV)
        if key != existing.get(WEBUI_SECRET_ENV):
            write_env_file(self._env_path, {WEBUI_SECRET_ENV: key})
        return key

    def db_password(self) -> str:
        """The managed Postgres password for LiteLLM's model store.

        Same managed-secret pattern as :meth:`master_key`: reused if already
        present in the state-dir ``.env`` (you may pin your own), else generated
        and persisted. ``docker compose --env-file`` interpolates it into the
        postgres + litellm services, so it never appears literally in the YAML.
        Only used when ``dynamic_routing`` is on. ``token_urlsafe`` output is safe
        inside the ``postgresql://`` URL (no ``@ : /`` characters).
        """
        existing = parse_env_file(self._env_path)
        pw = ensure_secret(existing, DB_PASSWORD_ENV)
        if pw != existing.get(DB_PASSWORD_ENV):
            write_env_file(self._env_path, {DB_PASSWORD_ENV: pw})
        return pw

    def _load_route_registry(self) -> dict[str, Any]:
        """Read the route registry, tolerantly (fail-open — a broken registry
        must never block a converge).

        Missing file → seed from the live ``litellm_config.yaml`` if present
        (upgrade migration, §6), else an empty registry. An *unknown* schema
        version whose ``entries`` still parses as a name→row map is preserved
        as-is (render what's understood, warn, do NOT rewrite) rather than
        reseeded, so a binary rollback doesn't discard the accumulated union.
        Only a structurally unusable file (not a map / garbage JSON) falls back
        to seeding."""
        from .._log import logger

        if not self._registry_file.exists():
            config = self.state_dir / LITELLM_CONFIG_FILENAME
            if config.exists():
                seeded, warnings = _seed_registry_from_litellm_config(
                    config.read_text()
                )
                for w in warnings:
                    logger.warning('  route registry: {}', w)
                logger.info(
                    '  route registry: seeded {} vLLM route(s) from {}',
                    len(seeded['entries']), config.name,
                )
                return seeded
            return {'version': LITELLM_REGISTRY_VERSION, 'entries': {}}
        try:
            data = json.loads(self._registry_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                '  route registry: {} is unreadable ({}); rebuilding from seed',
                self._registry_file.name, exc,
            )
            data = None
        if not isinstance(data, dict) or not isinstance(
            data.get('entries'), dict
        ):
            config = self.state_dir / LITELLM_CONFIG_FILENAME
            if config.exists():
                seeded, warnings = _seed_registry_from_litellm_config(
                    config.read_text()
                )
                for w in warnings:
                    logger.warning('  route registry: {}', w)
                return seeded
            return {'version': LITELLM_REGISTRY_VERSION, 'entries': {}}
        version = data.get('version')
        if version != LITELLM_REGISTRY_VERSION:
            logger.warning(
                '  route registry: unknown schema version {!r} in {} — '
                'rendering as-is without rewrite (fields this renderer does '
                'not understand are ignored)',
                version, self._registry_file.name,
            )
        return data

    def _save_route_registry(self, merged: dict[str, Any]) -> None:
        """Persist a merged registry if it changed (under the converge flock)."""
        from .._log import logger

        existing = self._load_route_registry()
        if merged == existing:
            return
        prior = existing.get('entries', {}) if isinstance(existing, dict) else {}
        added = sorted(set(merged['entries']) - set(prior))
        updated = sorted(
            k for k in merged['entries'] if k in prior and merged['entries'][k] != prior[k]
        )
        if added:
            logger.info('  route registry: +{} route(s): {}', len(added), ', '.join(added))
        if updated:
            logger.info('  route registry: updated route(s): {}', ', '.join(updated))
        self._atomic_write(self._registry_file, _dump_route_registry(merged))

    def merge_route_registry(
        self, incoming: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Public write path for out-of-converge registry seeds (``routes seed``).

        Takes the converge flock, read-merge-writes the registry, and returns the
        merged dict. ``converge`` only ever merges the invoking process's own
        catalog, so a standalone caller (seeding a *sibling* runbook's catalog)
        needs this to fold extra rows in before the follow-up ``reconcile``
        renders+applies. The flock here and the one the subsequent converge takes
        are sequential acquisitions, not nested — no reentrancy concern."""
        from .._log import logger

        with self._converge_lock():
            existing = self._load_route_registry()
            merged, warnings = _merge_route_registry(existing, incoming)
            for w in warnings:
                logger.warning('  route registry: {}', w)
            if merged != existing:
                self._atomic_write(
                    self._registry_file, _dump_route_registry(merged)
                )
        return merged

    def _gateway_base(self) -> str:
        where = self.base_url
        if where is None:
            base = f'http://127.0.0.1:{self.litellm_port}'
        elif isinstance(where, str):
            base = where
        else:
            base = where()
        return base.rstrip('/')

    def _auth_headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.master_key()}'}

    def _desired_routes(self) -> list[dict[str, Any]]:
        """The rendered desired route set (litellm_routes.json), or empty."""
        try:
            data = json.loads(self._routes_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _reconcile_routes(
        self, *, deadline_s: float = ROUTE_RECONCILE_BOOTSTRAP_S, delay: float = 2.0,
    ) -> bool:
        """Make the live gateway's managed routes match the rendered route set.

        The render half wrote the desired routes (one per live deployment×
        endpoint) to ``litellm_routes.json``; this is the apply half. List the
        gateway's current models, add the missing routes and delete the ones no
        longer desired -- through the admin API, with **no** container restart --
        then list again to verify both ids and routing semantics, re-diffing and
        retrying failed calls until the table matches or the budget runs out.

        Properties this relies on:

        * **Idempotent.** A redundant apply re-diffs to the same set and does
          nothing.
        * **Drift-healing.** Routes lost to a gateway/DB restart reappear in the
          diff and are re-added; stale routes from a prior run (still in the DB)
          are deleted because they're no longer desired.
        * **Co-existence.** Only routes infer-stack created (id prefix ``isr-``)
          are ever deleted, so a model added by hand through the UI/API is left
          alone.

        **Bounded and reported.** Everything -- listing retries while the gateway
        starts, every POST, and the final verification -- shares one wall-clock
        budget, ``deadline_s``. A retry count alone would not bound it: a listing
        can take 10 s and a POST 30 s. Returns ``True`` only when the verified
        managed route set equals the desired set; any failure is logged and
        returns ``False`` rather than raising, so the caller decides whether an
        unverified route set blocks anything.
        """
        from .._log import logger

        deadline = self._clock() + max(0.0, deadline_s)
        desired = {
            r['model_info']['id']: r
            for r in self._desired_routes()
            if isinstance(r.get('model_info'), dict) and r['model_info'].get('id')
        }
        desired_semantics = {rid: self._route_semantics(route)
                             for rid, route in desired.items()}
        rounds = 0
        while True:
            current = self._list_managed_routes(deadline=deadline, delay=delay)
            if current is None:
                logger.warning(
                    'dynamic routing: route set not reconciled and verified within '
                    '{:g}s; leaving it for the next apply', deadline_s,
                )
                return False
            mismatched = sorted(
                rid for rid in desired.keys() & current.keys()
                if desired_semantics[rid] != current[rid]
            )
            to_add_ids = sorted((desired.keys() - current.keys()) | set(mismatched))
            to_delete = sorted((current.keys() - desired.keys()) | set(mismatched))
            to_add = [desired[rid] for rid in to_add_ids]
            if not (to_add or to_delete):
                return True            # this listing is the verification
            if rounds:
                logger.info('dynamic routing: route set still differs; retrying')
            rounds += 1
            ok = True
            # A same-id semantic drift must be removed before it can be re-added;
            # model/new is not an update API on every LiteLLM release.
            for rid in to_delete:
                # ok_if_missing: with a shared gateway, another converge may have
                # deleted this route already; "not found in db" means the desired
                # end-state (route gone) is reached, so don't treat it as an error.
                ok &= self._post_route(
                    '/model/delete', {'id': rid}, rid, ok_if_missing=True,
                    deadline=deadline,
                )
            for route in to_add:
                ok &= self._post_route(
                    '/model/new', route, route.get('model_name'), deadline=deadline,
                )
            logger.info(
                'dynamic routing: +{} route(s), -{} route(s), ~{} replacement(s) '
                '(now {} desired)',
                len(to_add), len(to_delete), len(mismatched), len(desired),
            )
            if not ok:
                # A transient admin-API failure: spend the rest of the budget
                # re-diffing rather than giving up with most of it unused.
                if deadline - self._clock() <= delay:
                    return False
                self._sleep(delay)

    @staticmethod
    def _route_semantics(route: dict[str, Any]) -> dict[str, Any]:
        """Observable route fields infer-stack owns and must verify.

        LiteLLM's model-info response contains additional database/runtime fields
        and may redact credentials.  The public alias, upstream model, and
        upstream base URL are the routing semantics infer-stack can both set and
        reliably observe.  A matching managed id with different values here is
        drift and is replaced, not accepted as healthy.
        """
        params = route.get('litellm_params') or {}
        return {
            'model_name': route.get('model_name'),
            'model': params.get('model'),
            'api_base': params.get('api_base'),
        }

    def _list_managed_routes(
        self, *, deadline: float, delay: float
    ) -> dict[str, dict[str, Any]] | None:
        """Observable semantics of infer-stack-managed gateway routes.

        Retries while the gateway is unreachable, until ``deadline`` (a value of
        ``self._clock``). Each request's own timeout is capped by the time left.
        Returns ``None`` if no listing succeeded in time.
        """
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return None
            resp = None
            try:
                resp = self.http.get(
                    f'{self._gateway_base()}/v1/model/info',
                    headers=self._auth_headers(),
                    timeout=min(10.0, remaining),
                )
            except Exception:  # noqa: BLE001 - the gateway may still be starting
                resp = None
            if resp is not None and getattr(resp, 'status_code', 0) == 200:
                routes: dict[str, dict[str, Any]] = {}
                for m in (resp.json().get('data') or []):
                    rid = (m.get('model_info') or {}).get('id')
                    if isinstance(rid, str) and rid.startswith(ROUTE_ID_PREFIX):
                        routes[rid] = self._route_semantics(m)
                return routes
            if deadline - self._clock() <= delay:
                return None
            self._sleep(delay)

    def _post_route(
        self,
        path: str,
        payload: dict[str, Any],
        label: Any,
        *,
        ok_if_missing: bool = False,
        deadline: float | None = None,
    ) -> bool:
        """POST one admin-API call (``/model/new`` or ``/model/delete``).

        Returns whether it reached its desired end state. A failure is logged,
        not raised, so the remaining calls still run. ``ok_if_missing`` accepts a
        "model not found" response (a delete whose target is already gone).
        The request timeout is capped by the time left before ``deadline``.
        """
        from .._log import logger

        timeout = 30.0
        if deadline is not None:
            remaining = deadline - self._clock()
            if remaining <= 0:
                logger.warning(
                    'dynamic routing: POST {} {} skipped: route deadline passed',
                    path, label,
                )
                return False
            timeout = min(timeout, remaining)
        try:
            resp = self.http.post(
                f'{self._gateway_base()}{path}',
                headers=self._auth_headers(),
                json=payload,
                timeout=timeout,
            )
        except Exception as ex:  # noqa: BLE001 - one bad call must not abort apply
            logger.warning('dynamic routing: POST {} {} error: {}', path, label, ex)
            return False
        if getattr(resp, 'status_code', 0) >= 300:
            body = str(getattr(resp, 'text', ''))
            if ok_if_missing and 'not found' in body.lower():
                return True
            logger.warning(
                'dynamic routing: POST {} {} -> {} {}',
                path, label, resp.status_code, body[:200],
            )
            return False
        return True

    def access(self, endpoints: list[str]) -> dict[str, Any] | None:
        """Where a client reaches these endpoints, for the env-file descriptor.

        With the LiteLLM front door, that is one ``base_url`` and the request
        model name is the endpoint alias itself. With LiteLLM off there is no
        single base URL, but a managed Open WebUI (if on) is still a useful
        access point, so report just its URL rather than ``None``.
        """
        if not self.litellm:
            if self.ui:
                return {'ui_url': f'http://127.0.0.1:{self.ui_port}'}
            return None
        info: dict[str, Any] = {
            'base_url': f'{self._gateway_base()}/v1',
            'api_key_env': API_KEY_ENV,
            'api_key': self.master_key(),
            'request_names': {ep: ep for ep in endpoints},
        }
        if self.ui:
            info['ui_url'] = f'http://127.0.0.1:{self.ui_port}'
        if self.reverse_proxy:
            # The unified front door: one origin, UI at / and the API at /v1.
            info['proxy_url'] = f'http://127.0.0.1:{self.reverse_proxy_port}'
        return info

    def merged_route_registry(self, incoming: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """The registry with ``incoming`` rows merged in, in memory (no write)."""
        from .._log import logger

        merged, warnings = _merge_route_registry(self._load_route_registry(), incoming)
        for w in warnings:
            logger.warning('  route registry: {}', w)
        return merged
