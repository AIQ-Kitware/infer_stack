"""The front door: the LiteLLM gateway, its routes and keys, and the UI and
proxy beside it.

Engines (vLLM, Ollama) are the backend's business; everything a client
talks to is here. A backend hands the gateway where its models are
(:class:`~infer_stack.leasing.routes.GatewayRoute` s), and the gateway renders
its Compose services and config, reconciles dynamic routes through LiteLLM's
admin API, and manages the master key. The compose backend runs it beside its engines; the
kubeai backend runs it alone, in front of a cluster.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from ..config import PINNED_IMAGES
from ..config import DEFAULT_PORTS
from ..env_utils import ensure_secret, parse_env_file, write_env_file
from .backend import ConnectionInfo, ConvergeScaffold
from .endpoints import ExternalTarget
from .launch import effective_max_model_len
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
from .profile import ProfileMismatch
from .routes import GatewayRoute, route_table

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
# The route registry: routes of deployments no published catalog defines,
# kept past release (see remembered_rows and docs/litellm-gateway-routing.md).
LITELLM_REGISTRY_FILENAME = 'litellm_registry.json'
LITELLM_REGISTRY_VERSION = 1
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


def compose_catalog_route(alias: str, request: Any) -> GatewayRoute | None:
    """Where a Compose front door sends a managed catalog endpoint: the
    engine service its served name determines (the same one every
    deployment of it gets without ``--dedicated``). A vLLM route advertises
    the endpoint's effective ``max_model_len`` contract, so clients read the
    managed context window from the gateway rather than guessing."""
    if request.engine == 'vllm':
        served = request.served.get('served_model_name') or alias
        return GatewayRoute(alias, 'openai', served,
                            f'http://{vllm_service_name_for(served)}:{VLLM_CONTAINER_PORT}/v1',
                            max_input_tokens=effective_max_model_len(request.spec.get('runtime')))
    if request.engine == 'ollama':
        host = request.spec.get('host') or request.host
        tag = request.served.get('model') or alias
        return GatewayRoute(alias, 'ollama', tag,
                            f'http://{ollama_service_name_for(host)}:{OLLAMA_CONTAINER_PORT}')
    return None


def deployment_route_max_input_tokens(
    deployment: Deployment, endpoint: str, *, catalog: Any = None,
) -> int:
    """Context window to advertise for one alias of ``deployment``.

    ``max_model_len`` is a capacity field, so shared-compatible acquisition may
    satisfy a smaller endpoint with a larger existing deployment.  The public
    endpoint must keep its own contract in that case: a 65K alias coalesced onto
    a 262K process still advertises 65K.  New catalog requests persist that
    per-alias value in ``deployment.served``.  For deployments written by older
    infer-stack versions, recover a published alias from the current catalog;
    ad-hoc/unknown aliases fall back to the actual deployment window.
    """
    payload = deployment.served.get(endpoint)
    if isinstance(payload, dict):
        value = payload.get('max_input_tokens')
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    if catalog is not None and endpoint in (getattr(catalog, 'endpoints', None) or {}):
        try:
            request = catalog.resolve_endpoint(endpoint).to_request()
            if request.engine == 'vllm':
                return effective_max_model_len(request.spec.get('runtime'))
        except Exception:  # noqa: BLE001 - route rendering must survive a bad catalog row
            pass
    return effective_max_model_len(deployment.spec.get('runtime'))


def catalog_routes(
    catalog: Any,
    managed: Callable[[str, Any], GatewayRoute | None] = compose_catalog_route,
    *, dynamic: bool = False,
) -> list[GatewayRoute]:
    """One route per endpoint of ``catalog``, sorted by alias.

    An external target routes to its own server, whatever the backend;
    ``managed`` maps a managed endpoint's request to where this backend runs
    it (or ``None``). With ``dynamic`` only the external routes, each with
    its stable managed id: dynamic routing routes managed endpoints per
    deployment (:func:`deployment_routes`). An endpoint that does not
    resolve is skipped: a bad endpoint must not break the gateway.
    """
    routes: list[GatewayRoute] = []
    for alias in sorted(getattr(catalog, 'endpoints', None) or {}):
        try:
            resolved = catalog.resolve_endpoint(alias)
            target = resolved.target
            if isinstance(target, ExternalTarget):
                routes.append(GatewayRoute(
                    alias, 'openai', target.model, target.api_base,
                    key_env=target.api_key_env, origin='external',
                    route_id=_route_id('external', alias) if dynamic else None))
                continue
            if dynamic:
                continue
            route = managed(alias, resolved.to_request())
        except Exception:  # noqa: BLE001 - a bad endpoint must not break the gateway
            continue
        if route is not None:
            routes.append(route)
    return routes


def deployment_routes(
    deployments: list[Deployment], assignments: dict[str, list[int]],
    *, dynamic: bool = False, catalog: Any = None,
) -> list[GatewayRoute]:
    """One route per (placed deployment, served endpoint alias).

    Static (the default): to the engine service its served name determines,
    so a catalog endpoint acquired live routes exactly as its catalog route
    does and live-vs-released never moves the rendered bytes. ``dynamic``: to
    the deployment's **own** service, with a managed id per (deployment,
    alias) (:func:`_route_id`), so same-model ``--dedicated`` deployments
    share the alias and LiteLLM balances across them. Only ``vllm`` and
    ``ollama`` render a service, so only they route.
    """
    routes: list[GatewayRoute] = []
    for deployment in sorted(deployments, key=lambda g: (g.created_at, g.id)):
        if deployment.id not in assignments:
            continue
        if deployment.engine == 'vllm':
            served = served_name(deployment)
            service = (vllm_service_name(deployment, unique=True) if dynamic
                       else vllm_service_name_for(served))
            api_base = f'http://{service}:{VLLM_CONTAINER_PORT}/v1'
            for endpoint in sorted(deployment.served):
                context = deployment_route_max_input_tokens(
                    deployment, endpoint, catalog=catalog)
                routes.append(GatewayRoute(
                    endpoint, 'openai', served, api_base, origin='deployment',
                    route_id=_route_id(deployment.id, endpoint) if dynamic else None,
                    max_input_tokens=context))
        elif deployment.engine == 'ollama':
            host = deployment.spec.get('host') or deployment.id
            service = (ollama_service_name(deployment) if dynamic
                       else ollama_service_name_for(host))
            api_base = f'http://{service}:{OLLAMA_CONTAINER_PORT}'
            for endpoint, payload in sorted(deployment.served.items()):
                routes.append(GatewayRoute(
                    endpoint, 'ollama', payload.get('model', endpoint), api_base,
                    origin='deployment',
                    route_id=_route_id(deployment.id, endpoint) if dynamic else None))
    return routes


class MissingRouteKey(ProfileMismatch):
    """A route sends a key variable the managed ``.env`` does not set.

    Raised by the render, before anything is approved or applied, whichever
    command published: a gateway recreated without the key would send an
    empty one (docs/planning/external-endpoints.md, decision 4).
    """

    def __init__(self, missing: dict[str, list[str]], env_path: Path):
        self.missing = dict(missing)
        name, aliases = next(iter(sorted(self.missing.items())))
        super().__init__(
            f'{", ".join(repr(a) for a in aliases)} sends ${name} as its upstream '
            f'key, which {env_path} does not set; set it first '
            f'(`infer-stack env {name}=...`) or unpublish the endpoint '
            '(`infer-stack routes prune`)')


def route_key_envs(routes: Sequence[GatewayRoute]) -> list[str]:
    """The variables ``routes`` send as upstream keys, sorted."""
    return sorted({r.key_env for r in routes if r.key_env})


def front_door_routes(
    deployments: list[Deployment], assignments: dict[str, list[int]], *,
    catalog: Any = None, dynamic: bool = False,
    registry: Sequence[GatewayRoute] = (),
    extra: Sequence[GatewayRoute] = (),
    extra_dynamic: Sequence[GatewayRoute] = (),
) -> tuple[list[GatewayRoute], list[GatewayRoute]]:
    """``(static route table, dynamic routes)`` of a front door: one is empty.

    Static: ``registry`` < the catalog's endpoints < placed deployments <
    ``extra`` (routes another backend supplies), one per alias. Dynamic: a
    route per (placed deployment, alias), per external endpoint, and each of
    ``extra_dynamic``, one per managed id.
    """
    if dynamic:
        by_id: dict[str | None, GatewayRoute] = {}
        for route in [*deployment_routes(
                          deployments, assignments, dynamic=True, catalog=catalog),
                      *catalog_routes(catalog, dynamic=True), *extra_dynamic]:
            by_id[route.route_id] = route
        return [], list(by_id.values())
    return route_table(registry, catalog_routes(catalog),
                       deployment_routes(deployments, assignments, catalog=catalog),
                       extra), []


# -- Route registry --------------------------------------------------------
#
# ``litellm_registry.json`` keeps the routes of deployments no published
# catalog defines, past their release: an ad-hoc acquire's alias stays
# routable and releasing it does not recreate the gateway. Catalog endpoints
# are not stored here; they are derived from the published union at every
# render. A registry from before that also holds catalog rows; they are read
# like the rest (lowest precedence, so a published definition wins), and
# ``routes prune`` drops what nothing serves. Rows are semantic inputs
# (served name / engine / host), never rendered entries.


def remembered_rows(
    deployments: list[Deployment], assignments: dict[str, list[int]], *,
    defined: set[str], extra: Sequence[GatewayRoute] = (),
) -> dict[str, dict[str, Any]]:
    """Registry rows for every placed deployment's alias, and every ``extra``
    route of a Model another backend runs, that ``defined`` (the published
    catalog aliases) does not cover.

    A vLLM row carries ``served`` (the host is re-derived from it) and the
    effective ``max_input_tokens`` its server ran with; an Ollama row the tag
    and ``host``; an upstream row its ``api_base`` (and the window when
    known): each renders back (:func:`registry_route`) to the route the
    deployment has now.
    """
    rows: dict[str, dict[str, Any]] = {}
    for deployment in deployments:
        if deployment.id not in assignments:
            continue
        if deployment.engine == 'vllm':
            served = served_name(deployment)
            for endpoint in sorted(deployment.served):
                # The per-alias contract travels with the row: a row outlives
                # its deployment, and a smaller alias may have been coalesced
                # onto a larger shared-compatible process.
                context = deployment_route_max_input_tokens(deployment, endpoint)
                rows[endpoint] = {'engine': 'vllm', 'served': served,
                                  'max_input_tokens': context}
        elif deployment.engine == 'ollama':
            host = deployment.spec.get('host') or deployment.id
            for endpoint, payload in sorted(deployment.served.items()):
                rows[endpoint] = {'engine': 'ollama',
                                  'model': payload.get('model', endpoint), 'host': host}
    for route in extra:
        if route.origin == 'upstream' and route.kind == 'openai':
            row: dict[str, Any] = {'engine': 'upstream', 'served': route.model,
                                   'api_base': route.api_base}
            if route.max_input_tokens is not None:
                row['max_input_tokens'] = route.max_input_tokens
            rows[route.alias] = row
    return {alias: row for alias, row in rows.items() if alias not in defined}


def _remembered_context(row: dict[str, Any]) -> int | None:
    """The context a registry row remembers: a positive whole number, or None.

    A row written by an older infer-stack carries no ``max_input_tokens`` and
    a hand-edited one may carry junk: both render as "window unknown" (the
    field is simply not advertised) rather than a wrong number.
    """
    value = row.get('max_input_tokens')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def registry_route(name: str, row: Any) -> GatewayRoute | None:
    """The route one registry row renders to, or ``None`` if it cannot."""
    if not isinstance(row, dict):
        return None
    engine = row.get('engine')
    if engine == 'vllm':
        served = row.get('served') or name
        return GatewayRoute(
            name, 'openai', served,
            f'http://{vllm_service_name_for(served)}:{VLLM_CONTAINER_PORT}/v1',
            origin='registry',
            max_input_tokens=_remembered_context(row))
    if engine == 'ollama':
        host = row.get('host') or name
        return GatewayRoute(
            name, 'ollama', row.get('model') or name,
            f'http://{ollama_service_name_for(host)}:{OLLAMA_CONTAINER_PORT}',
            origin='registry')
    if engine == 'upstream' and row.get('api_base'):
        # A server this project does not run (a KubeAI cluster's gateway).
        return GatewayRoute(name, 'openai', row.get('served') or name,
                            str(row['api_base']), origin='registry',
                            max_input_tokens=_remembered_context(row))
    return None


def registry_routes(registry: dict[str, Any] | None) -> list[GatewayRoute]:
    """The routes a registry's rows render to, sorted by alias."""
    rows = (registry or {}).get('entries') if isinstance(registry, dict) else None
    if not isinstance(rows, dict):
        return []
    return [r for r in (registry_route(name, rows[name]) for name in sorted(rows))
            if r is not None]


def upstream_route(deployment_id: str, endpoint: str, served: str,
                   api_base: str,
                   max_input_tokens: int | None = None) -> GatewayRoute:
    """A dynamic route to a server this project does not run (a KubeAI Model),
    with the same deterministic id as a Compose engine's dynamic route.

    ``max_input_tokens`` is the window that Model runs with (the effective
    ``max_model_len`` of its rendered spec), advertised like any other route;
    absent when the caller does not know it.
    """
    return GatewayRoute(endpoint, 'openai', served, api_base, origin='upstream',
                        route_id=_route_id(deployment_id, endpoint),
                        max_input_tokens=max_input_tokens)


def _dump_route_registry(registry: dict[str, Any]) -> str:
    """Canonical, byte-stable serialization: sorted keys + trailing newline."""
    return json.dumps(registry, sort_keys=True, indent=2) + '\n'


def _route_id(owner: str, endpoint: str) -> str:
    """Deterministic LiteLLM model id for one logical route.

    ``owner`` is the deployment id for a managed route (one alias can have
    several dedicated deployments), or ``'external'`` for an external
    target's. Stable across converges, so route reconcile
    (:meth:`Gateway._reconcile_routes`) identifies one logical route across
    renders: a route whose id disappears is deleted by exactly this id; one
    whose id remains but whose semantics drifted is replaced under it. The
    ``isr-`` prefix marks it infer-stack-managed so reconcile never deletes a
    model someone added by hand.
    """
    digest = hashlib.sha256(f'{owner}|{endpoint}'.encode()).hexdigest()
    return f'{ROUTE_ID_PREFIX}{digest[:32]}'


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
    host_port: int,
    images: dict[str, str],
    aux_dir: str,
    master_key: str | None = None,
    config_hash: str | None = None,
    *,
    dynamic_routing: bool = False,
    salt_key: bool = False,
    key_envs: Sequence[str] = (),
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
    for name in sorted(set(key_envs)):
        # An external route's key, by name: LiteLLM resolves `os.environ/NAME`
        # from its own environment, and Compose interpolates the value from the
        # managed .env, so the fingerprint (which hashes the values a stanza
        # references) recreates the gateway when the key changes.
        environment[name] = '${' + name + '}'
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
class RenderedFrontDoor:
    """The rendered front door: its Compose services and config files."""

    services: dict[str, Any]
    litellm_config: str | None
    nginx_config: str | None
    litellm_routes: list[dict[str, Any]] | None


def render_front_door(
    routes: list[GatewayRoute],
    *,
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
    dynamic_routing: bool,
    dynamic_routes: list[GatewayRoute] | None = None,
    ui_run_as: str | None = None,
) -> RenderedFrontDoor:
    """Render the gateway, its database, Open WebUI and the reverse proxy.

    ``routes`` is the static route table (one per alias,
    :func:`~infer_stack.leasing.routes.route_table`); with
    ``dynamic_routing`` the config has none and ``dynamic_routes`` are
    reconciled through the admin API instead. The engines are the caller's:
    it passes the in-network URLs a UI with no gateway can talk to directly.
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
        # Two route-table strategies:
        #  * DYNAMIC ROUTING: the rendered config is a STATIC base (empty
        #    model_list); the real routes live in Postgres and are applied to the
        #    running gateway via the admin API (see _reconcile_routes). The config
        #    hash never changes as models come/go, so the gateway is never
        #    recreated -- no blip, and per-deployment routing works (so same-model
        #    --dedicated deployments each get their own upstream).
        #  * STATIC: the model_list is the route table the caller derived from
        #    the published catalogs, the placed deployments and any legacy
        #    registry. It depends on endpoint definitions, not on which models
        #    are up, so the gateway is not recreated as models come and go.
        if dynamic_routing:
            entries: list[dict[str, Any]] = []
            litellm_routes = [r.entry() for r in dynamic_routes or []]
        else:
            entries = [r.entry() for r in routes]
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
            litellm_port,
            images,
            str(aux_dir or '.'),
            master_key=litellm_master_key,
            config_hash=config_hash,
            dynamic_routing=dynamic_routing,
            salt_key=litellm_salt_key,
            key_envs=route_key_envs([*routes, *(dynamic_routes or [])]),
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

    return RenderedFrontDoor(services=services, litellm_config=litellm_config,
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

    #: Secrets a preview generated and has not written (see staging_secrets).
    _staged: dict[str, str] | None = None
    _staging_depth: int = 0

    def staging_secrets(self):
        """While active, a missing managed secret is generated in memory, not
        written: a preview (admission, the idle-host feasibility check) leaves
        the ``.env`` as it was, even when the admission is then refused. The
        next writing call persists the same value, so the commit's render
        matches what the preview approved."""
        import contextlib

        @contextlib.contextmanager
        def staging():
            self._staging_depth += 1
            try:
                yield
            finally:
                self._staging_depth -= 1

        return staging()

    def managed_env(self) -> dict[str, str]:
        """The managed ``.env`` as a render sees it: the file, plus secrets a
        preview has staged and not yet written (so a preview's fingerprints
        match the commit's render)."""
        values = parse_env_file(self._env_path) if self._env_path.exists() else {}
        return {**(self._staged or {}), **values}

    def _managed_secret(self, name: str, *, prefix: str = '') -> str:
        """A secret from the managed ``.env``, made on first use (see
        :meth:`staging_secrets` for when "made" is not yet "written")."""
        existing = parse_env_file(self._env_path)
        if existing.get(name):
            return existing[name]
        if self._staged is None:
            self._staged = {}
        value = self._staged.get(name) or ensure_secret({}, name, prefix=prefix)
        if self._staging_depth:
            self._staged[name] = value
        else:
            write_env_file(self._env_path, {name: value})
            self._staged.pop(name, None)
        return value

    def master_key(self) -> str:
        """The managed LiteLLM master key.

        infer-stack manages this secret in the state dir's ``.env``: reused if
        already present (you may pin your own ``sk-`` key there), otherwise
        generated and persisted. The caller doesn't need to invent or export it
        — it is baked into the LiteLLM service, used by the readiness probe, and
        shipped in the env-file descriptor (``infer-stack env KEY`` prints it).
        """
        return self._managed_secret(API_KEY_ENV, prefix='sk-')

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
        return self._managed_secret(WEBUI_SECRET_ENV)

    def db_password(self) -> str:
        """The managed Postgres password for LiteLLM's model store.

        Same managed-secret pattern as :meth:`master_key`: reused if already
        present in the state-dir ``.env`` (you may pin your own), else generated
        and persisted. ``docker compose --env-file`` interpolates it into the
        postgres + litellm services, so it never appears literally in the YAML.
        Only used when ``dynamic_routing`` is on. ``token_urlsafe`` output is safe
        inside the ``postgresql://`` URL (no ``@ : /`` characters).
        """
        pw = self._managed_secret(DB_PASSWORD_ENV)
        return pw

    def _load_route_registry(self) -> dict[str, Any]:
        """Read the route registry, tolerantly (fail-open: a broken
        registry must never block a converge). Missing or structurally
        unusable -> empty. An unknown schema version whose ``entries`` parses
        is read as-is, rendering what is understood."""
        from .._log import logger

        empty = {'version': LITELLM_REGISTRY_VERSION, 'entries': {}}
        if not self._registry_file.exists():
            return empty
        try:
            data = json.loads(self._registry_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning('  route registry: {} is unreadable ({}); ignoring it',
                           self._registry_file.name, exc)
            return empty
        if not isinstance(data, dict) or not isinstance(data.get('entries'), dict):
            return empty
        if data.get('version') != LITELLM_REGISTRY_VERSION:
            logger.warning('  route registry: unknown schema version {!r} in {}; '
                           'rendering the rows it understands, never rewriting it',
                           data.get('version'), self._registry_file.name)
        return data

    def route_registry(self) -> dict[str, Any]:
        """The registry as stored: ``{'version': ..., 'entries': {...}}``."""
        registry = self._load_route_registry()
        return registry if isinstance(registry, dict) else {}

    def route_entries(self) -> dict[str, dict[str, Any]]:
        """The registry's entries: public alias -> route row."""
        entries = self.route_registry().get('entries')
        return dict(entries) if isinstance(entries, dict) else {}

    def missing_keys(self, names: Sequence[str]) -> list[str]:
        """Which of ``names`` the managed ``.env`` does not set (or sets empty)."""
        env = self.managed_env()
        return [n for n in names if not env.get(n)]

    @property
    def env_path(self) -> Path:
        """The managed ``.env`` (public spelling of ``_env_path``)."""
        return self._env_path

    def require_route_keys(self, routes: Sequence[GatewayRoute]) -> None:
        """Raise :class:`MissingRouteKey` if any route's key has no value."""
        missing = set(self.missing_keys(route_key_envs(routes)))
        if not missing:
            return
        users: dict[str, list[str]] = {}
        for route in routes:
            if route.key_env in missing:
                users.setdefault(route.key_env, []).append(route.alias)
        raise MissingRouteKey(users, self._env_path)

    def registry_routes(self) -> list[GatewayRoute]:
        """The registry's routes, the lowest-precedence route layer."""
        return registry_routes(self._load_route_registry())

    def remember(self, rows: dict[str, dict[str, Any]]) -> None:
        """Add or update ``rows`` in the registry. Called after an approved
        render, by a caller already holding the converge flock (taking it
        again would block on itself); nothing is ever removed here."""
        from .._log import logger

        if not rows:
            return
        existing = self._load_route_registry()
        if existing.get('version', LITELLM_REGISTRY_VERSION) != LITELLM_REGISTRY_VERSION:
            return                      # a newer binary's file: read, never rewrite
        entries = dict(existing.get('entries') or {})
        changed = sorted(k for k, v in rows.items() if entries.get(k) != v)
        if not changed:
            return
        entries.update(rows)
        logger.info('  route registry: remembered {}', ', '.join(changed))
        self._atomic_write(self._registry_file, _dump_route_registry(
            {'version': existing.get('version', LITELLM_REGISTRY_VERSION),
             'entries': entries}))

    def replace_route_entries(self, entries: dict[str, dict[str, Any]]) -> None:
        """Write the registry's entries (under the converge flock, atomically);
        its schema version is kept."""
        with self._converge_lock():
            existing = self._load_route_registry()
            version = (existing.get('version', LITELLM_REGISTRY_VERSION)
                       if isinstance(existing, dict) else LITELLM_REGISTRY_VERSION)
            self._atomic_write(self._registry_file, _dump_route_registry(
                {'version': version, 'entries': dict(entries)}))

    def urls(self) -> tuple[str | None, str | None]:
        """``(OpenAI base URL, Open WebUI URL)``; ``None`` for one that is off.

        The one derivation of where a client goes: the env-file descriptor,
        ``env``, ``test`` and the TUI all read it (see :func:`front_door_urls`).

        >>> import tempfile
        >>> gw = Gateway(tempfile.mkdtemp(), ports={'litellm': 14042, 'open_webui': 13000},
        ...              litellm=True, ui=True)
        >>> gw.urls()
        ('http://127.0.0.1:14042/v1', 'http://127.0.0.1:13000')
        >>> Gateway(tempfile.mkdtemp(), litellm=True, ui=False,
        ...         base_url='http://10.0.0.5:30442/').urls()
        ('http://10.0.0.5:30442/v1', None)
        """
        base = f'{self._gateway_base()}/v1' if self.litellm else None
        ui = f'http://127.0.0.1:{self.ui_port}' if self.ui else None
        return base, ui

    def _gateway_base(self) -> str:
        where = self.base_url
        if where is None:
            base = f'http://127.0.0.1:{self.litellm_port}'
        elif isinstance(where, str):
            base = where
        else:
            base = where()
        return base.rstrip('/')

    #: The key the RUNNING gateway holds, while it differs from the managed
    #: one (see :meth:`live_credential`); ``None``: the managed key.
    _live_key: str | None = None

    def live_credential(self, key: str | None):
        """While active, admin calls authenticate with ``key``.

        A rotation writes the new key to the ``.env`` before the gateway is
        recreated with it, and an apply's first phase (retiring routes) talks
        to the gateway still running on the old one. ``None`` (the running
        key is unknown) keeps the managed key.
        """
        import contextlib

        @contextlib.contextmanager
        def scope():
            previous, self._live_key = self._live_key, key
            try:
                yield
            finally:
                self._live_key = previous

        return scope()

    def _auth_headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self._live_key or self.master_key()}'}

    def _desired_routes(self) -> list[dict[str, Any]]:
        """The rendered desired route set (litellm_routes.json), or empty."""
        try:
            data = json.loads(self._routes_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _reconcile_routes(
        self, *, deadline_s: float = ROUTE_RECONCILE_BOOTSTRAP_S, delay: float = 2.0,
        retire_only: bool = False,
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

        ``retire_only`` is the first phase of an apply that tears upstreams
        down: delete only the managed routes the render no longer has (no
        adds, no replacements), and verify they are gone. It runs before the
        runtime removes anything, so a route is never left pointing at an
        upstream this apply stopped.

        A replacement (same id, different semantics) is a delete then an add,
        since ``/model/new`` does not update on every LiteLLM release. If the
        add fails after the delete, the route is missing until the next apply:
        that gap is logged by name, and the result is ``False``.
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
            mismatched = [] if retire_only else sorted(
                rid for rid in desired.keys() & current.keys()
                if not self._route_semantics_match(
                    desired_semantics[rid], current[rid])
            )
            to_add_ids = [] if retire_only else sorted(
                (desired.keys() - current.keys()) | set(mismatched))
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
                added = self._post_route(
                    '/model/new', route, route.get('model_name'), deadline=deadline,
                )
                rid = route['model_info']['id']
                if not added and rid in mismatched:
                    logger.warning(
                        'dynamic routing: route {} ({}) was removed for replacement '
                        'and not re-added; it is missing until the next apply',
                        rid, route.get('model_name'))
                ok &= added
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
        reliably observe; so is the name of an external route's key variable,
        kept in ``model_info`` because LiteLLM redacts the key itself.  So is the
        advertised context window when infer-stack supplied one: a changed
        managed window is a different route and is replaced under its id.
        LiteLLM may synthesize a window for routes where infer-stack supplied
        none; that extra metadata is deliberately ignored. A matching managed
        id with different infer-stack-owned values here is drift and is
        replaced, not accepted as healthy.
        """
        params = route.get('litellm_params') or {}
        info = route.get('model_info') or {}
        return {
            'model_name': route.get('model_name'),
            'model': params.get('model'),
            'api_base': params.get('api_base'),
            'key_env': info.get('infer_stack_key_env'),
            'max_input_tokens': info.get('max_input_tokens'),
        }

    @staticmethod
    def _route_semantics_match(
        desired: dict[str, Any], current: dict[str, Any],
    ) -> bool:
        """Whether observable LiteLLM route semantics satisfy ``desired``.

        LiteLLM can populate ``model_info.max_input_tokens`` from its bundled
        model metadata even when infer-stack did not set that field (notably for
        external routes with published model names).  Such a synthesized value
        is not infer-stack-owned state and must not trigger an endless
        delete/re-add loop.  When infer-stack *does* advertise a context window,
        it remains strict and any difference is semantic drift.
        """
        if desired.get('max_input_tokens') is None:
            current = dict(current)
            current['max_input_tokens'] = None
        return desired == current

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

    def connection_info(self) -> ConnectionInfo | None:
        """Where a client reaches the front door.

        With LiteLLM, one ``base_url``, the master key and its variable's
        name. With LiteLLM off there is no single base URL, but a managed Open
        WebUI (if on) is still a useful access point, so report just its URL
        rather than ``None``.
        """
        base_url, ui_url = self.urls()
        proxy_url = (f'http://127.0.0.1:{self.reverse_proxy_port}'
                     if self.reverse_proxy and base_url is not None else None)
        if base_url is None:
            return ConnectionInfo(None, ui_url=ui_url) if ui_url else None
        return ConnectionInfo(base_url, api_key_env=API_KEY_ENV,
                              api_key=self.master_key(), ui_url=ui_url,
                              proxy_url=proxy_url)


def front_door_urls(backend) -> tuple[str | None, str | None]:
    """``(OpenAI base URL, Open WebUI URL)`` of ``backend``'s front door.

    Whatever holds the gateway answers: the compose project itself, or on
    kubeai the gateway on this host or in the cluster. ``(None, None)`` for a
    backend with no gateway, or when asking fails (the cluster's node address
    is read with kubectl): a URL lookup must not fail its caller.

    >>> from infer_stack.leasing import NullBackend
    >>> front_door_urls(NullBackend())
    (None, None)
    """
    try:
        front = getattr(backend, 'front_door', lambda: None)()
        gateway = getattr(front, 'gateway', None)
        return gateway.urls() if gateway is not None else (None, None)
    except Exception:  # noqa: BLE001 - see the docstring
        return None, None
