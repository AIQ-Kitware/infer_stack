"""Focused Compose backend for the leasing model.

Renders a docker-compose project straight from the live set of
:class:`Deployment` s — not the legacy resolved-deployment schema — using
the placement planner for GPU assignment and reusing ``profile_runtime.vllm_args``
for the vLLM CLI flags. It *converges the whole union* on every reconcile:
render the file, then ``docker compose up -d --remove-orphans``. Adding or
removing a deployment re-renders and converges; pinned placement (persisted in a
sidecar) keeps already-running models on their GPUs, and ``--remove-orphans``
tears down services whose deployment is gone.

A **LiteLLM front door** (default on) gives one stable ``base_url`` and routes
each endpoint *alias* to its upstream vLLM/Ollama service, so a client always
talks to ``http://host:<litellm>/v1`` and asks for the public endpoint name.
That is what makes the endpoint descriptor's ``base_url`` correct (the backend
supplies it via :meth:`ComposeBackend.access`).

Docker and HTTP are invoked through injected seams (``run`` / ``http_get``), so
all logic here is unit-testable without docker or a network. The real
docker/GPU path is validated on a GPU host. ``converge`` is serialized with a
file lock so concurrent processes don't clobber the shared compose file.

Slice status: readiness probes the LiteLLM ``/v1/models`` listing (model is
routable). The Ollama tag pull/warmup rung is a follow-up.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config import DEFAULT_PORTS, PINNED_IMAGES, default_state_paths
from ..env_utils import ensure_secret, parse_env_file, write_env_file
from ..probe import openai_ready
from ..profile_runtime import vllm_args
from .backend import Readiness
from .models import Deployment
from .placement import plan_placement

LEASING_PROJECT = 'infer-stack'  # docker compose project name for leased stacks
VLLM_HOST_PORT_BASE = 18000
VLLM_CONTAINER_PORT = 8000
OLLAMA_CONTAINER_PORT = 11434
LITELLM_CONTAINER_PORT = 4000
STATE_FILENAME = 'leasing-compose-state.json'
COMPOSE_FILENAME = 'docker-compose.yml'
LITELLM_CONFIG_FILENAME = 'litellm_config.yaml'
LOCK_FILENAME = '.converge.lock'
LITELLM_SERVICE = 'litellm'
API_KEY_ENV = 'LITELLM_MASTER_KEY'

VLLM_DEFAULTS = {
    'gpu_memory_utilization': 0.9,
    'max_model_len': 8192,
    'max_num_batched_tokens': 8192,
    'max_num_seqs': 256,
}

DEPLOYMENT_LABEL = 'infer-stack.deployment'
ENGINE_LABEL = 'infer-stack.engine'


def _dns_slug(text: str) -> str:
    """A lowercase ``[a-z0-9-]`` label safe as a compose service/DNS name."""
    out = re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')
    return out or 'model'


def vllm_service_name_for(served: str) -> str:
    """Deterministic compose/DNS service name for a vLLM upstream: ``vllm-<served>``.

    Derived purely from the served model name (vLLM's ``--served-model-name``,
    the Open WebUI label, the alias the user chose), so it is identical whether
    computed from a live :class:`Deployment` or from a catalog endpoint. That
    stability is what lets the LiteLLM gateway carry a *static* route table (one
    per catalog endpoint) whose upstream hosts match the containers when they
    come up — so adding/removing models does not rewrite the gateway's config and
    the gateway is never recreated (no "blip"); see :func:`_litellm_model_list`.
    """
    return f'vllm-{_dns_slug(served)}'


def vllm_service_name(deployment: Deployment) -> str:
    """Compose service name for a vLLM deployment (see :func:`vllm_service_name_for`).

    Deterministic from the served model name only — *no* deployment-id suffix —
    so it matches the gateway's pre-rendered route for that endpoint. Trade-off:
    two *simultaneously desired* deployments that share a served name (an endpoint
    re-pointed at a new model while the old one is still live) would collide on
    this name; that interactive case is unsupported under the static-gateway
    model (the catalog endpoint is the unit). ``observe`` correlates a running
    container back to its deployment id via the ``infer-stack.deployment`` label,
    not this name, so the dropped suffix does not affect reconcile bookkeeping.
    """
    served = deployment.spec.get('served_model_name') or (
        sorted(deployment.served)[0] if deployment.served else deployment.id
    )
    return vllm_service_name_for(served)


def ollama_service_name_for(host: str) -> str:
    """Deterministic service name for an Ollama daemon: ``ollama-<host>``.

    One daemon per host (Ollama coalesces tags onto it), so the host is the
    stable key — matching :func:`vllm_service_name_for`'s role for vLLM so the
    gateway's static route table addresses it regardless of which tags are live.
    """
    return f'ollama-{_dns_slug(host)}'


def ollama_service_name(deployment: Deployment) -> str:
    host = deployment.spec.get('host') or deployment.id
    return ollama_service_name_for(host)


@dataclass
class RenderedCompose:
    compose: dict[str, Any]
    services: dict[str, str] = field(default_factory=dict)  # service -> deployment id
    litellm_config: str | None = None
    nginx_config: str | None = None


def _gpu_reservation(indices: list[int]) -> dict[str, Any]:
    return {
        'resources': {
            'reservations': {
                'devices': [
                    {
                        'driver': 'nvidia',
                        'device_ids': [str(i) for i in indices],
                        'capabilities': ['gpu'],
                    }
                ]
            }
        }
    }


def _vllm_service_dict(deployment: Deployment) -> dict[str, Any]:
    """Build the dict ``vllm_args`` consumes from a deployment's runtime spec."""
    runtime = deployment.spec.get('runtime', {}) or {}
    served = deployment.spec.get('served_model_name') or (
        sorted(deployment.served)[0] if deployment.served else deployment.id
    )
    return {
        'served_model_name': served,
        'tensor_parallel_size': int(runtime.get('tensor_parallel_size', 1) or 1),
        'data_parallel_size': int(runtime.get('data_parallel_size', 1) or 1),
        'max_model_len': runtime.get('max_model_len', VLLM_DEFAULTS['max_model_len']),
        'gpu_memory_utilization': runtime.get(
            'gpu_memory_utilization', VLLM_DEFAULTS['gpu_memory_utilization']
        ),
        'max_num_batched_tokens': runtime.get(
            'max_num_batched_tokens', VLLM_DEFAULTS['max_num_batched_tokens']
        ),
        'max_num_seqs': runtime.get(
            'max_num_seqs', VLLM_DEFAULTS['max_num_seqs']
        ),
        'enable_prefix_caching': bool(runtime.get('enable_prefix_caching', False)),
        'extra_args': list(runtime.get('extra_args', []) or []),
    }


def _vllm_service(
    deployment: Deployment,
    gpus: list[int],
    host_port: int,
    images: dict[str, str],
    state: dict[str, str],
) -> dict[str, Any]:
    svc = _vllm_service_dict(deployment)
    command = [
        deployment.spec['hf_model_id'],
        '--host',
        '0.0.0.0',
        '--port',
        '8000',
        *vllm_args(svc),
    ]
    service: dict[str, Any] = {
        'image': images['vllm'],
        'command': command,
        'ports': [f'{host_port}:8000'],
        'environment': {'HF_TOKEN': '${HF_TOKEN:-}'},
        'volumes': [f'{state["hf_cache"]}:/root/.cache/huggingface'],
        'restart': 'unless-stopped',
        'labels': {DEPLOYMENT_LABEL: deployment.id, ENGINE_LABEL: 'vllm'},
        'healthcheck': {
            'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
            'interval': '30s',
            'timeout': '10s',
            'retries': 5,
            'start_period': '1800s',
        },
    }
    if gpus:
        service['deploy'] = _gpu_reservation(gpus)
    return service


def _ollama_service(
    deployment: Deployment,
    gpus: list[int],
    host_port: int,
    images: dict[str, str],
    state: dict[str, str],
) -> dict[str, Any]:
    settings = deployment.spec.get('settings', {}) or {}
    env: dict[str, str] = {}
    if settings.get('keep_alive'):
        env['OLLAMA_KEEP_ALIVE'] = str(settings['keep_alive'])
    if settings.get('num_parallel') is not None:
        env['OLLAMA_NUM_PARALLEL'] = str(settings['num_parallel'])
    if settings.get('max_loaded_models') is not None:
        env['OLLAMA_MAX_LOADED_MODELS'] = str(settings['max_loaded_models'])
    if settings.get('context_length') is not None:
        env['OLLAMA_CONTEXT_LENGTH'] = str(settings['context_length'])
    # GPU pinning is done by the device reservation below (``device_ids``), which
    # exposes *only* those physical GPUs to the container — and the NVIDIA
    # runtime renumbers them to 0..n-1 inside it. So we must NOT also set
    # ``CUDA_VISIBLE_DEVICES`` to the host indices: pinning to host GPU 1 would
    # leave the container seeing one GPU as device 0 while CUDA_VISIBLE_DEVICES=1
    # points at nothing, and ollama silently falls back to CPU. vLLM relies on
    # the reservation alone; ollama does the same.
    service: dict[str, Any] = {
        'image': deployment.spec.get('image') or images['ollama'],
        'ports': [f'{host_port}:11434'],
        'environment': env,
        'volumes': [f'{state["ollama"]}:/root/.ollama'],
        'restart': 'unless-stopped',
        'labels': {DEPLOYMENT_LABEL: deployment.id, ENGINE_LABEL: 'ollama'},
        'healthcheck': {
            'test': ['CMD', 'ollama', 'list'],
            'interval': '30s',
            'timeout': '10s',
            'retries': 5,
        },
    }
    if gpus:
        service['deploy'] = _gpu_reservation(gpus)
    return service


def _litellm_model_list(
    deployments: list[Deployment], assignments: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """One LiteLLM ``model_list`` entry per served endpoint alias."""
    entries: list[dict[str, Any]] = []
    for deployment in sorted(deployments, key=lambda g: (g.created_at, g.id)):
        if deployment.id not in assignments:
            continue
        if deployment.engine == 'vllm':
            served = deployment.spec.get('served_model_name') or deployment.id
            api_base = f'http://{vllm_service_name(deployment)}:8000/v1'
            for endpoint in sorted(deployment.served):
                entries.append(
                    {
                        'model_name': endpoint,
                        'litellm_params': {
                            'model': f'openai/{served}',
                            'api_base': api_base,
                            'api_key': 'EMPTY',
                        },
                    }
                )
        elif deployment.engine == 'ollama':
            api_base = f'http://{ollama_service_name(deployment)}:{OLLAMA_CONTAINER_PORT}'
            for endpoint, payload in sorted(deployment.served.items()):
                tag = payload.get('model', endpoint)
                entries.append(
                    {
                        'model_name': endpoint,
                        'litellm_params': {
                            'model': f'ollama/{tag}',
                            'api_base': api_base,
                        },
                    }
                )
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

    Direction (see ``docs/litellm-gateway-routing.md``): static superset is the
    default and is fine for now, but the intended evolution is to add/remove
    routes at runtime via LiteLLM's admin API (``/model/new`` / ``/model/delete``)
    so that non-catalog / interactive acquires can route with zero blip too. Add
    that as an opt-in reconcile, not a replacement for this static path.
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
            entries.append(
                {
                    'model_name': name,
                    'litellm_params': {
                        'model': f'openai/{served}',
                        'api_base': api_base,
                        'api_key': 'EMPTY',
                    },
                }
            )
        elif req.engine == 'ollama':
            host = req.spec.get('host') or req.host
            tag = req.served.get('model') or name
            api_base = (
                f'http://{ollama_service_name_for(host)}:{OLLAMA_CONTAINER_PORT}'
            )
            entries.append(
                {
                    'model_name': name,
                    'litellm_params': {
                        'model': f'ollama/{tag}',
                        'api_base': api_base,
                    },
                }
            )
    return entries


CONFIG_HASH_LABEL = 'infer-stack.config-hash'


def _litellm_service(
    service_names: list[str],
    host_port: int,
    images: dict[str, str],
    aux_dir: str,
    master_key: str | None = None,
    config_hash: str | None = None,
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
        'environment': {API_KEY_ENV: key_value},
        'restart': 'unless-stopped',
        'labels': labels,
    }
    # Only wait on upstreams when there are any (zero models -> empty gateway).
    if service_names:
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
) -> dict[str, Any]:
    """A managed Open WebUI pointed at whatever front door is available.

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
    }
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
    if depends_on:
        service['depends_on'] = sorted(depends_on)
    return service


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


def render_compose(
    deployments: list[Deployment],
    assignments: dict[str, list[int]],
    *,
    images: dict[str, str],
    ports: dict[str, int],
    state: dict[str, str],
    litellm: bool = False,
    litellm_port: int = 14042,
    litellm_master_key: str | None = None,
    ui: bool = False,
    ui_port: int = 13000,
    reverse_proxy: bool = False,
    reverse_proxy_port: int = 80,
    reverse_proxy_config: str | None = None,
    aux_dir: str | Path | None = None,
    project: str = LEASING_PROJECT,
    catalog: Any = None,
) -> RenderedCompose:
    """Render a compose project for the placed deployments.

    Deployments absent from ``assignments`` (placement failures) are skipped. When
    ``litellm`` is set, a front-door service + config is added so every endpoint
    alias is reachable at one ``base_url``. When ``ui`` is also set, a managed
    Open WebUI is rendered in front of that gateway.

    The project name is baked into the file as a top-level ``name:`` so a plain
    ``docker compose -f docker-compose.yml up`` (infer-stack not involved) lands
    in the *same* project — same container names, same network — as
    ``infer-stack``'s own ``-p`` invocations. That makes "drop the tool and run
    docker yourself" a true equivalent of ``apply`` rather than a sibling
    project the tool can no longer see.
    """
    services: dict[str, Any] = {}
    service_map: dict[str, str] = {}
    # In-network upstreams Open WebUI can connect to directly when there is no
    # LiteLLM gateway (or, for Ollama, *in addition* to it — see below).
    vllm_v1_urls: list[str] = []      # OpenAI /v1 of each vLLM process
    ollama_native_urls: list[str] = []  # native Ollama API of each daemon
    ordered = sorted(deployments, key=lambda g: (g.created_at, g.id))
    vllm_i = 0
    ollama_i = 0
    for deployment in ordered:
        if deployment.id not in assignments:
            continue
        gpus = assignments[deployment.id]
        if deployment.engine == 'vllm':
            name = vllm_service_name(deployment)
            port = VLLM_HOST_PORT_BASE + vllm_i
            vllm_i += 1
            services[name] = _vllm_service(deployment, gpus, port, images, state)
            vllm_v1_urls.append(f'http://{name}:{VLLM_CONTAINER_PORT}/v1')
        elif deployment.engine == 'ollama':
            name = ollama_service_name(deployment)
            port = ports.get('ollama', DEFAULT_PORTS['ollama']) + ollama_i
            ollama_i += 1
            services[name] = _ollama_service(deployment, gpus, port, images, state)
            ollama_native_urls.append(f'http://{name}:{OLLAMA_CONTAINER_PORT}')
        else:
            continue
        service_map[name] = deployment.id

    litellm_config = None
    # The front door (gateway + UI) is rendered whenever it's enabled, even with
    # zero models — it's a standing entry point, not a per-model service. So
    # releasing/evicting every model leaves an empty gateway (and an empty Open
    # WebUI picker) up instead of tearing the whole stack down; only an explicit
    # `stack down` removes it. With no models the model_list is simply empty.
    if litellm:
        # Prefer a STATIC superset route table from the catalog: one route per
        # catalog endpoint, addressing a deterministic upstream host. That config
        # depends only on the catalog, not on which models are placed — so
        # acquiring/releasing a model leaves the LiteLLM config (hence its
        # container) untouched and the gateway is not recreated ("no blip"). The
        # config_hash still changes if the *catalog* changes, correctly
        # recreating the gateway to pick up new/removed endpoints. Without a
        # catalog (legacy / tests) fall back to routing only the placed
        # deployments, which does churn per model change.
        if catalog is not None:
            entries = _litellm_model_list_from_catalog(catalog)
            litellm_depends: list[str] = []  # no per-model depends_on -> no churn
        else:
            entries = _litellm_model_list(deployments, assignments)
            litellm_depends = list(service_map)
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
        services[LITELLM_SERVICE] = _litellm_service(
            litellm_depends,
            litellm_port,
            images,
            str(aux_dir or '.'),
            master_key=litellm_master_key,
            config_hash=config_hash,
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

    return RenderedCompose(
        compose={'name': project, 'services': services},
        services=service_map,
        litellm_config=litellm_config,
        nginx_config=nginx_config,
    )


def _default_docker_run(args: list[str]) -> str:
    import subprocess

    return subprocess.check_output(args, text=True)


def _parse_ps(out: str) -> set[str]:
    """Parse running service names from ``docker compose ps --format json``.

    Handles both a JSON array and newline-delimited JSON objects.
    """
    out = (out or '').strip()
    if not out:
        return set()
    rows: list[dict[str, Any]] = []
    try:
        data = json.loads(out)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    running: set[str] = set()
    for row in rows:
        if str(row.get('State', '')).lower().startswith('running'):
            name = row.get('Service') or row.get('Name')
            if name:
                running.add(str(name))
    return running


class ComposeBackend:
    """Single-host docker compose backend (converge-style).

    Driven by the controller's ``converge`` path. ``run`` / ``http_get`` are the
    injected docker/HTTP seams; defaults shell out to ``docker compose`` and
    ``requests``.
    """

    def __init__(
        self,
        *,
        state_dir: str | Path,
        inventory: dict[str, Any],
        run: Callable[[list[str]], str] | None = None,
        http: Any = None,
        images: dict[str, str] | None = None,
        ports: dict[str, int] | None = None,
        state: dict[str, str] | None = None,
        allowed_gpus: list[int] | None = None,
        reserved: list[int] | tuple[int, ...] = (),
        project: str = LEASING_PROJECT,
        skip_display: bool = False,
        litellm: bool = True,
        ui: bool = True,
        reverse_proxy: bool = False,
        reverse_proxy_port: int = 80,
        reverse_proxy_config: str | None = None,
        require_generation: bool = False,
        assume_yes: bool = True,
        catalog: Any = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.inventory = inventory
        self.run = run or _default_docker_run
        if http is None:
            import requests
            http = requests
        self.http = http
        self.images = {**PINNED_IMAGES, **(images or {})}
        self.ports = {**DEFAULT_PORTS, **(ports or {})}
        self.state = state or default_state_paths()
        self.allowed_gpus = allowed_gpus
        self.reserved = tuple(reserved)
        self.project = project
        self.skip_display = skip_display
        self.litellm = litellm
        self.ui = ui
        self.reverse_proxy = reverse_proxy
        self.reverse_proxy_port = reverse_proxy_port
        self.reverse_proxy_config = reverse_proxy_config
        # Retained for API/CLI compatibility but no longer consulted: probe_ready
        # always verifies a real generation now (the only trustworthy readiness).
        self.require_generation = require_generation
        self.assume_yes = assume_yes
        # Optional catalog: when present, the LiteLLM gateway is rendered with a
        # static superset route table (one route per catalog endpoint) so the
        # gateway is never recreated as models come and go. See render_compose.
        self.catalog = catalog
        self.last_errors: list[str] = []
        self.last_unplaced: set[str] = set()  # desired deployment ids placement skipped
        self.last_assignments: dict[str, list[int]] = {}  # deployment id -> GPU ids
        self._pulled: set[str] = set()  # (deployment:tag) pulled this process

    @property
    def compose_file(self) -> Path:
        return self.state_dir / COMPOSE_FILENAME

    @property
    def _state_file(self) -> Path:
        return self.state_dir / STATE_FILENAME

    @property
    def litellm_port(self) -> int:
        return self.ports.get('litellm', DEFAULT_PORTS['litellm'])

    @property
    def ui_port(self) -> int:
        return self.ports.get('open_webui', DEFAULT_PORTS['open_webui'])

    @property
    def _env_path(self) -> Path:
        return self.state_dir / '.env'

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

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write ``text`` to ``path`` atomically (temp + ``os.replace``).

        The render half (``converge(apply=False)``) and the apply half
        (:meth:`apply`) run under *different* locks so a render can proceed while
        another process applies. A reader (``apply`` / ``docker compose``) must
        therefore never see a half-written file — ``os.replace`` swaps it in one
        atomic step, so every read sees either the old or the new file whole.
        """
        tmp = path.with_name(f'{path.name}.tmp')
        tmp.write_text(text)
        os.replace(tmp, path)

    def _load_sidecar(self) -> dict[str, Any]:
        if self._state_file.exists():
            return json.loads(self._state_file.read_text())
        return {}

    def _save_sidecar(self, data: dict[str, Any]) -> None:
        self._atomic_write(self._state_file, json.dumps(data, indent=2))

    def _compose(self, args: list[str]) -> str:
        cmd = ['docker', 'compose']
        # Resolve ${LITELLM_MASTER_KEY} (and any other managed secret) from the
        # sidecar .env beside the compose file, so secrets stay out of the YAML.
        # docker compose's default .env discovery keys off the *current working
        # directory* (wherever infer-stack was invoked), not the state dir, so we
        # point it explicitly. Only when present: a litellm-less stack never
        # writes one, and a missing --env-file path is a hard error.
        if self._env_path.exists():
            cmd += ['--env-file', str(self._env_path)]
        cmd += ['-p', self.project, '-f', str(self.compose_file)]
        return self.run([*cmd, *args])

    @contextlib.contextmanager
    def _converge_lock(self):
        """Serialize converge across processes sharing this state dir."""
        handle = open(self.state_dir / LOCK_FILENAME, 'w')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def _approve_changes(self, planned: dict[Path, str]) -> None:
        """Show pending compose/litellm changes and confirm them.

        ``planned`` maps target paths to their new content. When nothing
        actually changed, this is a quiet no-op. When ``assume_yes`` (scripts /
        non-interactive / ``--yes``), it applies after a one-line log. Otherwise
        it renders a per-file diff and prompts; a decline raises
        :class:`ConvergeAborted` so the caller can roll back.
        """
        from .._log import logger
        from .backend import ConvergeAborted

        changed = {
            p: text
            for p, text in planned.items()
            if (p.read_text() if p.exists() else '') != text
        }
        if not changed:
            logger.debug('compose project already up to date')
            return
        names = ', '.join(p.name for p in changed)
        if self.assume_yes:
            logger.info('Updating compose project ({})', names)
            return
        from ..diff_prompt import confirm_writes

        if not confirm_writes(
            changed,
            assume_yes=False,
            title='infer-stack will update the compose project',
        ):
            raise ConvergeAborted('compose changes were not approved')

    def plan(self, desired: list[Deployment]):
        """Compute GPU placement for ``desired`` without writing or applying.

        Read-only and side-effect free: it honors the persisted pins, so the
        result reflects where deployments are (for running ones) or *would* be (for
        not-yet-started ones) placed. ``leases`` uses it to show actual/slated
        GPUs; ``converge`` uses it as the first step of render.
        """
        pinned = self._load_sidecar().get('assignments', {})
        return plan_placement(
            list(desired),
            self.inventory,
            allowed_gpus=self.allowed_gpus,
            reserved=self.reserved,
            pinned=pinned,
            skip_display=self.skip_display,
        )

    def converge(self, desired: list[Deployment], *, apply: bool = True):
        """Place + render the desired union, then optionally apply it.

        The work splits into *render* (decide placement, write the
        ``docker-compose.yml`` / LiteLLM config / placement sidecar to the state
        dir) and *apply* (``docker compose up -d`` / ``down``). With
        ``apply=False`` only the render half runs, so the on-disk project shows
        exactly what *would* execute — inspect it, or bring it up yourself
        (``infer-stack apply``). Either way the placement plan is returned and
        ``last_errors`` / ``last_unplaced`` / ``last_assignments`` are updated.
        """
        from .._log import logger

        desired = list(desired)
        with self._converge_lock():
            logger.info(
                'Converging {} deployment(s): {}',
                len(desired),
                ', '.join(sorted(g.id for g in desired)) or '(none)',
            )
            plan = self.plan(desired)
            self.last_errors = plan.errors
            self.last_unplaced = {
                g.id for g in desired if g.id not in plan.assignments
            }
            self.last_assignments = dict(plan.assignments)
            for gid, gpus in sorted(plan.assignments.items()):
                logger.info('  placed {} on GPU(s) {}', gid, gpus or '(cpu)')
            for err in plan.errors:
                logger.warning('  placement: {}', err)
            rendered = render_compose(
                desired,
                plan.assignments,
                images=self.images,
                ports=self.ports,
                state=self.state,
                litellm=self.litellm,
                litellm_port=self.litellm_port,
                litellm_master_key=self.master_key() if self.litellm else None,
                ui=self.ui,
                ui_port=self.ui_port,
                reverse_proxy=self.reverse_proxy,
                reverse_proxy_port=self.reverse_proxy_port,
                reverse_proxy_config=self.reverse_proxy_config,
                aux_dir=self.state_dir,
                project=self.project,
                catalog=self.catalog,
            )

            compose_text = yaml.safe_dump(rendered.compose, sort_keys=False)
            planned: dict[Path, str] = {self.compose_file: compose_text}
            if rendered.litellm_config is not None:
                planned[self.state_dir / LITELLM_CONFIG_FILENAME] = (
                    rendered.litellm_config
                )
            if rendered.nginx_config is not None:
                planned[self.state_dir / NGINX_CONFIG_FILENAME] = (
                    rendered.nginx_config
                )
            self._approve_changes(planned)  # may raise ConvergeAborted

            if rendered.litellm_config is not None:
                self._atomic_write(
                    self.state_dir / LITELLM_CONFIG_FILENAME,
                    rendered.litellm_config,
                )
            if rendered.nginx_config is not None:
                self._atomic_write(
                    self.state_dir / NGINX_CONFIG_FILENAME, rendered.nginx_config
                )
            self._atomic_write(self.compose_file, compose_text)
            self._save_sidecar(
                {'assignments': plan.assignments, 'services': rendered.services}
            )
            services = rendered.compose.get('services')
            if not apply:
                logger.info(
                    'rendered {} service(s) to {} (not applied; '
                    '`infer-stack apply` to bring it up)',
                    len(services or {}), self.compose_file,
                )
                return plan
        # Apply OUTSIDE the converge (render) lock: the controller coalesces and
        # serializes applies via its own apply-lock, so re-taking the render lock
        # here would needlessly serialize renders against this slow `up`.
        self.apply()
        return plan

    def apply(self) -> None:
        """Bring the already-rendered compose project up (``docker compose up -d``).

        Reads the on-disk compose file last written by :meth:`converge` (render)
        and applies it — it does NOT re-render. Deliberately does **not** take the
        converge (render) lock: the controller serializes and coalesces applies
        via its apply-lock + the ledger generation, and taking the render lock
        here would re-serialize renders against this slow step (the whole point of
        the split). Idempotent — a no-op when reality already matches the file,
        which is what makes coalescing safe (a redundant apply costs ~nothing).
        """
        from .._log import logger

        if not self.compose_file.exists():
            return
        try:
            doc = yaml.safe_load(self.compose_file.read_text()) or {}
        except Exception:  # noqa: BLE001 - a torn/old file must not brick apply
            return
        services = doc.get('services') or {}
        if services:
            logger.info(
                'docker compose up -d ({} service(s): {})',
                len(services), ', '.join(sorted(services)),
            )
            self._compose(['up', '-d', '--remove-orphans'])
        else:
            # Nothing at all to run — only reachable with the gateway off
            # (litellm=False) and zero models, since the front door otherwise
            # keeps the project non-empty. `docker compose up` errors with "no
            # service selected" on a services-less file, so tear the project down
            # instead (`down` works on the empty file). With the gateway on,
            # releasing every model lands in the `up` branch above and leaves the
            # front door standing; `stack down` is the way to take everything off.
            logger.info('no services desired -> docker compose down')
            self._compose(['down', '--remove-orphans'])

    def observe(self) -> set[str]:
        if not self.compose_file.exists():
            return set()
        try:
            out = self._compose(['ps', '--format', 'json'])
        except Exception:  # noqa: BLE001 - observe is best-effort
            # A stale/invalid compose file on disk (e.g. left by an older
            # version) or a transient docker error must not brick acquire:
            # `docker compose ps` validates the file, so a bad file would raise
            # here *before* converge gets to overwrite it. Treat as "nothing
            # observed" and let converge rewrite + reconcile.
            return set()
        running = _parse_ps(out)
        services = self._load_sidecar().get('services', {})
        return {services[name] for name in running if name in services}

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
            'base_url': f'http://127.0.0.1:{self.litellm_port}/v1',
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

    def _ensure_ollama_tag(
        self, deployment: Deployment, endpoint: str
    ) -> str | None:
        """Pull the endpoint's Ollama tag into its daemon (idempotent).

        An Ollama daemon loads tags lazily, so a tag must be present before it
        can serve. Returns an error reason if the pull failed (retry next poll),
        else ``None``.
        """
        tag = (deployment.served.get(endpoint) or {}).get('model')
        if not tag:
            return None
        key = f'{deployment.id}:{tag}'
        if key in self._pulled:
            return None
        try:
            # Exec into the daemon by the SAME name it is rendered/observed under
            # (ollama-<host>), not ollama-<deployment.id> — a host runs one daemon
            # that coalesces tags, so the service is keyed by host. Using the
            # deployment id targets a non-existent service ("is not running") and
            # the tag is never pulled, so --require-generation times out.
            service = ollama_service_name(deployment)
            self._compose(['exec', '-T', service, 'ollama', 'pull', tag])
        except Exception as ex:  # noqa: BLE001 - readiness is retryable
            return f'pulling {tag}: {ex}'
        self._pulled.add(key)
        return None

    def _published_v1_url(self, deployment: Deployment) -> str | None:
        """Host-reachable ``/v1`` of a deployment, from its published port.

        Used to probe a vLLM upstream directly when there is no gateway. Reads
        the rendered compose so it matches whatever port was actually published.
        """
        try:
            doc = yaml.safe_load(self.compose_file.read_text()) or {}
        except FileNotFoundError:
            return None
        for svc in (doc.get('services') or {}).values():
            if (svc.get('labels') or {}).get(DEPLOYMENT_LABEL) != deployment.id:
                continue
            for mapping in svc.get('ports') or []:
                host = str(mapping).split(':')[0]
                if host:
                    return f'http://127.0.0.1:{host}/v1'
        return None

    def probe_ready(
        self, deployment: Deployment, endpoint: str
    ) -> Readiness:
        """Ready == the model actually served a (protocol-aware) request.

        A real generation is the only trustworthy signal. A container can be
        ``running`` — even Docker-``healthy`` (the vLLM healthcheck has a long
        ``start_period`` grace) — and the gateway advertises every alias from its
        *static superset* route table, all long before vLLM has loaded the model
        and can serve. So we gate on the container existing, then require a
        successful generation, using the endpoint's protocol (a completions-only
        model never answers a chat probe). The probe goes through the gateway when
        present, else straight to the vLLM upstream's own published ``/v1``.
        """
        if deployment.id not in self.observe():
            return Readiness(False, 'container not running')
        served = deployment.served.get(endpoint) or {}
        protocol = served.get('protocol') or 'chat'
        if deployment.engine == 'ollama':
            # An Ollama daemon loads tags lazily; make the declared tag present
            # first (this also confirms the daemon is up).
            error = self._ensure_ollama_tag(deployment, endpoint)
            if error:
                return Readiness(False, error)
            protocol = 'chat'  # Ollama's OpenAI surface is chat
        if self.litellm:
            # The alias must be routable (require_listed) AND actually serve
            # (require_generation) — listing alone is trivially true here.
            ok, reason = openai_ready(
                base_url=f'http://127.0.0.1:{self.litellm_port}/v1',
                headers={'Authorization': f'Bearer {self.master_key()}'},
                model=endpoint,
                protocol=protocol,
                require_listed=True,
                require_generation=True,
                http=self.http,
            )
            return Readiness(ok, reason)
        # No gateway: probe the engine's own published API. Ollama loads on first
        # request, so the pull above is the readiness we can confirm for it; a
        # vLLM upstream we probe directly with a real generation.
        if deployment.engine == 'ollama':
            return Readiness(True, 'ollama tag pulled')
        base = self._published_v1_url(deployment)
        if base is None:
            return Readiness(False, 'vLLM upstream has no published port yet')
        ok, reason = openai_ready(
            base_url=base,
            model=served.get('served_model_name') or endpoint,
            protocol=protocol,
            require_listed=False,
            require_generation=True,
            http=self.http,
        )
        return Readiness(ok, reason)

    def down(self) -> None:
        """Tear the whole project down (for an explicit stop)."""
        if self.compose_file.exists():
            self._compose(['down', '--remove-orphans'])
