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

In static-superset mode the gateway's ``model_list`` is rendered from an
**append-only route registry** (``litellm_registry.json`` in the shared state
dir): every converge merges the invoking catalog plus every live deployment
(across all runbooks sharing the stack) into the registry and renders from the
whole thing. That makes the render a function of accumulated shared state — not
of which runbook invoked the converge — so a cross-catalog converge can no
longer strip another's live routes and, once every catalog has merged once, the
config is byte-stable (the gateway is never recreated). See
:meth:`ComposeBackend._update_route_registry` and
:func:`_litellm_model_list_from_registry`; ``infer-stack routes`` inspects/seeds/
prunes it; ``docs/litellm-gateway-routing.md`` has the full story.

Docker and HTTP are invoked through injected seams (``run`` / ``http_get``), so
all logic here is unit-testable without docker or a network. The real
docker/GPU path is validated on a GPU host. ``converge`` is serialized with a
file lock so concurrent processes don't clobber the shared compose file.

Slice status: readiness probes the LiteLLM ``/v1/models`` listing (model is
routable). The Ollama tag pull/warmup rung is a follow-up.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config import DEFAULT_PORTS, PINNED_IMAGES, default_state_paths
from ..env_utils import ensure_secret, parse_env_file, write_env_file
from ..probe import openai_ready
from ..profile_runtime import simulator_args, vllm_args
from .backend import ConvergeScaffold, Readiness
from .models import Deployment, is_reservation
from .placement import plan_placement

LEASING_PROJECT = 'infer-stack'  # docker compose project name for leased stacks
VLLM_HOST_PORT_BASE = 18000
VLLM_CONTAINER_PORT = 8000
OLLAMA_CONTAINER_PORT = 11434
LITELLM_CONTAINER_PORT = 4000
STATE_FILENAME = 'leasing-compose-state.json'
COMPOSE_FILENAME = 'docker-compose.yml'
LITELLM_CONFIG_FILENAME = 'litellm_config.yaml'
LITELLM_SERVICE = 'litellm'
API_KEY_ENV = 'LITELLM_MASTER_KEY'

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
POSTGRES_SERVICE = 'postgres-litellm'
POSTGRES_CONTAINER_PORT = 5432
POSTGRES_DB_NAME = 'litellm'
POSTGRES_DB_USER = 'litellm'
DB_PASSWORD_ENV = 'LITELLM_DB_PASSWORD'  # managed secret in the sidecar .env
# Marks a LiteLLM route as infer-stack-managed, so reconcile only ever deletes
# routes it created (never a model added by hand through the UI/admin API).
ROUTE_ID_PREFIX = 'isr-'

VLLM_DEFAULTS = {
    'gpu_memory_utilization': 0.9,
    'max_model_len': 8192,
    'max_num_batched_tokens': 8192,
    'max_num_seqs': 256,
}

DEPLOYMENT_LABEL = 'infer-stack.deployment'
ENGINE_LABEL = 'infer-stack.engine'


def dns_slug(text: str) -> str:
    """A lowercase ``[a-z0-9-]`` label safe as a compose service / DNS /
    Kubernetes object name (shared by the compose and kubeai backends)."""
    out = re.sub(r'[^a-z0-9]+', '-', str(text).lower()).strip('-')
    return out or 'model'


_dns_slug = dns_slug  # historical internal name


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


def _unique_vllm_service_name(served: str, deployment_id: str) -> str:
    """Per-deployment vLLM service/DNS name: ``vllm-<served>-<id-tail>``.

    The static-superset gateway needs a name derivable from the served model
    *alone* (so a catalog route can address it without knowing the live
    deployment) — but that deliberately drops the deployment id, which
    **collapses every** ``--dedicated`` **deployment of one model onto a single
    container** (hence one GPU). Dynamic routing manages the gateway's routes
    live via the admin API, so the upstream host no longer has to be predictable
    from the catalog. That frees us to give each deployment its **own** service,
    so N dedicated deployments of one model become N containers on N GPUs. The
    suffix is the deployment id's hex tail, keeping the name short and DNS-safe.
    """
    tail = deployment_id.rsplit('-', 1)[-1][:8] or 'x'
    return f'{vllm_service_name_for(served)}-{_dns_slug(tail)}'


def vllm_service_name(deployment: Deployment, *, unique: bool = False) -> str:
    """Compose service name for a vLLM deployment (see :func:`vllm_service_name_for`).

    Default (``unique=False``, static-superset mode): deterministic from the
    served model name only — *no* deployment-id suffix — so it matches the
    gateway's pre-rendered route for that endpoint. Trade-off: two
    *simultaneously desired* deployments that share a served name collide on this
    name; under the static gateway the catalog endpoint is the unit, so that case
    (including same-model ``--dedicated``) is unsupported.

    ``unique=True`` (dynamic-routing mode): append the deployment-id tail
    (:func:`_unique_vllm_service_name`) so same-model dedicated deployments get
    distinct containers/GPUs; the admin-API route table addresses each by name.

    Either way ``observe`` correlates a running container back to its deployment
    id via the ``infer-stack.deployment`` label, not this name, so the choice of
    suffix does not affect reconcile bookkeeping.
    """
    served = deployment.spec.get('served_model_name') or (
        sorted(deployment.served)[0] if deployment.served else deployment.id
    )
    if unique:
        return _unique_vllm_service_name(served, deployment.id)
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
    # Desired LiteLLM route set for dynamic routing (None in static-superset
    # mode). The render half writes it to litellm_routes.json; the apply half
    # reconciles it against the live gateway via the admin API.
    litellm_routes: list[dict[str, Any]] | None = None
    # Placed deployments the render had to EXCLUDE (e.g. a compose service-name
    # collision) + the per-deployment reasons, each prefixed with the deployment
    # id like placement errors. The backend folds these into last_unplaced /
    # last_errors so a colliding acquire fails loudly instead of the later
    # deployment silently never getting a container.
    unrenderable: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


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


def vllm_service_dict(deployment: Deployment) -> dict[str, Any]:
    """Build the dict ``vllm_args`` consumes from a deployment's runtime spec
    (shared by the compose and kubeai backends, so every serving knob renders
    identically on both)."""
    runtime = deployment.spec.get('runtime', {}) or {}
    served = deployment.spec.get('served_model_name') or (
        sorted(deployment.served)[0] if deployment.served else deployment.id
    )
    return {
        'served_model_name': served,
        'tensor_parallel_size': int(runtime.get('tensor_parallel_size', 1) or 1),
        'pipeline_parallel_size': int(
            runtime.get('pipeline_parallel_size', 1) or 1
        ),
        'data_parallel_size': int(runtime.get('data_parallel_size', 1) or 1),
        # Model-level knobs (compat-key members; see catalog._resolve_vllm).
        'revision': deployment.spec.get('revision'),
        'quantization': deployment.spec.get('quantization'),
        'dtype': deployment.spec.get('dtype'),
        'chat_template': runtime.get('chat_template'),
        'trust_remote_code': bool(runtime.get('trust_remote_code', False)),
        'image': runtime.get('image'),
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
        # Selected via the VLLM_ATTENTION_BACKEND env var, not a CLI flag — the
        # backend renderer (compose environment / kubeai CR env) turns it into env.
        'attention_backend': runtime.get('attention_backend'),
        # Present => this deployment runs a simulator image whose CLI is not
        # vLLM's (see profile_runtime.simulator_args). Absent => a real engine.
        'simulator': runtime.get('simulator') or None,
        'extra_args': list(runtime.get('extra_args', []) or []),
    }


_vllm_service_dict = vllm_service_dict  # historical internal name


def _serve_config_hash(
    svc: dict[str, Any],
    images: dict[str, str],
    command: list[str],
    environment: dict[str, str],
) -> str:
    """Stable short hash of everything that shapes the serve's compiled graphs.

    Used to key the vLLM compile-cache mount per serve config (see the volumes
    comment in ``_vllm_service``). Image + rendered command + generation-
    relevant env cover every knob that can reach the traced graph, including
    ``extra_args`` that are deliberately non-structural for deployment
    identity (e.g. ``--limit-mm-per-prompt``).
    """
    material = "\x00".join(
        [
            str(svc.get("image") or images["vllm"]),
            *command,
            *(f"{k}={v}" for k, v in sorted(environment.items()) if k != "HF_TOKEN"),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def _vllm_service(
    deployment: Deployment,
    gpus: list[int],
    host_port: int | None,
    images: dict[str, str],
    state: dict[str, str],
) -> dict[str, Any]:
    # Merge over the defaults so direct callers (tests, embedders) with a
    # partial state dict still resolve every cache-mount key below.
    state = {**default_state_paths(), **(state or {})}
    svc = _vllm_service_dict(deployment)
    simulated = bool(svc.get('simulator'))
    if simulated:
        command = simulator_args(svc)
    else:
        command = [
            deployment.spec['hf_model_id'],
            '--host',
            '0.0.0.0',
            '--port',
            '8000',
            *vllm_args(svc),
        ]
    environment: dict[str, str] = {'HF_TOKEN': '${HF_TOKEN:-}'}
    # Attention backend is a vLLM env var (VLLM_ATTENTION_BACKEND), not a CLI
    # flag; forward it verbatim when the endpoint sets one (e.g. TORCH_SDPA to
    # match a HuggingFace-eager deployment's numerics).
    if svc.get('attention_backend'):
        environment['VLLM_ATTENTION_BACKEND'] = str(svc['attention_backend'])
    service: dict[str, Any] = {
        # A runtime image override is structural (distinct deployments), so it
        # must also pick the container image — like _ollama_service does.
        'image': svc.get('image') or images['vllm'],
        'command': command,
        'environment': environment,
        # Weights AND compile artifacts persist across container recreations:
        # with `reclaim: stop`, every re-acquire cold-starts the container, and
        # without these mounts vLLM re-pays its full torch.compile / Triton /
        # CUDA-jit pass (~10-20 min on big models) on every lease. The state
        # dirs have existed in default_state_paths all along — they were simply
        # never mounted.
        #
        # The vLLM compile cache is keyed by a hash of the FULL serve config
        # (image + command + attention env): vLLM's own cache key omits at
        # least limit_mm_per_prompt, so a config change can silently reload a
        # graph traced under different inputs — observed as an
        # AttributeError('NoneType'.size) engine crash when the mm limits
        # changed, and the quiet failure mode would be wrong numerics. A
        # per-config subdir makes any arg change start a fresh cache while
        # identical configs keep the reuse. Weights (hf) and the triton/cuda
        # jit caches are content-addressed internally and stay shared.
        'volumes': [
            f'{state["hf_cache"]}:/root/.cache/huggingface',
            f'{state["vllm_cache"]}/cfg-{_serve_config_hash(svc, images, command, environment)}'
            ':/root/.cache/vllm',
            f'{state["torch_cache"]}:/root/.cache/torch',
            f'{state["triton_cache"]}:/root/.triton',
            f'{state["cuda_cache"]}:/root/.nv',
        ],
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
    # Only publish a host port when there's no gateway to front the upstream.
    # Behind LiteLLM the upstream is internal (reached by compose-network DNS at
    # :8000), and a published port would have to be unique across the live set,
    # which reintroduces the set-dependence this avoids. See render_compose.
    if host_port is not None:
        service['ports'] = [f'{host_port}:8000']
    if gpus:
        service['deploy'] = _gpu_reservation(gpus)
    if simulated:
        # A simulator downloads no weights and compiles no graphs, so the
        # caches above are dead weight -- and worse, the images ship
        # distroless and run as a non-root uid that cannot write /root, so
        # mounting them invites a permission failure for no benefit. The
        # healthcheck goes for the same reason: it shells out to `curl`,
        # which a distroless image does not contain, so it would mark a
        # perfectly healthy container unhealthy forever. Nothing depends on
        # it -- `probe_ready` gates on a real generation over HTTP, which is
        # a stronger signal and works against any image.
        service.pop('volumes', None)
        service['healthcheck'] = {'disable': True}
    return service


def _ollama_service(
    deployment: Deployment,
    gpus: list[int],
    host_port: int | None,
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
    # See _vllm_service: only publish a host port when there is no gateway.
    if host_port is not None:
        service['ports'] = [f'{host_port}:11434']
    if gpus:
        service['deploy'] = _gpu_reservation(gpus)
    return service


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
            served = deployment.spec.get('served_model_name') or deployment.id
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
            served = deployment.spec.get('served_model_name') or (
                sorted(deployment.served)[0] if deployment.served else deployment.id
            )
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
        row = rows[name]
        if not isinstance(row, dict):
            continue
        engine = row.get('engine')
        if engine == 'vllm':
            served = row.get('served') or name
            api_base = (
                f'http://{vllm_service_name_for(served)}:{VLLM_CONTAINER_PORT}/v1'
            )
            entries.append(_vllm_route_entry(name, served, api_base))
        elif engine == 'ollama':
            tag = row.get('model') or name
            host = row.get('host') or name
            api_base = (
                f'http://{ollama_service_name_for(host)}:{OLLAMA_CONTAINER_PORT}'
            )
            entries.append(_ollama_route_entry(name, tag, api_base))
    return entries


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
    _reconcile_routes`) is a pure set-diff: the same logical route always has the
    same id (added once, never churned), and a route that drops out of the
    desired set is deleted by exactly this id. The ``isr-`` prefix marks it
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
            served = deployment.spec.get('served_model_name') or deployment.id
            api_base = (
                f'http://{vllm_service_name(deployment, unique=True)}'
                f':{VLLM_CONTAINER_PORT}/v1'
            )
            for endpoint in sorted(deployment.served):
                entries.append(
                    {
                        'model_name': endpoint,
                        'litellm_params': {
                            'model': f'openai/{served}',
                            'api_base': api_base,
                            'api_key': 'EMPTY',
                        },
                        'model_info': {'id': _route_id(deployment.id, endpoint)},
                    }
                )
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
    route_registry: dict[str, Any] | None = None,
    dynamic_routing: bool = False,
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
    unrenderable: set[str] = set()
    errors: list[str] = []
    ordered = sorted(deployments, key=lambda g: (g.created_at, g.id))
    vllm_i = 0
    ollama_i = 0
    for deployment in ordered:
        if deployment.id not in assignments:
            continue
        gpus = assignments[deployment.id]
        if deployment.engine == 'vllm':
            # Unique-per-deployment names in dynamic-routing mode so same-model
            # --dedicated deployments don't collapse onto one container/GPU.
            name = vllm_service_name(deployment, unique=dynamic_routing)
        elif deployment.engine == 'ollama':
            name = ollama_service_name(deployment)
        else:
            continue
        # Two live deployments can map to one service name (same served name in
        # static-superset mode — e.g. same-model --dedicated — or two Ollama
        # deployments on one host with different structural settings). Writing
        # both would silently drop the earlier one: its container never exists,
        # observe() never sees it, and its probes fail until lease timeout with
        # no error anywhere. The oldest deployment keeps the name; later ones
        # are excluded and reported like placement failures.
        if name in services:
            unrenderable.add(deployment.id)
            errors.append(
                f'{deployment.id}: compose service name {name!r} is already '
                f'used by deployment {service_map[name]} — simultaneously '
                'live deployments must have distinct served names '
                '(use dynamic routing for same-model dedicated deployments)'
            )
            continue
        # Host ports are published ONLY when there is no LiteLLM gateway. Behind
        # the gateway every upstream is reached by compose-network DNS, so a host
        # port is unnecessary — and harmful: it was assigned by position in the
        # live set (BASE + i), so adding/removing any deployment renumbered the
        # survivors' ports, which changed their service definitions and made
        # `docker compose up -d` recreate unrelated, in-flight containers (a blip
        # that killed readiness mid-request). Omitting it makes each upstream's
        # rendered service depend only on the deployment itself -> no churn, the
        # same no-blip property the static gateway config already has.
        if deployment.engine == 'vllm':
            port = None if litellm else VLLM_HOST_PORT_BASE + vllm_i
            vllm_i += 0 if litellm else 1
            services[name] = _vllm_service(deployment, gpus, port, images, state)
            vllm_v1_urls.append(f'http://{name}:{VLLM_CONTAINER_PORT}/v1')
        else:
            base = ports.get('ollama', DEFAULT_PORTS['ollama'])
            port = None if litellm else base + ollama_i
            ollama_i += 0 if litellm else 1
            services[name] = _ollama_service(deployment, gpus, port, images, state)
            ollama_native_urls.append(f'http://{name}:{OLLAMA_CONTAINER_PORT}')
        service_map[name] = deployment.id

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
            litellm_routes = _litellm_routes(deployments, assignments)
            litellm_depends: list[str] = []
        elif route_registry is not None:
            entries = _litellm_model_list_from_registry(route_registry)
            litellm_depends = []  # no per-model depends_on -> no churn
        elif catalog is not None:
            entries = _litellm_model_list_from_catalog(catalog)
            litellm_depends = []  # no per-model depends_on -> no churn
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
        litellm_routes=litellm_routes,
        unrenderable=unrenderable,
        errors=errors,
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


class ComposeBackend(ConvergeScaffold):
    """Single-host docker compose backend (converge-style).

    Driven by the controller's ``converge`` path. ``run`` / ``http_get`` are the
    injected docker/HTTP seams; defaults shell out to ``docker compose`` and
    ``requests``. State-dir plumbing (atomic writes, the converge lock, the
    sidecar, diff-confirm) comes from :class:`ConvergeScaffold`.
    """

    _approve_title = 'infer-stack will update the compose project'
    _state_noun = 'compose project'

    def __init__(
        self,
        *,
        state_dir: str | Path,
        inventory: dict[str, Any] | None = None,
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
        dynamic_routing: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # None => detect lazily on first use (see the `inventory` property):
        # startup paths (notably the TUI) must never wait on nvidia-smi
        # before the first frame.
        self._inventory = inventory
        self.run = run or _default_docker_run
        if http is None:
            import requests
            http = requests
        self.http = http
        self.images = {**PINNED_IMAGES, **(images or {})}
        self.ports = {**DEFAULT_PORTS, **(ports or {})}
        # Merge over the defaults (not replace) so a caller-supplied partial
        # state dict — tests, embedders — still resolves every cache-mount key.
        self.state = {**default_state_paths(), **(state or {})}
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
        # Dynamic routing: manage the gateway's routes live via the admin API
        # against a Postgres-backed model store, instead of a static config file.
        # Gives each deployment its own upstream (so same-model --dedicated
        # deployments land on distinct GPUs) with no gateway recreation/blip.
        self.dynamic_routing = dynamic_routing
        self._sleep = sleep
        self.last_errors: list[str] = []
        self.last_unplaced: set[str] = set()  # desired deployment ids placement skipped
        self.last_assignments: dict[str, list[int]] = {}  # deployment id -> GPU ids
        self._pulled: set[str] = set()  # (deployment:tag) pulled this process
        # VRAM facts (docs/planning/vram-aware-placement.md Phase 3): the
        # measured-requirement overlay + a per-process cache of weight-bytes
        # floors from the local HF hub cache.
        from .vram import Measurements

        self.measurements = Measurements(self.state_dir / 'measurements.json')
        self._floor_cache: dict[str, float | None] = {}

    @property
    def inventory(self) -> dict[str, Any]:
        """GPU inventory, detected lazily on first use.

        Startup paths (notably the TUI) construct the backend with
        ``inventory=None`` so nothing waits on the nvidia-smi subprocess before
        the first frame; the first placement (``plan``/``converge``) pays the
        detection instead — off the UI thread when driven from the TUI's
        workers. Tests and callers that pass an explicit inventory are
        unaffected.
        """
        if self._inventory is None:
            from ..hardware import detect_inventory

            self._inventory = detect_inventory()
        return self._inventory

    @inventory.setter
    def inventory(self, value: dict[str, Any] | None) -> None:
        self._inventory = value

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

    def plan(self, desired: list[Deployment]):
        """Compute GPU placement for ``desired`` without writing or applying.

        Read-only and side-effect free: it honors the persisted pins, so the
        result reflects where deployments are (for running ones) or *would* be (for
        not-yet-started ones) placed. ``leases`` uses it to show actual/slated
        GPUs; ``converge`` uses it as the first step of render.
        """
        pinned = self._load_sidecar().get('assignments', {})
        desired = list(desired)
        self._enrich_placement(desired)
        return plan_placement(
            desired,
            self.inventory,
            allowed_gpus=self.allowed_gpus,
            reserved=self.reserved,
            pinned=pinned,
            skip_display=self.skip_display,
        )

    def _enrich_placement(self, desired: list[Deployment]) -> None:
        """Attach VRAM facts to vLLM deployments before planning (in-memory).

        Resolution order (docs/planning/vram-aware-placement.md §3):
        a catalog-declared ``min_vram_gib`` wins; else a recorded measurement
        from the overlay fills it; the weight-bytes floor rides alongside
        (``max(declared-or-measured, floor)`` is applied by the planner, and
        a floor-only deployment keeps legacy index-order selection). Purely
        best-effort and never persisted — a missing overlay or an
        un-downloaded model just means placement runs exactly as before.
        """
        from .vram import measurement_key_for_spec, weight_floor_gib

        for deployment in desired:
            if deployment.engine != 'vllm':
                continue
            try:
                spec = deployment.spec
                placement = dict(spec.get('placement') or {})
                if not placement.get('min_vram_gib'):
                    measured = self.measurements.get_min_vram_gib(
                        measurement_key_for_spec(spec)
                    )
                    if measured:
                        placement['min_vram_gib'] = measured
                        placement['min_vram_source'] = 'measured'
                if not placement.get('floor_vram_gib'):
                    model_id = spec.get('hf_model_id') or ''
                    if model_id not in self._floor_cache:
                        self._floor_cache[model_id] = weight_floor_gib(
                            model_id, self.state.get('hf_cache')
                        )
                    floor = self._floor_cache[model_id]
                    if floor:
                        placement['floor_vram_gib'] = floor
                if placement:
                    spec['placement'] = placement
            except Exception:
                continue  # enrichment must never block placement

    def deployment_logs(self, deployment: Deployment, *, tail: int = 400) -> str:
        """Recent engine logs for a deployment's compose service.

        Fail-open to ``''`` — this feeds diagnosis paths (OOM classification,
        ``infer-stack measure``), which must degrade silently when the
        container is already gone.
        """
        try:
            if deployment.engine == 'vllm':
                name = vllm_service_name(
                    deployment, unique=self.dynamic_routing
                )
            elif deployment.engine == 'ollama':
                name = ollama_service_name(deployment)
            else:
                return ''
            return self._compose(
                ['logs', '--no-color', '--tail', str(tail), name]
            )
        except Exception:
            return ''

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

    def _update_route_registry(
        self, desired: list[Deployment], assignments: dict[str, list[int]]
    ) -> dict[str, Any]:
        """Load, merge the invoking catalog (if any) + all live deployments,
        persist iff changed, and return the merged registry.

        Called under the converge flock (:meth:`_converge_lock`), so the
        read-merge-write is race-safe against concurrent converges from other
        runbooks with no new locking."""
        from .._log import logger

        existing = self._load_route_registry()
        incoming: dict[str, dict[str, Any]] = {}
        if self.catalog is not None:
            incoming.update(_registry_incoming_from_catalog(self.catalog))
        # `desired` spans all runbooks via the shared ledger, so this keeps every
        # live cross-runbook deployment routable (and, via persistence, routable
        # past release).
        incoming.update(
            _registry_incoming_from_deployments(desired, assignments)
        )
        merged, warnings = _merge_route_registry(existing, incoming)
        for w in warnings:
            logger.warning('  route registry: {}', w)
        if merged != existing:
            prior = existing.get('entries', {}) if isinstance(existing, dict) else {}
            added = sorted(set(merged['entries']) - set(prior))
            updated = sorted(
                k for k in merged['entries']
                if k in prior and merged['entries'][k] != prior[k]
            )
            if added:
                logger.info('  route registry: +{} route(s): {}',
                            len(added), ', '.join(added))
            if updated:
                logger.info('  route registry: updated route(s): {}',
                            ', '.join(updated))
            self._atomic_write(
                self._registry_file, _dump_route_registry(merged)
            )
        return merged

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
            self.last_assignments = dict(plan.assignments)
            for gid, gpus in sorted(plan.assignments.items()):
                logger.info('  placed {} on GPU(s) {}', gid, gpus or '(cpu)')
            for err in plan.errors:
                logger.warning('  placement: {}', err)
            for note in plan.warnings:
                # Honored-but-suspect decisions (a pin/explicit index that
                # contradicts a declared min_vram_gib): never fail the plan,
                # never be silent either.
                logger.warning('  placement: {}', note)
            if self.litellm and self.dynamic_routing:
                # Persist the DB secret to the sidecar .env *before* rendering, so
                # docker compose --env-file can interpolate ${LITELLM_DB_PASSWORD}
                # into the postgres + litellm services at apply time.
                self.db_password()
            route_registry = None
            if self.litellm and not self.dynamic_routing:
                # Unconditional in static-superset mode: `self.catalog` may be
                # None (a bare release/gc with no discoverable config dir) — the
                # incoming set is then deployments-only, and the render still
                # comes from the accumulated registry, so a catalog-less converge
                # cannot strip routes or blip. This retires the legacy
                # per-deployment `_litellm_model_list` branch from the backend
                # path entirely (it survives in render_compose for direct callers).
                route_registry = self._update_route_registry(
                    desired, plan.assignments
                )
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
                route_registry=route_registry,
                dynamic_routing=self.dynamic_routing,
            )
            # A deployment the render excluded (service-name collision) is as
            # undeliverable as an unplaced one: fold it into last_unplaced /
            # last_errors so acquire fails loudly and rolls the lease back.
            self.last_errors = list(plan.errors) + list(rendered.errors)
            self.last_unplaced = {
                g.id for g in desired if g.id not in plan.assignments
            } | set(rendered.unrenderable)
            for err in rendered.errors:
                logger.warning('  render: {}', err)

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
            routes_text = (
                json.dumps(rendered.litellm_routes, indent=2)
                if rendered.litellm_routes is not None
                else None
            )
            if routes_text is not None:
                planned[self._routes_file] = routes_text
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
            if routes_text is not None:
                # The rendered desired route set for the running gateway (applied
                # via the admin API in apply() -> _reconcile_routes).
                self._atomic_write(self._routes_file, routes_text)
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
            if self.litellm and self.dynamic_routing:
                # Apply the rendered desired route set to the now-running gateway
                # via the admin API (the dynamic-routing half of apply).
                self._reconcile_routes()
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

    # -- dynamic routing (admin API) --------------------------------------

    def _gateway_base(self) -> str:
        return f'http://127.0.0.1:{self.litellm_port}'

    def _auth_headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.master_key()}'}

    def _desired_routes(self) -> list[dict[str, Any]]:
        """The rendered desired route set (litellm_routes.json), or empty."""
        try:
            data = json.loads(self._routes_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _reconcile_routes(self, *, attempts: int = 90, delay: float = 2.0) -> None:
        """Make the live gateway's managed routes match the rendered route set.

        The render half wrote the desired routes (one per live deployment×
        endpoint) to ``litellm_routes.json``; this is the apply half. List the
        gateway's current models, then add the missing routes and delete the ones
        no longer desired — through the admin API, with **no** container restart.

        Properties this relies on:

        * **Idempotent / coalescing-safe.** A redundant apply re-diffs to the
          same set and does nothing, which is what makes the controller's
          coalesced apply correct for routes too.
        * **Drift-healing.** Routes lost to a gateway/DB restart reappear in the
          diff and are re-added; stale routes from a prior run (still in the DB)
          are deleted because they're no longer desired.
        * **Co-existence.** Only routes infer-stack created (id prefix ``isr-``)
          are ever deleted, so a model added by hand through the UI/API is left
          alone.

        Best-effort: the gateway may still be starting (it waits on Postgres
        health, then boots — and on first-ever bring-up runs LiteLLM's Prisma DB
        migrations, which can take a while), so the initial listing is retried
        generously. A persistent failure is logged and left for the next converge
        rather than raised — apply must stay non-fatal, like ``docker compose up``.
        """
        from .._log import logger

        desired = {
            r['model_info']['id']: r
            for r in self._desired_routes()
            if isinstance(r.get('model_info'), dict) and r['model_info'].get('id')
        }
        current = self._list_managed_routes(attempts=attempts, delay=delay)
        if current is None:
            logger.warning(
                'dynamic routing: gateway not reachable to reconcile routes; '
                'leaving it for the next converge'
            )
            return
        to_add = [desired[i] for i in desired if i not in current]
        to_delete = [i for i in current if i not in desired]
        for route in to_add:
            self._post_route('/model/new', route, route.get('model_name'))
        for rid in to_delete:
            # ok_if_missing: with a shared gateway, another converge may have
            # deleted this route already; "not found in db" means the desired
            # end-state (route gone) is reached, so don't treat it as an error.
            self._post_route(
                '/model/delete', {'id': rid}, rid, ok_if_missing=True
            )
        if to_add or to_delete:
            logger.info(
                'dynamic routing: +{} route(s), -{} route(s) (now {} desired)',
                len(to_add), len(to_delete), len(desired),
            )

    def _list_managed_routes(
        self, *, attempts: int, delay: float
    ) -> set[str] | None:
        """Ids of infer-stack-managed routes currently on the gateway.

        Returns ``None`` if the gateway never became reachable within
        ``attempts`` (so the caller can skip the diff and retry next converge).
        """
        for attempt in range(max(1, attempts)):
            resp = None
            try:
                resp = self.http.get(
                    f'{self._gateway_base()}/v1/model/info',
                    headers=self._auth_headers(),
                    timeout=10,
                )
            except Exception:  # noqa: BLE001 - the gateway may still be starting
                resp = None
            if resp is not None and getattr(resp, 'status_code', 0) == 200:
                ids: set[str] = set()
                for m in (resp.json().get('data') or []):
                    rid = (m.get('model_info') or {}).get('id')
                    if isinstance(rid, str) and rid.startswith(ROUTE_ID_PREFIX):
                        ids.add(rid)
                return ids
            if attempt < attempts - 1:
                self._sleep(delay)
        return None

    def _post_route(
        self,
        path: str,
        payload: dict[str, Any],
        label: Any,
        *,
        ok_if_missing: bool = False,
    ) -> None:
        """POST one admin-API call (``/model/new`` or ``/model/delete``).

        Per-call best-effort: a failure is logged and the rest still run; the
        next converge re-reconciles, so a transient error self-heals.
        ``ok_if_missing`` swallows a "model not found" response (a delete whose
        target is already gone has already reached its desired end-state).
        """
        from .._log import logger

        try:
            resp = self.http.post(
                f'{self._gateway_base()}{path}',
                headers=self._auth_headers(),
                json=payload,
                timeout=30,
            )
        except Exception as ex:  # noqa: BLE001 - one bad call must not abort apply
            logger.warning('dynamic routing: POST {} {} error: {}', path, label, ex)
            return
        if getattr(resp, 'status_code', 0) >= 300:
            body = str(getattr(resp, 'text', ''))
            if ok_if_missing and 'not found' in body.lower():
                return
            logger.warning(
                'dynamic routing: POST {} {} -> {} {}',
                path, label, resp.status_code, body[:200],
            )

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
        if is_reservation(deployment):
            # A reservation holds a GPU but runs no server, so there is nothing to
            # probe — it is "ready" the moment placement assigned it a GPU.
            return Readiness(True, 'gpu reserved (no server to probe)')
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
