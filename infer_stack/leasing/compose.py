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
from .gateway import (
    API_KEY_ENV,
    DB_PASSWORD_ENV,
    _dump_route_registry,
    LITELLM_CONFIG_FILENAME,
    LITELLM_CONTAINER_PORT,
    _litellm_model_list,
    _litellm_model_list_from_catalog,
    _litellm_model_list_from_registry,
    LITELLM_REGISTRY_FILENAME,
    LITELLM_REGISTRY_VERSION,
    _litellm_routes,
    LITELLM_ROUTES_FILENAME,
    _litellm_service,
    LITELLM_SERVICE,
    _merge_route_registry,
    _nginx_conf,
    NGINX_CONFIG_FILENAME,
    _nginx_service,
    NGINX_SERVICE,
    _open_webui_service,
    OPEN_WEBUI_SERVICE,
    _postgres_service,
    POSTGRES_SERVICE,
    _registry_incoming_from_catalog,
    _registry_incoming_from_deployments,
    ROUTE_ID_PREFIX,
    ROUTE_RECONCILE_BOOTSTRAP_S,
    ROUTE_RECONCILE_STEADY_S,
    SALT_KEY_ENV,
    _seed_registry_from_litellm_config,
    set_master_key,
)
from .launch import env_string, fill, translate_legacy
from .models import Deployment, is_reservation, served_name
from .naming import (  # noqa: F401  (public names re-exported for callers)
    OLLAMA_CONTAINER_PORT,
    VLLM_CONTAINER_PORT,
    _dns_slug,
    _unique_vllm_service_name,
    dns_slug,
    ollama_service_name,
    ollama_service_name_for,
    vllm_service_name,
    vllm_service_name_for,
)
from .placement import plan_placement
from .residency import (  # labels live beside the code that reads them back
    ENGINE_LABEL,
    FINGERPRINT_LABEL,
    SERVICE_LABEL,
    COMPOSE_PROJECT_LABEL,
    DEPLOYMENT_LABEL,
    Residency,
    ResidencyUnknown,
    residency_from_inspect,
)

LEASING_PROJECT = 'infer-stack'  # docker compose project name for leased stacks
VLLM_HOST_PORT_BASE = 18000
STATE_FILENAME = 'leasing-compose-state.json'
COMPOSE_FILENAME = 'docker-compose.yml'

VLLM_DEFAULTS = {
    'gpu_memory_utilization': 0.9,
    'max_model_len': 8192,
    'max_num_batched_tokens': 8192,
    'max_num_seqs': 256,
}




class ApplyAborted(RuntimeError):
    """Selective apply refused to act; the change stays pending."""


@dataclass
class SelectiveApplyOutcome:
    kept_services: set[str]
    removed: list[str]
    started: list[str]
    orphans: list[str]


def profile_images(profile: dict[str, Any]) -> list[str]:
    """Every image a Compose profile can need, so publication pulls them all.

    Infrastructure images for what the profile enables, plus the engine image
    of every endpoint reachable from its published catalogs, including
    per-endpoint image overrides. With no catalog any vLLM request is possible,
    so the default vLLM image is included.

    Example:
        >>> p = {'images': {'vllm': 'v', 'ollama': 'o', 'litellm': 'l', 'postgres': 'p',
        ...                 'open_webui': 'w', 'nginx': 'n'},
        ...      'litellm': True, 'dynamic_routing': True, 'ui': False,
        ...      'reverse_proxy': {'enabled': False}, 'catalogs': []}
        >>> profile_images(p)
        ['l', 'p', 'v']
    """
    from .profile import CatalogUnion

    images = profile['images']
    wanted: set[str] = set()
    if profile.get('litellm'):
        wanted.add(images['litellm'])
        if profile.get('dynamic_routing'):
            wanted.add(images['postgres'])
        if (profile.get('reverse_proxy') or {}).get('enabled'):
            wanted.add(images['nginx'])
    if profile.get('ui'):
        wanted.add(images['open_webui'])
    sources = profile.get('catalogs') or []
    if not sources:
        wanted.add(images['vllm'])
        return sorted(wanted)
    union = CatalogUnion.from_sources(sources)
    for name in union.endpoints:
        try:
            req = union.resolve_endpoint(name)
        except Exception:  # noqa: BLE001 - an unresolvable endpoint serves nothing
            continue
        if req.engine == 'vllm':      # same sources the render uses
            wanted.add((req.spec.get('runtime') or {}).get('image') or images['vllm'])
        elif req.engine == 'ollama':
            wanted.add(req.spec.get('image') or images['ollama'])
    return sorted(wanted)


# Crash diagnosis is backend-neutral (the kubeai backend uses it too); these
# names stay importable from here.
from .diagnosis import (  # noqa: E402,F401
    CRASH_LOOP_RESTARTS,
    _ENGINE_ERROR_HINTS,
    _TRANSIENT_ENGINE_SIGNATURES,
    _engine_error_summary,
    classify_engine_log,
    diagnose_startup,
)


def _network_name() -> str:
    from .network import NETWORK_NAME

    return NETWORK_NAME


def _stanza_gpus(service: dict[str, Any]) -> list[int]:
    devices = (((service.get('deploy') or {}).get('resources') or {})
               .get('reservations') or {}).get('devices') or []
    out = []
    for dev in devices:
        for raw in dev.get('device_ids') or []:
            try:
                out.append(int(raw))
            except ValueError:
                pass
    return out


#: Bounds for selective apply's waits under the controller's lock.
APPLY_REMOVAL_WAIT_S = 60.0
APPLY_HEALTH_WAIT_S = 180.0


def _depends_on(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``depends_on`` as ``{service: {condition...}}`` for list and map forms."""
    deps = service.get('depends_on') or {}
    if isinstance(deps, list):
        return {d: {} for d in deps}
    return {d: dict(v or {}) for d, v in deps.items()}


def _dependency_levels(services: dict[str, Any], names: list[str]) -> list[list[str]]:
    """Group ``names`` into start order: each level depends only on earlier ones.

    Example:
        >>> svcs = {'db': {}, 'gw': {'depends_on': {'db': {}}}, 'm': {}}
        >>> _dependency_levels(svcs, ['gw', 'db', 'm'])
        [['db', 'm'], ['gw']]
    """
    pending = list(dict.fromkeys(names))
    started: set[str] = set()
    levels = []
    while pending:
        level = []
        for name in pending:
            deps = services.get(name, {}).get('depends_on') or []
            deps = set(deps) & set(pending)
            if deps <= started:
                level.append(name)
        if not level:                      # a cycle: start the rest together
            level = list(pending)
        levels.append(sorted(level))
        started.update(level)
        pending = [n for n in pending if n not in started]
    return levels


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


def stamp_fingerprints(
    compose: dict[str, Any], *, files: dict[Path, str], env_file: Path | None = None,
) -> dict[str, str]:
    """Label every service with a behavioural fingerprint; return service -> fingerprint.

    The fingerprint is ``sha256(canonical stanza without the fingerprint label ||
    sha256 of each generated file the service bind-mounts)``, plus the managed
    ``.env`` when the stanza interpolates from it. It changes exactly when the
    service's behaviour does: an unchanged fingerprint keeps a container, a
    changed one makes selective apply recreate it. (A per-render label would
    recreate everything on every apply.)

    ``files`` holds content about to be written (path -> text); a mounted file
    that is not in it is read from disk if it exists.

    Example:
        >>> doc = {'services': {'a': {'image': 'x', 'labels': {}}}}
        >>> fps = stamp_fingerprints(doc, files={})
        >>> doc['services']['a']['labels'][FINGERPRINT_LABEL] == fps['a']
        True
        >>> doc['services']['a']['image'] = 'y'
        >>> stamp_fingerprints(doc, files={})['a'] != fps['a']
        True
    """
    import re

    by_path = {str(Path(p)): text for p, text in files.items()}
    env_values: dict[str, str] = {}
    if env_file is not None and Path(env_file).exists():
        from ..env_utils import parse_env_file

        env_values = parse_env_file(Path(env_file))
    out: dict[str, str] = {}
    for name, svc in (compose.get('services') or {}).items():
        labels = dict(svc.get('labels') or {})
        labels.pop(FINGERPRINT_LABEL, None)
        stanza = {**svc, 'labels': labels}
        material = [json.dumps(stanza, sort_keys=True, default=str)]
        for volume in svc.get('volumes') or []:
            source = str(volume).split(':', 1)[0] if isinstance(volume, str) else ''
            text = by_path.get(str(Path(source))) if source else None
            if text is None and source and Path(source).is_file():
                text = Path(source).read_text(errors='replace')
            if text is not None:
                material.append(hashlib.sha256(text.encode('utf-8')).hexdigest())
        # Only the variables this stanza interpolates: an unrelated key in the
        # managed .env must not recreate the service.
        referenced = sorted(set(re.findall(r'\$\{([A-Za-z_][A-Za-z0-9_]*)', material[0])))
        if referenced:
            values = json.dumps({k: env_values.get(k) for k in referenced}, sort_keys=True)
            material.append(hashlib.sha256(values.encode('utf-8')).hexdigest())
        fingerprint = hashlib.sha256('\x00'.join(material).encode('utf-8')).hexdigest()[:16]
        svc.setdefault('labels', {})[FINGERPRINT_LABEL] = fingerprint
        out[name] = fingerprint
    return out


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
    # A deployment recorded before the generic launch fields may still carry
    # `serve_recipe`; read it as the fields it meant.
    runtime = translate_legacy(deployment.spec.get('runtime', {}) or {})
    served = served_name(deployment)
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
        # Generic launch fields (see leasing.launch).
        'command': list(runtime.get('command') or []),
        'env': dict(runtime.get('env') or {}),
        'mounts': dict(runtime.get('mounts') or {}),
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
    elif svc['command']:
        # An image with its own launcher: it gets exactly this, nothing added.
        command = [fill(part, svc) for part in svc['command']]
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
    # Catalog mappings are semantically unordered.  A freshly resolved request
    # preserves YAML insertion order, while the same deployment reloaded from
    # sqlite has passed through JSON ``sort_keys=True``.  Render a canonical
    # order so admission preview and post-commit render are byte-identical; the
    # approval guard hashes the rendered files, not their parsed semantics.
    for key in sorted(svc['env'], key=str):
        value = svc['env'][key]
        # `$$` is a literal `$` to Compose: catalog values are never
        # interpolated, so they cannot pull secrets out of the managed .env.
        environment[str(key)] = fill(env_string(value), svc).replace('$', '$$')
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
    if svc['mounts']:
        # An image that keeps prepared weights or caches outside vLLM's
        # /root/.cache layout persists them here, so a release and re-acquire
        # is a restart rather than a re-download and re-preparation. A custom
        # launcher gets only these; stock vLLM keeps its caches as well.
        mounts = [
            f'{state["runtime"]}/{svc["mounts"][target]}:{target}'
            for target in sorted(svc['mounts'], key=str)
        ]
        service['volumes'] = mounts if svc['command'] else [*service['volumes'], *mounts]
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
    litellm_salt_key: bool = False,
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

    for name, svc in services.items():
        svc.setdefault('labels', {})[SERVICE_LABEL] = name

    return RenderedCompose(
        compose={'name': project, 'services': services},
        services=service_map,
        litellm_config=litellm_config,
        nginx_config=nginx_config,
        litellm_routes=litellm_routes,
        unrenderable=unrenderable,
        errors=errors,
    )


#: Wall-clock bounds for Docker commands, by kind (seconds). Controller
#: operations run these under a host-wide lock, so none may be unbounded.
#: Generous for now: ``up`` may still pull images. Tune from host measurements.
DOCKER_TIMEOUT_QUERY = 60.0        # ps, inspect, version, images
DOCKER_TIMEOUT_LIFECYCLE = 300.0   # stop, rm, start, unpause, exec
DOCKER_TIMEOUT_CONVERGE = 1800.0   # compose up / down
DOCKER_TIMEOUT_PULL = 3600.0       # pull, manifest inspect, in-container model pulls



def _docker_timeout(args: list[str]) -> float:
    """Pick the time bound for one ``docker`` invocation from its arguments.

    Example:
        >>> _docker_timeout(['docker', 'inspect', 'abc'])
        60.0
        >>> _docker_timeout(['docker', 'compose', '-p', 'x', '-f', 'y', 'up', '-d'])
        1800.0
        >>> _docker_timeout(['docker', 'compose', '-p', 'x', 'exec', '-T', 's', 'ollama', 'pull', 't'])
        3600.0
    """
    words = set(args[1:])
    if 'pull' in words or 'manifest' in words:
        return DOCKER_TIMEOUT_PULL
    if 'up' in words or 'down' in words:
        return DOCKER_TIMEOUT_CONVERGE
    if words & {'stop', 'rm', 'start', 'unpause', 'exec', 'kill'}:
        return DOCKER_TIMEOUT_LIFECYCLE
    return DOCKER_TIMEOUT_QUERY


#: Caller environment variables Docker commands may inherit. Everything else,
#: notably ``HF_TOKEN`` and ``DOCKER_HOST``, is dropped: a variable exported in
#: one caller's shell must not change what Compose interpolates
#: (``${HF_TOKEN:-}`` would otherwise override the managed ``.env``) or which
#: daemon a recovery talks to. ``DOCKER_CONTEXT`` is then forced to
#: ``default``, because ``$HOME/.docker/config.json`` can select another.
DOCKER_ENV_ALLOWLIST = (
    'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR',
    'XDG_RUNTIME_DIR', 'TERM',
)


def docker_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The explicit process environment for every Docker command infer-stack runs.

    Example:
        >>> env = docker_environment({'PATH': '/bin', 'HF_TOKEN': 'x', 'DOCKER_HOST': 'tcp://y',
        ...                           'DOCKER_CONTEXT': 'remote'})
        >>> env
        {'PATH': '/bin', 'DOCKER_CONTEXT': 'default'}
    """
    import os

    source = os.environ if environ is None else environ
    env = {k: source[k] for k in DOCKER_ENV_ALLOWLIST if k in source}
    env['DOCKER_CONTEXT'] = 'default'
    return env


def _default_docker_run(
    args: list[str], *, timeout: float | None = None,
    stderr_lines: Callable[[str], None] | None = None,
    stdout_lines: Callable[[str], None] | None = None,
) -> str:
    """Run a docker command and return stdout, bounded in wall-clock time.

    Same contract as ``subprocess.check_output(args, text=True)`` -- stdout is
    returned, stderr is inherited, a non-zero exit raises ``CalledProcessError``
    -- plus a time bound. The command runs in its own process group so that, on
    timeout, the whole group (``docker compose`` spawns children) is killed and
    :class:`~infer_stack.leasing.backend.BackendTimeout` is raised.
    """
    import os
    import signal
    import subprocess

    from .backend import BackendTimeout

    bound = _docker_timeout(args) if timeout is None else timeout
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, text=True, start_new_session=True,
        env=docker_environment(),
        stderr=subprocess.PIPE if stderr_lines is not None else None,
    )
    def kill_group():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()

    try:
        if stdout_lines is None:
            out, err = proc.communicate(timeout=bound)
        else:
            out, err = _communicate_streaming(proc, bound, stdout_lines)
    except subprocess.TimeoutExpired:
        kill_group()
        raise BackendTimeout(
            f'{" ".join(args)} did not finish within {bound:g}s and was killed; '
            'runtime state is unknown until observed again'
        ) from None
    except BaseException:
        # Ctrl-C reaches only our process group; the command runs in its own
        # session, so without this it would keep running unattended.
        kill_group()
        raise
    if stderr_lines is not None:
        for line in (err or '').splitlines():
            if line.strip():
                stderr_lines(line.rstrip())
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, output=out, stderr=err)
    return out


def _communicate_streaming(proc, bound: float, on_line) -> tuple[str, str | None]:
    """``proc.communicate(timeout=bound)``, handing each stdout line to ``on_line``.

    The callback runs on a reader thread and must not raise; a progress
    report that fails is not a reason to fail the command.
    """
    import threading

    lines: list[str] = []

    def read():
        for line in proc.stdout:
            lines.append(line)
            try:
                on_line(line.rstrip('\n'))
            except Exception:  # noqa: BLE001
                pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    proc.wait(timeout=bound)
    reader.join(timeout=5.0)
    err = proc.stderr.read() if proc.stderr is not None else None
    return ''.join(lines), err


class PullProgress:
    """Turn non-TTY ``docker pull`` output into "N of M layers, X of Y GB".

    A pull that is not on a terminal prints one line per layer event
    (``<id>: Pulling fs layer`` ... ``Download complete`` ... ``Pull
    complete``) and no byte counts. Sizes come from the registry manifest,
    keyed by the same 12-character digest prefix the pull prints; without
    them the report is layers only.

    Example:
        >>> p = PullProgress('img', {'aaaaaaaaaaaa': 2 * 10**9, 'bbbbbbbbbbbb': 10**9})
        >>> p.feed('aaaaaaaaaaaa: Pulling fs layer')
        >>> p.feed('bbbbbbbbbbbb: Already exists')
        'pulling img: 1 of 2 layers downloaded, 1.0 GB of 3.0 GB'
        >>> p.feed('cccccccccccc: Download complete')   # not in the manifest: ignored
        >>> p.feed('aaaaaaaaaaaa: Download complete')
        'pulling img: 2 of 2 layers downloaded, 3.0 GB of 3.0 GB'
    """

    _DONE = ('Download complete', 'Pull complete', 'Already exists')

    def __init__(self, image: str, sizes: dict[str, int] | None = None):
        self.image = image
        self.sizes = dict(sizes or {})
        self.layers: set[str] = set(self.sizes)
        self.downloaded: set[str] = set()

    def feed(self, line: str) -> str | None:
        """A new report when a layer finished downloading, else ``None``."""
        layer, sep, event = line.partition(': ')
        if not sep or len(layer) != 12 or any(c not in '0123456789abcdef' for c in layer):
            return None
        if self.sizes and layer not in self.sizes:
            return None             # an attestation blob, not an image layer
        self.layers.add(layer)
        if event.strip() in self._DONE and layer not in self.downloaded:
            self.downloaded.add(layer)
            return self.report()
        return None

    def report(self) -> str:
        text = (f'pulling {self.image}: {len(self.downloaded)} of '
                f'{len(self.layers)} layers downloaded')
        if self.sizes:
            done = sum(self.sizes.get(layer, 0) for layer in self.downloaded)
            text += f', {_size(done)} of {_size(sum(self.sizes.values()))}'
        return text


def _size(n: int) -> str:
    return f'{n / 1e9:.1f} GB' if n >= 1e9 else f'{n / 1e6:.0f} MB'


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
        run: Callable[..., str] | None = None,
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
        clock: Callable[[], float] = time.monotonic,
    ):
        self.state_dir = Path(state_dir)
        # Constructing a backend must not touch the filesystem fatally. `doctor`
        # builds one before it runs a single check, so an unwritable state dir
        # used to kill the very command whose job is to say "the state dir is
        # unwritable" -- the preflight required the thing it was preflighting.
        # Record the failure instead and let doctor() report it; anything that
        # actually writes calls _ensure_state_dir() and gets a real error.
        self._state_dir_error: str | None = None
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            self._state_dir_error = str(ex)
        # None => detect lazily on first use (see the `inventory` property):
        # startup paths (notably the TUI) must never wait on nvidia-smi
        # before the first frame.
        self._inventory = inventory
        self.run = run or _default_docker_run
        if http is None:
            import requests
            http = requests
        self.http = http
        # Called with a one-line message during long steps (image pulls); the
        # TUI sets it. Always also logged.
        self.progress: Callable[[str], None] | None = None
        # alias -> UPSTREAM_ROUTE row, set by a backend that uses this one only
        # as its gateway (the kubeai backend). Merged into the route registry.
        self.upstream_routes: dict[str, dict[str, Any]] = {}
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
        # Set by use_profile(): the published proxy config content.
        self._profile_proxy_text: str | None = None
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
        self._clock = clock
        self.last_errors: list[str] = []
        self.last_unplaced: set[str] = set()  # desired deployment ids placement skipped
        self.last_assignments: dict[str, list[int]] = {}  # deployment id -> GPU ids
        self.last_displaced: list[str] = []  # optional residents that yielded
        self.last_degraded: list[str] = []   # invalid committed allocations
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

    def _ensure_state_dir(self) -> None:
        """Raise a legible error if the state dir could not be created.

        The constructor records rather than raises (see ``__init__``), so the
        failure has to resurface at the first real write instead of turning
        into a confusing downstream error about a missing compose file.
        """
        if self._state_dir_error is None:
            return
        raise RuntimeError(
            f'infer-stack state dir is not usable: {self._state_dir_error}\n'
            f'  path: {self.state_dir}\n'
            '  Run `infer-stack doctor` for the full preflight, and check '
            'that the data dir from `infer-stack paths` is writable by you.'
        )

    # -- preflight -----------------------------------------------------------

    def doctor(self) -> list[tuple[str, bool, str]]:
        """Preflight everything ``acquire`` needs: ``(check, ok, detail)`` rows.

        Checked cheaply and in dependency order so a fresh host fails as a
        checklist instead of a mid-converge traceback, and — critically —
        *before* anything is placed or any lease is recorded. Never raises.

        The image check is the one that earns its keep. A converge brings up
        every enabled service at once, so an unpullable image belonging to a
        service that has nothing to do with serving (the Open WebUI frontend,
        say) aborts a model lease *after* placement has already succeeded, and
        the ledger is left describing containers that do not exist. Resolving
        the tags first turns that into one line here.
        """
        checks: list[tuple[str, bool, str]] = []

        # 1. State dir. Everything else writes here, so it comes first.
        if self._state_dir_error is None:
            probe = self.state_dir / '.doctor-write-probe'
            try:
                probe.touch()
                probe.unlink()
                checks.append(('state dir writable', True, str(self.state_dir)))
            except OSError as ex:
                checks.append((
                    'state dir writable', False,
                    f'{self.state_dir}: {ex} — chown/chgrp it to you; on a '
                    'shared box use a group with the setgid bit so the ledger '
                    'stays shared (that is how concurrent runs avoid placing '
                    'two models on one GPU)',
                ))
        else:
            checks.append((
                'state dir writable', False,
                f'{self.state_dir}: {self._state_dir_error} — create it and '
                'make it writable by you',
            ))

        # 2. Docker daemon, then compose. No point checking images without them.
        try:
            ver = self.run(['docker', 'version', '--format', '{{.Server.Version}}']).strip()
            checks.append(('docker daemon reachable', True, f'server {ver}'))
        except Exception as ex:  # noqa: BLE001 - report, don't raise
            checks.append((
                'docker daemon reachable', False,
                f'{ex} — is dockerd running, and are you in the docker group? '
                '(a fresh group membership needs a new login session)',
            ))
            return checks

        try:
            cver = self.run(['docker', 'compose', 'version', '--short']).strip()
            checks.append(('docker compose available', True, f'v{cver}'))
        except Exception as ex:  # noqa: BLE001
            checks.append((
                'docker compose available', False,
                f'{ex} — the compose v2 plugin is required (docker-compose v1 '
                'is not enough)',
            ))
            return checks

        # 3. Every image this configuration would actually bring up. Checking
        #    the full PINNED_IMAGES set would fail on services that are off.
        wanted: dict[str, str] = {'vllm': self.images['vllm']}
        if self.litellm:
            wanted['litellm'] = self.images['litellm']
            if self.dynamic_routing:
                wanted['postgres'] = self.images['postgres']
        if self.ui:
            wanted['open_webui'] = self.images['open_webui']
        if self.reverse_proxy:
            wanted['nginx'] = self.images['nginx']

        for role, image in sorted(wanted.items()):
            try:
                # `images -q` prints the id or nothing and always exits 0.
                # `image inspect` would work too but writes "No such image" to
                # stderr, which is alarming noise in a preflight that is about
                # to report the same fact calmly.
                if self.run(['docker', 'images', '-q', image]).strip():
                    checks.append((f'image {role}', True, f'{image} (local)'))
                    continue
            except Exception:  # noqa: BLE001 - fall through to the remote check
                pass
            try:
                self.run(['docker', 'manifest', 'inspect', image])
                checks.append((f'image {role}', True, f'{image} (pullable)'))
            except Exception as ex:  # noqa: BLE001
                hint = 'not present locally and could not be resolved'
                if 'denied' in str(ex).lower():
                    # A public image answering `denied` almost always means a
                    # stale credential is being sent: docker will not fall back
                    # to anonymous once it has an auth entry for the registry.
                    registry = image.split('/')[0]
                    hint += (
                        f' — `denied` on a public image usually means a stale '
                        f'credential; try `docker logout {registry}`'
                    )
                checks.append((f'image {role}', False, f'{image}: {hint}'))

        # 4. GPUs. Last because a CPU-only stack is legitimate (mock endpoints,
        #    the null backend), so this is informational rather than fatal.
        try:
            gpus = self.inventory.get('gpus') or []
            if gpus:
                checks.append(('GPUs visible', True, f'{len(gpus)} device(s)'))
            else:
                checks.append((
                    'GPUs visible', True,
                    'none detected — fine for mock/simulator endpoints, but a '
                    'real vLLM endpoint will not place',
                ))
        except Exception as ex:  # noqa: BLE001
            checks.append(('GPUs visible', False, f'inventory failed: {ex}'))

        return checks

    def _compose(self, args: list[str]) -> str:
        self._ensure_state_dir()
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

    def plan(self, desired: list[Deployment], placement=None):
        """Compute GPU placement for ``desired`` without writing or applying.

        Read-only and side-effect free: it honors the persisted pins, so the
        result reflects where deployments are (for running ones) or *would* be (for
        not-yet-started ones) placed. ``leases`` uses it to show actual/slated
        GPUs; ``converge`` uses it as the first step of render.
        """
        pinned = self._load_sidecar().get('assignments', {})
        desired = list(desired)
        self._enrich_placement(desired)
        keywords: dict[str, Any] = {}
        if placement is not None:
            # Admission mode: committed allocations and residency decide; the
            # sidecar's pins only keep unresolved LIVE deployments stable.
            keywords = dict(
                required_ids=set(placement.required_ids),
                hard=dict(placement.hard),
                optional_hints=dict(placement.optional_hints),
            )
        return plan_placement(
            desired,
            self.inventory,
            allowed_gpus=self.allowed_gpus,
            reserved=self.reserved,
            pinned=pinned,
            skip_display=self.skip_display,
            **keywords,
        )

    def preview(self, desired: list[Deployment], placement=None, *, approve: bool = False):
        """Place and render ``desired`` exactly as :meth:`converge` would; write nothing.

        Returns ``(plan, rendered)``. Admission uses it to decide, before any
        commit, whether a candidate lease is placeable **and** renderable. With
        ``approve``, the operator is shown the resulting diff now (raising
        :class:`ConvergeAborted` on decline); the render that follows the commit
        then does not ask again as long as it produces the same files.
        """
        docs = self._render_documents(list(desired), placement)
        self.last_preview_digest = self._planned_digest(docs['planned'])
        if approve:
            self._approve_changes(docs['planned'])
            self._preapproved = self.last_preview_digest
        return docs['plan'], docs['rendered']

    def _render_documents(self, desired: list[Deployment], placement) -> dict[str, Any]:
        """Placement, render, generated files and fingerprints, in memory."""
        plan = self.plan(desired, placement)
        if self.litellm and self.dynamic_routing:
            # The DB secret must exist before rendering, so docker compose
            # --env-file can interpolate ${LITELLM_DB_PASSWORD} at apply time.
            self.db_password()
        route_registry = None
        if self.litellm and not self.dynamic_routing:
            # Unconditional in static-superset mode: `self.catalog` may be None;
            # the incoming set is then deployments-only, and the render still
            # comes from the accumulated registry, so a catalog-less converge
            # cannot strip routes or blip.
            route_registry = self._merged_route_registry(desired, plan.assignments)
        rendered = render_compose(
            desired, plan.assignments, images=self.images, ports=self.ports,
            state=self.state, litellm=self.litellm, litellm_port=self.litellm_port,
            litellm_master_key=self.master_key() if self.litellm else None,
            litellm_salt_key=SALT_KEY_ENV in parse_env_file(self._env_path),
            ui=self.ui, ui_port=self.ui_port, reverse_proxy=self.reverse_proxy,
            reverse_proxy_port=self.reverse_proxy_port,
            reverse_proxy_config=self.reverse_proxy_config, aux_dir=self.state_dir,
            project=self.project, catalog=self.catalog,
            route_registry=route_registry, dynamic_routing=self.dynamic_routing,
        )
        addresses = None
        if self.network is not None:
            from .network import allocate, stamp_network

            addresses = allocate(self.network['subnet'], self.network['addresses'],
                                 (rendered.compose.get('services') or {}).keys())
            stamp_network(rendered.compose, self.network['subnet'], addresses)
        planned: dict[Path, str] = {}
        if (
            self.reverse_proxy
            and self._profile_proxy_text is not None
            and self.reverse_proxy_config is not None
        ):
            planned[Path(self.reverse_proxy_config)] = self._profile_proxy_text
        if rendered.litellm_config is not None:
            planned[self.state_dir / LITELLM_CONFIG_FILENAME] = rendered.litellm_config
        if rendered.nginx_config is not None:
            planned[self.state_dir / NGINX_CONFIG_FILENAME] = rendered.nginx_config
        if rendered.litellm_routes is not None:
            planned[self._routes_file] = json.dumps(rendered.litellm_routes, indent=2)
        fingerprints = stamp_fingerprints(
            rendered.compose, files=planned, env_file=self._env_path,
        )
        planned[self.compose_file] = yaml.safe_dump(rendered.compose, sort_keys=False)
        return {'plan': plan, 'rendered': rendered, 'planned': planned,
                'fingerprints': fingerprints, 'route_registry': route_registry,
                'addresses': addresses}

    #: Stable addressing (plan step P7), set by the controller once
    #: `network migrate` has run: ``{'subnet': ..., 'addresses': {service: ip}}``.
    network: dict[str, Any] | None = None
    #: Called with the full address table after an approved render, to persist
    #: newly allocated addresses (append-only).
    on_addresses: Any = None

    @staticmethod
    def _planned_digest(planned: dict) -> str:
        material = json.dumps({str(k): v for k, v in planned.items()}, sort_keys=True)
        return hashlib.sha256(material.encode('utf-8')).hexdigest()

    #: Digest of files an admission preview already had approved.
    _preapproved: str | None = None
    #: Digest of the files the last render produced (approved-digest guard).
    last_planned_digest: str | None = None
    #: Digest of the files the last preview produced.
    last_preview_digest: str | None = None

    def pull_images(self, images) -> list[str]:
        """Pull ``images``; ``config publish`` passes :func:`profile_images`."""
        from .._log import logger

        for image in sorted(set(images)):
            logger.info('docker pull {}', image)
            self.run(['docker', 'pull', image])
        return sorted(set(images))

    def _approve_changes(self, planned: dict) -> None:
        if self._preapproved is not None and self._planned_digest(planned) == self._preapproved:
            self._preapproved = None
            return
        self._preapproved = None
        super()._approve_changes(planned)

    def plan_on_idle_host(self, desired: list[Deployment]):
        """Placement for ``desired`` alone, as if nothing else were running.

        The question this answers is "could this request EVER be satisfied
        here", separate from "is there room right now". Same planner, same
        inventory and allow-list; the difference from :meth:`plan` is which
        pins survive, and that the caller passes only the deployments it is
        asking about. So an unplaced result means the host cannot serve the
        request at all, not that it is busy.

        "Idle" means *free of everything unrelated to this request*, not
        *empty*. Pins are kept for the requested deployments and dropped for
        everything else:

        * An unrelated deployment's pin is contention — precisely what the
          check exists to see past — so it goes.
        * A requested deployment that is ALREADY placed keeps its GPU. Under
          Slurm that GPU may sit outside this call's ``allowed_gpus``: a shared
          extractor another job started is reusable exactly as it is (see
          ``docs/slurm-compatibility.md``, and ``pin_pool_set`` in
          :func:`~infer_stack.leasing.placement.plan_placement`, which
          validates pins against the whole host). Dropping that pin would force
          the extractor back inside our own slice and count it against our
          budget, rejecting a lease that is merely waiting for a GPU to free.

        The distinction matters because the two failures look identical while
        waiting: an admission queue that waits out its timeout for capacity
        that could never exist is indistinguishable, from the outside, from one
        waiting on a GPU that is about to free.
        """
        desired = list(desired)
        requested = {deployment.id for deployment in desired}
        pinned = {
            gid: gpus
            for gid, gpus in self._load_sidecar().get('assignments', {}).items()
            if gid in requested
        }
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

    def startup_failure(self, deployment: Deployment) -> str | None:
        """Diagnose an engine that cannot start, or ``None`` if it may still load.

        ``restart: unless-stopped`` makes a container that exits immediately
        restart forever, and the lenient ``observe()`` only counts *running*
        services -- so a model that can never start looks exactly like one that
        is still loading, and an acquire waits out its whole timeout holding a
        GPU. Docker's bookkeeping says a container crashed; its LOG says whether
        another attempt could ever work (:func:`classify_engine_log`):

        * an unrecoverable error -- a rejected config, an architecture this
          build does not implement, a gated repo, a CUDA OOM -- is fatal on the
          FIRST crash, because the restart reproduces it exactly;
        * a transient one -- an unreachable hub, a truncated download, a port
          still held by the container being replaced -- is never fatal here;
          that is what ``restart: unless-stopped`` is for;
        * an unrecognised crash keeps the blunt budget of
          :data:`CRASH_LOOP_RESTARTS` restarts, so a genuinely stuck engine
          still fails fast without a signature for every possible error.

        The returned string carries the engine's last words, because the cause
        is in its log and nowhere else ("set trust_remote_code=True", a CUDA
        OOM, an unsupported architecture).
        """
        try:
            residency = self.residency()
        except ResidencyUnknown:
            return None                      # cannot read Docker: say nothing
        return diagnose_startup(
            residency.containers(deployment.id),
            lambda: self.deployment_logs(deployment, tail=200),
        )

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

    def _merged_route_registry(
        self, desired: list[Deployment], assignments: dict[str, list[int]]
    ) -> dict[str, Any]:
        """The route registry merged with the catalog and ``desired``, in memory."""
        from .._log import logger

        existing = self._load_route_registry()
        incoming: dict[str, dict[str, Any]] = {}
        if self.catalog is not None:
            incoming.update(_registry_incoming_from_catalog(self.catalog))
        # `desired` spans all runbooks via the shared ledger, so this keeps every
        # live cross-runbook deployment routable (and, via persistence, routable
        # past release).
        incoming.update(_registry_incoming_from_deployments(desired, assignments))
        # Routes to servers another backend runs (see UPSTREAM_ROUTE).
        incoming.update(self.upstream_routes)
        merged, warnings = _merge_route_registry(existing, incoming)
        for w in warnings:
            logger.warning('  route registry: {}', w)
        return merged

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

    def _update_route_registry(
        self, desired: list[Deployment], assignments: dict[str, list[int]]
    ) -> dict[str, Any]:
        """Merge and persist the route registry; return it (kept for callers)."""
        merged = self._merged_route_registry(desired, assignments)
        self._save_route_registry(merged)
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

    def converge(self, desired: list[Deployment], *, apply: bool = True, placement=None):
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
            docs = self._render_documents(desired, placement)
            plan, rendered, planned = docs['plan'], docs['rendered'], docs['planned']
            fingerprints = docs['fingerprints']
            self.last_assignments = dict(plan.assignments)
            self.last_displaced = list(plan.displaced)
            self.last_degraded = list(plan.degraded)
            for gid, gpus in sorted(plan.assignments.items()):
                logger.info('  placed {} on GPU(s) {}', gid, gpus or '(cpu)')
            for err in plan.errors:
                logger.warning('  placement: {}', err)
            for note in plan.warnings:
                # Honored-but-suspect decisions (a pin/explicit index that
                # contradicts a declared min_vram_gib): never fail the plan,
                # never be silent either.
                logger.warning('  placement: {}', note)
            # A deployment the render excluded (service-name collision) is as
            # undeliverable as an unplaced one: fold it into last_unplaced /
            # last_errors so acquire fails loudly and rolls the lease back.
            self.last_errors = list(plan.errors) + list(rendered.errors)
            self.last_unplaced = {
                g.id for g in desired
                if g.id not in plan.assignments and g.id not in plan.displaced
            } | set(rendered.unrenderable)
            for err in rendered.errors:
                logger.warning('  render: {}', err)

            self.last_planned_digest = self._planned_digest(planned)
            self._approve_changes(planned)  # may raise ConvergeAborted
            # Only after approval: persist new addresses and the merged route
            # registry, then the files.
            if docs['addresses'] is not None and self.on_addresses is not None:
                self.on_addresses(docs['addresses'])
            if docs['route_registry'] is not None:
                self._save_route_registry(docs['route_registry'])
            for path, text in planned.items():
                if path != self.compose_file:
                    self._atomic_write(path, text)
            self._atomic_write(self.compose_file, planned[self.compose_file])
            optional = set((placement.optional_hints if placement is not None else {}))
            self._save_sidecar({
                'assignments': plan.assignments,
                'services': rendered.services,
                # Selective apply (see apply()): what each service must look
                # like, which deployments are degraded (never started or
                # removed), and which services are optional residents (kept if
                # present, never started).
                'fingerprints': fingerprints,
                'degraded': list(plan.degraded),
                'displaced': list(plan.displaced),
                'optional_services': sorted(
                    svc for svc, gid in rendered.services.items() if gid in optional
                ),
            })
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

    def apply(self) -> bool:
        """Bring the already-rendered compose project up; return whether it fully succeeded.

        Reads the on-disk compose file last written by :meth:`converge` (render)
        and applies it -- it does NOT re-render. The controller serialises render
        and apply under its host-wide lock, so the file cannot change underneath
        this call. Idempotent: a no-op when reality already matches the file.

        Returns ``False`` when the apply did not fully take effect, so the
        controller keeps the change pending and retries it:

        * the rendered file cannot be read;
        * in dynamic-routing mode, the gateway's routes could not be reconciled
          and verified within budget. The budget is short when the gateway was
          already running (steady state), and long when this apply is bringing
          it up (bootstrap: it waits on Postgres health and runs DB migrations).

        Docker failures and timeouts raise, and leave the change pending too.
        """
        from .._log import logger

        if not self.compose_file.exists():
            return True
        try:
            doc = yaml.safe_load(self.compose_file.read_text()) or {}
        except Exception:  # noqa: BLE001 - reported as "not applied", never raised
            logger.warning('apply: rendered compose file is unreadable; change stays pending')
            return False
        services = doc.get('services') or {}
        sidecar = self._load_sidecar()
        fingerprints = sidecar.get('fingerprints')
        if fingerprints is None or set(fingerprints) != set(services):
            # A render from before fingerprints: re-render (any mutation) first.
            logger.warning('apply: the render predates fingerprints; re-render, then apply')
            return False
        dynamic = bool(self.litellm and self.dynamic_routing)
        outcome = self.selective_apply(
            services, fingerprints,
            degraded=set(sidecar.get('degraded') or ()),
            optional=set(sidecar.get('optional_services') or ()),
            networks=doc.get('networks') or {},
        )
        if dynamic and 'litellm' in services:
            return self._reconcile_routes(
                deadline_s=ROUTE_RECONCILE_STEADY_S if 'litellm' in outcome.kept_services
                else ROUTE_RECONCILE_BOOTSTRAP_S,
            )
        return True

    #: Service-level ownership adopted at migration: container id ->
    #: {service, fingerprint}. Set by the controller from the ledger.
    adopted: dict[str, dict[str, str]] = {}

    def _wait_until(self, predicate, *, deadline_s: float, what: str, interval: float = 1.0):
        """Poll strict residency until ``predicate(snapshot)``; abort at the deadline."""
        deadline = self._clock() + deadline_s
        while True:
            if predicate(self.residency()):
                return
            if self._clock() + interval > deadline:
                raise ApplyAborted(f'timed out after {deadline_s:g}s waiting for {what}')
            self._sleep(interval)

    def _network_state(self, name: str) -> tuple[str | None, list[str]] | None:
        """``(subnet, attached container ids)`` of a Docker network, or ``None`` if absent."""
        found = (self.run(['docker', 'network', 'ls', '-q', '--filter', f'name=^{name}$'])
                 or '').split()
        if not found:
            return None
        info = json.loads(self.run(['docker', 'network', 'inspect', name]) or '[]')
        item = info[0] if info else {}
        configs = ((item.get('IPAM') or {}).get('Config') or [])
        subnet = next((c.get('Subnet') for c in configs if c.get('Subnet')), None)
        return subnet, sorted((item.get('Containers') or {}).keys())

    def _reconcile_network(self, networks: dict[str, Any]) -> None:
        """Recreate the fixed network when its actual subnet differs from the render.

        Runs after departing containers are confirmed gone. Compose never
        changes an existing network's IPAM, so a subnet migration must remove
        the old network first; any container still attached (unmanaged, or a
        degraded deployment's) blocks it. Checked on every apply, so an
        interrupted migration is completed by the next one.
        """
        from .._log import logger
        from .network import NETWORK_NAME

        spec = networks.get(NETWORK_NAME)
        if not spec:
            return
        wanted = next((c.get('subnet') for c in (spec.get('ipam') or {}).get('config') or []), None)
        state = self._network_state(NETWORK_NAME)
        if state is None or wanted is None or state[0] == wanted:
            return
        actual, attached = state
        if attached:
            raise ApplyAborted(
                f'network {NETWORK_NAME} must move from {actual} to {wanted}, but '
                f'{len(attached)} container(s) are still attached '
                f'({", ".join(c[:12] for c in attached)}); remove them first'
            )
        logger.info('apply: recreating network {} ({} -> {})', NETWORK_NAME, actual, wanted)
        self.run(['docker', 'network', 'rm', NETWORK_NAME])

    def _report(self, message: str) -> None:
        """Progress for a long step: the log, and ``self.progress`` if set (the TUI)."""
        from .._log import logger

        logger.info(message)
        if self.progress is not None:
            try:
                self.progress(message)
            except Exception:  # noqa: BLE001 - a display failure must not fail an apply
                pass

    def _image_present(self, image: str) -> bool:
        import subprocess

        args = ['docker', 'image', 'inspect', '--format', '{{.Id}}', image]
        try:
            try:
                # Capture "No such image" rather than let it reach the terminal
                # (under the TUI it would draw over the screen).
                self.run(args, stderr_lines=lambda _line: None)
            except TypeError:           # an injected runner without the keyword
                self.run(args)
            return True
        except subprocess.CalledProcessError:
            return False

    def _layer_sizes(self, image: str) -> dict[str, int]:
        """Layer sizes by 12-char digest prefix, for this host's platform. Best effort."""
        import platform

        arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine(),
                                                           platform.machine())
        try:
            data = json.loads(self.run(['docker', 'manifest', 'inspect', '-v', image],
                                       timeout=30.0) or 'null')
        except Exception:  # noqa: BLE001 - sizes only improve the report
            return {}
        for entry in data if isinstance(data, list) else [data]:
            if not isinstance(entry, dict):
                continue
            plat = (entry.get('Descriptor') or {}).get('platform') or {}
            if plat and (plat.get('os'), plat.get('architecture')) != ('linux', arch):
                continue
            manifest = entry.get('OCIManifest') or entry.get('SchemaV2Manifest') or {}
            layers = manifest.get('layers') or []
            if layers:
                return {str(l['digest']).split(':', 1)[-1][:12]: int(l.get('size') or 0)
                        for l in layers if l.get('digest')}
        return {}

    def _pull_missing(self, images) -> None:
        """Pull every image not present locally, reporting as layers arrive.

        ``docker compose up`` would pull them itself, silently: on a first
        use that is many minutes of an apply that looks hung. Pulling first
        also means a failed pull aborts before anything is removed.
        """
        for image in sorted({str(i) for i in images if i}):
            if self._image_present(image):
                continue
            progress = PullProgress(image, self._layer_sizes(image))
            size = f' ({_size(sum(progress.sizes.values()))})' if progress.sizes else ''
            self._report(f'pulling {image}{size}: not present locally')

            def on_line(line, progress=progress):
                message = progress.feed(line)
                if message:
                    self._report(message)

            self.run(['docker', 'pull', image], timeout=DOCKER_TIMEOUT_PULL,
                     stdout_lines=on_line)
            self._report(f'pulled {image}')

    def selective_apply(self, services, fingerprints, *, degraded=frozenset(),
                        optional=frozenset(), networks=None):
        """Make the project match the render, touching only what differs.

        * **keep** a managed container whose (service, fingerprint) is wanted,
          is the only one with that key, and is running, restarting or paused;
        * **remove** other managed containers, except those of degraded
          deployments;
        * **report** unmanaged containers (orphans), never remove them;
        * **start** each service with no kept container, except optional
          residents, which are never started;
        * **barrier**: nothing starts on a GPU still occupied by another
          container. A managed occupant is removed first; an unmanaged or
          degraded one aborts the apply.

        Services start with ``up -d --no-deps``, dependency level by level, so
        a fresh dynamic stack brings Postgres up before the gateway. Raises
        :class:`ApplyAborted` (the change stays pending) on ambiguity, a
        blocked GPU, or a container still being removed.
        """
        from .._log import logger

        res = self.residency()
        adopted = dict(self.adopted or {})
        wanted = {name: fingerprints[name] for name in services}

        def key(c):
            if c.labelled:
                return (c.service, c.fingerprint)
            if c.container_id in adopted:
                info = adopted[c.container_id]
                return (info['service'], info['fingerprint'])
            return None

        containers = res.all_containers()
        managed = [c for c in containers if key(c) is not None]
        orphans = [c for c in containers if key(c) is None]
        by_key: dict[tuple, list] = {}
        for c in managed:
            by_key.setdefault(key(c), []).append(c)
        for name, fp in wanted.items():
            if len(by_key.get((name, fp), ())) > 1:
                raise ApplyAborted(
                    f'service {name!r} has {len(by_key[(name, fp)])} containers with its '
                    'wanted configuration; refusing to guess which one serves'
                )
        keep = [
            c for c in managed
            if wanted.get(key(c)[0]) == key(c)[1]
            and c.state in {'running', 'restarting', 'paused'}
        ]
        kept_ids = {c.container_id for c in keep}
        kept_services = {key(c)[0] for c in keep}
        departing = [
            c for c in managed
            if c.container_id not in kept_ids and c.deployment_id not in degraded
        ]
        for c in departing:
            if c.state == 'removing':
                raise ApplyAborted(
                    f'container {c.container_id[:12]} ({key(c)[0]}) is still being '
                    'removed; retry when it is gone'
                )
        to_start = [
            name for name in services
            if name not in kept_services and name not in optional
        ]
        departing_ids = {c.container_id for c in departing}
        for name in to_start:
            for gpu in _stanza_gpus(services[name]):
                for c in res.occupants(gpu):
                    if c.container_id in kept_ids or c.container_id in departing_ids:
                        continue
                    who = 'an unmanaged' if key(c) is None else 'a degraded'
                    raise ApplyAborted(
                        f'{who} container {c.container_id[:12]} occupies GPU {gpu} '
                        f'needed by {name!r}' + (
                            '; see `infer-stack gc --orphans`' if key(c) is None else '')
                    )
                for c in res.occupants(gpu):
                    if c.container_id in kept_ids and key(c)[0] != name:
                        raise ApplyAborted(
                            f'GPU {gpu} is held by kept service {key(c)[0]!r} but '
                            f'rendered for {name!r}'
                        )
        for name in to_start:
            address = ((services[name].get('networks') or {}).get(_network_name()) or {}).get('ipv4_address')
            if not address:
                continue
            for c in containers:
                if address in c.ips and key(c) is None:
                    raise ApplyAborted(
                        f'an unmanaged container {c.container_id[:12]} holds address '
                        f'{address} of {name!r}; see `infer-stack gc --orphans`'
                    )
        if networks:
            # Preflight a subnet change BEFORE removing anything: a foreign
            # attachment would otherwise take the stack down and only then abort.
            from .network import NETWORK_NAME

            spec = networks.get(NETWORK_NAME) or {}
            wanted_subnet = next((c.get('subnet') for c in
                                  (spec.get('ipam') or {}).get('config') or []), None)
            state = self._network_state(NETWORK_NAME) if wanted_subnet else None
            if state is not None and state[0] != wanted_subnet:
                foreign = [cid for cid in state[1] if cid not in departing_ids]
                if foreign:
                    raise ApplyAborted(
                        f'network {NETWORK_NAME} must move from {state[0]} to '
                        f'{wanted_subnet}, but {len(foreign)} container(s) not managed '
                        f'for removal are attached ({", ".join(c[:12] for c in foreign)})'
                    )
        self._pull_missing(services[name].get('image') for name in to_start)
        if orphans:
            logger.warning(
                'apply: {} unmanaged container(s) in the project left alone ({}); '
                '`infer-stack gc --orphans` removes them',
                len(orphans), ', '.join(c.container_id[:12] for c in orphans),
            )
        if departing:
            logger.info('apply: removing {} container(s): {}', len(departing),
                        ', '.join(f'{key(c)[0]}' for c in departing))
            self.run(['docker', 'rm', '-f', *[c.container_id for c in departing]])
            # The barrier: confirm they are really gone before anything starts
            # on their GPUs or addresses.
            self._wait_until(
                lambda snap: not ({c.container_id for c in snap.all_containers()}
                                  & departing_ids),
                deadline_s=APPLY_REMOVAL_WAIT_S,
                what='removed containers to disappear',
            )
        if networks:
            self._reconcile_network(networks)
        paused = [c for c in keep if c.state == 'paused' and key(c)[0] not in optional]
        if paused:
            self.run(['docker', 'unpause', *[c.container_id for c in paused]])
        for level in _dependency_levels(services, to_start):
            # `--no-deps` keeps unrelated services untouched but also skips
            # Compose's `condition: service_healthy`, so wait for it here.
            healthy_deps = sorted({
                dep for name in level
                for dep, cond in _depends_on(services[name]).items()
                if cond.get('condition') == 'service_healthy'
            })
            if healthy_deps:
                logger.info('apply: waiting for {} to be healthy', ', '.join(healthy_deps))
                self._wait_until(
                    lambda snap, deps=healthy_deps: all(
                        any(c.service == dep and c.health == 'healthy'
                            for c in snap.all_containers())
                        for dep in deps),
                    deadline_s=APPLY_HEALTH_WAIT_S,
                    what=f'{", ".join(healthy_deps)} to become healthy',
                )
            logger.info('docker compose up -d --no-deps {}', ' '.join(level))
            self._compose(['up', '-d', '--no-deps', *level])
        return SelectiveApplyOutcome(
            kept_services=kept_services, removed=[c.container_id for c in departing],
            started=list(to_start), orphans=[c.container_id for c in orphans],
        )

    def _service_running(self, service: str) -> bool:
        """Whether a compose service of this project is running (best-effort)."""
        try:
            return service in _parse_ps(self._compose(['ps', '--format', 'json']))
        except Exception:  # noqa: BLE001 - unknown -> treat as a bootstrap (long budget)
            return False

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

    # -- published profile (see leasing/profile.py) ---------------------------

    REVERSE_PROXY_SNAPSHOT = 'reverse-proxy.conf'

    def render_profile(self) -> dict[str, Any]:
        """This backend's current render inputs, as a publishable profile.

        ``allowed_gpus`` is deliberately absent: it is per-caller admission
        scope, not host configuration. A BYO reverse-proxy config is captured
        by content, so editing the file later cannot change a recovery.
        """
        from .profile import PROFILE_VERSION, catalog_sources

        config_text = None
        if self.reverse_proxy_config:
            if self._profile_proxy_text is not None:
                config_text = self._profile_proxy_text
            else:
                path = Path(self.reverse_proxy_config).expanduser()
                try:
                    config_text = path.read_text()
                except OSError as ex:
                    raise RuntimeError(
                        f'reverse-proxy config {path} is unreadable: {ex}'
                    ) from ex
        return {
            'version': PROFILE_VERSION,
            'backend': 'compose',
            'project': self.project,
            'litellm': bool(self.litellm),
            'ui': bool(self.ui),
            'dynamic_routing': bool(self.dynamic_routing),
            'skip_display': bool(self.skip_display),
            'reverse_proxy': {
                'enabled': bool(self.reverse_proxy),
                'port': int(self.reverse_proxy_port),
                'config_text': config_text,
            },
            'images': dict(sorted(self.images.items())),
            'ports': dict(sorted(self.ports.items())),
            'state': dict(sorted(self.state.items())),
            'catalogs': catalog_sources(self.catalog),
        }

    def use_profile(self, profile: dict[str, Any]) -> None:
        """Render from ``profile`` from now on, instead of this process's settings."""
        from .profile import CatalogUnion

        if profile.get('backend') != 'compose':
            from .profile import ProfileMismatch

            raise ProfileMismatch(
                f"the active recovery snapshot is for the {profile.get('backend')!r} backend; "
                'tear down the old backend before switching backend kinds'
            )
        self.project = profile['project']
        self.litellm = profile['litellm']
        self.ui = profile['ui']
        self.dynamic_routing = profile['dynamic_routing']
        self.skip_display = profile['skip_display']
        proxy = profile['reverse_proxy']
        self.reverse_proxy = proxy['enabled']
        self.reverse_proxy_port = proxy['port']
        self._profile_proxy_text = proxy.get('config_text')
        if self._profile_proxy_text is not None:
            # Applying a candidate profile must be pure: config-publish preview
            # can call use_profile() and then be declined or crash.  Point the
            # render at the stable managed path now; _render_documents() adds
            # the bytes to its planned files so converge writes them only after
            # approval.
            self.reverse_proxy_config = str(
                self.state_dir / self.REVERSE_PROXY_SNAPSHOT
            )
        else:
            self.reverse_proxy_config = None
        self.images = dict(profile['images'])
        self.ports = dict(profile['ports'])
        self.state = dict(profile['state'])
        sources = profile.get('catalogs') or []
        self.catalog = CatalogUnion.from_sources(sources) if sources else None

    def placement_context(self) -> dict[str, Any] | None:
        """This caller's admission scope, stored with a pending acquire."""
        if self.allowed_gpus is None:
            return None
        return {'allowed_gpus': list(self.allowed_gpus)}

    def placement_scope(self, context: dict[str, Any] | None):
        """Temporarily render with another caller's admission scope."""
        import contextlib

        @contextlib.contextmanager
        def scope():
            saved = self.allowed_gpus
            if context and 'allowed_gpus' in context:
                self.allowed_gpus = context['allowed_gpus']
            try:
                yield
            finally:
                self.allowed_gpus = saved

        return scope()

    def validate_requests(self, requests) -> None:
        """Refuse requests the published catalog union does not define identically."""
        from .profile import validate_requests_against

        validate_requests_against(self.catalog, requests)

    def upstream_check(self) -> dict[str, dict[str, Any]]:
        """Probe each model upstream by name from inside the gateway's network.

        Returns ``{service: {'deployment', 'expected', 'status', 'answer'}}``
        with status ``healthy``, ``not-ready`` or ``routing-fault`` (the name
        answers, but with another model: the misroute signature). The gateway
        image has ``python3`` and no ``curl``.
        """
        from .network import UPSTREAM_CHECK_SCRIPT, classify_upstream

        doc = yaml.safe_load(self.compose_file.read_text()) if self.compose_file.exists() else {}
        services = (doc or {}).get('services') or {}
        by_service = self._load_sidecar().get('services') or {}
        out: dict[str, dict[str, Any]] = {}
        for name, gid in sorted(by_service.items()):
            svc = services.get(name) or {}
            expected = next((a.split('=', 1)[1] for a in svc.get('command') or []
                             if str(a).startswith('--served-model-name=')), None)
            if expected is None:
                continue
            url = f'http://{name}:{VLLM_CONTAINER_PORT}/v1/models'
            try:
                raw = self._compose(['exec', '-T', LITELLM_SERVICE, 'python3', '-c',
                                     UPSTREAM_CHECK_SCRIPT, url])
                answer = json.loads(raw.strip().splitlines()[-1])
            except Exception as ex:  # noqa: BLE001 - reported, never raised
                answer = {'error': str(ex)}
            out[name] = {'deployment': gid, 'expected': expected,
                         'status': classify_upstream(expected, answer), 'answer': answer}
        return out

    def settle_snapshot(self) -> tuple[tuple[str, str], ...]:
        """Every container of this Compose project as sorted ``(id, state)`` pairs.

        Used after an interrupted apply to wait until the daemon has finished
        work a killed client started. Covers infrastructure (gateway, database)
        as well as deployments. Raises ``ResidencyUnknown`` if Docker cannot be
        read.
        """
        try:
            out = self.run([
                'docker', 'ps', '-a', '--no-trunc',
                '--filter', f'label={COMPOSE_PROJECT_LABEL}={self.project}',
                '--format', '{{.ID}} {{.State}}',
            ])
        except Exception as ex:  # noqa: BLE001 - unknown, never "empty"
            raise ResidencyUnknown(f'docker ps failed: {ex}') from ex
        pairs = []
        for line in (out or '').splitlines():
            parts = line.split()
            if len(parts) != 2:
                raise ResidencyUnknown(f'unexpected docker ps line: {line!r}')
            pairs.append((parts[0], parts[1].lower()))
        return tuple(sorted(pairs))

    def residency(self) -> Residency:
        """Strict snapshot of this project's deployment containers and their GPUs.

        Unlike :meth:`observe`, this never reports "nothing" for "could not look":
        any Docker error, or output that cannot be parsed, raises
        :class:`~infer_stack.leasing.residency.ResidencyUnknown`. Containers are
        found by label (this Compose project and ``infer-stack.deployment``), in
        every state, so a container absent from the current render or sidecar is
        still seen. GPUs come from each container's device reservation. See
        :mod:`infer_stack.leasing.residency` for the ambiguity rules.

        A container removed between the listing and the inspect makes the inspect
        fail, which is reported as unknown; the caller retries.
        """
        try:
            listing = self.run([
                'docker', 'ps', '-a', '--no-trunc',
                '--filter', f'label={COMPOSE_PROJECT_LABEL}={self.project}',
                '--format', '{{.ID}}',
            ])
        except Exception as ex:  # noqa: BLE001 - any failure is "unknown", never "empty"
            raise ResidencyUnknown(f'docker ps failed: {ex}') from ex
        ids = [line.strip() for line in (listing or '').splitlines() if line.strip()]
        if not ids:
            return Residency({})
        try:
            raw = self.run(['docker', 'inspect', *ids])
        except Exception as ex:  # noqa: BLE001 - see above
            raise ResidencyUnknown(f'docker inspect failed: {ex}') from ex
        return residency_from_inspect(raw, project=self.project)

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
            failure = self.startup_failure(deployment)
            if failure is not None:
                return Readiness(False, failure, fatal=True)
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
