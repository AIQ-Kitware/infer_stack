"""Focused Compose backend for the leasing model.

Renders a docker-compose project straight from the live set of
:class:`DeploymentGroup` s — not the legacy resolved-deployment schema — using
the placement planner for GPU assignment and reusing ``profile_runtime.vllm_args``
for the vLLM CLI flags. It *converges the whole union* on every reconcile:
render the file, then ``docker compose up -d --remove-orphans``. Adding or
removing a group re-renders and converges; pinned placement (persisted in a
sidecar) keeps already-running models on their GPUs, and ``--remove-orphans``
tears down services whose group is gone.

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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

from ..config import DEFAULT_PORTS, PINNED_IMAGES, default_state_paths
from ..env_utils import ensure_secret, parse_env_file, write_env_file
from ..probe import openai_ready
from ..profile_runtime import vllm_args
from .backend import Readiness
from .models import DeploymentGroup
from .placement import plan_placement

LEASING_PROJECT = 'infer-stack'  # docker compose project name for leased stacks
VLLM_HOST_PORT_BASE = 18000
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

GROUP_LABEL = 'infer-stack.group'
ENGINE_LABEL = 'infer-stack.engine'


@dataclass
class RenderedCompose:
    compose: dict[str, Any]
    services: dict[str, str] = field(default_factory=dict)  # service -> group id
    litellm_config: str | None = None


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


def _vllm_service_dict(group: DeploymentGroup) -> dict[str, Any]:
    """Build the dict ``vllm_args`` consumes from a group's runtime spec."""
    runtime = group.spec.get('runtime', {}) or {}
    served = group.spec.get('served_model_name') or (
        sorted(group.served)[0] if group.served else group.id
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
    group: DeploymentGroup,
    gpus: list[int],
    host_port: int,
    images: dict[str, str],
    state: dict[str, str],
) -> dict[str, Any]:
    svc = _vllm_service_dict(group)
    command = [
        group.spec['hf_model_id'],
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
        'labels': {GROUP_LABEL: group.id, ENGINE_LABEL: 'vllm'},
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
    group: DeploymentGroup,
    gpus: list[int],
    host_port: int,
    images: dict[str, str],
    state: dict[str, str],
) -> dict[str, Any]:
    settings = group.spec.get('settings', {}) or {}
    env: dict[str, str] = {}
    if settings.get('keep_alive'):
        env['OLLAMA_KEEP_ALIVE'] = str(settings['keep_alive'])
    if settings.get('num_parallel') is not None:
        env['OLLAMA_NUM_PARALLEL'] = str(settings['num_parallel'])
    if settings.get('max_loaded_models') is not None:
        env['OLLAMA_MAX_LOADED_MODELS'] = str(settings['max_loaded_models'])
    if settings.get('context_length') is not None:
        env['OLLAMA_CONTEXT_LENGTH'] = str(settings['context_length'])
    if gpus:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpus)
    service: dict[str, Any] = {
        'image': group.spec.get('image') or images['ollama'],
        'ports': [f'{host_port}:11434'],
        'environment': env,
        'volumes': [f'{state["ollama"]}:/root/.ollama'],
        'restart': 'unless-stopped',
        'labels': {GROUP_LABEL: group.id, ENGINE_LABEL: 'ollama'},
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
    groups: list[DeploymentGroup], assignments: dict[str, list[int]]
) -> list[dict[str, Any]]:
    """One LiteLLM ``model_list`` entry per served endpoint alias."""
    entries: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda g: (g.created_at, g.id)):
        if group.id not in assignments:
            continue
        if group.engine == 'vllm':
            served = group.spec.get('served_model_name') or group.id
            api_base = f'http://vllm-{group.id}:8000/v1'
            for endpoint in sorted(group.served):
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
        elif group.engine == 'ollama':
            api_base = f'http://ollama-{group.id}:11434'
            for endpoint, payload in sorted(group.served.items()):
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


def _litellm_service(
    service_names: list[str],
    host_port: int,
    images: dict[str, str],
    aux_dir: str,
    master_key: str | None = None,
) -> dict[str, Any]:
    # Bake the managed key in literally (not ${...}) so the container and the
    # readiness probe always agree regardless of the caller's shell env.
    key_value = (
        master_key
        if master_key is not None
        else '${' + API_KEY_ENV + ':-sk-local}'
    )
    return {
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
        'depends_on': sorted(service_names),
        'restart': 'unless-stopped',
        'labels': {ENGINE_LABEL: 'litellm'},
    }


def render_compose(
    groups: list[DeploymentGroup],
    assignments: dict[str, list[int]],
    *,
    images: dict[str, str],
    ports: dict[str, int],
    state: dict[str, str],
    litellm: bool = False,
    litellm_port: int = 14042,
    litellm_master_key: str | None = None,
    aux_dir: str | Path | None = None,
) -> RenderedCompose:
    """Render a compose project for the placed groups.

    Groups absent from ``assignments`` (placement failures) are skipped. When
    ``litellm`` is set, a front-door service + config is added so every endpoint
    alias is reachable at one ``base_url``.
    """
    services: dict[str, Any] = {}
    service_map: dict[str, str] = {}
    ordered = sorted(groups, key=lambda g: (g.created_at, g.id))
    vllm_i = 0
    ollama_i = 0
    for group in ordered:
        if group.id not in assignments:
            continue
        gpus = assignments[group.id]
        if group.engine == 'vllm':
            name = f'vllm-{group.id}'
            port = VLLM_HOST_PORT_BASE + vllm_i
            vllm_i += 1
            services[name] = _vllm_service(group, gpus, port, images, state)
        elif group.engine == 'ollama':
            name = f'ollama-{group.id}'
            port = ports.get('ollama', DEFAULT_PORTS['ollama']) + ollama_i
            ollama_i += 1
            services[name] = _ollama_service(group, gpus, port, images, state)
        else:
            continue
        service_map[name] = group.id

    litellm_config = None
    if litellm and service_map:
        entries = _litellm_model_list(groups, assignments)
        litellm_config = yaml.safe_dump(
            {
                'model_list': entries,
                'general_settings': {
                    'master_key': f'os.environ/{API_KEY_ENV}'
                },
            },
            sort_keys=False,
        )
        services[LITELLM_SERVICE] = _litellm_service(
            list(service_map),
            litellm_port,
            images,
            str(aux_dir or '.'),
            master_key=litellm_master_key,
        )
    return RenderedCompose(
        compose={'services': services},
        services=service_map,
        litellm_config=litellm_config,
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
        skip_display: bool = True,
        litellm: bool = True,
        require_generation: bool = False,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.inventory = inventory
        self.run = run or _default_docker_run
        self.http = http or requests
        self.images = {**PINNED_IMAGES, **(images or {})}
        self.ports = {**DEFAULT_PORTS, **(ports or {})}
        self.state = state or default_state_paths()
        self.allowed_gpus = allowed_gpus
        self.reserved = tuple(reserved)
        self.project = project
        self.skip_display = skip_display
        self.litellm = litellm
        self.require_generation = require_generation
        self.last_errors: list[str] = []
        self._pulled: set[str] = set()  # (group:tag) pulled this process

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
    def _env_path(self) -> Path:
        return self.state_dir / '.env'

    def master_key(self) -> str:
        """The managed LiteLLM master key.

        infer-stack manages this secret in the state dir's ``.env``: reused if
        already present (you may pin your own ``sk-`` key there), otherwise
        generated and persisted. The caller doesn't need to invent or export it
        — it is baked into the LiteLLM service, used by the readiness probe, and
        shipped in the env-file descriptor (``infer-stack secrets`` prints it).
        """
        existing = parse_env_file(self._env_path)
        key = ensure_secret(existing, API_KEY_ENV, prefix='sk-')
        if key != existing.get(API_KEY_ENV):
            write_env_file(self._env_path, {API_KEY_ENV: key})
        return key

    def _load_sidecar(self) -> dict[str, Any]:
        if self._state_file.exists():
            return json.loads(self._state_file.read_text())
        return {}

    def _save_sidecar(self, data: dict[str, Any]) -> None:
        self._state_file.write_text(json.dumps(data, indent=2))

    def _compose(self, args: list[str]) -> str:
        return self.run(
            [
                'docker',
                'compose',
                '-p',
                self.project,
                '-f',
                str(self.compose_file),
                *args,
            ]
        )

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

    def converge(self, desired: list[DeploymentGroup]):
        """Place, render, and ``docker compose up`` the desired union."""
        desired = list(desired)
        with self._converge_lock():
            pinned = self._load_sidecar().get('assignments', {})
            plan = plan_placement(
                desired,
                self.inventory,
                allowed_gpus=self.allowed_gpus,
                reserved=self.reserved,
                pinned=pinned,
                skip_display=self.skip_display,
            )
            self.last_errors = plan.errors
            rendered = render_compose(
                desired,
                plan.assignments,
                images=self.images,
                ports=self.ports,
                state=self.state,
                litellm=self.litellm,
                litellm_port=self.litellm_port,
                litellm_master_key=self.master_key() if self.litellm else None,
                aux_dir=self.state_dir,
            )
            if rendered.litellm_config is not None:
                (self.state_dir / LITELLM_CONFIG_FILENAME).write_text(
                    rendered.litellm_config
                )
            self.compose_file.write_text(
                yaml.safe_dump(rendered.compose, sort_keys=False)
            )
            self._save_sidecar(
                {'assignments': plan.assignments, 'services': rendered.services}
            )
            self._compose(['up', '-d', '--remove-orphans'])
        return plan

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
        model name is the endpoint alias itself. Returns ``None`` (let the CLI
        fall back) when LiteLLM is off, since there is then no single base URL.
        """
        if not self.litellm:
            return None
        return {
            'base_url': f'http://127.0.0.1:{self.litellm_port}/v1',
            'api_key_env': API_KEY_ENV,
            'api_key': self.master_key(),
            'request_names': {ep: ep for ep in endpoints},
        }

    def _ensure_ollama_tag(
        self, group: DeploymentGroup, endpoint: str
    ) -> str | None:
        """Pull the endpoint's Ollama tag into its daemon (idempotent).

        An Ollama daemon loads tags lazily, so a tag must be present before it
        can serve. Returns an error reason if the pull failed (retry next poll),
        else ``None``.
        """
        tag = (group.served.get(endpoint) or {}).get('model')
        if not tag:
            return None
        key = f'{group.id}:{tag}'
        if key in self._pulled:
            return None
        try:
            self._compose(['exec', '-T', f'ollama-{group.id}', 'ollama', 'pull', tag])
        except Exception as ex:  # noqa: BLE001 - readiness is retryable
            return f'pulling {tag}: {ex}'
        self._pulled.add(key)
        return None

    def probe_ready(
        self, group: DeploymentGroup, endpoint: str
    ) -> Readiness:
        """Ready == container running and (with LiteLLM) the alias is routable.

        Delegates the HTTP check to the shared :func:`infer_stack.probe.openai_ready`
        — ``require_listed`` confirms the alias is advertised by the gateway, and
        ``require_generation`` additionally runs a tiny chat. For Ollama the tag
        is pulled first and a generation is forced, so readiness means the tag is
        actually loaded and serving (not just lazily configured).
        """
        if group.id not in self.observe():
            return Readiness(False, 'container not running')
        if not self.litellm:
            return Readiness(True, 'container running')
        require_generation = self.require_generation
        if group.engine == 'ollama':
            error = self._ensure_ollama_tag(group, endpoint)
            if error:
                return Readiness(False, error)
            require_generation = True  # warm the tag so it is resident
        headers = {'Authorization': f'Bearer {self.master_key()}'}
        ok, reason = openai_ready(
            base_url=f'http://127.0.0.1:{self.litellm_port}/v1',
            headers=headers,
            model=endpoint,
            require_listed=True,
            require_generation=require_generation,
            http=self.http,
        )
        return Readiness(ok, reason)

    def down(self) -> None:
        """Tear the whole project down (for an explicit stop)."""
        if self.compose_file.exists():
            self._compose(['down', '--remove-orphans'])
