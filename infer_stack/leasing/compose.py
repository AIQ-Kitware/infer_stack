"""Focused Compose backend for the leasing model.

Renders a docker-compose project straight from the live set of
:class:`DeploymentGroup` s — not the legacy resolved-deployment schema — using
the placement planner for GPU assignment and reusing ``profile_runtime.vllm_args``
for the vLLM CLI flags. It *converges the whole union* on every reconcile:
render the file, then ``docker compose up -d --remove-orphans``. Adding or
removing a group re-renders and converges; pinned placement (persisted in a
sidecar) keeps already-running models on their GPUs, and ``--remove-orphans``
tears down services whose group is gone.

Docker is invoked through an injected ``run(args) -> str`` seam, so all of the
logic here is unit-testable without docker. The default seam shells out to
``docker compose``; the real docker/GPU path is validated on a GPU host.

Readiness in this slice is "the container is running" (via ``observe``); the
HTTP generation probe and the Ollama pull/warmup rung land in the next slice,
together with the LiteLLM gateway that maps endpoint aliases onto served names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from ..config import DEFAULT_PORTS, PINNED_IMAGES, default_state_paths
from ..profile_runtime import vllm_args
from .backend import Readiness
from .models import DeploymentGroup
from .placement import plan_placement

VLLM_HOST_PORT_BASE = 18000
STATE_FILENAME = 'leasing-compose-state.json'
COMPOSE_FILENAME = 'docker-compose.yml'

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


def _gpu_reservation(indices: list[int]) -> dict[str, Any]:
    return {
        'resources': {
            'reservations': {
                'devices': [
                    {
                        'driver': 'nvidia',
                        'device_ids': [str(i) for i in indices],
                        'capabilities': [['gpu']],
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


def render_compose(
    groups: list[DeploymentGroup],
    assignments: dict[str, list[int]],
    *,
    images: dict[str, str],
    ports: dict[str, int],
    state: dict[str, str],
) -> RenderedCompose:
    """Render a compose project for the placed groups.

    Groups absent from ``assignments`` (placement failures) are skipped.
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
    return RenderedCompose(compose={'services': services}, services=service_map)


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

    Driven by the controller's ``converge`` path. ``run`` is the injected docker
    seam; default shells out to ``docker compose``.
    """

    def __init__(
        self,
        *,
        state_dir: str | Path,
        inventory: dict[str, Any],
        run: Callable[[list[str]], str] | None = None,
        images: dict[str, str] | None = None,
        ports: dict[str, int] | None = None,
        state: dict[str, str] | None = None,
        allowed_gpus: list[int] | None = None,
        reserved: list[int] | tuple[int, ...] = (),
        project: str = 'infer-stack',
        skip_display: bool = True,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.inventory = inventory
        self.run = run or _default_docker_run
        self.images = {**PINNED_IMAGES, **(images or {})}
        self.ports = {**DEFAULT_PORTS, **(ports or {})}
        self.state = state or default_state_paths()
        self.allowed_gpus = allowed_gpus
        self.reserved = tuple(reserved)
        self.project = project
        self.skip_display = skip_display
        self.last_errors: list[str] = []

    @property
    def compose_file(self) -> Path:
        return self.state_dir / COMPOSE_FILENAME

    @property
    def _state_file(self) -> Path:
        return self.state_dir / STATE_FILENAME

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

    def converge(self, desired: list[DeploymentGroup]):
        """Place, render, and ``docker compose up`` the desired union."""
        desired = list(desired)
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
        running = _parse_ps(self._compose(['ps', '--format', 'json']))
        services = self._load_sidecar().get('services', {})
        return {services[name] for name in running if name in services}

    def probe_ready(
        self, group: DeploymentGroup, endpoint: str
    ) -> Readiness:
        # Slice 2: readiness == container running. The HTTP generation probe and
        # the Ollama pull/warmup rung arrive with the LiteLLM gateway slice.
        if group.id in self.observe():
            return Readiness(True, 'container running')
        return Readiness(False, 'container not running')

    def down(self) -> None:
        """Tear the whole project down (for an explicit stop)."""
        if self.compose_file.exists():
            self._compose(['down', '--remove-orphans'])
