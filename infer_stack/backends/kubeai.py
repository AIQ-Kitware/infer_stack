"""KubeAI leasing backend: converge the ledger's desired set onto a cluster.

The modern sibling of :class:`infer_stack.leasing.compose.ComposeBackend`,
driven by the same controller through the same converge-style surface:

* ``converge(desired, apply=False)`` — the RENDER half: one KubeAI ``Model``
  custom resource per desired vLLM deployment, written to ``models.yaml`` in the
  state dir (diffed + confirmable, atomic).
* ``apply()`` — the slow half: ``kubectl apply`` the rendered manifest, then
  prune stale infer-stack-managed Models the render no longer wants.
* ``observe()`` / ``probe_ready()`` / ``access()`` — same contracts as compose.

Where compose plans GPU placement locally, KubeAI/Kubernetes schedules: each
Model's ``resourceProfile`` (``<profile>:<gpu-count>``) tells the cluster what
to reserve, so ``last_assignments`` stays empty and placement can only fail at
admission (a Model that never gets a pod shows up as not-ready, not unplaced).

Model naming mirrors the compose service-name contract: the CR name is derived
purely from the served model name (``_dns_slug``), so it is stable across
releases/re-acquires, and it doubles as the *request* name — the KubeAI gateway
routes ``model=<CR name>`` to the backing pods. Collisions between
simultaneously desired deployments are reported loudly (``last_unplaced`` /
``last_errors``), never silently overwritten.

Cluster prerequisites (once per cluster, not per acquire): a reachable
kubeconfig, the KubeAI helm chart installed (``scripts/install_kubeai.sh``)
with ``resourceProfiles`` matching the ``resource_profile`` names your catalog
uses, and a route to the gateway (the default ``base_url`` assumes
``kubectl port-forward svc/kubeai 8000:80``). See ``docs/kubeai-backend.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from ..leasing.backend import ConvergeScaffold, Readiness
from ..leasing.compose import dns_slug, vllm_service_dict
from ..leasing.models import Deployment
from ..probe import openai_ready
from ..profile_runtime import vllm_args

MODELS_FILENAME = 'models.yaml'
STATE_FILENAME = 'leasing-kubeai-state.json'
MANAGED_LABEL = 'infer-stack/managed'
DEPLOYMENT_LABEL = 'infer-stack/deployment'
DEFAULT_NAMESPACE = 'kubeai'
# The standard local access path: `kubectl port-forward svc/kubeai 8000:80`.
# An ingress-fronted cluster overrides this via the kubeai_base_url setting.
DEFAULT_BASE_URL = 'http://127.0.0.1:8000/openai/v1'


def model_name_for(served: str) -> str:
    """Deterministic Model CR name for a served model name: ``<dns-slug>``.

    Derived purely from the served name (like the compose service name), so it
    is identical across releases/re-acquires of the same endpoint — the request
    name clients use through the KubeAI gateway never changes.
    """
    return dns_slug(served)


def _served_name(deployment: Deployment) -> str:
    return deployment.spec.get('served_model_name') or (
        sorted(deployment.served)[0] if deployment.served else deployment.id
    )


def _gpu_count(deployment: Deployment) -> int:
    runtime = deployment.spec.get('runtime', {}) or {}
    tp = int(runtime.get('tensor_parallel_size', 1) or 1)
    pp = int(runtime.get('pipeline_parallel_size', 1) or 1)
    dp = int(runtime.get('data_parallel_size', 1) or 1)
    return max(1, tp * pp * dp)


def _model_doc(
    deployment: Deployment,
    *,
    namespace: str,
    resource_profile: str,
) -> dict[str, Any]:
    """Build one KubeAI ``Model`` CR for a vLLM deployment.

    ``spec.args`` reuses the exact arg pipeline the compose backend renders
    with (``_vllm_service_dict`` + ``vllm_args``), so every serving knob the
    compat key distinguishes (revision/quantization/dtype/pp/...) reaches the
    engine here too. ``served_model_name`` is overridden to the CR name so the
    gateway's request name and vLLM's served name agree.
    """
    name = model_name_for(_served_name(deployment))
    svc = vllm_service_dict(deployment)
    svc['served_model_name'] = name
    profile = resource_profile
    if ':' not in profile:
        profile = f'{profile}:{_gpu_count(deployment)}'
    runtime = deployment.spec.get('runtime', {}) or {}
    min_replicas = int(runtime.get('min_replicas', 1) or 1)
    max_replicas = max(min_replicas, int(runtime.get('max_replicas', 1) or 1))
    doc: dict[str, Any] = {
        'apiVersion': 'kubeai.org/v1',
        'kind': 'Model',
        'metadata': {
            'name': name,
            'namespace': namespace,
            'labels': {
                MANAGED_LABEL: 'true',
                DEPLOYMENT_LABEL: deployment.id,
            },
        },
        'spec': {
            'features': ['TextGeneration'],
            'url': f'hf://{deployment.spec["hf_model_id"]}',
            'engine': 'VLLM',
            'resourceProfile': profile,
            'minReplicas': min_replicas,
            'maxReplicas': max_replicas,
            'args': vllm_args(svc),
        },
    }
    return doc


class RenderedModels:
    """Output of the render half: manifest text + bookkeeping maps."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.models: dict[str, str] = {}  # CR name -> deployment id
        self.request_names: dict[str, str] = {}  # endpoint -> CR name
        self.unrenderable: set[str] = set()
        self.errors: list[str] = []

    @property
    def text(self) -> str:
        return '---\n'.join(
            yaml.safe_dump(doc, sort_keys=False) for doc in self.docs
        )


def render_models(
    deployments: list[Deployment],
    *,
    namespace: str,
    default_resource_profile: str | None,
) -> RenderedModels:
    """Render the desired set into KubeAI ``Model`` docs (pure, no I/O).

    Follows the compose render conventions: the oldest deployment keeps a
    contested name; anything the render must exclude (collision, unsupported
    engine, no resource profile) lands in ``unrenderable`` + ``errors`` with
    the deployment id prefixed, so the controller fails the acquire loudly and
    rolls the lease back instead of leasing a Model that never exists.
    """
    out = RenderedModels()
    ordered = sorted(deployments, key=lambda g: (g.created_at, g.id))
    for deployment in ordered:
        if deployment.engine != 'vllm':
            out.unrenderable.add(deployment.id)
            out.errors.append(
                f'{deployment.id}: engine {deployment.engine!r} is not '
                'supported by the kubeai backend yet (KubeAI serves models, '
                'not daemons — the catalog\'s host-centric ollama endpoints '
                'do not map onto it). Use --backend compose for ollama.'
            )
            continue
        runtime = deployment.spec.get('runtime', {}) or {}
        profile = (
            runtime.get('resource_profile') or default_resource_profile or ''
        )
        if not str(profile).strip():
            out.unrenderable.add(deployment.id)
            out.errors.append(
                f'{deployment.id}: no resource profile — set '
                '`runtime.resource_profile` on the catalog endpoint (a '
                'resourceProfiles key from your KubeAI helm values, e.g. '
                "'nvidia-gpu-rtx-4090') or `config set "
                'kubeai_resource_profile <name>` as the default.'
            )
            continue
        name = model_name_for(_served_name(deployment))
        if name in out.models:
            out.unrenderable.add(deployment.id)
            out.errors.append(
                f'{deployment.id}: Model name {name!r} is already used by '
                f'deployment {out.models[name]} — simultaneously live '
                'deployments must have distinct served names.'
            )
            continue
        out.docs.append(
            _model_doc(
                deployment,
                namespace=namespace,
                resource_profile=str(profile),
            )
        )
        out.models[name] = deployment.id
        for endpoint in deployment.served:
            out.request_names[endpoint] = name
    return out


def _default_kubectl_run(args: list[str]) -> str:
    """Run kubectl, returning stdout; a failure raises with stderr attached.

    ``check_output`` alone would lose kubectl's actual complaint ("connection
    refused", "the server doesn't have a resource type models") — exactly the
    text needed to debug a cluster that isn't set up yet.
    """
    import subprocess

    proc = subprocess.run(
        args, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or '').strip()
        raise RuntimeError(
            f'{" ".join(args)} failed ({proc.returncode}): {detail[:500]}'
        )
    return proc.stdout


class KubeaiBackend(ConvergeScaffold):
    """Cluster KubeAI backend (converge-style).

    Driven by the controller's ``converge`` path exactly like
    :class:`ComposeBackend`. ``run`` / ``http`` are the injected kubectl/HTTP
    seams; defaults shell out to ``kubectl`` and ``requests``. State-dir
    plumbing (atomic writes, the converge lock, the sidecar, diff-confirm)
    comes from :class:`ConvergeScaffold`.
    """

    _approve_title = 'infer-stack will update the KubeAI models'
    _state_noun = 'kubeai manifests'

    def __init__(
        self,
        *,
        state_dir: str | Path,
        namespace: str = DEFAULT_NAMESPACE,
        base_url: str | None = None,
        default_resource_profile: str | None = None,
        run: Callable[[list[str]], str] | None = None,
        http: Any = None,
        assume_yes: bool = True,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip('/')
        self.default_resource_profile = default_resource_profile
        self.run = run or _default_kubectl_run
        if http is None:
            import requests

            http = requests
        self.http = http
        self.assume_yes = assume_yes
        self.last_errors: list[str] = []
        self.last_unplaced: set[str] = set()
        # KubeAI/k8s schedules; there are no host GPU indices to report.
        self.last_assignments: dict[str, list[int]] = {}

    # -- state-dir plumbing --------------------------------------------------

    @property
    def models_file(self) -> Path:
        return self.state_dir / MODELS_FILENAME

    @property
    def _state_file(self) -> Path:
        return self.state_dir / STATE_FILENAME

    # -- kubectl plumbing ------------------------------------------------------

    def _kubectl(self, args: list[str]) -> str:
        return self.run(['kubectl', '-n', self.namespace, *args])

    def _cluster_models(self) -> dict[str, str | None]:
        """Managed Model CRs on the cluster: ``{name: deployment id}``.

        Raises on kubectl failure — callers decide whether that is best-effort
        (observe) or fatal (apply's prune step must not guess).
        """
        out = self._kubectl(
            [
                'get',
                'models.kubeai.org',
                '-l',
                f'{MANAGED_LABEL}=true',
                '-o',
                'json',
            ]
        )
        items = (json.loads(out or '{}') or {}).get('items') or []
        models: dict[str, str | None] = {}
        for item in items:
            meta = item.get('metadata') or {}
            name = meta.get('name')
            if not name:
                continue
            models[name] = (meta.get('labels') or {}).get(DEPLOYMENT_LABEL)
        return models

    # -- converge-style surface ------------------------------------------------

    def converge(self, desired: list[Deployment], *, apply: bool = True):
        """Render the desired Model set, then optionally apply it."""
        from .._log import logger

        desired = list(desired)
        with self._converge_lock():
            logger.info(
                'Converging {} deployment(s) onto kubeai/{}: {}',
                len(desired),
                self.namespace,
                ', '.join(sorted(g.id for g in desired)) or '(none)',
            )
            rendered = render_models(
                desired,
                namespace=self.namespace,
                default_resource_profile=self.default_resource_profile,
            )
            self.last_errors = list(rendered.errors)
            self.last_unplaced = set(rendered.unrenderable)
            self.last_assignments = {}
            for err in rendered.errors:
                logger.warning('  render: {}', err)
            self._approve_changes({self.models_file: rendered.text})
            self._atomic_write(self.models_file, rendered.text)
            self._save_sidecar(
                {
                    'models': rendered.models,
                    'request_names': rendered.request_names,
                }
            )
            if not apply:
                logger.info(
                    'rendered {} Model(s) to {} (not applied; '
                    '`infer-stack apply` to converge the cluster)',
                    len(rendered.models),
                    self.models_file,
                )
                return None
        self.apply()
        return None

    def apply(self) -> None:
        """Converge the cluster to the last render: apply + prune.

        Reads the on-disk manifest last written by :meth:`converge` (render)
        and applies it — it does NOT re-render (compose parity: the controller
        coalesces applies under its own lock/generation). Then deletes any
        infer-stack-managed Model the render no longer contains. Idempotent.
        """
        from .._log import logger

        if not self.models_file.exists():
            return
        text = self.models_file.read_text()
        wanted = set(self._load_sidecar().get('models') or {})
        if text.strip():
            logger.info(
                'kubectl apply ({} Model(s): {})',
                len(wanted),
                ', '.join(sorted(wanted)) or '?',
            )
            try:
                self._kubectl(['apply', '-f', str(self.models_file)])
            except Exception as ex:
                raise RuntimeError(
                    f'kubectl apply failed: {ex}\n'
                    'Is the KubeAI chart installed and the kubeconfig '
                    'reachable? See scripts/install_kubeai.sh and '
                    'docs/kubeai-backend.md.'
                ) from ex
        # Prune: managed Models on the cluster that the render dropped.
        stale = [
            name for name in self._cluster_models() if name not in wanted
        ]
        for name in sorted(stale):
            logger.info('kubectl delete model {}', name)
            self._kubectl(
                ['delete', 'models.kubeai.org', name, '--ignore-not-found']
            )

    def observe(self) -> set[str]:
        """Deployment ids with a managed Model CR on the cluster (best-effort)."""
        try:
            models = self._cluster_models()
        except Exception:  # noqa: BLE001 - observe must not brick acquire
            return set()
        return {gid for gid in models.values() if gid}

    def probe_ready(self, deployment: Deployment, endpoint: str) -> Readiness:
        """Ready == the model actually served a (protocol-aware) generation.

        Same philosophy as compose: a Model CR existing (or even reporting
        ready replicas) is not proof it can serve, so gate on the CR existing
        and then require a real generation through the KubeAI gateway.
        """
        if deployment.id not in self.observe():
            return Readiness(False, 'Model CR not on the cluster')
        name = model_name_for(_served_name(deployment))
        served = deployment.served.get(endpoint) or {}
        protocol = served.get('protocol') or 'chat'
        ok, reason = openai_ready(
            base_url=self.base_url,
            model=name,
            protocol=protocol,
            require_listed=True,
            require_generation=True,
            http=self.http,
        )
        return Readiness(ok, reason)

    def access(self, endpoints: list[str]) -> dict[str, Any] | None:
        """Where a client reaches these endpoints (env-file descriptor).

        One gateway ``base_url`` for everything; the request name is the Model
        CR name (from the render sidecar), not the endpoint alias — KubeAI has
        no alias layer. The gateway is unauthenticated, so the api key is the
        literal ``EMPTY`` placeholder and no key env var is advertised.
        """
        request_names = self._load_sidecar().get('request_names') or {}
        return {
            'base_url': self.base_url,
            'api_key_env': None,
            'api_key': 'EMPTY',
            'request_names': {
                ep: request_names.get(ep, model_name_for(ep))
                for ep in endpoints
            },
        }

    # -- realize/teardown (Protocol completeness; converge path supersedes) ---

    def realize(self, deployment: Deployment) -> None:  # pragma: no cover
        # The controller always drives converge-style backends through
        # converge()/apply(); realize exists only so the Backend Protocol is
        # satisfied. Deliberately NOT converge([deployment]) — a one-element
        # desired set would prune every other managed Model.
        pass

    def teardown(self, deployment: Deployment) -> None:
        name = model_name_for(_served_name(deployment))
        self._kubectl(
            ['delete', 'models.kubeai.org', name, '--ignore-not-found']
        )

    def down(self) -> None:
        """Delete every infer-stack-managed Model (explicit stop)."""
        for name in sorted(self._cluster_models()):
            self._kubectl(
                ['delete', 'models.kubeai.org', name, '--ignore-not-found']
            )

    # -- preflight -------------------------------------------------------------

    def doctor(self) -> list[tuple[str, bool, str]]:
        """Preflight the cluster prerequisites: ``(check, ok, detail)`` rows.

        Everything ``acquire`` needs, checked cheaply and in dependency order,
        so a fresh setup fails as a checklist instead of a mid-acquire
        traceback: cluster reachable -> KubeAI CRD installed -> namespace
        exists -> gateway answering at ``base_url``. Never raises.
        """
        checks: list[tuple[str, bool, str]] = []

        def _run_check(name: str, args: list[str], hint: str) -> bool:
            try:
                self.run(['kubectl', *args])
            except Exception as ex:  # noqa: BLE001 - report, don't raise
                checks.append((name, False, f'{ex} — {hint}'))
                return False
            checks.append((name, True, ''))
            return True

        if not _run_check(
            'cluster reachable',
            ['version', '--client=false', '-o', 'json'],
            'is the kubeconfig set up? (scripts/bootstrap_k3s.sh)',
        ):
            return checks
        if not _run_check(
            'KubeAI Model CRD installed',
            ['get', 'crd', 'models.kubeai.org', '-o', 'name'],
            'install the chart: scripts/install_kubeai.sh',
        ):
            return checks
        _run_check(
            f'namespace {self.namespace!r} exists',
            ['get', 'namespace', self.namespace, '-o', 'name'],
            f'kubectl create namespace {self.namespace} (or install the '
            'chart there)',
        )
        try:
            resp = self.http.get(f'{self.base_url}/models', timeout=10)
            code = getattr(resp, 'status_code', 0)
            ok = code == 200
            detail = '' if ok else (
                f'HTTP {code} from {self.base_url}/models'
            )
        except Exception as ex:  # noqa: BLE001 - report, don't raise
            ok = False
            detail = (
                f'{ex} — is the gateway routed? e.g. '
                f'`kubectl -n {self.namespace} port-forward svc/kubeai '
                '8000:80` (or set kubeai_base_url)'
            )
        checks.append((f'gateway at {self.base_url}', ok, detail))
        return checks
