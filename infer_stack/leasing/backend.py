"""Backend protocol: the seam between the ledger and real serving.

The ledger decides *what should be running* (the desired deployments); a
backend makes it so. :class:`ServingBackend` is everything the
:class:`Controller` asks of one, and all of it has a meaning on every
backend: render a desired set, apply it, say what is resident, probe
readiness, preview an admission. Compose, KubeAI, and the in-process
backends (through :class:`SimpleAdmission`) implement it.

What only a backend that runs containers on this host can do (a stable
address network, adopting containers from before ownership labels, removing
unmanaged containers) is :class:`HostRuntime`, reached through
``backend.host_runtime``, which is ``None`` everywhere else. The controller
asks for the capability by meaning; the ``docker`` commands and the sidecar
stay inside Compose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .models import Deployment


@dataclass
class Readiness:
    """Result of a single readiness probe for one served endpoint.

    ``fatal`` means this endpoint can never become ready without operator
    action -- the engine is crash-looping, not loading slowly. The controller
    stops waiting immediately rather than holding the GPU for the whole
    timeout, and ``detail`` carries the engine's own error.
    """

    ready: bool
    detail: str = ''
    fatal: bool = False
    #: The endpoint is waiting for capacity held by others: the scheduler says
    #: a node is short of a resource, not that no node could ever host it.
    #: The controller then evicts an idle keep-warm deployment the backend
    #: names in ``reclaim_candidates`` (none: nothing is evicted).
    needs_room: bool = False


@dataclass(frozen=True)
class ApplyResult:
    """How far an apply got: the runtime, then the gateway's routes.

    A publication is complete only when both hold. ``runtime`` false: the
    runtime was not brought to the render (an unreadable render, a render
    from before fingerprints). ``routes`` false: the runtime reached the
    render but the gateway's routes (dynamic routing) were not verified. In
    either case the publication stays pending, and so does the approval of
    the render it is applying. Failures raise instead; this is for the
    outcomes that are not errors.

    >>> ApplyResult.of(None).complete, ApplyResult.of(False).runtime
    (True, False)
    >>> ApplyResult(routes=False).complete
    False
    """

    runtime: bool = True
    routes: bool = True
    detail: str = ''

    @property
    def complete(self) -> bool:
        return self.runtime and self.routes

    @classmethod
    def of(cls, value: ApplyResult | bool | None) -> ApplyResult:
        """An apply's return value as a result; ``None``/``True`` is complete."""
        if isinstance(value, ApplyResult):
            return value
        return cls() if value is not False else cls(runtime=False)

    def __and__(self, other: ApplyResult) -> ApplyResult:
        """Both parts of one apply (KubeAI's Models, then its gateway)."""
        return ApplyResult(self.runtime and other.runtime, self.routes and other.routes,
                           '; '.join(d for d in (self.detail, other.detail) if d))


class BackendTimeout(RuntimeError):
    """A backend command exceeded its time bound and was killed.

    Raised instead of hanging, because backend calls run under the controller's
    host-wide lock: an unbounded ``docker`` command would hold that lock forever.
    Killing the client does not undo work a daemon had already started, so the
    caller must treat the runtime state as unknown until it is observed again.
    """


class RuntimeUnsettled(RuntimeError):
    """The runtime was still changing (or unreadable) after an interrupted apply.

    Raised instead of starting another apply on top of work a killed client
    left in flight. The change stays pending; retry once the runtime settles.
    """


class ConvergeAborted(Exception):
    """A backend's ``converge`` was declined by the user (diff not approved).

    Raised by an interactive backend when the operator rejects the pending
    compose changes. The controller rolls back the just-created lease so a
    declined ``acquire`` doesn't leave dangling ledger state.
    """


class PlacementError(Exception):
    """An ``acquire`` requested a deployment the backend could not place.

    Raised by the controller when reconcile leaves one of the just-requested
    deployments unplaced (e.g. no free GPU). Like :class:`ConvergeAborted`, the
    controller rolls back the just-created lease before raising, so a request
    that cannot be satisfied does not linger as a phantom ``live`` deployment with no
    container behind it. ``reasons`` carries the planner's per-deployment messages.

    ``capacity`` says whether free GPUs would have let it in, as opposed to a
    request no amount of room admits (a render refusal, an unreadable
    runtime, a host too small): only then is "free a GPU" the advice.
    """

    def __init__(self, deployment_ids, reasons, *, capacity: bool = True):
        self.deployment_ids = list(deployment_ids)
        self.reasons = list(reasons)
        self.capacity = bool(capacity)
        super().__init__(
            '; '.join(self.reasons)
            or f'could not place: {", ".join(self.deployment_ids)}'
        )


def allocates_gpus(backend) -> bool:
    """Whether ``backend`` allocates host GPU indices itself.

    True for Compose, which places on this host and records the GPUs each
    deployment holds. False for a backend whose cluster schedules (KubeAI):
    admission commits an empty allocation, and no GPU is ever "unresolved".
    The one place the controller and the CLI ask this.
    """
    return bool(getattr(backend, 'allocates_gpus', True))


class HostRuntime(Protocol):
    """Capabilities of a backend that runs containers on this host (Compose).

    ``backend.host_runtime`` is one of these, or ``None`` for a backend with
    nothing of the kind (KubeAI: the cluster owns the network and the pods;
    the in-process backends). Each method is a meaning; how it is done
    (``docker network``, ``docker rm``, the render sidecar) is the
    implementation's business.
    """

    def network_table(self) -> dict[str, Any] | None:
        """The stable-address table renders use, or ``None`` before a migration."""
        ...

    def configure_network(self, table: dict[str, Any] | None, *,
                          on_addresses: Callable[[dict[str, str]], None] | None = None,
                          ) -> None:
        """Render with this table (``{'subnet', 'addresses'}``); ``on_addresses``
        persists addresses an approved render allocates."""
        ...

    def subnet_clashes(self, subnet: str) -> list[str]:
        """Why ``subnet`` cannot be used here: overlapping networks or routes."""
        ...

    def rendered_services(self) -> dict[str, tuple[str, str]]:
        """``{service: (fingerprint, deployment id or '')}`` of the last render."""
        ...

    def set_adopted(self, table: dict[str, dict[str, str]]) -> None:
        """Containers from before ownership labels that count as managed."""
        ...

    def remove_containers(self, container_ids: list[str]) -> None:
        """Remove these containers now (orphans the operator agreed to remove)."""
        ...


@runtime_checkable
class ServingBackend(Protocol):
    """What the :class:`Controller` needs from a serving backend.

    Every acquire goes through admission: a strict :meth:`residency`, an
    in-memory :meth:`preview`, then :meth:`converge` inside the render lock
    (fast, writes the backend's state only) and :meth:`apply` in the same
    lock hold (slow, bounded; see ``Controller._apply_pending``). After a
    converge the controller reads the ``last_*`` attributes:

    * ``last_unplaced``: desired deployment ids the render could not deliver;
      an acquire whose deployment lands here fails (:class:`PlacementError`)
      and rolls its lease back;
    * ``last_errors``: per-deployment reasons, each prefixed with the id;
    * ``last_assignments``: deployment id -> GPU indices (empty where the
      cluster schedules).

    Compose and KubeAI implement it directly; :class:`SimpleAdmission`
    supplies it for the in-process backends.
    """

    last_unplaced: set[str]
    last_errors: list[str]
    last_assignments: dict[str, list[int]]
    last_preview_digest: str | None

    @property
    def host_runtime(self) -> HostRuntime | None:
        """This host's container capabilities, or ``None`` (see :class:`HostRuntime`)."""
        ...

    def observe(self) -> set[str]:
        """Deployment ids the backend has rendered and brought up (lenient)."""
        ...

    def residency(self) -> Any:
        """A strict :class:`~infer_stack.leasing.residency.Residency`, or raise."""
        ...

    def instances(self) -> list[Any]:
        """Every running unit, as :class:`~infer_stack.leasing.instances.Instance`.

        Built from :meth:`residency`, so it raises when that does.
        """
        ...

    def probe_ready(self, deployment: Deployment, endpoint: str) -> Readiness:
        """Whether one served ``endpoint`` of ``deployment`` answers."""
        ...

    def preview(self, desired: list[Deployment], placement: Any = None, *,
                approve: bool = False) -> Any:
        """Place and render ``desired`` without writing; ``(plan, rendered)``."""
        ...

    def converge(self, desired: list[Deployment], *, apply: bool = True,
                 placement: Any = None) -> Any:
        """Render the desired set to backend state; optionally apply it."""
        ...

    def apply(self) -> ApplyResult | bool | None:
        """Converge reality to the last render (idempotent, slow half).

        Return an :class:`ApplyResult` saying how far it got; ``None`` or
        ``True`` means complete, ``False`` means the runtime was not reached.
        The controller keeps the change pending, and its approval, until a
        complete result. Raise on backend failure, which also leaves the
        change pending.
        """
        ...

    def placement_notes(self) -> dict[str, list[str]]:
        """What the last render recorded about placement, for health views:
        ``degraded`` (committed GPUs no longer valid) and ``displaced`` (idle
        residents that yielded their GPUs). Empty where nothing is placed."""
        ...

    def reclaim_candidates(self, blocked: Deployment,
                           idle: list[Deployment]) -> list[str]:
        """Which of ``idle`` could free room ``blocked`` is waiting for.

        Asked only when a probe said ``needs_room``. The backend knows the
        scheduling domain (which nodes, which resources); the controller
        keeps the policy (only idle deployments, longest idle first, one at
        a time). Return in ``idle``'s order; empty means evicting would not help.
        """
        ...


#: The name ``infer_stack.leasing`` has always exported for the controller's
#: backend protocol.
Backend = ServingBackend

class Realizer(Protocol):
    """The per-deployment primitives :class:`SimpleAdmission` converges with.

    Only the in-process backends implement these; a real backend converges a
    whole desired set at once, where realizing one deployment alone would
    prune the others.
    """

    def realize(self, deployment: Deployment) -> None:
        """Ensure ``deployment`` exists (idempotent)."""
        ...

    def teardown(self, deployment: Deployment) -> None:
        """Ensure ``deployment`` is gone (idempotent)."""
        ...

    def observe(self) -> set[str]:
        """The deployment ids realized now."""
        ...


class ConvergeScaffold:
    """Shared state-dir plumbing for converge-style backends.

    Hosts the pieces ComposeBackend and KubeaiBackend would otherwise copy:
    atomic writes (a concurrent ``apply`` must never read a half-written
    render), the per-state-dir converge flock, the tolerant JSON sidecar, and
    the diff-confirm gate. Subclasses set ``state_dir``, ``assume_yes``, a
    ``_state_file`` property, and may override ``_approve_title`` /
    ``_state_noun`` for their prompt/log wording.
    """

    CONVERGE_LOCK_FILENAME = '.converge.lock'
    _approve_title = 'infer-stack will update the rendered state'
    _state_noun = 'rendered state'

    # Provided by subclasses, as the docstring says. Annotation-only, so
    # nothing is created at runtime -- this states the contract to a type
    # checker, which otherwise flags every use of them in this class.
    state_dir: Path
    assume_yes: bool
    _state_file: Path

    @staticmethod
    def _atomic_write(path, text: str) -> None:
        """Write atomically (temp + ``os.replace``): the render half and the
        apply half run under different locks, so a reader must see either the
        old or the new file whole, never a torn one."""
        import os

        tmp = path.with_name(f'{path.name}.tmp')
        tmp.write_text(text)
        os.replace(tmp, path)

    def _converge_lock(self):
        """Serialize converge across processes sharing this state dir."""
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def _lock():
            handle = open(self.state_dir / self.CONVERGE_LOCK_FILENAME, 'w')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()

        return _lock()

    def _load_sidecar(self) -> dict:
        """The render-time bookkeeping sidecar (tolerant: a corrupt/absent
        file reads as empty rather than bricking every verb)."""
        import json

        state_file = self._state_file
        if state_file.exists():
            try:
                return json.loads(state_file.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_sidecar(self, data: dict) -> None:
        import json

        self._atomic_write(self._state_file, json.dumps(data, indent=2))

    def placement_notes(self) -> dict[str, list[str]]:
        """``degraded`` / ``displaced`` as the last render recorded them."""
        sidecar = self._load_sidecar()
        return {key: list(sidecar.get(key) or ()) for key in ('degraded', 'displaced')}

    #: Digest of files an admission preview already had approved.
    _preapproved: str | None = None
    #: Digest of the files the last render produced (approved-digest guard).
    last_planned_digest: str | None = None
    #: Digest of the files the last preview produced.
    last_preview_digest: str | None = None

    @staticmethod
    def _planned_digest(planned: dict) -> str:
        import hashlib
        import json

        material = json.dumps({str(k): v for k, v in planned.items()}, sort_keys=True)
        return hashlib.sha256(material.encode('utf-8')).hexdigest()

    def _preview_approval(self, planned: dict, *, approve: bool) -> None:
        """Record a preview's digest; with ``approve``, ask now, not after commit.

        The render that follows the commit skips the prompt when it produces
        the same files (see :meth:`_approve_changes`).
        """
        self.last_preview_digest = self._planned_digest(planned)
        if approve:
            self._approve_changes(planned)
            self._preapproved = self.last_preview_digest

    def _approve_changes(self, planned: dict) -> None:
        """Show pending rendered-state changes and confirm them.

        Files a preview already had approved (same digest) pass silently once.

        ``planned`` maps target paths to their new content. When nothing
        actually changed, this is a quiet no-op. When ``assume_yes`` (scripts /
        non-interactive / ``--yes``), it applies after a one-line log.
        Otherwise it renders a per-file diff and prompts; a decline raises
        :class:`ConvergeAborted` so the caller can roll back.
        """
        from .._log import logger

        preapproved, self._preapproved = self._preapproved, None
        if preapproved is not None and self._planned_digest(planned) == preapproved:
            return
        changed = {
            p: text
            for p, text in planned.items()
            if (p.read_text() if p.exists() else '') != text
        }
        if not changed:
            logger.debug('{} already up to date', self._state_noun)
            return
        names = ', '.join(p.name for p in changed)
        if self.assume_yes:
            logger.info('Updating {} ({})', self._state_noun, names)
            return
        from ..diff_prompt import confirm_writes

        if not confirm_writes(
            changed, assume_yes=False, title=self._approve_title
        ):
            raise ConvergeAborted(
                f'{self._state_noun} changes were not approved'
            )


@dataclass
class RenderPreview:
    """The render half of a :meth:`SimpleAdmission.preview`: what it refused."""

    unrenderable: set[str]
    errors: list[str]


class SimpleAdmission:
    """:class:`ServingBackend` for a backend that neither places nor inspects.

    For in-process backends (the dry-run and test backends): they allocate no
    GPUs, their :meth:`residency` is what :meth:`observe` reports (each
    deployment one warm instance), and converge/apply is the subclass's
    :class:`Realizer` primitives over the rendered set. A subclass that
    emulates capacity overrides :meth:`plan`; one that emulates render
    failures overrides :meth:`refuse`. Everything else goes through the one
    admission path the real backends use.
    """

    allocates_gpus = False
    last_preview_digest: str | None = None
    #: No containers on this host to manage.
    host_runtime: HostRuntime | None = None

    def placement_notes(self) -> dict[str, list[str]]:
        return {}

    def reclaim_candidates(self, blocked: Deployment,
                           idle: list[Deployment]) -> list[str]:
        """One scheduling domain in-process: any idle deployment frees room."""
        return [g.id for g in idle]

    def plan(self, desired: list[Deployment], placement: Any = None):
        """Which of ``desired`` fit; here, all of them, on no GPU."""
        from .placement import GpuPlan

        return GpuPlan(assignments={g.id: [] for g in desired})

    def refuse(self, desired: list[Deployment]) -> dict[str, str]:
        """``{deployment id: reason}`` for what cannot be rendered; none here."""
        return {}

    def residency(self):
        from .residency import Container, Residency

        return Residency(by_deployment={
            gid: (Container(container_id=gid, deployment_id=gid,
                            state='running', labelled=True),)
            for gid in sorted(self.observe())
        })

    def instances(self):
        """One in-process instance per realized deployment; no log to read."""
        from .instances import MEMORY, from_residency

        return from_residency(self.residency(), runtime=MEMORY)

    def _preview(self, desired: list[Deployment], placement: Any = None):
        desired = list(desired)
        plan = self.plan(desired, placement)
        refused = self.refuse(desired)
        return plan, RenderPreview(set(refused), [f'{g}: {why}' for g, why in refused.items()])

    def preview(self, desired: list[Deployment], placement: Any = None, *,
                approve: bool = False):
        plan, rendered = self._preview(desired, placement)
        self.last_preview_digest = repr(sorted(
            (gid, list(gpus)) for gid, gpus in plan.assignments.items()
            if gid not in rendered.unrenderable))
        return plan, rendered

    def converge(self, desired: list[Deployment], *, apply: bool = True,
                 placement: Any = None):
        """Record the renderable, placed part of ``desired``; ``apply`` realizes it."""
        desired = list(desired)
        plan, rendered = self._preview(desired, placement)
        self.last_errors = list(plan.errors) + list(rendered.errors)
        self.last_unplaced = {
            g.id for g in desired
            if g.id not in plan.assignments or g.id in rendered.unrenderable}
        self.last_assignments = (dict(plan.assignments)
                                 if self.allocates_gpus else {})
        self._rendered = {g.id: g for g in desired if g.id not in self.last_unplaced}
        known = getattr(self, '_known', {})
        known.update({g.id: g for g in desired})
        self._known = known
        if apply:
            self.apply()

    def apply(self) -> None:
        """Realize what the last converge rendered; tear down the rest."""
        rendered = getattr(self, '_rendered', {})
        for gid in sorted(self.observe() - set(rendered)):
            deployment = getattr(self, '_known', {}).get(gid)
            if deployment is not None:
                self.teardown(deployment)
        for gid, deployment in rendered.items():
            if gid not in self.observe():
                self.realize(deployment)

    # Set by converge (the controller reads them with a default before then).
    last_unplaced: set[str]
    last_errors: list[str]
    last_assignments: dict[str, list[int]]

    def observe(self) -> set[str]:  # pragma: no cover - provided by the backend
        raise NotImplementedError

    def realize(self, deployment: Deployment) -> None:  # pragma: no cover
        raise NotImplementedError

    def teardown(self, deployment: Deployment) -> None:  # pragma: no cover
        raise NotImplementedError


class MemoryBackend(SimpleAdmission):
    """In-memory backend that records calls and has configurable readiness.

    Not a real serving backend — it never starts a process. It exists so the
    controller's reconcile/wait logic can be tested deterministically, and as a
    ``--dry-run`` backend.

    Example:
        >>> from infer_stack.leasing.models import Deployment, DeploymentState
        >>> b = MemoryBackend(ready=True)
        >>> g = Deployment('grp-1', 'ck', 'vllm', 'shared-compatible',
        ...     {}, {}, {'qwen': {}}, DeploymentState.LIVE, 0.0, 0.0)
        >>> b.realize(g); sorted(b.observe())
        ['grp-1']
        >>> b.probe_ready(g, 'qwen').ready
        True
        >>> b.teardown(g); sorted(b.observe())
        []
    """

    def __init__(self, *, ready: bool = True):
        self.ready_default = ready
        self.realized: dict[str, Deployment] = {}
        self.ready_overrides: dict[object, bool] = {}
        self.realize_calls: list[str] = []
        self.teardown_calls: list[str] = []

    def realize(self, deployment: Deployment) -> None:
        self.realized[deployment.id] = deployment
        self.realize_calls.append(deployment.id)

    def teardown(self, deployment: Deployment) -> None:
        self.realized.pop(deployment.id, None)
        self.teardown_calls.append(deployment.id)

    def observe(self) -> set[str]:
        return set(self.realized)

    def probe_ready(
        self, deployment: Deployment, endpoint: str
    ) -> Readiness:
        if deployment.id not in self.realized:
            return Readiness(False, 'not realized')
        ready = self.ready_overrides.get((deployment.id, endpoint))
        if ready is None:
            ready = self.ready_overrides.get(deployment.id, self.ready_default)
        return Readiness(bool(ready), 'ok' if ready else 'warming up')

    def set_ready(
        self,
        ready: bool,
        *,
        deployment_id: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Override readiness globally, per deployment, or per (deployment, endpoint)."""
        if deployment_id is None:
            self.ready_default = ready
        elif endpoint is None:
            self.ready_overrides[deployment_id] = ready
        else:
            self.ready_overrides[(deployment_id, endpoint)] = ready


class NullBackend(SimpleAdmission):
    """A no-op backend that serves nothing — for ``--dry-run`` and ``leases``.

    It never starts a process. ``observe`` returns the empty set, so the
    controller treats every desired deployment as freshly realized (a no-op) and
    never tears anything down; readiness is immediate. Because it keeps no
    in-memory state, behaviour stays coherent across separate CLI invocations:
    the persistent ledger is the only source of truth. Use this to exercise
    acquire/release/run plumbing before the Compose backend exists.
    """

    def realize(self, deployment: Deployment) -> None:
        pass

    def teardown(self, deployment: Deployment) -> None:
        pass

    def observe(self) -> set[str]:
        return set()

    def probe_ready(
        self, deployment: Deployment, endpoint: str
    ) -> Readiness:
        return Readiness(True, 'dry-run')


def _conforms() -> None:  # pragma: no cover - read by the type checker only
    """Each backend the controller drives is a :class:`ServingBackend`.

    Never called: ``ty`` checks these returns, so a backend that drifts from
    the protocol fails the type check rather than a run.
    """
    from ..backends.kubeai import KubeaiBackend
    from .compose import ComposeBackend

    def compose(backend: ComposeBackend) -> ServingBackend:
        return backend

    def kubeai(backend: KubeaiBackend) -> ServingBackend:
        return backend

    def memory(backend: MemoryBackend) -> ServingBackend:
        return backend

    def null(backend: NullBackend) -> ServingBackend:
        return backend

    def host(backend: ComposeBackend) -> HostRuntime:
        return backend
