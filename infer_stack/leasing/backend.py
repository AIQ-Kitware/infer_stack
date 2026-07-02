"""Backend protocol: the seam between the ledger and real serving.

The ledger decides *what should be running* (desired deployment deployments); a
backend makes it so. The :class:`Controller` reconciles between them through the
four methods below. Keeping this surface tiny is deliberate — it is the only
thing a new backend (Compose, KubeAI, ...) must implement, and it is where the
redesign draws the line between "infer-stack coordinates" and "the backend /
KubeAI / k8s schedules".

All four methods MUST be idempotent: the reconciler may ``realize`` a deployment that
is already up, or ``teardown`` one that is already gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import Deployment


@dataclass
class Readiness:
    """Result of a single readiness probe for one served endpoint."""

    ready: bool
    detail: str = ''


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
    """

    def __init__(self, deployment_ids, reasons):
        self.deployment_ids = list(deployment_ids)
        self.reasons = list(reasons)
        super().__init__(
            '; '.join(self.reasons)
            or f'could not place: {", ".join(self.deployment_ids)}'
        )


@runtime_checkable
class Backend(Protocol):
    """What the :class:`Controller` needs from a serving backend.

    Implementations: :class:`MemoryBackend` (here, for tests/dry-runs), and
    later ``ComposeBackend`` / ``KubeAIBackend``.
    """

    def realize(self, deployment: Deployment) -> None:
        """Ensure a deployment for ``deployment`` exists and is converging."""
        ...

    def teardown(self, deployment: Deployment) -> None:
        """Ensure ``deployment``'s deployment is stopped/removed."""
        ...

    def observe(self) -> set[str]:
        """Return the set of deployment ids currently realized in the backend."""
        ...

    def probe_ready(
        self, deployment: Deployment, endpoint: str
    ) -> Readiness:
        """Report whether one served ``endpoint`` of ``deployment`` is ready."""
        ...


@runtime_checkable
class ConvergeBackend(Backend, Protocol):
    """The optional converge-style surface the controller prefers.

    A backend exposing these drives the render/apply split: the controller
    calls ``converge(desired, apply=False)`` inside the render lock (fast,
    writes on-disk state only) and ``apply()`` under the separate apply lock
    (slow, coalesced across processes via the ledger generation). After a
    converge the controller reads the three ``last_*`` attributes:

    * ``last_unplaced`` — desired deployment ids the render/placement could not
      deliver; an acquire whose deployment lands here fails loudly
      (:class:`PlacementError`) and rolls its lease back.
    * ``last_errors`` — per-deployment reasons, each prefixed with the
      deployment id.
    * ``last_assignments`` — deployment id -> GPU indices (empty for backends
      where the cluster schedules).

    Backends without this surface fall back to the per-deployment
    ``realize``/``teardown`` path in :meth:`Controller._render`.
    """

    last_unplaced: set[str]
    last_errors: list[str]
    last_assignments: dict[str, list[int]]

    def converge(self, desired: list[Deployment], *, apply: bool = True):
        """Render the desired set to backend state; optionally apply it."""
        ...

    def apply(self) -> None:
        """Converge reality to the last render (idempotent, slow half)."""
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

    def _approve_changes(self, planned: dict) -> None:
        """Show pending rendered-state changes and confirm them.

        ``planned`` maps target paths to their new content. When nothing
        actually changed, this is a quiet no-op. When ``assume_yes`` (scripts /
        non-interactive / ``--yes``), it applies after a one-line log.
        Otherwise it renders a per-file diff and prompts; a decline raises
        :class:`ConvergeAborted` so the caller can roll back.
        """
        from .._log import logger

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


class MemoryBackend:
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


class NullBackend:
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
