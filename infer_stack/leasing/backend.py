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
    """An ``acquire``/``serve`` requested a deployment the backend could not place.

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
