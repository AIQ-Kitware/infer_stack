"""Backend protocol: the seam between the ledger and real serving.

The ledger decides *what should be running* (desired deployment groups); a
backend makes it so. The :class:`Controller` reconciles between them through the
four methods below. Keeping this surface tiny is deliberate — it is the only
thing a new backend (Compose, KubeAI, ...) must implement, and it is where the
redesign draws the line between "infer-stack coordinates" and "the backend /
KubeAI / k8s schedules".

All four methods MUST be idempotent: the reconciler may ``realize`` a group that
is already up, or ``teardown`` one that is already gone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import DeploymentGroup


@dataclass
class Readiness:
    """Result of a single readiness probe for one served endpoint."""

    ready: bool
    detail: str = ''


@runtime_checkable
class Backend(Protocol):
    """What the :class:`Controller` needs from a serving backend.

    Implementations: :class:`MemoryBackend` (here, for tests/dry-runs), and
    later ``ComposeBackend`` / ``KubeAIBackend``.
    """

    def realize(self, group: DeploymentGroup) -> None:
        """Ensure a deployment for ``group`` exists and is converging."""
        ...

    def teardown(self, group: DeploymentGroup) -> None:
        """Ensure ``group``'s deployment is stopped/removed."""
        ...

    def observe(self) -> set[str]:
        """Return the set of group ids currently realized in the backend."""
        ...

    def probe_ready(
        self, group: DeploymentGroup, endpoint: str
    ) -> Readiness:
        """Report whether one served ``endpoint`` of ``group`` is ready."""
        ...


class MemoryBackend:
    """In-memory backend that records calls and has configurable readiness.

    Not a real serving backend — it never starts a process. It exists so the
    controller's reconcile/wait logic can be tested deterministically, and as a
    ``--dry-run`` backend.

    Example:
        >>> from infer_stack.leasing.models import DeploymentGroup, GroupState
        >>> b = MemoryBackend(ready=True)
        >>> g = DeploymentGroup('grp-1', 'ck', 'vllm', 'shared-compatible',
        ...     {}, {}, {'qwen': {}}, GroupState.LIVE, 0.0, 0.0)
        >>> b.realize(g); sorted(b.observe())
        ['grp-1']
        >>> b.probe_ready(g, 'qwen').ready
        True
        >>> b.teardown(g); sorted(b.observe())
        []
    """

    def __init__(self, *, ready: bool = True):
        self.ready_default = ready
        self.realized: dict[str, DeploymentGroup] = {}
        self.ready_overrides: dict[object, bool] = {}
        self.realize_calls: list[str] = []
        self.teardown_calls: list[str] = []

    def realize(self, group: DeploymentGroup) -> None:
        self.realized[group.id] = group
        self.realize_calls.append(group.id)

    def teardown(self, group: DeploymentGroup) -> None:
        self.realized.pop(group.id, None)
        self.teardown_calls.append(group.id)

    def observe(self) -> set[str]:
        return set(self.realized)

    def probe_ready(
        self, group: DeploymentGroup, endpoint: str
    ) -> Readiness:
        if group.id not in self.realized:
            return Readiness(False, 'not realized')
        ready = self.ready_overrides.get((group.id, endpoint))
        if ready is None:
            ready = self.ready_overrides.get(group.id, self.ready_default)
        return Readiness(bool(ready), 'ok' if ready else 'warming up')

    def set_ready(
        self,
        ready: bool,
        *,
        group_id: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Override readiness globally, per group, or per (group, endpoint)."""
        if group_id is None:
            self.ready_default = ready
        elif endpoint is None:
            self.ready_overrides[group_id] = ready
        else:
            self.ready_overrides[(group_id, endpoint)] = ready
