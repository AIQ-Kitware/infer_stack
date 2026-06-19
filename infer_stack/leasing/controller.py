"""The controller: reconcile the ledger's desired state onto a backend.

This is the thin orchestration layer the design doc (§11) calls for. The ledger
is pure bookkeeping; a backend realizes deployments; the controller is what
turns ``acquire`` / ``release`` into "the right things are running and ready".

The reconcile loop is the standard desired-vs-actual converge:

    desired = LIVE groups  +  IDLE groups whose reclaim policy is keep-warm
    actual  = backend.observe()
    realize(desired - actual);  teardown(actual - desired)

``reclaim`` policy lives on each group's spec (from the catalog): ``keep-warm``
(default — survive idle until pressure, avoids cold-start thrash) keeps an idle
group running; ``stop`` / ``scale-to-zero`` let it be torn down as soon as demand
hits zero. TTL expiry is enforced here too: every ``reconcile`` first
``sweep``s the ledger, so a crashed job's lease stops protecting its group once
its TTL elapses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .backend import Backend
from .ledger import Ledger
from .models import DeploymentGroup, EndpointRequest, GroupState, Lease

KEEP_WARM = 'keep-warm'


@dataclass
class ReconcileResult:
    realized: list[str] = field(default_factory=list)
    torn_down: list[str] = field(default_factory=list)
    # Desired groups the backend could not place (e.g. no free GPU) plus the
    # planner's per-group reasons; empty for backends without placement.
    unplaced: list[str] = field(default_factory=list)
    placement_errors: list[str] = field(default_factory=list)
    # group id -> GPU indices it is on / slated for (placement backends only).
    assignments: dict[str, list[int]] = field(default_factory=dict)
    # False when reconcile only rendered the on-disk state (no docker up/down).
    applied: bool = True


@dataclass
class WaitResult:
    ready: bool
    pending: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class AcquireOutcome:
    lease: Lease
    groups: list[DeploymentGroup]
    reconcile: ReconcileResult
    wait: WaitResult | None = None
    applied: bool = True  # False for a --no-apply (staged, not brought up) acquire


@dataclass
class ReleaseOutcome:
    idled_group_ids: list[str]
    reconcile: ReconcileResult


@dataclass
class EvictOutcome:
    evicted_group_ids: list[str]
    reconcile: ReconcileResult


class Controller:
    """Ties a :class:`Ledger` to a :class:`Backend`.

    Args:
        ledger: the lease bookkeeping store.
        backend: realizes/observes/probes deployments.
        clock: epoch-seconds source for wait timing; defaults to the ledger's.
        sleep: how to wait between readiness polls; injectable for tests.
        reclaim_default: policy for groups whose spec omits one.
    """

    def __init__(
        self,
        ledger: Ledger,
        backend: Backend,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        reclaim_default: str = KEEP_WARM,
    ):
        self.ledger = ledger
        self.backend = backend
        self.clock = clock or ledger.clock
        self.sleep = sleep
        self.reclaim_default = reclaim_default

    # -- reconcile ---------------------------------------------------------

    def desired_groups(self) -> list[DeploymentGroup]:
        """Groups that should currently be running."""
        _, groups = self.ledger.status()
        desired: list[DeploymentGroup] = []
        for group in groups:
            if group.state == GroupState.LIVE:
                desired.append(group)
            elif group.state == GroupState.IDLE:
                policy = group.spec.get('reclaim', self.reclaim_default)
                if policy == KEEP_WARM:
                    desired.append(group)
        return desired

    def reconcile(self, *, apply: bool = True) -> ReconcileResult:
        """Converge the backend to the ledger's desired state.

        A backend may implement a single ``converge(desired)`` (whole-union
        convergence — e.g. Compose renders one file and ``up``s it); otherwise
        the per-group ``realize``/``teardown`` loop is used.

        ``apply=False`` asks a converge-style backend to render the on-disk
        project *without* bringing it up — the "see what would execute" / staged
        path. Backends without that capability ignore it (render and apply are
        inseparable for them).
        """
        self.ledger.sweep()
        desired = self.desired_groups()
        if hasattr(self.backend, 'converge'):
            before = set(self.backend.observe())
            # Only pass apply= when staging, so backends/fakes with the older
            # ``converge(desired)`` signature keep working on the default path.
            if apply:
                self.backend.converge(desired)
            else:
                self.backend.converge(desired, apply=False)
            after = set(self.backend.observe())
            return ReconcileResult(
                realized=sorted(after - before),
                torn_down=sorted(before - after),
                unplaced=sorted(getattr(self.backend, 'last_unplaced', ()) or ()),
                placement_errors=list(
                    getattr(self.backend, 'last_errors', ()) or ()
                ),
                assignments=dict(
                    getattr(self.backend, 'last_assignments', {}) or {}
                ),
                applied=apply,
            )
        desired_ids = {g.id for g in desired}
        actual = self.backend.observe()
        result = ReconcileResult()
        for group in desired:
            if group.id not in actual:
                self.backend.realize(group)
                result.realized.append(group.id)
        stale = actual - desired_ids
        if stale:
            by_id = {g.id: g for g in self.ledger.status()[1]}
            for gid in stale:
                group = by_id.get(gid)
                if group is not None:
                    self.backend.teardown(group)
                result.torn_down.append(gid)
        return result

    # -- readiness ---------------------------------------------------------

    def wait_ready(
        self,
        groups: Iterable[DeploymentGroup],
        *,
        endpoints: set[str] | None = None,
        timeout: float = 300.0,
        interval: float = 2.0,
    ) -> WaitResult:
        """Block until the requested served endpoints are ready or ``timeout``.

        ``endpoints`` filters which served names to wait on (a coalesced group
        may serve more than this caller asked for); ``None`` waits for all.
        """
        pairs = [
            (group, ep)
            for group in groups
            for ep in sorted(group.served)
            if endpoints is None or ep in endpoints
        ]
        deadline = self.clock() + timeout
        while True:
            pending = [
                (g, ep)
                for (g, ep) in pairs
                if not self.backend.probe_ready(g, ep).ready
            ]
            if not pending:
                return WaitResult(ready=True)
            if self.clock() >= deadline:
                return WaitResult(
                    ready=False,
                    pending=[(g.id, ep) for g, ep in pending],
                )
            self.sleep(interval)
            pairs = pending

    # -- thin acquire / release -------------------------------------------

    def acquire(
        self,
        owner: str,
        requests: list[EndpointRequest],
        *,
        ttl_seconds: float | None = None,
        wait: bool = True,
        timeout: float = 300.0,
        interval: float = 2.0,
        apply: bool = True,
    ) -> AcquireOutcome:
        """Create a lease, realize its groups, and (optionally) block on ready.

        ``apply=False`` stages the lease and renders the on-disk project without
        bringing it up (and skips the readiness wait, since nothing is running).
        Placement is still computed, so an unplaceable request still fails fast.
        """
        from .backend import ConvergeAborted, PlacementError

        result = self.ledger.acquire(
            owner, requests, ttl_seconds=ttl_seconds
        )
        try:
            rec = self.reconcile(apply=apply)
        except ConvergeAborted:
            # The operator declined the compose changes — don't leave the
            # just-created lease dangling in the ledger.
            self.ledger.release(result.lease.id)
            raise
        # If a group this lease just requested could not be placed (e.g. no free
        # GPU), don't block forever waiting on a container that will never come
        # up — roll the lease back and report the planner's reason, so the group
        # never lingers as a phantom ``live`` with nothing behind it.
        requested = {g.id for g in result.groups}
        unplaced = requested & set(rec.unplaced)
        if unplaced:
            self.ledger.release(result.lease.id)
            reasons = [
                e
                for e in rec.placement_errors
                if any(e.startswith(gid) for gid in unplaced)
            ]
            raise PlacementError(sorted(unplaced), reasons)
        groups = [self.ledger.get_group(g.id) for g in result.groups]
        groups = [g for g in groups if g is not None]
        wait_result = None
        if wait and apply:  # nothing to wait on when we only staged the render
            wait_result = self.wait_ready(
                groups,
                endpoints=set(result.lease.endpoints),
                timeout=timeout,
                interval=interval,
            )
        return AcquireOutcome(
            lease=result.lease,
            groups=groups,
            reconcile=rec,
            wait=wait_result,
            applied=apply,
        )

    def release(self, lease_id: str) -> ReleaseOutcome:
        """Release a lease and converge (tearing down per reclaim policy)."""
        rel = self.ledger.release(lease_id)
        rec = self.reconcile()
        return ReleaseOutcome(
            idled_group_ids=rel.idled_group_ids, reconcile=rec
        )

    def evict(self, group_ids: Iterable[str] | None = None) -> EvictOutcome:
        """Force-evict idle (released) groups now, overriding keep-warm.

        Marks the matching IDLE groups STOPPED and reconciles, so a keep-warm
        model that is merely resident gets torn down and its GPU freed.
        ``group_ids=None`` evicts every idle group.
        """
        ids = None if group_ids is None else list(group_ids)
        self.ledger.sweep()
        evicted = self.ledger.evict_idle(ids)
        rec = self.reconcile()
        return EvictOutcome(evicted_group_ids=evicted, reconcile=rec)
