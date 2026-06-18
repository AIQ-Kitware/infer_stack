"""High-level lease bookkeeping over the sqlite store.

This is the backend-agnostic heart of the controller. It owns the *invariants*:

* **Coalescing.** Two requests for the same compatible deployment share one
  :class:`~infer_stack.leasing.models.DeploymentGroup` (demand reference-counted)
  unless one asks for ``dedicated``. Compatibility is structural identity
  (:func:`~infer_stack.leasing.models.compatibility_key`) plus capacity
  subsumption (:func:`~infer_stack.leasing.models.capacity_satisfies`).
* **Soft TTL.** A lease protects its groups while ACTIVE and not past its TTL.
  Expiry does not kill anything; it merely stops protecting, so the group can be
  reclaimed once nothing else needs it. TTL is the crash-recovery backstop for
  leases whose explicit ``release`` never ran.
* **Demand-driven group state.** A group is LIVE while demand > 0 and IDLE when
  it reaches 0. Actually tearing an IDLE group down is a reconciler/backend
  concern (per reclaim policy); the ledger only computes the candidates.

It does **not** talk to docker, GPUs, or KubeAI — that is the reconciler/backend
layer that consumes :class:`AcquireResult`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..paths import data_root
from .models import (
    DeploymentGroup,
    EndpointRequest,
    GroupState,
    Lease,
    LeaseState,
    Sharing,
    capacity_satisfies,
)
from .store import SqliteStore


def default_ledger_path() -> Path:
    """Default shared ledger location under the infer-stack data root."""
    return data_root() / 'leasing' / 'ledger.db'


@dataclass
class AcquireResult:
    """What :meth:`Ledger.acquire` returns to the reconciler.

    ``groups`` are the deployments the caller must ensure are realized and ready
    before handing the lease's env-file to the job.
    """

    lease: Lease
    groups: list[DeploymentGroup] = field(default_factory=list)


@dataclass
class ReleaseResult:
    idled_group_ids: list[str] = field(default_factory=list)


@dataclass
class SweepResult:
    expired_lease_ids: list[str] = field(default_factory=list)
    idled_group_ids: list[str] = field(default_factory=list)


class Ledger:
    """Lease/deployment bookkeeping.

    Args:
        store: the sqlite store.
        clock: returns epoch seconds; injectable for tests.
        id_factory: ``(prefix) -> id``; injectable for deterministic tests.

    Example:
        >>> from infer_stack.leasing.store import SqliteStore
        >>> from infer_stack.leasing.models import EndpointRequest, vllm_structural
        >>> led = Ledger(SqliteStore(':memory:'))
        >>> req = EndpointRequest(
        ...     endpoint='qwen-coder', engine='vllm',
        ...     structural=vllm_structural(model_ref='qwen-coder-32b'),
        ...     capacity={'max_model_len': 32768})
        >>> a = led.acquire('alice', [req])
        >>> b = led.acquire('bob', [req])            # same model -> coalesces
        >>> a.groups[0].id == b.groups[0].id
        True
        >>> led.get_group(a.groups[0].id).demand     # two protecting leases
        2
        >>> _ = led.release(a.lease.id)
        >>> led.get_group(a.groups[0].id).demand
        1
        >>> res = led.release(b.lease.id)            # last one out -> idle
        >>> res.idled_group_ids == [a.groups[0].id]
        True
    """

    def __init__(
        self,
        store: SqliteStore,
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[str], str] | None = None,
    ):
        self.store = store
        self.clock = clock
        self.id_factory = id_factory or self._default_id

    @staticmethod
    def _default_id(prefix: str) -> str:
        return f'{prefix}-{uuid.uuid4().hex[:12]}'

    # -- public API --------------------------------------------------------

    def acquire(
        self,
        owner: str,
        requests: list[EndpointRequest],
        *,
        ttl_seconds: float | None = None,
    ) -> AcquireResult:
        """Create a lease and coalesce its endpoints onto deployment groups.

        ``ttl_seconds=None`` is an infinite (standing) lease — the shape the
        legacy ``switch <profile>`` maps onto.
        """
        now = self.clock()
        lease_id = self.id_factory('sess')
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        group_ids: list[str] = []
        with self.store.transaction():
            self.store.insert_lease(
                lease_id=lease_id,
                owner=owner,
                created_at=now,
                ttl_seconds=ttl_seconds,
                expires_at=expires_at,
                heartbeat_at=now,
            )
            for req in requests:
                group = self._find_or_create_group(req, now)
                self.store.insert_claim(
                    lease_id=lease_id, endpoint=req.endpoint, group_id=group.id
                )
                group_ids.append(group.id)
        lease = self.store.get_lease(lease_id)
        groups = [
            self.store.get_group(gid, now=now)
            for gid in dict.fromkeys(group_ids)
        ]
        return AcquireResult(lease=lease, groups=[g for g in groups if g])

    def release(self, lease_id: str) -> ReleaseResult:
        """Mark a lease released and idle any group whose demand hits zero."""
        now = self.clock()
        idled: list[str] = []
        with self.store.transaction():
            lease = self.store.get_lease(lease_id)
            if lease is None or lease.state == LeaseState.RELEASED:
                return ReleaseResult()
            self.store.set_lease_state(lease_id, LeaseState.RELEASED)
            idled = self._idle_groups(lease.group_ids, now)
        return ReleaseResult(idled_group_ids=idled)

    def renew(self, lease_id: str, *, ttl_seconds: float | None) -> Lease | None:
        """Extend (or make infinite) a lease's protection window."""
        now = self.clock()
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        with self.store.transaction():
            if self.store.get_lease(lease_id) is None:
                return None
            self.store.renew_lease(
                lease_id,
                ttl_seconds=ttl_seconds,
                expires_at=expires_at,
                heartbeat_at=now,
            )
        return self.store.get_lease(lease_id)

    def sweep(self) -> SweepResult:
        """Materialize TTL expiry and recompute idle groups.

        Safe to call periodically. Expiring a lease is the crash backstop: a job
        that died without ``release`` stops protecting once its TTL elapses.
        """
        now = self.clock()
        expired: list[str] = []
        idled: list[str] = []
        with self.store.transaction():
            due = self.store.active_leases_past(now)
            affected: list[str] = []
            for lease in due:
                self.store.set_lease_state(lease.id, LeaseState.EXPIRED)
                expired.append(lease.id)
                affected.extend(lease.group_ids)
            idled = self._idle_groups(affected, now)
        return SweepResult(expired_lease_ids=expired, idled_group_ids=idled)

    def reclaimable_groups(self) -> list[DeploymentGroup]:
        """IDLE groups the reconciler may tear down per reclaim policy."""
        now = self.clock()
        return [
            g
            for g in self.store.list_groups(now=now)
            if g.state == GroupState.IDLE
        ]

    def evict_idle(self, group_ids: list[str] | None = None) -> list[str]:
        """Force IDLE groups to STOPPED so the next reconcile tears them down.

        This overrides ``keep-warm``: a released group normally stays resident
        (IDLE) to avoid cold-start thrash, but evicting it frees its GPU now.
        ``group_ids=None`` evicts every idle group; a list restricts it (ids not
        currently idle are skipped). Returns the ids actually evicted.
        """
        now = self.clock()
        wanted = None if group_ids is None else set(group_ids)
        evicted: list[str] = []
        with self.store.transaction():
            for g in self.store.list_groups(now=now):
                if g.state != GroupState.IDLE:
                    continue
                if wanted is not None and g.id not in wanted:
                    continue
                self.store.set_group_state(g.id, GroupState.STOPPED, now)
                evicted.append(g.id)
        return evicted

    def status(self) -> tuple[list[Lease], list[DeploymentGroup]]:
        """Snapshot for ``infer-stack status`` (leases, groups-with-demand)."""
        now = self.clock()
        return self.store.list_leases(), self.store.list_groups(now=now)

    def get_lease(self, lease_id: str) -> Lease | None:
        return self.store.get_lease(lease_id)

    def get_group(self, group_id: str) -> DeploymentGroup | None:
        return self.store.get_group(group_id, now=self.clock())

    # -- internals ---------------------------------------------------------

    def _find_or_create_group(
        self, req: EndpointRequest, now: float
    ) -> DeploymentGroup:
        if req.sharing == Sharing.DEDICATED:
            return self._create_group(req, now)
        candidates = self.store.groups_by_compat(
            req.compat_key,
            sharing=Sharing.SHARED,
            states=(GroupState.LIVE, GroupState.IDLE),
        )
        for group in candidates:
            if capacity_satisfies(group.capacity, req.capacity):
                if group.state == GroupState.IDLE:
                    self.store.set_group_state(
                        group.id, GroupState.LIVE, now
                    )
                self._merge_served(group, req, now)
                return self.store.get_group(group.id) or group
        return self._create_group(req, now)

    def _create_group(
        self, req: EndpointRequest, now: float
    ) -> DeploymentGroup:
        group = DeploymentGroup(
            id=self.id_factory('grp'),
            compat_key=req.compat_key,
            engine=req.engine,
            sharing=req.sharing,
            capacity=dict(req.capacity),
            spec=dict(req.spec),
            served={req.endpoint: dict(req.served)},
            state=GroupState.LIVE,
            created_at=now,
            updated_at=now,
        )
        self.store.insert_group(group)
        return group

    def _merge_served(
        self, group: DeploymentGroup, req: EndpointRequest, now: float
    ) -> None:
        """Add an endpoint (e.g. a new Ollama tag) to a coalesced group."""
        served = dict(group.served)
        if served.get(req.endpoint) != req.served:
            served[req.endpoint] = dict(req.served)
            self.store.update_group_served(group.id, served, now)

    def _idle_groups(self, group_ids: list[str], now: float) -> list[str]:
        idled: list[str] = []
        for gid in dict.fromkeys(group_ids):
            group = self.store.get_group(gid)
            if group is None or group.state != GroupState.LIVE:
                continue
            if self.store.demand(gid, now) == 0:
                self.store.set_group_state(gid, GroupState.IDLE, now)
                idled.append(gid)
        return idled
