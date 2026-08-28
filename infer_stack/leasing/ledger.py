"""High-level lease bookkeeping over the sqlite store.

This is the backend-agnostic heart of the controller. It owns the *invariants*:

* **Coalescing.** Two requests for the same compatible deployment share one
  :class:`~infer_stack.leasing.models.Deployment` (demand reference-counted)
  unless one asks for ``dedicated``. Compatibility is structural identity
  (:func:`~infer_stack.leasing.models.compatibility_key`) plus capacity
  subsumption (:func:`~infer_stack.leasing.models.capacity_satisfies`).
* **Soft TTL.** A lease protects its deployments while ACTIVE and not past its TTL.
  Expiry does not kill anything; it merely stops protecting, so the deployment can be
  reclaimed once nothing else needs it. TTL is the crash-recovery backstop for
  leases whose explicit ``release`` never ran.
* **Demand-driven deployment state.** A deployment is LIVE while demand > 0 and IDLE when
  it reaches 0. Actually tearing an IDLE deployment down is a reconciler/backend
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
    Deployment,
    DeploymentState,
    EndpointRequest,
    Lease,
    LeaseState,
    Sharing,
    capacity_satisfies,
    is_reservation,
)
from .store import SqliteStore


def default_ledger_path() -> Path:
    """Default shared ledger location under the infer-stack data root."""
    return data_root() / 'leasing' / 'ledger.db'


@dataclass
class AcquireResult:
    """What :meth:`Ledger.acquire` returns to the reconciler.

    ``deployments`` are the deployments the caller must ensure are realized and ready
    before handing the lease's env-file to the job.
    """

    lease: Lease
    deployments: list[Deployment] = field(default_factory=list)


@dataclass
class ReleaseResult:
    # ``found=False`` means the lease id does not exist in the ledger at all —
    # callers reporting success/failure must distinguish that from a release
    # that simply idled nothing. ``already_released=True`` is the idempotent
    # re-release of a RELEASED lease (a cleanup trap firing twice is fine).
    idled_deployment_ids: list[str] = field(default_factory=list)
    found: bool = True
    already_released: bool = False


@dataclass
class SweepResult:
    expired_lease_ids: list[str] = field(default_factory=list)
    idled_deployment_ids: list[str] = field(default_factory=list)


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
        >>> a.deployments[0].id == b.deployments[0].id
        True
        >>> led.get_deployment(a.deployments[0].id).demand     # two protecting leases
        2
        >>> _ = led.release(a.lease.id)
        >>> led.get_deployment(a.deployments[0].id).demand
        1
        >>> res = led.release(b.lease.id)            # last one out -> idle
        >>> res.idled_deployment_ids == [a.deployments[0].id]
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
        """Create a lease and coalesce its endpoints onto deployment deployments.

        ``ttl_seconds=None`` is an infinite (standing) lease — the shape the
        legacy ``switch <profile>`` maps onto.
        """
        now = self.clock()
        lease_id = self.id_factory('lease')
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        deployment_ids: list[str] = []
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
                deployment = self._find_or_create_deployment(req, now)
                self.store.insert_claim(
                    lease_id=lease_id,
                    endpoint=req.endpoint,
                    deployment_id=deployment.id,
                    kind='reserved-gpu' if is_reservation(req) else 'endpoint',
                )
                deployment_ids.append(deployment.id)
            # Every acquire bumps the desired generation, even when it coalesces
            # onto an already-live deployment: this is the signal that the caller
            # needs an apply to have run including its claim (and it heals drift —
            # a crashed container gets re-upped by the apply this forces).
            self.store.bump_desired_generation()
        lease = self.store.get_lease(lease_id)
        if lease is None:
            # Written moments ago in this method. Missing means the store was
            # mutated underneath us, which is worth saying plainly.
            raise KeyError(f'lease {lease_id!r} vanished during acquire')
        deployments = [
            self.store.get_deployment(gid, now=now)
            for gid in dict.fromkeys(deployment_ids)
        ]
        return AcquireResult(lease=lease, deployments=[g for g in deployments if g])

    def release(self, lease_id: str) -> ReleaseResult:
        """Mark a lease released and idle any deployment whose demand hits zero."""
        now = self.clock()
        idled: list[str] = []
        with self.store.transaction():
            lease = self.store.get_lease(lease_id)
            if lease is None:
                return ReleaseResult(found=False)
            if lease.state == LeaseState.RELEASED:
                return ReleaseResult(already_released=True)
            self.store.set_lease_state(lease_id, LeaseState.RELEASED)
            idled = self._idle_deployments(lease.deployment_ids, now)
            if idled:  # desired set shrank -> an apply must run to tear them down
                self.store.bump_desired_generation()
        return ReleaseResult(idled_deployment_ids=idled)

    def renew(self, lease_id: str, *, ttl_seconds: float | None) -> Lease | None:
        """Extend (or make infinite) a lease's protection window.

        Only an ACTIVE lease renews; a RELEASED/EXPIRED one returns ``None``
        (like an unknown id). Silently re-ACTIVATING a lapsed lease would
        "protect" deployments that were already idled, evicted, or torn down
        behind it — the caller must re-acquire instead, which re-realizes them.

        An ACTIVE lease *can* legitimately find its deployment IDLE (its TTL
        lapsed unswept while the other demand vanished, then a heartbeat
        arrived before the reclaim). Renewing re-LIVEs such deployments and bumps
        the desired generation so the next apply re-ups anything torn down.
        """
        now = self.clock()
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        with self.store.transaction():
            lease = self.store.get_lease(lease_id)
            if lease is None or lease.state != LeaseState.ACTIVE:
                return None
            self.store.renew_lease(
                lease_id,
                ttl_seconds=ttl_seconds,
                expires_at=expires_at,
                heartbeat_at=now,
            )
            revived = []
            for gid in dict.fromkeys(lease.deployment_ids):
                deployment = self.store.get_deployment(gid)
                if (
                    deployment is not None
                    and deployment.state == DeploymentState.IDLE
                ):
                    self.store.set_deployment_state(
                        gid, DeploymentState.LIVE, now
                    )
                    revived.append(gid)
            if revived:  # demand is back -> an apply must run to re-up them
                self.store.bump_desired_generation()
        return self.store.get_lease(lease_id)

    def sweep(self) -> SweepResult:
        """Materialize TTL expiry and recompute idle deployments.

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
                affected.extend(lease.deployment_ids)
            idled = self._idle_deployments(affected, now)
            if expired or idled:  # TTL reclaim changed the desired set
                self.store.bump_desired_generation()
        return SweepResult(expired_lease_ids=expired, idled_deployment_ids=idled)

    def reclaimable_deployments(self) -> list[Deployment]:
        """IDLE deployments the reconciler may tear down per reclaim policy."""
        now = self.clock()
        return [
            g
            for g in self.store.list_deployments(now=now)
            if g.state == DeploymentState.IDLE
        ]

    def evict_idle(self, deployment_ids: list[str] | None = None) -> list[str]:
        """Force IDLE deployments to STOPPED so the next reconcile tears them down.

        This overrides ``keep-warm``: a released deployment normally stays resident
        (IDLE) to avoid cold-start thrash, but evicting it frees its GPU now.
        ``deployment_ids=None`` evicts every idle deployment; a list restricts it (ids not
        currently idle are skipped). Returns the ids actually evicted.
        """
        now = self.clock()
        wanted = None if deployment_ids is None else set(deployment_ids)
        evicted: list[str] = []
        with self.store.transaction():
            for g in self.store.list_deployments(now=now):
                if g.state != DeploymentState.IDLE:
                    continue
                if wanted is not None and g.id not in wanted:
                    continue
                # IDLE with demand > 0 is an anomaly (e.g. a heartbeat landed
                # in the idle->reclaim window); never stop a protected deployment.
                if g.demand > 0:
                    continue
                self.store.set_deployment_state(g.id, DeploymentState.STOPPED, now)
                evicted.append(g.id)
            if evicted:  # desired set shrank -> an apply must run to tear them down
                self.store.bump_desired_generation()
        return evicted

    def prune(self) -> tuple[int, int]:
        """Forget terminal entries: released/expired leases + stopped deployments.

        Sweep/evict leave a tail of RELEASED/EXPIRED leases and STOPPED deployments
        in the ledger so history stays inspectable; this clears that tail once
        you no longer care. Returns ``(n_leases, n_deployments)`` removed.
        """
        return self.store.prune(
            lease_states=(LeaseState.RELEASED, LeaseState.EXPIRED),
            deployment_states=(DeploymentState.STOPPED,),
        )

    def status(self) -> tuple[list[Lease], list[Deployment]]:
        """Snapshot for ``infer-stack status`` (leases, deployments-with-demand)."""
        now = self.clock()
        return self.store.list_leases(), self.store.list_deployments(now=now)

    # -- generation (coalesced-apply coordination, see store + controller) --

    def desired_generation(self) -> int:
        return self.store.desired_generation()

    def applied_generation(self) -> int:
        return self.store.applied_generation()

    def set_applied_generation(self, gen: int) -> None:
        self.store.set_applied_generation(gen)

    def get_lease(self, lease_id: str) -> Lease | None:
        return self.store.get_lease(lease_id)

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        return self.store.get_deployment(deployment_id, now=self.clock())

    # -- internals ---------------------------------------------------------

    def _find_or_create_deployment(
        self, req: EndpointRequest, now: float
    ) -> Deployment:
        if req.sharing == Sharing.DEDICATED:
            return self._create_deployment(req, now)
        candidates = self.store.deployments_by_compat(
            req.compat_key,
            sharing=Sharing.SHARED,
            states=(DeploymentState.LIVE, DeploymentState.IDLE),
        )
        for deployment in candidates:
            if capacity_satisfies(deployment.capacity, req.capacity):
                if deployment.state == DeploymentState.IDLE:
                    self.store.set_deployment_state(
                        deployment.id, DeploymentState.LIVE, now
                    )
                self._merge_served(deployment, req, now)
                return self.store.get_deployment(deployment.id) or deployment
        return self._create_deployment(req, now)

    def _create_deployment(
        self, req: EndpointRequest, now: float
    ) -> Deployment:
        deployment = Deployment(
            id=self.id_factory('grp'),
            compat_key=req.compat_key,
            engine=req.engine,
            sharing=req.sharing,
            capacity=dict(req.capacity),
            spec=dict(req.spec),
            served={req.endpoint: dict(req.served)},
            state=DeploymentState.LIVE,
            created_at=now,
            updated_at=now,
        )
        self.store.insert_deployment(deployment)
        return deployment

    def _merge_served(
        self, deployment: Deployment, req: EndpointRequest, now: float
    ) -> None:
        """Add an endpoint (e.g. a new Ollama tag) to a coalesced deployment."""
        served = dict(deployment.served)
        if served.get(req.endpoint) != req.served:
            served[req.endpoint] = dict(req.served)
            self.store.update_deployment_served(deployment.id, served, now)

    def _idle_deployments(self, deployment_ids: list[str], now: float) -> list[str]:
        idled: list[str] = []
        for gid in dict.fromkeys(deployment_ids):
            deployment = self.store.get_deployment(gid)
            if deployment is None or deployment.state != DeploymentState.LIVE:
                continue
            if self.store.demand(gid, now) == 0:
                self.store.set_deployment_state(gid, DeploymentState.IDLE, now)
                idled.append(gid)
        return idled
