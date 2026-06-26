"""The controller: reconcile the ledger's desired state onto a backend.

This is the thin orchestration layer the design doc (§11) calls for. The ledger
is pure bookkeeping; a backend realizes deployments; the controller is what
turns ``acquire`` / ``release`` into "the right things are running and ready".

The reconcile loop is the standard desired-vs-actual converge:

    desired = LIVE deployments  +  IDLE deployments whose reclaim policy is keep-warm
    actual  = backend.observe()
    realize(desired - actual);  teardown(actual - desired)

``reclaim`` policy lives on each deployment's spec (from the catalog): ``keep-warm``
(default — survive idle until pressure, avoids cold-start thrash) keeps an idle
deployment running; ``stop`` / ``scale-to-zero`` let it be torn down as soon as demand
hits zero. TTL expiry is enforced here too: every ``reconcile`` first
``sweep``s the ledger, so a crashed job's lease stops protecting its deployment once
its TTL elapses.
"""

from __future__ import annotations

import contextlib
import fcntl
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .backend import Backend
from .ledger import Ledger
from .models import Deployment, DeploymentState, EndpointRequest, Lease

KEEP_WARM = 'keep-warm'

LOCK_FILENAME = '.leasing.lock'


@dataclass
class ReconcileResult:
    realized: list[str] = field(default_factory=list)
    torn_down: list[str] = field(default_factory=list)
    # Desired deployments the backend could not place (e.g. no free GPU) plus the
    # planner's per-deployment reasons; empty for backends without placement.
    unplaced: list[str] = field(default_factory=list)
    placement_errors: list[str] = field(default_factory=list)
    # deployment id -> GPU indices it is on / slated for (placement backends only).
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
    deployments: list[Deployment]
    reconcile: ReconcileResult
    wait: WaitResult | None = None
    applied: bool = True  # False for a --no-apply (staged, not brought up) acquire


@dataclass
class ReleaseOutcome:
    idled_deployment_ids: list[str]
    reconcile: ReconcileResult


@dataclass
class EvictOutcome:
    evicted_deployment_ids: list[str]
    reconcile: ReconcileResult


@dataclass
class GcOutcome:
    expired_lease_ids: list[str]
    idled_deployment_ids: list[str]
    evicted_deployment_ids: list[str]
    reconcile: ReconcileResult


class Controller:
    """Ties a :class:`Ledger` to a :class:`Backend`.

    Args:
        ledger: the lease bookkeeping store.
        backend: realizes/observes/probes deployments.
        clock: epoch-seconds source for wait timing; defaults to the ledger's.
        sleep: how to wait between readiness polls; injectable for tests.
        reclaim_default: policy for deployments whose spec omits one.
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
        self._lock_path = self._resolve_lock_path()
        # Intra-process serialization (reentrant for nested acquire->reconcile in
        # one thread; blocks other threads, e.g. the TUI's converge-while-monitor).
        self._tlock = threading.RLock()
        self._flock_handle = None
        self._flock_depth = 0

    # -- cross-process lock ------------------------------------------------

    def _resolve_lock_path(self) -> Path | None:
        """Where the global mutate lock lives: beside the shared ledger db.

        ``None`` for an in-memory ledger (tests/fakes) — there is no shared
        state to guard, so the lock degrades to a no-op.
        """
        path = getattr(getattr(self.ledger, 'store', None), 'path', None)
        if path and path != ':memory:':
            return Path(path).expanduser().parent / LOCK_FILENAME
        return None

    def _open_lock_handle(self):
        """Open an ``flock``-able handle for the cross-process lock.

        Normally the lock file sits beside the ledger. If that directory is not
        writable for a *new* file (e.g. a service-owned shared data dir that this
        user can read but not write), fall back to a host-shared temp path keyed
        by the ledger location, so same-host processes still serialize on a
        consistent file. If even that fails, return ``None`` and the caller
        proceeds with only the in-process lock. Opened append-mode (never
        truncated): the file is a pure ``flock`` token, its content is unused.
        """
        import hashlib
        import tempfile
        import warnings

        assert self._lock_path is not None
        digest = hashlib.sha1(str(self._lock_path).encode()).hexdigest()[:16]
        fallback = Path(tempfile.gettempdir()) / f'infer-stack-{digest}.lock'
        for path in (self._lock_path, fallback):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                return open(path, 'a')
            except OSError:
                continue
        warnings.warn(
            'infer-stack: could not open a cross-process lock file (tried '
            f'{self._lock_path} and {fallback}); proceeding with in-process '
            'locking only — concurrent CLIs on this host may race.'
        )
        return None

    @contextlib.contextmanager
    def _global_lock(self):
        """Serialize the whole state-mutating critical section, single-writer.

        Every verb that mutates shared state — ``acquire``/``release``/``gc``/
        ``evict`` — does a read-modify-write: a sqlite ledger write (``BEGIN
        IMMEDIATE``) **then** ``reconcile`` (render the compose project +
        ``docker compose up``). Without one lock over the *entire* sequence, two
        CLIs race: their ledger writes collide (sqlite ``database is locked``
        once one holds the write past the busy-timeout during a slow converge),
        and their renders diff against a target the other just moved. So the
        second caller must **block here before it touches sqlite**, not fail.

        Reentrant within a thread (nested ``acquire``->``reconcile`` is one
        flock), serialized across threads via ``_tlock``, and across processes
        via an exclusive ``flock`` on a file beside the ledger. Taken before the
        backend's own ``converge`` lock (a different file) so ordering is
        consistent and deadlock-free.

        NOT held during the readiness wait or the admission-queue sleep, so
        queued waiters and ``leases``/``status`` readers (WAL, lock-free) coexist.
        """
        if self._lock_path is None:
            yield
            return
        self._tlock.acquire()
        try:
            if self._flock_depth == 0:
                self._flock_handle = self._open_lock_handle()
                if self._flock_handle is not None:
                    fcntl.flock(self._flock_handle, fcntl.LOCK_EX)
            self._flock_depth += 1
            try:
                yield
            finally:
                self._flock_depth -= 1
                if self._flock_depth == 0 and self._flock_handle is not None:
                    try:
                        fcntl.flock(self._flock_handle, fcntl.LOCK_UN)
                    finally:
                        self._flock_handle.close()
                        self._flock_handle = None
        finally:
            self._tlock.release()

    # -- reconcile ---------------------------------------------------------

    def desired_deployments(self) -> list[Deployment]:
        """Deployments that should currently be running."""
        _, deployments = self.ledger.status()
        desired: list[Deployment] = []
        for deployment in deployments:
            if deployment.state == DeploymentState.LIVE:
                desired.append(deployment)
            elif deployment.state == DeploymentState.IDLE:
                policy = deployment.spec.get('reclaim', self.reclaim_default)
                if policy == KEEP_WARM:
                    desired.append(deployment)
        return desired

    def reconcile(self, *, apply: bool = True) -> ReconcileResult:
        """Converge the backend to the ledger's desired state.

        A backend may implement a single ``converge(desired)`` (whole-union
        convergence — e.g. Compose renders one file and ``up``s it); otherwise
        the per-deployment ``realize``/``teardown`` loop is used.

        ``apply=False`` asks a converge-style backend to render the on-disk
        project *without* bringing it up — the "see what would execute" / staged
        path. Backends without that capability ignore it (render and apply are
        inseparable for them).

        The entire desired-read + converge is held under :meth:`_global_lock` so
        concurrent CLIs cannot diff/apply against each other's stale target.
        """
        with self._global_lock():
            self.ledger.sweep()
            desired = self.desired_deployments()
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
                    unplaced=sorted(
                        getattr(self.backend, 'last_unplaced', ()) or ()
                    ),
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
            for deployment in desired:
                if deployment.id not in actual:
                    self.backend.realize(deployment)
                    result.realized.append(deployment.id)
            stale = actual - desired_ids
            if stale:
                by_id = {g.id: g for g in self.ledger.status()[1]}
                for gid in stale:
                    deployment = by_id.get(gid)
                    if deployment is not None:
                        self.backend.teardown(deployment)
                    result.torn_down.append(gid)
            return result

    # -- readiness ---------------------------------------------------------

    def wait_ready(
        self,
        deployments: Iterable[Deployment],
        *,
        endpoints: set[str] | None = None,
        timeout: float = 300.0,
        interval: float = 2.0,
    ) -> WaitResult:
        """Block until the requested served endpoints are ready or ``timeout``.

        ``endpoints`` filters which served names to wait on (a coalesced deployment
        may serve more than this caller asked for); ``None`` waits for all.
        """
        pairs = [
            (deployment, ep)
            for deployment in deployments
            for ep in sorted(deployment.served)
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
        wait_for_placement: bool = False,
        placement_timeout: float | None = None,
        placement_interval: float | None = None,
    ) -> AcquireOutcome:
        """Create a lease, realize its deployments, and (optionally) block on ready.

        ``apply=False`` stages the lease and renders the on-disk project without
        bringing it up (and skips the readiness wait, since nothing is running).
        Placement is still computed, so an unplaceable request still fails fast.

        ``wait_for_placement`` turns acquire into an *admission queue*: instead of
        failing fast when every GPU is busy, it polls until a deployment frees one.
        Each retry ``reconcile``s, which first ``sweep``s the ledger — so a crashed
        job's TTL-expired lease is reclaimed while we wait, and the freed GPU lets
        the queued request through. It is bounded by ``placement_timeout`` (default:
        ``timeout``); a request that can never fit (one exceeding total capacity)
        simply waits out the timeout and then fails. Default off, so interactive
        ``acquire``/``serve`` keep their fail-fast behavior; the pipeline opts in.

        .. note::
            Queueing is plain (no reservation): a multi-GPU request can be starved
            by a steady stream of single-GPU ones, since each freed GPU is up for
            grabs. Head-of-line GPU reservation is a follow-up; for the small-fleet
            case (few GPUs, rare multi-GPU jobs) plain queueing is sufficient.
        """
        from .backend import ConvergeAborted, PlacementError

        # Mutation (ledger write + first reconcile) under one lock, so a second
        # caller blocks here before touching sqlite rather than racing BEGIN
        # IMMEDIATE. The readiness wait and the admission-queue sleep stay OUTSIDE.
        with self._global_lock():
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
            # If a deployment this lease just requested could not be placed (e.g. no
            # free GPU), either queue for one (wait_for_placement) or — the default —
            # roll the lease back and report the planner's reason, so the deployment
            # never lingers as a phantom ``live`` with nothing behind it.
            requested = {g.id for g in result.deployments}
            unplaced = requested & set(rec.unplaced)
        if unplaced and wait_for_placement and apply:
            p_timeout = timeout if placement_timeout is None else placement_timeout
            p_interval = (
                interval if placement_interval is None else placement_interval
            )
            deadline = self.clock() + p_timeout
            while unplaced and self.clock() < deadline:
                self.sleep(p_interval)
                rec = self.reconcile(apply=apply)
                unplaced = requested & set(rec.unplaced)
        if unplaced:
            with self._global_lock():
                self.ledger.release(result.lease.id)
            reasons = [
                e
                for e in rec.placement_errors
                if any(e.startswith(gid) for gid in unplaced)
            ]
            raise PlacementError(sorted(unplaced), reasons)
        deployments = [self.ledger.get_deployment(g.id) for g in result.deployments]
        deployments = [g for g in deployments if g is not None]
        wait_result = None
        if wait and apply:  # nothing to wait on when we only staged the render
            wait_result = self.wait_ready(
                deployments,
                endpoints=set(result.lease.endpoints),
                timeout=timeout,
                interval=interval,
            )
        return AcquireOutcome(
            lease=result.lease,
            deployments=deployments,
            reconcile=rec,
            wait=wait_result,
            applied=apply,
        )

    def release(self, lease_id: str) -> ReleaseOutcome:
        """Release a lease and converge (tearing down per reclaim policy)."""
        with self._global_lock():
            rel = self.ledger.release(lease_id)
            rec = self.reconcile()
        return ReleaseOutcome(
            idled_deployment_ids=rel.idled_deployment_ids, reconcile=rec
        )

    def evict(self, deployment_ids: Iterable[str] | None = None) -> EvictOutcome:
        """Force-evict idle (released) deployments now, overriding keep-warm.

        Marks the matching IDLE deployments STOPPED and reconciles, so a keep-warm
        model that is merely resident gets torn down and its GPU freed.
        ``deployment_ids=None`` evicts every idle deployment.
        """
        ids = None if deployment_ids is None else list(deployment_ids)
        with self._global_lock():
            self.ledger.sweep()
            evicted = self.ledger.evict_idle(ids)
            rec = self.reconcile()
        return EvictOutcome(evicted_deployment_ids=evicted, reconcile=rec)

    def gc(self, *, evict_idle: bool = False) -> GcOutcome:
        """Reclaim TTL-expired leases and converge — the standalone leak backstop.

        Sweeps the ledger (a TTL-expired lease stops protecting its deployments),
        then reconciles so ``stop``-policy deployments left with no demand are torn
        down and their GPUs freed. This is what a blocking ``acquire`` does
        implicitly on each retry; as a standalone verb it cleans up after a
        hard-killed job — whose ``teardown``/``release`` never ran — on a schedule
        or as a final pipeline step. ``evict_idle`` additionally tears down idle
        *keep-warm* deployments (like ``evict --all``); without it, healthy
        keep-warm models are left resident and only leaked/expired demand is
        reclaimed.
        """
        with self._global_lock():
            swept = self.ledger.sweep()
            evicted = self.ledger.evict_idle(None) if evict_idle else []
            rec = self.reconcile()
        return GcOutcome(
            expired_lease_ids=list(swept.expired_lease_ids),
            idled_deployment_ids=list(swept.idled_deployment_ids),
            evicted_deployment_ids=list(evicted),
            reconcile=rec,
        )
