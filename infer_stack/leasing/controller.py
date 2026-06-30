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

# Two cross-process locks, beside the ledger:
#  - render lock: serializes the fast read-modify-write (sqlite ledger mutation +
#    placement + compose-file render). Reentrant (acquire->render nests).
#  - apply lock: serializes the slow `docker compose up` and doubles as the
#    coalescing wait-queue (see Controller._ensure_applied). Split from render so
#    a second caller can render while the first is still applying.
LOCK_FILENAME = '.leasing.lock'
APPLY_LOCK_FILENAME = '.apply.lock'


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
    # True when the readiness wait timed out and the controller rolled the lease
    # back (released + reconciled) so a never-ready acquire doesn't pin a GPU.
    released_on_timeout: bool = False


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
        self._apply_lock_path = self._resolve_apply_lock_path()
        # Intra-process serialization (reentrant for nested acquire->reconcile in
        # one thread; blocks other threads, e.g. the TUI's converge-while-monitor).
        self._tlock = threading.RLock()
        self._flock_handle = None
        self._flock_depth = 0

    # -- cross-process lock ------------------------------------------------

    def _resolve_lock_path(self) -> Path | None:
        """Where the render (mutate) lock lives: beside the shared ledger db.

        ``None`` for an in-memory ledger (tests/fakes) — there is no shared
        state to guard, so the lock degrades to a no-op.
        """
        return self._lock_beside_ledger(LOCK_FILENAME)

    def _resolve_apply_lock_path(self) -> Path | None:
        """Where the apply (coalescing) lock lives: beside the shared ledger db."""
        return self._lock_beside_ledger(APPLY_LOCK_FILENAME)

    def _lock_beside_ledger(self, filename: str) -> Path | None:
        path = getattr(getattr(self.ledger, 'store', None), 'path', None)
        if path and path != ':memory:':
            return Path(path).expanduser().parent / filename
        return None

    def _open_lock_handle(self):
        """Open an ``flock``-able handle for the render lock (back-compat shim)."""
        assert self._lock_path is not None
        return self._open_flock(self._lock_path)

    def _open_flock(self, lock_path: Path):
        """Open an ``flock``-able handle for ``lock_path``.

        Normally the lock file sits beside the ledger. If that directory is not
        writable for a *new* file (e.g. a service-owned shared data dir that this
        user can read but not write), fall back to a host-shared temp path keyed
        by the lock location, so same-host processes still serialize on a
        consistent file. If even that fails, return ``None`` and the caller
        proceeds with only the in-process lock. Opened append-mode (never
        truncated): the file is a pure ``flock`` token, its content is unused.
        """
        import hashlib
        import tempfile
        import warnings

        digest = hashlib.sha1(str(lock_path).encode()).hexdigest()[:16]
        fallback = Path(tempfile.gettempdir()) / f'infer-stack-{digest}.lock'
        for path in (lock_path, fallback):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                return open(path, 'a')
            except OSError:
                continue
        warnings.warn(
            'infer-stack: could not open a cross-process lock file (tried '
            f'{lock_path} and {fallback}); proceeding with in-process '
            'locking only — concurrent CLIs on this host may race.'
        )
        return None

    @contextlib.contextmanager
    def _global_lock(self):
        """Serialize the RENDER critical section, single-writer.

        Every verb that mutates shared state — ``acquire``/``release``/``gc``/
        ``evict`` — does a read-modify-write: a sqlite ledger write (``BEGIN
        IMMEDIATE``) **then** the render (placement + compose-file write). Without
        one lock over *that* sequence, two CLIs race: their ledger writes collide
        (sqlite ``database is locked``) and their renders diff against a target
        the other just moved. So the second caller must **block here before it
        touches sqlite**, not fail.

        Held only for the FAST render. The slow ``docker compose up`` is applied
        afterward under the separate :meth:`_apply_lock` (coalesced), so a second
        caller can render while the first is still applying. NOT held during the
        readiness wait or the admission-queue sleep either.

        Reentrant within a thread (nested ``acquire``->``_render`` is one flock),
        serialized across threads via ``_tlock``, and across processes via an
        exclusive ``flock`` on a file beside the ledger.
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

    @contextlib.contextmanager
    def _apply_lock(self):
        """Serialize ``docker compose up`` across processes (the apply queue).

        A fresh ``flock`` handle per call: ``flock(LOCK_EX)`` blocks until the
        current applier finishes, so waiters form a natural queue. Whoever wakes
        first re-checks ``applied_generation`` (in :meth:`_ensure_applied`) and
        either finds itself already covered or does the one apply for the batch.
        Auto-released if the holder dies (crash-safe; a redundant idempotent apply
        is the worst case). Distinct file from the render lock and never nested
        inside it, so the two cannot deadlock.
        """
        if self._apply_lock_path is None:
            yield
            return
        handle = self._open_flock(self._apply_lock_path)
        if handle is None:
            yield
            return
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()

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

    def _render(self) -> ReconcileResult:
        """Render the desired state to disk WITHOUT the slow apply.

        The caller MUST hold :meth:`_global_lock`. For a converge-style backend
        (Compose) this writes the compose project (placement + files) but does
        not ``docker compose up`` -- that is the separate, coalesced apply step
        (:meth:`_ensure_applied`). A per-deployment ``realize``/``teardown``
        backend has no such split, so it realizes here (old behavior); those
        backends expose no ``apply`` and ``_ensure_applied`` is a no-op for them.
        """
        self.ledger.sweep()
        desired = self.desired_deployments()
        if hasattr(self.backend, 'converge'):
            before = set(self.backend.observe())
            try:
                self.backend.converge(desired, apply=False)
            except TypeError:
                # Legacy converge(desired) with no apply kwarg renders+applies
                # in one shot (no separate apply()); accept that here.
                self.backend.converge(desired)
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
                applied=False,
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

    def _ensure_applied(
        self, g_target: int, rec: ReconcileResult | None = None
    ) -> None:
        """Coalesced apply: ensure an apply reflecting desired-gen ``g_target``
        has run, then return — without re-applying if someone already covered us.

        See :mod:`infer_stack.leasing.store` for the generation contract. One
        apply satisfies every waiter that rendered before it; a late render
        (``g_target`` advanced during an in-flight apply) takes the apply-lock
        next and re-applies. The snapshot ``g`` is read *before* the apply so it
        is a guaranteed-covered floor (a render that lands mid-apply is handled by
        the next iteration, never silently dropped). Backends without ``apply``
        (realize/teardown, null) applied during render, so this is a no-op.
        """
        apply_fn = getattr(self.backend, 'apply', None)
        if apply_fn is None:
            return
        while self.ledger.applied_generation() < g_target:
            with self._apply_lock():
                if self.ledger.applied_generation() >= g_target:
                    break  # an applier we queued behind already covered us
                g = self.ledger.desired_generation()  # floor snapshot BEFORE up
                before = set(self.backend.observe())
                apply_fn()
                after = set(self.backend.observe())
                self.ledger.set_applied_generation(g)
                if rec is not None:
                    rec.realized = sorted(set(rec.realized) | (after - before))
                    rec.torn_down = sorted(set(rec.torn_down) | (before - after))
                    rec.applied = True
                break  # g >= g_target, so we are now covered

    def reconcile(self, *, apply: bool = True) -> ReconcileResult:
        """Render the ledger's desired state, then (if ``apply``) bring it up.

        Render runs under :meth:`_global_lock`; the apply is coalesced under
        :meth:`_ensure_applied` (separate apply-lock), so renders are not blocked
        by another caller's slow ``docker compose up``. ``apply=False`` stages the
        on-disk project without bringing it up (the "see what would execute" path).
        """
        with self._global_lock():
            rec = self._render()
            g_target = self.ledger.desired_generation()
        if apply:
            self._ensure_applied(g_target, rec)
        return rec

    def apply_now(self) -> ReconcileResult:
        """Render, then FORCE an apply even if the generation has not advanced.

        This is the manual ``infer-stack apply``: reconcile reality to the
        rendered desired state unconditionally (drift healing — re-up a container
        that died out-of-band). The automatic acquire/release path uses the
        coalesced :meth:`_ensure_applied` instead, which skips a redundant apply.
        """
        with self._global_lock():
            rec = self._render()
        apply_fn = getattr(self.backend, 'apply', None)
        if apply_fn is not None:
            with self._apply_lock():
                before = set(self.backend.observe())
                apply_fn()
                after = set(self.backend.observe())
                self.ledger.set_applied_generation(
                    self.ledger.desired_generation()
                )
                rec.realized = sorted(set(rec.realized) | (after - before))
                rec.torn_down = sorted(set(rec.torn_down) | (before - after))
                rec.applied = True
        return rec

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

        A readiness wait that *times out* (``wait=True`` and the endpoints never
        become ready within ``timeout``) is the third "acquire couldn't deliver"
        path, alongside :class:`ConvergeAborted` and :class:`PlacementError`: the
        lease is rolled back (released + reconciled, tearing down per reclaim
        policy) so a never-ready acquire doesn't leave a LIVE deployment pinning a
        GPU. The outcome is returned with ``released_on_timeout=True`` (not raised,
        so callers can still inspect ``wait.pending``). Use ``wait=False`` to hold
        a lease while a slow model loads and wait for it separately.

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

        # Ledger write + RENDER under the render lock, so a second caller blocks
        # here before touching sqlite rather than racing BEGIN IMMEDIATE. The slow
        # apply (docker compose up) is coalesced under the SEPARATE apply lock, and
        # the readiness wait + admission-queue sleep stay OUTSIDE both -- so a
        # second caller can render while this one is still applying/waiting.
        with self._global_lock():
            result = self.ledger.acquire(
                owner, requests, ttl_seconds=ttl_seconds
            )
            try:
                rec = self._render()
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
            g_target = self.ledger.desired_generation()
        if unplaced and wait_for_placement and apply:
            p_timeout = timeout if placement_timeout is None else placement_timeout
            p_interval = (
                interval if placement_interval is None else placement_interval
            )
            deadline = self.clock() + p_timeout
            while unplaced and self.clock() < deadline:
                self.sleep(p_interval)
                # Re-render under the lock: each retry sweeps (reclaiming a crashed
                # job's TTL-expired lease) and re-plans against the freed GPUs.
                with self._global_lock():
                    rec = self._render()
                    unplaced = requested & set(rec.unplaced)
                    g_target = self.ledger.desired_generation()
        if unplaced:
            with self._global_lock():
                self.ledger.release(result.lease.id)
            reasons = [
                e
                for e in rec.placement_errors
                if any(e.startswith(gid) for gid in unplaced)
            ]
            raise PlacementError(sorted(unplaced), reasons)
        # Coalesced apply (bring the rendered project up), OUTSIDE the render lock.
        if apply:
            self._ensure_applied(g_target, rec)
        deployments = [self.ledger.get_deployment(g.id) for g in result.deployments]
        deployments = [g for g in deployments if g is not None]
        wait_result = None
        released_on_timeout = False
        if wait and apply:  # nothing to wait on when we only staged the render
            wait_result = self.wait_ready(
                deployments,
                endpoints=set(result.lease.endpoints),
                timeout=timeout,
                interval=interval,
            )
            if not wait_result.ready:
                # The endpoints never became ready. Roll the lease back (release +
                # reconcile) so a timed-out acquire doesn't leave the deployment
                # LIVE holding a GPU indefinitely -- the readiness analogue of the
                # ConvergeAborted / PlacementError rollbacks above.
                self.release(result.lease.id)
                released_on_timeout = True
        return AcquireOutcome(
            lease=result.lease,
            deployments=deployments,
            reconcile=rec,
            wait=wait_result,
            applied=apply,
            released_on_timeout=released_on_timeout,
        )

    def release(self, lease_id: str) -> ReleaseOutcome:
        """Release a lease and converge (tearing down per reclaim policy)."""
        with self._global_lock():
            rel = self.ledger.release(lease_id)
            rec = self._render()
            g_target = self.ledger.desired_generation()
        self._ensure_applied(g_target, rec)
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
            rec = self._render()
            g_target = self.ledger.desired_generation()
        self._ensure_applied(g_target, rec)
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
            rec = self._render()
            g_target = self.ledger.desired_generation()
        self._ensure_applied(g_target, rec)
        return GcOutcome(
            expired_lease_ids=list(swept.expired_lease_ids),
            idled_deployment_ids=list(swept.idled_deployment_ids),
            evicted_deployment_ids=list(evicted),
            reconcile=rec,
        )
