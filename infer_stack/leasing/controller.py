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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .backend import Backend
from .ledger import Ledger
from .models import Deployment, DeploymentState, EndpointRequest, Lease

KEEP_WARM = 'keep-warm'

# One cross-process lock, beside the ledger. It serialises every desired-state
# publication: the ledger mutation, the render, and the apply of that exact
# render (see Controller._publish). Reentrant (acquire->rollback nests). The
# readiness wait and the admission-queue sleep stay outside it.
LOCK_FILENAME = '.leasing.lock'


class LeaseLockError(RuntimeError):
    """A mutating verb could not obtain the cross-process lock, so it refuses to
    touch the shared ledger.

    The render critical section (ledger write + placement + compose render) is
    single-writer across processes via an ``flock`` on a file beside the shared
    ledger. When neither that file nor the host-temp fallback can be opened for
    writing, degrading to an in-process lock would let concurrent CLIs in other
    tmux/slurm sessions corrupt the ledger (colliding ``BEGIN IMMEDIATE`` writes,
    stale-diff renders) — the in-process lock only serializes threads of *one*
    process, and the concurrency here is separate CLI processes. We raise this
    instead. The message names the exact paths tried and the ``chgrp``/``chmod``
    fix that makes the lock dir shareable by every CLI identity.
    """


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
    # True when a publication marker is still set after this operation: the
    # desired state was staged (--no-apply, render) or its apply did not fully
    # succeed. The next applying operation, or `infer-stack apply`, publishes it.
    publication_pending: bool = False


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
        """Open an ``flock``-able handle for ``lock_path`` (``None`` if neither
        candidate is openable).

        Normally the lock file sits beside the ledger. If that directory is not
        writable for a *new* file (e.g. a service-owned shared data dir that this
        user can read but not write), fall back to a host-shared temp path keyed
        by the lock location, so same-host processes still serialize on a
        consistent file. A lock file (and the dir) we create is made
        group-writable (best-effort, see :meth:`_ensure_group_writable`) so the
        *other* tmux / slurm sessions running the CLI — which may be a different
        uid in the same group — can open the same file rather than colliding on
        permissions (a 0644 file owned by the first user is the classic ``/tmp``
        fallback collision).

        Returns ``None`` if neither path can be opened; the caller (a mutating
        verb) raises :class:`LeaseLockError` rather than silently degrading to an
        in-process lock, which would not serialize separate CLI processes.
        Opened append-mode (never truncated): the file is a pure ``flock`` token,
        its content is unused.
        """
        import hashlib
        import tempfile

        digest = hashlib.sha1(str(lock_path).encode()).hexdigest()[:16]
        fallback = Path(tempfile.gettempdir()) / f'infer-stack-{digest}.lock'
        for path in (lock_path, fallback):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(path, 'a')
            except OSError:
                continue
            self._ensure_group_writable(handle, path)
            return handle
        return None

    def _ensure_group_writable(self, handle, path: Path) -> None:
        """Best-effort: give the lock file and its directory group read+write so
        the next session in the owning group can share the same ``flock`` file.

        Only touches paths we own — a non-owner cannot ``chmod`` (and must not:
        a service-owned lock dir is fixed by an admin, not silently widened
        here). The directory also gets the setgid bit so files siblings create
        inherit the group. Idempotent and never raises: a lock we cannot widen
        is surfaced by :meth:`_diagnose_lock_failure` on the raise path, not here.
        """
        import os
        import stat

        uid = os.getuid()
        # The lock file: fchmod the open fd (race-free; skip if it is not ours).
        try:
            st = os.fstat(handle.fileno())
            if st.st_uid == uid:
                mode = stat.S_IMODE(st.st_mode)
                want = mode | stat.S_IRGRP | stat.S_IWGRP
                if want != mode:
                    os.fchmod(handle.fileno(), want)
        except OSError:
            pass
        # The directory: group rwx + setgid, so new lock/db files stay shareable.
        try:
            directory = path.parent
            dst = directory.stat()
            if dst.st_uid == uid:
                mode = stat.S_IMODE(dst.st_mode)
                want = (
                    mode
                    | stat.S_IRGRP
                    | stat.S_IWGRP
                    | stat.S_IXGRP
                    | stat.S_ISGID
                )
                if want != mode:
                    directory.chmod(want)
        except OSError:
            pass

    def _diagnose_lock_failure(self, lock_path: Path) -> str:
        """Build the actionable message for :class:`LeaseLockError`.

        Re-inspects the two candidate lock paths (beside-ledger + host-temp
        fallback) and reports, for each, *why* it could not be opened for
        writing — the parent dir is not writable, or the existing file is owned
        by another uid without group write — then names the ``chgrp``/``chmod``
        fix that makes the shared lock dir usable by every CLI identity.
        """
        import hashlib
        import os
        import tempfile

        digest = hashlib.sha1(str(lock_path).encode()).hexdigest()[:16]
        fallback = Path(tempfile.gettempdir()) / f'infer-stack-{digest}.lock'
        tried = '\n'.join(
            f'  - {p}: {self._lock_path_reason(p)}' for p in (lock_path, fallback)
        )
        directory = lock_path.parent
        return (
            f'infer-stack: refusing to mutate the shared ledger as uid={os.getuid()} '
            '— no usable cross-process lock.\n'
            f'{tried}\n'
            'A mutating verb (acquire/release/gc/evict) needs an flock file every '
            'CLI on this host can open for writing; without it, concurrent CLIs in '
            'other tmux/slurm sessions can corrupt the ledger. Make the lock '
            'directory shareable by every CLI identity, e.g.:\n'
            f'  chgrp -R <shared-group> {directory} && chmod -R g+rwX {directory} '
            f'&& chmod g+s {directory}\n'
            '  (and run the CLIs under `umask 002` so new files keep group write)\n'
            'or run all sessions as the same user.'
        )

    def _lock_path_reason(self, path: Path) -> str:
        """One-line reason a single candidate lock path is not openable."""
        import os
        import stat

        try:
            if path.exists():
                st = path.stat()
                mode = stat.S_IMODE(st.st_mode)
                grp_w = (
                    'group-writable'
                    if mode & stat.S_IWGRP
                    else 'NOT group-writable'
                )
                return (
                    f'file exists (owner uid={st.st_uid}, mode={oct(mode)}, '
                    f'{grp_w}); not openable for append by uid={os.getuid()}'
                )
            parent = path.parent
            if not parent.exists():
                return f'parent dir {parent} is missing and could not be created'
            pst = parent.stat()
            pmode = stat.S_IMODE(pst.st_mode)
            if not os.access(parent, os.W_OK | os.X_OK):
                return (
                    f'parent dir {parent} not writable by uid={os.getuid()} '
                    f'(owner uid={pst.st_uid}, mode={oct(pmode)})'
                )
            return 'open failed'
        except OSError as exc:
            return f'inspection failed: {exc}'

    @contextlib.contextmanager
    def _global_lock(self):
        """Serialize desired-state publication, single-writer.

        Every verb that mutates shared state — ``acquire``/``release``/``gc``/
        ``evict`` — does a read-modify-write: a sqlite ledger write (``BEGIN
        IMMEDIATE``) **then** the render (placement + compose-file write). Without
        one lock over *that* sequence, two CLIs race: their ledger writes collide
        (sqlite ``database is locked``) and their renders diff against a target
        the other just moved. So the second caller must **block here before it
        touches sqlite**, not fail.

        Also held for the apply of that render (:meth:`_publish`), so no render
        can change the files an apply is reading. Backend calls under it are
        bounded (see ``compose.DOCKER_TIMEOUT_*``). NOT held during the readiness
        wait or the admission-queue sleep.

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
                if self._flock_handle is None:
                    raise LeaseLockError(
                        self._diagnose_lock_failure(self._lock_path)
                    )
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

    def _infeasible_alone(self, deployments, requested: set) -> dict:
        """Which of this lease's deployments cannot be placed even on an idle host.

        Returns ``{deployment_id: reason}``, empty when the whole set fits.
        Backends without an idle-host planner (null, kubeai) return empty --
        the check is an optimisation over waiting, never a new failure mode.
        """
        plan_alone = getattr(self.backend, 'plan_on_idle_host', None)
        if plan_alone is None:
            return {}
        try:
            plan = plan_alone(list(deployments))
        except Exception as ex:  # noqa: BLE001 -- never fail an acquire on the check
            # Waiting (the old behavior) is still correct, so a broken check
            # must not break acquire. But it must not be silent either: a
            # swallowed AttributeError here disables the check on every call
            # and is indistinguishable from "everything is feasible".
            warnings.warn(f'feasibility check skipped: {ex!r}',
                          RuntimeWarning, stacklevel=2)
            return {}
        # GpuPlan reports what it PLACED; unplaced is the complement. It has no
        # ``unplaced``/``placement_errors`` -- those belong to ReconcileResult.
        blocked = requested - set(plan.assignments)
        if not blocked:
            return {}
        reasons = {}
        for gid in sorted(blocked):
            why = next((e for e in plan.errors if e.startswith(gid)), gid)
            reasons[gid] = (
                f'{why} -- and that is the whole host with nothing else '
                f'running, so waiting cannot help'
            )
        return reasons

    def _render(self) -> ReconcileResult:
        """Render the desired state to disk WITHOUT the slow apply.

        The caller MUST hold :meth:`_global_lock`. For a converge-style backend
        (Compose) this writes the compose project (placement + files) but does
        not ``docker compose up`` -- that is :meth:`_apply_pending`, under the
        same lock hold. A per-deployment ``realize``/``teardown`` backend has no
        such split, so it realizes here; those backends expose no ``apply``.
        """
        self.ledger.sweep()
        desired = self.desired_deployments()
        # Bound once rather than probed with hasattr: the capability check is
        # the same, and the bound method keeps its type instead of narrowing
        # to `object` the way an attribute reached through hasattr does.
        converge = getattr(self.backend, 'converge', None)
        if converge is not None:
            before = set(self.backend.observe())
            try:
                converge(desired, apply=False)
            except TypeError:
                # Legacy converge(desired) with no apply kwarg renders+applies
                # in one shot (no separate apply()); accept that here.
                converge(desired)
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

    # -- serialised publication --------------------------------------------
    #
    # Every desired-state mutation, under one _global_lock hold:
    #   1. record intent: set (or extend) the durable publication marker;
    #   2. mutate the ledger;
    #   3. render from the ledger;
    #   4. if the marker requests an apply, apply that exact render;
    #   5. clear the marker only after the whole apply succeeded (in dynamic
    #      routing mode, including verified routes).
    # A crash or failure anywhere leaves the marker set, and the next applying
    # operation re-renders and applies the whole pending desired state.

    def _mark_pending(self, *, apply: bool) -> dict:
        """Record that desired state is about to change (caller holds the lock).

        Written before the ledger mutation, so a crash between the two leaves at
        worst a redundant marker, never a mutation without one. ``apply`` only
        ever turns ``apply_requested`` on: a staged change never cancels an
        apply already requested (promotion, see the plan's D23).
        """
        return self.ledger.mark_publication_pending(apply_requested=apply)

    def _apply_pending(self, rec: ReconcileResult) -> ReconcileResult:
        """Apply the last render if the marker requests it; clear on success.

        The caller holds :meth:`_global_lock` and has just rendered. Backend
        exceptions propagate with the marker still set. A backend ``apply`` that
        returns ``False`` did not fully take effect: the marker stays, the
        result says so, and the operation carries on.
        """
        from .._log import logger

        marker = self.ledger.publication_pending()
        if marker is None:
            return rec
        apply_fn = getattr(self.backend, 'apply', None)
        if apply_fn is None:
            # realize/teardown backends applied during the render itself.
            self.ledger.clear_publication_pending(marker['version'])
            rec.publication_pending = False
            return rec
        if not marker['apply_requested']:
            rec.publication_pending = True    # staged; never applied here
            return rec
        before = set(self.backend.observe())
        ok = apply_fn()
        after = set(self.backend.observe())
        rec.realized = sorted(set(rec.realized) | (after - before))
        rec.torn_down = sorted(set(rec.torn_down) | (before - after))
        rec.applied = True
        if ok is False:
            rec.publication_pending = True
            logger.warning(
                'apply did not fully take effect; the change stays pending and '
                'the next acquire/release or `infer-stack apply` retries it'
            )
            return rec
        self.ledger.clear_publication_pending(marker['version'])
        rec.publication_pending = False
        return rec

    def _publish(self) -> ReconcileResult:
        """Render, then apply per the marker (caller holds the lock)."""
        return self._apply_pending(self._render())

    def reconcile(self, *, apply: bool = True) -> ReconcileResult:
        """Render the ledger's desired state, then (if ``apply``) bring it up.

        ``apply=False`` is the render-only path (``infer-stack render``): it
        writes the on-disk project and never applies, even if an apply is
        pending. ``apply=True`` requests an apply and publishes the whole
        pending desired state.
        """
        with self._global_lock():
            if not apply:
                # The render sweeps, which can change desired state: record it
                # as staged (this never requests an apply).
                self._mark_pending(apply=False)
                rec = self._render()
                rec.publication_pending = True
                return rec
            self._mark_pending(apply=True)
            return self._publish()

    def apply_now(self) -> ReconcileResult:
        """Render and apply unconditionally: the manual ``infer-stack apply``.

        Heals drift (re-ups a container that died out-of-band) and publishes
        anything pending, including leases staged with ``--no-apply``.
        """
        return self.reconcile(apply=True)

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

    def _rollback_acquire(self, lease_id: str, *, apply: bool) -> ReconcileResult | None:
        """Roll a failed acquire back and publish the result, under the lock.

        Releases the lease, evicts any deployment the release idled that is not
        actually running (keep-warm only means something for a deployment that
        came up: a pre-existing warm deployment this lease merely coalesced onto
        stays resident, but a never-ran one would pin a phantom in the desired
        set -- and an unplaceable one would be re-planned, and re-fail, on every
        future render), then re-renders and applies per the marker, tearing
        down anything of this lease an earlier apply brought up.

        Best-effort after the ledger change: the original failure must surface,
        not a failure of this cleanup. Anything that did not publish stays
        pending behind the marker.
        """
        from .._log import logger
        from .backend import ConvergeAborted

        with self._global_lock():
            self._mark_pending(apply=apply)
            rel = self.ledger.release(lease_id)
            if rel.idled_deployment_ids:
                running = set(self.backend.observe())
                never_ran = [
                    gid for gid in rel.idled_deployment_ids
                    if gid not in running
                ]
                if never_ran:
                    self.ledger.evict_idle(never_ran)
            try:
                return self._publish()
            except ConvergeAborted:
                # The rollback render normally diffs clean, but an operator can
                # still decline an unrelated swept-in change.
                return None
            except Exception as ex:  # noqa: BLE001 - see docstring
                logger.warning('rollback publication failed; it stays pending: {!r}', ex)
                return None

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

        # Intent, ledger write, render and (once placed) apply all under one lock
        # hold, so a second caller blocks before touching sqlite and no render
        # can change the files this apply reads. The readiness wait and the
        # admission-queue sleep stay OUTSIDE the lock.
        with self._global_lock():
            self._mark_pending(apply=apply)
            result = self.ledger.acquire(
                owner, requests, ttl_seconds=ttl_seconds
            )
            try:
                rec = self._render()
            except ConvergeAborted:
                # The operator declined the compose changes -- don't leave the
                # just-created lease dangling in the ledger.
                self._rollback_acquire(result.lease.id, apply=apply)
                raise
            # If a deployment this lease just requested could not be placed (e.g. no
            # free GPU), either queue for one (wait_for_placement) or -- the default --
            # roll the lease back and report the planner's reason, so the deployment
            # never lingers as a phantom ``live`` with nothing behind it.
            requested = {g.id for g in result.deployments}
            unplaced = requested & set(rec.unplaced)
            if not unplaced:
                rec = self._apply_pending(rec)
        if unplaced and wait_for_placement and apply:
            # Never queue for capacity that cannot exist. Re-plan this lease's
            # deployments ALONE on an idle host: if they do not fit there, no
            # amount of waiting will help, and waiting is actively harmful --
            # the lease holds whatever it did place for the whole timeout, so a
            # request that was never satisfiable can block ones that are.
            #
            # Only the aggregate case needs this. A single deployment too large
            # for any card is already caught by the planner's permanent branch;
            # what is missed is a lease whose deployments cannot fit TOGETHER,
            # e.g. a 4-GPU model plus a 1-GPU extractor on a 4-GPU host.
            infeasible = self._infeasible_alone(result.deployments, requested)
            if infeasible:
                self._rollback_acquire(result.lease.id, apply=apply)
                raise PlacementError(sorted(infeasible.keys()),
                                     sorted(infeasible.values()))
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
                    self._mark_pending(apply=True)   # the render sweeps
                    rec = self._render()
                    unplaced = requested & set(rec.unplaced)
                    if not unplaced:
                        rec = self._apply_pending(rec)
        if unplaced:
            self._rollback_acquire(result.lease.id, apply=apply)
            reasons = [
                e
                for e in rec.placement_errors
                if any(e.startswith(gid) for gid in unplaced)
            ]
            raise PlacementError(sorted(unplaced), reasons)
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
            self._mark_pending(apply=True)
            rel = self.ledger.release(lease_id)
            rec = self._publish()
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
            self._mark_pending(apply=True)
            self.ledger.sweep()
            evicted = self.ledger.evict_idle(ids)
            rec = self._publish()
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
            self._mark_pending(apply=True)
            swept = self.ledger.sweep()
            evicted = self.ledger.evict_idle(None) if evict_idle else []
            rec = self._publish()
        return GcOutcome(
            expired_lease_ids=list(swept.expired_lease_ids),
            idled_deployment_ids=list(swept.idled_deployment_ids),
            evicted_deployment_ids=list(evicted),
            reconcile=rec,
        )
