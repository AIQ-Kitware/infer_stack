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
from .models import Deployment, DeploymentState, EndpointRequest, Lease, LeaseState

KEEP_WARM = 'keep-warm'

#: After an interrupted apply, how long to wait for the runtime to stop changing
#: before applying again, and how often to sample it. Held under the lock.
SETTLE_DEADLINE_S = 60.0
SETTLE_INTERVAL_S = 2.0

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
    # Admission mode: idle keep-warm residents that yielded their GPUs, and
    # LIVE deployments whose committed allocation is no longer valid.
    displaced: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


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
class ReleaseLeasesOutcome:
    # Leases this call moved to RELEASED (already-released ones are not listed).
    released_lease_ids: list[str]
    # Requested ids that do not exist in the ledger at all.
    missing_lease_ids: list[str]
    idled_deployment_ids: list[str]
    evicted_deployment_ids: list[str]
    # None when nothing changed, so nothing was rendered or applied.
    reconcile: ReconcileResult | None = None


@dataclass
class RenewOutcome:
    # None when the lease is unknown or no longer ACTIVE.
    lease: Lease | None
    # IDLE deployments this renew made LIVE again (empty for a TTL-only renew).
    revived_deployment_ids: list[str]
    # None for a TTL-only renew, which changes no desired state.
    reconcile: ReconcileResult | None = None


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
        # Published profile (see leasing/profile.py): this process's own
        # resolved settings, captured before the backend is switched to the
        # published copy, and the copy the backend currently renders from.
        self._invocation_profile: dict | None = None
        self._applied_profile: dict | None = None
        self._profile_drift_warned = False
        # A published profile for another backend kind is reported on the
        # first mutation, not here: `config publish` must still be able to
        # open a controller in order to switch backends.
        self._profile_error: Exception | None = None
        # Set only by apply_now(): the operator explicitly re-approves a render
        # that differs from an earlier approved digest.
        self._explicit_apply = False
        self._admission_digest: str | None = None
        from .profile import ProfileMismatch

        try:
            self._sync_profile(create=False)
        except ProfileMismatch as ex:
            self._profile_error = ex

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
        placement = None
        if self._admission_mode():
            # Residency decides which idle keep-warm deployments are candidates
            # at all; unknown residency fails the render (the change stays
            # pending) rather than guessing which warm models exist.
            residency = self.backend.residency()
            self._backfill_allocations(residency)
            self._prepare_network()
            desired, placement = self._admission_view(residency)
            self.backend.adopted = self._prune_adopted(residency)
        else:
            desired = self.desired_deployments()
        # Bound once rather than probed with hasattr: the capability check is
        # the same, and the bound method keeps its type instead of narrowing
        # to `object` the way an attribute reached through hasattr does.
        converge = getattr(self.backend, 'converge', None)
        if converge is not None:
            before = set(self.backend.observe())
            try:
                if placement is not None:
                    converge(desired, apply=False, placement=placement)
                else:
                    converge(desired, apply=False)
            except TypeError:
                # Legacy converge(desired) with no apply kwarg renders+applies
                # in one shot (no separate apply()); accept that here.
                converge(desired)
            after = set(self.backend.observe())
            rec = ReconcileResult(
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
                displaced=list(getattr(self.backend, 'last_displaced', ()) or ()),
                degraded=list(getattr(self.backend, 'last_degraded', ()) or ()),
            )
            if placement is not None:
                self._adopt_existing(residency)
            return rec
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

    # -- admission (plan steps P5, P6, P9) ------------------------------------
    #
    # Backends with strict residency and an in-memory preview (Compose) get
    # admission semantics:
    #   * a LIVE deployment holds a committed allocation (assigned_gpus);
    #   * an IDLE keep-warm deployment is only an optional candidate, and only
    #     while it is uniquely resident; it yields its GPUs to demand and is
    #     never started;
    #   * an acquire is previewed in memory (placement AND render) and commits
    #     its lease together with its allocations, or commits nothing.
    # Other backends (KubeAI, where the cluster schedules; test fakes) keep the
    # previous behaviour.

    def _admission_mode(self) -> bool:
        return all(
            callable(getattr(self.backend, name, None))
            for name in ('residency', 'preview', 'converge')
        )

    def _admission_view(self, residency, *, overlay=None):
        """``(desired deployments, PlacementInputs)`` for a render or preview.

        ``overlay`` (an :class:`~infer_stack.leasing.ledger.AcquireOverlay`)
        adds a candidate acquire on top of the ledger, without writing it.
        """
        from .placement import PlacementInputs

        _, deployments = self.ledger.status()
        by_id = {g.id: g for g in deployments}
        fresh: set[str] = set()
        if overlay is not None:
            by_id.update(overlay.deployments)
            fresh = set(overlay.created) | set(overlay.revived)
        required: dict[str, Deployment] = {}
        hard: dict[str, list[int]] = {}
        hints: dict[str, list[int]] = {}
        optional: list[Deployment] = []
        for gid, deployment in by_id.items():
            if deployment.state == DeploymentState.LIVE:
                if deployment.assigned_gpus is not None:
                    required[gid] = deployment
                    hard[gid] = list(deployment.assigned_gpus)
                elif gid in fresh:
                    required[gid] = deployment      # the candidate: a new placement
                # else: unresolved (a LIVE row from before allocations with no
                # unique container). Never freshly placed; it blocks new
                # allocation until released (see _admit).
            elif deployment.state == DeploymentState.IDLE:
                if deployment.spec.get('reclaim', self.reclaim_default) != KEEP_WARM:
                    continue
                resident = residency.resident(gid) if residency is not None else None
                if resident is not None and not resident.all_gpus:
                    hints[gid] = list(resident.gpus)
                    optional.append(deployment)
        inputs = PlacementInputs(
            required_ids=set(required), hard=hard, optional_hints=hints,
        )
        return [*required.values(), *optional], inputs

    def _prepare_network(self) -> None:
        """Give the backend the stable-address table, once a network is migrated."""
        config = self.ledger.network_config()
        if config is None:
            self.backend.network = None
            return
        self.backend.network = {
            'subnet': config['subnet'], 'addresses': self.ledger.service_addresses()}
        self.backend.on_addresses = self.ledger.add_service_addresses

    def network_migrate(self, subnet: str, *, force: bool = False) -> ReconcileResult:
        """``infer-stack network migrate``: move the project to stable addresses.

        Operator-invoked. Refused while any lease is ACTIVE unless ``force``,
        because every container, the gateway included, is recreated once.
        Rejects a subnet that overlaps an existing Docker network or host
        route. Changing an already-migrated subnet reassigns every address.
        """
        import ipaddress

        from .network import overlapping_subnets
        from .profile import ProfileMismatch

        subnet = str(ipaddress.ip_network(subnet))
        with self._global_lock():
            leases, _ = self.ledger.status(virtual_expiry=True)
            active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
            if active and not force:
                raise ProfileMismatch(
                    f'network migrate recreates every container; {len(active)} lease(s) '
                    'are active (release them, or pass --force)'
                )
            clash = overlapping_subnets(subnet, self.backend.run)
            if clash:
                raise ProfileMismatch(f'subnet {subnet} overlaps: {"; ".join(clash)}')
            current = self.ledger.network_config()
            reset = current is not None and current['subnet'] != subnet
            # Preview the migrated render and take approval BEFORE any write:
            # a declined migration must leave the subnet and addresses as they were.
            self._sync_profile(create=True)
            residency = self.backend.residency()
            saved = self.backend.network
            self.backend.network = {
                'subnet': subnet,
                'addresses': {} if reset else self.ledger.service_addresses(),
            }
            try:
                desired, inputs = self._admission_view(residency)
                self.backend.preview(desired, inputs, approve=True)
            finally:
                self.backend.network = saved
            self.ledger.store.migrate_network(
                subnet=subnet, reset_addresses=reset,
                approved_digest=getattr(self.backend, 'last_preview_digest', None),
            )
            return self._publish()

    def observe_state(self) -> dict:
        """A read-only health view for ``leases`` / ``status`` (plan step P10).

        Never writes: TTL expiry is virtual, residency is a Docker read. Each
        deployment gets one condition:

        * ``unknown``: Docker could not be read;
        * ``ambiguous``: more than one container claims it;
        * ``degraded``: LIVE, but its committed GPUs are no longer valid (it is
          neither started nor removed until its lease is released);
        * ``displaced``: an idle keep-warm model that yielded its GPUs;
        * ``unresolved``: LIVE from before allocations, with no unique container;
        * ``running`` / ``not-running``: whether its single container serves.
        """
        from .profile import profile_drift
        from .residency import ResidencyUnknown

        leases, deployments = self.ledger.status(virtual_expiry=True)
        sidecar = {}
        loader = getattr(self.backend, '_load_sidecar', None)
        if loader is not None:
            try:
                sidecar = loader() or {}
            except Exception:  # noqa: BLE001 - a view must always render
                sidecar = {}
        residency, residency_error = None, None
        if callable(getattr(self.backend, 'residency', None)):
            try:
                residency = self.backend.residency()
            except ResidencyUnknown as ex:
                residency_error = str(ex)
        degraded = set(sidecar.get('degraded') or ())
        displaced = set(sidecar.get('displaced') or ())
        rows = []
        for g in deployments:
            if g.state not in (DeploymentState.LIVE, DeploymentState.IDLE):
                continue
            if residency is None and residency_error is not None:
                condition = 'unknown'
            elif residency is not None and residency.ambiguous(g.id):
                condition = 'ambiguous'
            elif g.id in degraded and g.state == DeploymentState.LIVE:
                condition = 'degraded'
            elif g.id in displaced and g.state == DeploymentState.IDLE:
                condition = 'displaced'
            elif (g.state == DeploymentState.LIVE and g.assigned_gpus is None
                  and self._admission_mode() and residency is not None
                  and residency.resident(g.id) is None):
                condition = 'unresolved'
            elif residency is not None:
                condition = 'running' if residency.resident(g.id) else 'not-running'
            else:
                condition = None
            rows.append({'id': g.id, 'state': g.state, 'condition': condition,
                         'assigned_gpus': g.assigned_gpus})
        adopted = self.ledger.adopted_containers() or {}
        orphans = [] if residency is None else [
            {'id': c.container_id, 'service': c.service, 'state': c.state}
            for c in residency.all_containers()
            if not c.labelled and c.container_id not in adopted
        ]
        drift = []
        stored = self.ledger.profile()
        if stored is not None and self._invocation_profile is not None:
            drift = profile_drift(stored, self._invocation_profile)
        return {
            'publication_pending': self.ledger.publication_pending(),
            'residency_error': residency_error,
            'deployments': rows,
            'orphans': orphans,
            'profile_drift': drift,
            'network': self.ledger.network_config(),
            'addresses': self.ledger.service_addresses(),
            'expired_unswept': [le.id for le in leases if le.state == LeaseState.EXPIRED
                                and (self.ledger.get_lease(le.id).state == LeaseState.ACTIVE)],
        }

    def _prune_adopted(self, residency) -> dict:
        """The adopted-container table, minus containers that no longer exist."""
        adopted = self.ledger.adopted_containers()
        if not adopted:
            return {}
        present = {c.container_id for c in residency.all_containers()}
        kept = {cid: info for cid, info in adopted.items() if cid in present}
        if kept != adopted:
            self.ledger.set_adopted_containers(kept)
        return kept

    def _adopt_existing(self, residency) -> None:
        """One-time migration: adopt project containers from before ownership labels.

        Runs after the first admission-mode render on a ledger. A container
        without infer-stack's service and fingerprint labels is adopted, with
        the fingerprint of this render, if it is infrastructure the render
        still has, or a deployment container whose deployment is LIVE on the
        same GPUs or is an idle resident. Adopted containers are kept until
        their service's fingerprint changes; the first recreation stamps the
        labels. Anything else stays an orphan: reported, never removed
        implicitly. Adoption recreates nothing.
        """
        if self.ledger.adopted_containers() is not None:
            return
        fingerprints = self.backend._load_sidecar().get('fingerprints') or {}
        services = self.backend._load_sidecar().get('services') or {}
        _, deployments = self.ledger.status()
        by_id = {g.id: g for g in deployments}
        adopted = {}
        for c in residency.all_containers():
            if c.labelled or c.service not in fingerprints:
                continue
            if c.deployment_id:
                deployment = by_id.get(c.deployment_id)
                if deployment is None or services.get(c.service) != c.deployment_id:
                    continue
                live_here = (deployment.state == DeploymentState.LIVE
                             and list(c.gpus) == list(deployment.assigned_gpus or []))
                resident = (deployment.state == DeploymentState.IDLE
                            and residency.resident(c.deployment_id) is not None)
                if not (live_here or resident):
                    continue
            adopted[c.container_id] = {
                'service': c.service, 'fingerprint': fingerprints[c.service]}
        self.ledger.set_adopted_containers(adopted)
        self.backend.adopted = adopted

    def remove_orphans(self, confirm: Callable[[list], bool]) -> list:
        """``gc --orphans``: remove the project's unmanaged containers, with consent.

        Takes a strict snapshot under the lock, lists containers infer-stack
        neither labelled nor adopted, and removes exactly those if ``confirm``
        (shown the list) returns True. Returns the removed containers.
        """
        with self._global_lock():
            residency = self.backend.residency()
            adopted = self.ledger.adopted_containers() or {}
            orphans = [c for c in residency.all_containers()
                       if not c.labelled and c.container_id not in adopted]
            if not orphans or not confirm(orphans):
                return []
            self.backend.run(['docker', 'rm', '-f', *[c.container_id for c in orphans]])
            return orphans

    def _backfill_allocations(self, residency) -> list[str]:
        """Adopt allocations for LIVE deployments that predate them.

        A LIVE deployment with no committed allocation adopts the GPUs of its
        unique warm container. One with no container, several, or unmappable
        GPUs stays unresolved: it keeps running where it is, but no new GPU is
        allocated to anyone until it is released. Returns the unresolved ids.
        """
        from .placement import required_gpu_count

        unresolved = []
        _, deployments = self.ledger.status()
        for deployment in deployments:
            if deployment.state != DeploymentState.LIVE or deployment.assigned_gpus is not None:
                continue
            if required_gpu_count(deployment) == 0:
                self.ledger.set_allocation(deployment.id, [])
                continue
            resident = residency.resident(deployment.id)
            if resident is not None and not resident.all_gpus and resident.gpus:
                self.ledger.set_allocation(deployment.id, list(resident.gpus))
            else:
                unresolved.append(deployment.id)
        return unresolved

    def _admit(self, overlay, residency):
        """Decide an acquire in memory: ``(allocations, reasons)``.

        ``reasons`` empty means admissible, and ``allocations`` are the GPUs to
        commit for the candidate's new and revived deployments. Nothing is
        written here.
        """
        from .placement import required_gpu_count

        adopted: dict[str, list[int]] = {}
        need: list[str] = []
        for gid, deployment in overlay.deployments.items():
            fresh = gid in overlay.created or gid in overlay.revived
            if not fresh:
                continue
            if required_gpu_count(deployment) == 0:
                continue
            resident = residency.resident(gid) if (
                residency is not None and gid in overlay.revived) else None
            if resident is not None and not resident.all_gpus and resident.gpus:
                adopted[gid] = list(resident.gpus)      # IDLE->LIVE keeps its GPUs
            else:
                need.append(gid)
        if residency is None and (need or adopted):
            return {}, [
                'Docker residency is unknown, so only requests that need no new '
                'GPU are admitted; retry when `docker ps` works'
            ]
        if need:
            unresolved = self._unresolved_allocations(exclude=set(overlay.deployments))
            if unresolved:
                return {}, [
                    'no new GPU is allocated while LIVE deployments have no '
                    f'committed allocation ({", ".join(unresolved[:3])}); '
                    'release their leases to resolve'
                ]
        for gid, gpus in adopted.items():
            overlay.deployments[gid].assigned_gpus = gpus
        self._prepare_network()
        desired, inputs = self._admission_view(residency, overlay=overlay)
        plan, rendered = self.backend.preview(desired, inputs)
        reasons = []
        unresolved = set(self._unresolved_allocations())
        # EVERY deployment the candidate claims must be placed and renderable,
        # including an existing one it only coalesces onto (whose served
        # aliases it would change).
        for gid in overlay.deployments:
            if gid in unresolved:
                reasons.append(
                    f'deployment {gid} is an unresolved pre-allocation deployment and '
                    'cannot accept new demand; release its existing lease first'
                )
                continue
            if gid in plan.degraded:
                reasons.append(f'{gid}: its GPUs are no longer available')
            elif gid not in plan.assignments:
                why = [e for e in plan.errors if e.startswith(gid)]
                reasons.extend(why or [f'{gid}: could not be placed'])
            elif gid in set(rendered.unrenderable):
                why = [e for e in rendered.errors if gid in e] or [
                    f'{gid}: could not be rendered ({e})' for e in rendered.errors]
                reasons.extend(why or [f'{gid}: could not be rendered'])
        allocations = {gid: list(plan.assignments.get(gid, [])) for gid in overlay.deployments
                       if gid in overlay.created or gid in overlay.revived}
        if reasons:
            holders = self._gpu_holders(plan)
            if holders:
                reasons.append('GPUs held by admitted demand: ' + '; '.join(holders))
        self._admission_digest = None
        if not reasons:
            # Approval happens now, before anything is committed; the render
            # after the commit produces the same files and does not ask again.
            # Its digest goes into the pending marker, so a recovery after a
            # crash (and perhaps an upgrade) cannot apply something else.
            self.backend.preview(desired, inputs, approve=True)
            self._admission_digest = getattr(self.backend, 'last_preview_digest', None)
        for gid in adopted:
            overlay.deployments[gid].assigned_gpus = None   # committed with the lease
        return allocations, reasons

    def _gpu_holders(self, plan) -> list[str]:
        """``GPU n: deployment (owner, ...)`` for every GPU placed in ``plan``."""
        leases, _ = self.ledger.status()
        owners: dict[str, set[str]] = {}
        for le in leases:
            if le.state == LeaseState.ACTIVE:
                for gid in le.deployment_ids:
                    owners.setdefault(gid, set()).add(le.owner)
        out = []
        for gid, gpus in sorted(plan.assignments.items(), key=lambda kv: kv[1]):
            if gpus and gid in owners:
                out.append(f'GPU {",".join(map(str, gpus))}: {gid} '
                           f'(owner {", ".join(sorted(owners[gid]))})')
        return out

    def _unresolved_allocations(self, *, exclude: set[str] = frozenset()) -> list[str]:
        from .placement import required_gpu_count

        _, deployments = self.ledger.status()
        return [
            g.id for g in deployments
            if g.state == DeploymentState.LIVE and g.assigned_gpus is None
            and g.id not in exclude and required_gpu_count(g) > 0
        ]

    def _sync_profile(self, *, create: bool) -> None:
        """Make the backend render from the published profile.

        With ``create`` (every mutation, under the lock), a ledger without a
        profile gets this invocation's resolved settings frozen as the initial
        one. Settings that differ from the published profile are ignored, with
        one warning per process. Backends without a profile (null, test fakes)
        are left alone.
        """
        from .._log import logger
        from .profile import profile_drift

        render = getattr(self.backend, 'render_profile', None)
        use = getattr(self.backend, 'use_profile', None)
        read = getattr(self.ledger, 'profile', None)
        if render is None or use is None or read is None:
            return
        if create and self._profile_error is not None:
            raise self._profile_error
        stored = read()
        if stored is None and not create:
            return
        if self._invocation_profile is None:
            self._invocation_profile = render()
        if stored is None:
            stored = self._invocation_profile
            self.ledger.set_profile(stored)
            logger.info(
                'Froze the initial render profile ({} backend, {} catalog(s)); '
                'change it with `infer-stack config publish` while no leases are active',
                stored.get('backend'), len(stored.get('catalogs') or []),
            )
        elif not self._profile_drift_warned:
            drift = profile_drift(stored, self._invocation_profile)
            if drift:
                self._profile_drift_warned = True
                logger.warning(
                    'Settings differ from the published profile ({}); rendering '
                    'from the published profile. Change it with `infer-stack '
                    'config publish` while no leases are active.', ', '.join(drift),
                )
        if stored != self._applied_profile:
            use(stored)
            self._applied_profile = stored

    def _mark_pending(
        self, *, apply: bool, placement_context: dict | None = None,
        create_profile: bool = True,
    ) -> dict:
        """Record that desired state is about to change (caller holds the lock).

        Written before the ledger mutation, so a crash between the two leaves at
        worst a redundant marker, never a mutation without one. ``apply`` only
        ever turns ``apply_requested`` on: a staged change never cancels an
        apply already requested (promotion, see the plan's D23).

        If an earlier acquire died between committing and its first render, its
        placement scope is still in the marker: render once with that scope
        first, so its deployment is placed where that caller was allowed.
        """
        if create_profile:
            self._sync_profile(create=True)
        current = self.ledger.publication_pending()
        if current and current.get('placement_context'):
            self._render_in_scope(current['placement_context'])
        return self.ledger.mark_publication_pending(
            apply_requested=apply, placement_context=placement_context,
        )

    def _render_in_scope(self, context: dict) -> None:
        """Render with another caller's admission scope, then forget the scope.

        The scope is cleared only after a render that succeeded (and so pinned
        the placement). A declined or failed render propagates and keeps it,
        so no later caller can place that deployment within its own scope.
        """
        scope = getattr(self.backend, 'placement_scope', None)
        if scope is not None:
            with scope(context):
                rec = self._render()
        self.ledger.clear_placement_context()

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
        approved = marker.get('approved_digest')
        rendered = getattr(self.backend, 'last_planned_digest', None)
        if approved and rendered and rendered != approved and not self._explicit_apply:
            from .profile import ProfileMismatch

            rec.publication_pending = True
            raise ProfileMismatch(
                'the rendered state differs from what was approved at publication '
                '(e.g. infer-stack was upgraded in between); review and approve it '
                'with `infer-stack apply`'
            )
        if marker['interrupted']:
            self._wait_for_settled_runtime()
        before = set(self.backend.observe())
        try:
            ok = apply_fn()
        except BaseException as ex:
            from .compose import ApplyAborted

            # A killed or failed client does not stop work the daemon already
            # started: the next apply must first wait for the runtime to settle.
            # A refusal (ApplyAborted) started nothing, so it needs no settling.
            self.ledger.mark_publication_pending(
                apply_requested=True, interrupted=not isinstance(ex, ApplyAborted))
            rec.publication_pending = True
            raise
        if approved:
            # The approved render reached Docker; a retry for routes (or any
            # later change) renders from newer state and needs no re-approval.
            self.ledger.store.clear_approved_digest()
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

    def _wait_for_settled_runtime(self) -> None:
        """Block (bounded) until two consecutive runtime samples are identical.

        Called before applying on top of an interrupted apply. A sample that
        cannot be read, a container still ``removing``, or a runtime that keeps
        changing past :data:`SETTLE_DEADLINE_S` raises
        :class:`~infer_stack.leasing.backend.RuntimeUnsettled` and leaves the
        change pending. Backends without ``settle_snapshot`` (KubeAI's apply is
        declarative server-side) skip the check.
        """
        from .backend import RuntimeUnsettled

        snapshot = getattr(self.backend, 'settle_snapshot', None)
        if snapshot is None:
            return
        deadline = self.clock() + SETTLE_DEADLINE_S
        previous = None
        while True:
            try:
                current = snapshot()
            except Exception as ex:  # noqa: BLE001 - unreadable is unsettled
                raise RuntimeUnsettled(
                    f'cannot read the runtime after an interrupted apply: {ex}; '
                    'the change stays pending'
                ) from ex
            busy = any(state == 'removing' for _, state in current)
            if previous is not None and current == previous and not busy:
                return
            if self.clock() + SETTLE_INTERVAL_S > deadline:
                raise RuntimeUnsettled(
                    'the runtime was still changing after an interrupted apply '
                    f'(waited {SETTLE_DEADLINE_S:g}s); the change stays pending -- '
                    'retry with `infer-stack apply`'
                )
            previous = current
            self.sleep(SETTLE_INTERVAL_S)

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
        anything pending, including leases staged with ``--no-apply``. It is
        also the explicit approval that clears an approved-digest mismatch.
        """
        self._explicit_apply = True
        try:
            return self.reconcile(apply=True)
        finally:
            self._explicit_apply = False

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

    def _never_ran(self, deployment_ids: list[str]) -> list[str]:
        """Which of these deployments definitely have no container at all.

        Uses strict residency where the backend has it: a deployment with any
        container, in any state, is kept, and if Docker cannot be read nothing
        is reported (so nothing warm is ever evicted on a failed look). Backends
        without residency fall back to ``observe()``.
        """
        residency = getattr(self.backend, 'residency', None)
        if residency is None:
            running = set(self.backend.observe())
            return [gid for gid in deployment_ids if gid not in running]
        from .residency import ResidencyUnknown

        try:
            snap = residency()
        except ResidencyUnknown:
            return []
        return [gid for gid in deployment_ids if not snap.containers(gid)]

    def _rollback_acquire(self, lease_id: str, *, apply: bool) -> ReconcileResult | None:
        """Roll a failed acquire back and publish the result, under the lock.

        Releases the lease, evicts any deployment the release idled that is not
        actually running (keep-warm only means something for a deployment that
        came up: a pre-existing warm deployment this lease merely coalesced onto
        stays resident, but a never-ran one would pin a phantom in the desired
        set -- and an unplaceable one would be re-planned, and re-fail, on every
        future render), then re-renders and applies per the marker, tearing
        down anything of this lease an earlier apply brought up.

        With ``apply=False`` the rollback only renders: it never applies, even
        if an apply is already pending (a ``--no-apply`` acquire, or a rollback
        right after a failed apply whose runtime state is unknown).

        Best-effort after the ledger change: the original failure must surface,
        not a failure of this cleanup. Anything that did not publish stays
        pending behind the marker.
        """
        from .._log import logger
        from .backend import ConvergeAborted

        with self._global_lock():
            self._mark_pending(apply=apply)
            # The approved admission is being compensated away: its digest no
            # longer describes the pending desired state.
            self.ledger.store.clear_approved_digest()
            rel = self.ledger.release(lease_id)
            if rel.idled_deployment_ids:
                never_ran = self._never_ran(list(rel.idled_deployment_ids))
                if never_ran:
                    self.ledger.evict_idle(never_ran)
            try:
                rec = self._render()
                return self._apply_pending(rec) if apply else rec
            except ConvergeAborted:
                # The rollback render normally diffs clean, but an operator can
                # still decline an unrelated swept-in change.
                return None
            except Exception as ex:  # noqa: BLE001 - see docstring
                logger.warning('rollback publication failed; it stays pending: {!r}', ex)
                return None

    def _apply_admitted(
        self, rec: ReconcileResult, lease_id: str, *, apply: bool
    ) -> ReconcileResult:
        """Apply a placed acquire's render (caller holds the lock).

        ``apply=False`` never applies, even over an older pending apply. If the
        apply raises, the lease is released in the ledger (re-rendered, not
        re-applied, since the runtime state is unknown) and the error re-raised,
        so a caller never loses the ID of a lease that is still ACTIVE. The
        release stays pending and the next apply publishes it.
        """
        if not apply:
            rec.publication_pending = True
            return rec
        try:
            return self._apply_pending(rec)
        except BaseException:
            self._rollback_acquire(lease_id, apply=False)
            raise

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

        if self._admission_mode():
            result, rec = self._acquire_by_admission(
                owner, requests, ttl_seconds=ttl_seconds, apply=apply,
                wait_for_placement=wait_for_placement,
                placement_timeout=timeout if placement_timeout is None else placement_timeout,
                placement_interval=interval if placement_interval is None else placement_interval,
            )
            return self._finish_acquire(result, rec, apply=apply, wait=wait,
                                        timeout=timeout, interval=interval)

        # Intent, ledger write, render and (once placed) apply all under one lock
        # hold, so a second caller blocks before touching sqlite and no render
        # can change the files this apply reads. The readiness wait and the
        # admission-queue sleep stay OUTSIDE the lock.
        with self._global_lock():
            self._sync_profile(create=True)
            validate = getattr(self.backend, 'validate_requests', None)
            if validate is not None:
                validate(requests)          # before anything is written
            context = getattr(self.backend, 'placement_context', lambda: None)()
            self._mark_pending(apply=apply, placement_context=context)
            result = self.ledger.acquire(
                owner, requests, ttl_seconds=ttl_seconds
            )
            try:
                try:
                    rec = self._render()
                finally:
                    # Rendered (placement pinned) or about to roll back: either
                    # way a recovery no longer needs this caller's scope.
                    if context is not None:
                        self.ledger.clear_placement_context()
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
                rec = self._apply_admitted(rec, result.lease.id, apply=apply)
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
                        rec = self._apply_admitted(rec, result.lease.id, apply=True)
        if unplaced:
            self._rollback_acquire(result.lease.id, apply=apply)
            reasons = [
                e
                for e in rec.placement_errors
                if any(e.startswith(gid) for gid in unplaced)
            ]
            raise PlacementError(sorted(unplaced), reasons)
        return self._finish_acquire(result, rec, apply=apply, wait=wait,
                                    timeout=timeout, interval=interval)

    def _acquire_by_admission(
        self, owner, requests, *, ttl_seconds, apply, wait_for_placement,
        placement_timeout, placement_interval,
    ):
        """Admission-mode acquire: preview in memory, commit only if admissible.

        Each attempt runs under the lock: sweep (itself a published mutation),
        observe residency, overlay the request on the ledger, and preview
        placement and render. An admissible attempt commits the lease with its
        allocations and publishes; an inadmissible one writes nothing, and a
        queued caller sleeps outside the lock and tries again.
        """
        from .backend import ConvergeAborted, PlacementError
        from .ledger import AdmissionConflict
        from .residency import ResidencyUnknown

        deadline = self.clock() + placement_timeout
        checked_feasible = False
        while True:
            with self._global_lock():
                self._sync_profile(create=True)
                validate = getattr(self.backend, 'validate_requests', None)
                if validate is not None:
                    validate(requests)
                leases, _ = self.ledger.status(virtual_expiry=True)
                if any(le.state == LeaseState.EXPIRED and
                       self.ledger.get_lease(le.id).state == LeaseState.ACTIVE
                       for le in leases):
                    # Maintenance sweep: TTL expiry is a desired-state change.
                    self._mark_pending(apply=True)
                    self.ledger.sweep()
                    self._publish()
                try:
                    residency = self.backend.residency()
                except ResidencyUnknown:
                    residency = None
                overlay = self.ledger.plan_acquire(requests)
                allocations, reasons = self._admit(overlay, residency)
                if not reasons:
                    # Allocations are committed with the lease, so no placement
                    # scope needs recording for recovery.
                    context = None
                    self._mark_pending(apply=apply)
                    try:
                        result = self.ledger.acquire(
                            owner, requests, ttl_seconds=ttl_seconds,
                            overlay=overlay, allocations=allocations,
                            # Nothing is applied for --no-apply, so there is no
                            # approval to guard; staged state stays discardable.
                            approved_digest=self._admission_digest if apply else None,
                        )
                    except AdmissionConflict:
                        continue            # the ledger moved; preview again
                    try:
                        try:
                            rec = self._render()
                        finally:
                            if context is not None:
                                self.ledger.clear_placement_context()
                    except ConvergeAborted:
                        self._rollback_acquire(result.lease.id, apply=apply)
                        raise
                    except BaseException:
                        # e.g. residency became unreadable between preview and
                        # render: never leave a lease the caller cannot see.
                        self._rollback_acquire(result.lease.id, apply=False)
                        raise
                    requested = {g.id for g in result.deployments}
                    unplaced = requested & set(rec.unplaced)
                    if unplaced:
                        # Admitted by preview but not placed by the render: the
                        # two disagree, which must never be silent.
                        self._rollback_acquire(result.lease.id, apply=apply)
                        raise PlacementError(sorted(unplaced), [
                            e for e in rec.placement_errors
                            if any(e.startswith(g) for g in unplaced)
                        ] or ['admission preview and render disagreed'])
                    rec = self._apply_admitted(rec, result.lease.id, apply=apply)
                    return result, rec
                blocked = sorted(
                    gid for gid in overlay.deployments
                    if gid in overlay.created or gid in overlay.revived
                )
            if not (wait_for_placement and apply):
                raise PlacementError(blocked, reasons)
            if not checked_feasible:
                checked_feasible = True
                infeasible = self._infeasible_alone(
                    list(overlay.deployments.values()), set(blocked))
                if infeasible:
                    raise PlacementError(sorted(infeasible), sorted(infeasible.values()))
            if self.clock() + placement_interval > deadline:
                raise PlacementError(blocked, reasons)
            self.sleep(placement_interval)

    def _finish_acquire(self, result, rec, *, apply, wait, timeout, interval) -> AcquireOutcome:
        """Readiness wait (outside the lock) and the outcome, for both paths."""
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

    def release_leases(
        self, lease_ids: Iterable[str] | None = None, *, evict: bool = False,
    ) -> ReleaseLeasesOutcome:
        """Release several leases (``None``: every ACTIVE one) in one publication.

        One render and one apply for the whole batch, so an interactive backend
        asks at most once. ``evict`` also tears down, now, the deployments the
        release idled -- or, with ``lease_ids=None``, every idle deployment --
        overriding keep-warm. Ids that do not exist are reported, not raised;
        if nothing at all would change, no marker is taken and nothing applies.
        """
        with self._global_lock():
            if lease_ids is None:
                self._mark_pending(apply=True)
                self.ledger.sweep()
                leases, _ = self.ledger.status()
                ids = [le.id for le in leases if le.state == LeaseState.ACTIVE]
                missing: list[str] = []
            else:
                requested = list(dict.fromkeys(lease_ids))
                missing = [s for s in requested if self.ledger.get_lease(s) is None]
                ids = [s for s in requested if s not in missing]
                if not ids:
                    return ReleaseLeasesOutcome([], missing, [], [])
                self._mark_pending(apply=True)
            released, idled = [], []
            for sid in ids:
                rel = self.ledger.release(sid)
                if not rel.already_released:
                    released.append(sid)
                idled.extend(rel.idled_deployment_ids)
            idled = list(dict.fromkeys(idled))
            evicted: list[str] = []
            if evict and lease_ids is None:
                evicted = self.ledger.evict_idle(None)     # every idle deployment
            elif evict and idled:
                evicted = self.ledger.evict_idle(idled)
            rec = self._publish()
        return ReleaseLeasesOutcome(released, missing, idled, list(evicted), rec)

    def publish_change(self, change: Callable[[], object]) -> tuple[object, ReconcileResult]:
        """Run ``change`` and publish, as one serialised desired-state mutation.

        For changes to backend state that the render reads (the route registry,
        via ``routes seed`` / ``routes prune``). ``change`` runs under the lock
        after the marker is set, so a crash leaves it pending like any other
        mutation. Returns ``(change's result, reconcile result)``.
        """
        with self._global_lock():
            self._mark_pending(apply=True)
            result = change()
            rec = self._publish()
        return result, rec

    def publish_profile(self, profile: dict) -> ReconcileResult:
        """Replace the published profile and publish, while the stack is quiescent.

        Refuses (:class:`~infer_stack.leasing.profile.ProfileMismatch`) with any
        ACTIVE lease or any deployment container, including warm idle ones:
        changing render inputs under a running workload is out of scope until
        live publication (plan step P4). The new profile's render is previewed
        through the backend's usual diff approval before the profile is stored;
        a declined or failed render stores nothing.
        """
        from .profile import ProfileMismatch
        from .residency import ResidencyUnknown

        use = getattr(self.backend, 'use_profile', None)
        if use is None:
            raise ProfileMismatch('this backend has no publishable profile')
        stored = self.ledger.profile()
        if stored is not None and stored.get('backend') != profile.get('backend'):
            # The quiescence check below can only see the NEW backend's
            # resources, so the old backend's would be left running unseen.
            raise ProfileMismatch(
                f"changing the backend ({stored.get('backend')} -> "
                f"{profile.get('backend')}) is not supported by config publish; "
                'tear the old stack down and use a new ledger for the new backend'
            )
        with self._global_lock():
            leases, _ = self.ledger.status(virtual_expiry=True)
            active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
            if active:
                raise ProfileMismatch(
                    f'config publish needs a quiescent stack: {len(active)} active '
                    f'lease(s) ({", ".join(active[:3])}); release them first'
                )
            residency = getattr(self.backend, 'residency', None)
            if residency is not None:
                try:
                    snap = residency()
                except ResidencyUnknown as ex:
                    raise ProfileMismatch(
                        f'config publish cannot confirm the stack is quiescent: {ex}'
                    ) from ex
                running = sorted({c.deployment_id for c in snap.all_containers()
                                  if c.deployment_id})
                if running:
                    raise ProfileMismatch(
                        'config publish needs a quiescent stack: deployment '
                        f'container(s) exist for {", ".join(running[:3])}; '
                        '`infer-stack evict --all` first'
                    )
            if self._admission_mode():
                # A PURE preview first: the real render persists append-only
                # state (route registry, addresses), which must not happen for
                # a candidate whose publication has not committed.
                previous = self._applied_profile
                use(profile)
                try:
                    residency = self.backend.residency()
                    self._prepare_network()
                    desired, inputs = self._admission_view(residency)
                    self.backend.preview(desired, inputs, approve=True)
                except BaseException:
                    if previous is not None:
                        use(previous)
                    raise
                self.ledger.store.publish_profile(
                    profile, approved_digest=self.backend.last_preview_digest)
                self._profile_error = None
                self._applied_profile = profile
                self._invocation_profile = profile
                self._profile_drift_warned = False
                return self._publish()
            # Backends without a preview (KubeAI): render, then commit.
            # No implicit profile here: on a fresh ledger a declined preview
            # must leave neither a profile nor a marker behind.
            existed = self.ledger.publication_pending() is not None
            marker = self._mark_pending(apply=True, create_profile=False)
            previous = self._applied_profile
            use(profile)
            try:
                rec = self._render()
            except BaseException:
                if previous is not None:
                    use(previous)
                if not existed:
                    self.ledger.clear_publication_pending(marker['version'])
                raise
            self.ledger.store.publish_profile(
                profile, approved_digest=getattr(self.backend, 'last_planned_digest', None))
            self._profile_error = None
            self._applied_profile = profile
            self._invocation_profile = profile
            self._profile_drift_warned = False
            return self._apply_pending(rec)

    def prune(self) -> tuple[int, int]:
        """Forget released/expired leases and stopped deployments, under the lock.

        Not a desired-state change (none of those are desired), so no marker and
        no apply. Serialised because an acquire may reuse a STOPPED deployment row.
        """
        with self._global_lock():
            return self.ledger.prune()

    def renew(self, lease_id: str, *, ttl_seconds: float | None) -> RenewOutcome:
        """Extend a lease's TTL; publish if that revives an idle deployment.

        **Fast path, lock-free:** an ACTIVE lease whose deployments are all LIVE
        only has its TTL and heartbeat updated, in one SQLite transaction that
        re-checks that condition. It changes no desired state and publishes
        nothing.

        **Slow path, under the lock:** a deployment went IDLE, so the renew is a
        desired-state change. In admission mode it is re-admitted: the IDLE
        deployment adopts its resident GPUs, or is placed fresh; if neither is
        possible the renew fails with :class:`PlacementError` and writes
        nothing. The lease is re-validated as ACTIVE under the lock first.
        """
        from .backend import PlacementError

        fast = self.ledger.renew_if_live(lease_id, ttl_seconds=ttl_seconds)
        if fast is not False:
            return RenewOutcome(fast, [])
        with self._global_lock():
            if self._admission_mode():
                lease = self.ledger.get_lease(lease_id)
                if lease is None or lease.state != LeaseState.ACTIVE:
                    return RenewOutcome(None, [])
                from .residency import ResidencyUnknown

                try:
                    residency = self.backend.residency()
                except ResidencyUnknown:
                    residency = None
                overlay = self.ledger.plan_acquire([])
                for gid in dict.fromkeys(lease.deployment_ids):
                    deployment = self.ledger.get_deployment(gid)
                    if deployment is not None and deployment.state == DeploymentState.IDLE:
                        deployment.state = DeploymentState.LIVE
                        overlay.deployments[gid] = deployment
                        overlay.revived.append(gid)
                if not overlay.revived:
                    return RenewOutcome(
                        self.ledger.renew(lease_id, ttl_seconds=ttl_seconds), [])
                allocations, reasons = self._admit(overlay, residency)
                if reasons:
                    raise PlacementError(list(overlay.revived), reasons)
                self._mark_pending(apply=True)
                if self._admission_digest:
                    self.ledger.mark_publication_pending(
                        apply_requested=True, approved_digest=self._admission_digest)
                renewed = self.ledger.renew(
                    lease_id, ttl_seconds=ttl_seconds, allocations=allocations)
                rec = self._publish()
                return RenewOutcome(renewed, list(overlay.revived), rec)
            lease = self.ledger.get_lease(lease_id)
            reviving = []
            if lease is not None and lease.state == LeaseState.ACTIVE:
                for gid in dict.fromkeys(lease.deployment_ids):
                    deployment = self.ledger.get_deployment(gid)
                    if deployment is not None and deployment.state == DeploymentState.IDLE:
                        reviving.append(gid)
            if not reviving:
                return RenewOutcome(
                    self.ledger.renew(lease_id, ttl_seconds=ttl_seconds), [],
                )
            self._mark_pending(apply=True)
            renewed = self.ledger.renew(lease_id, ttl_seconds=ttl_seconds)
            rec = self._publish()
        return RenewOutcome(renewed, reviving, rec)

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
