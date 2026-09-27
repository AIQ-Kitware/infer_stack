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
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from .backend import (
    ApplyResult,
    ConnectionInfo,
    FrontDoorControl,
    HostRuntime,
    ServingBackend,
)
from .routes import RoutePlan
from .ledger import Ledger
from .models import Deployment, DeploymentState, EndpointRequest, Lease, LeaseState

_T = TypeVar('_T')

KEEP_WARM = 'keep-warm'
#: Between two evictions made to fit leased demand: long enough for the
#: freed instance to terminate and the scheduler to retry.
ROOM_COOLDOWN_S = 30.0

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
    # False when reconcile only rendered the on-disk state (nothing brought up or taken down).
    applied: bool = True
    # True when a publication marker is still set after this operation: the
    # desired state was staged (--no-apply, render) or its apply did not fully
    # succeed. The next applying operation, or `infer-stack apply`, publishes it.
    publication_pending: bool = False
    # Whether the runtime reached the render (routes may still be unverified),
    # and the apply's own words when it did not fully take effect.
    runtime_applied: bool = False
    apply_detail: str = ''
    # Admission mode: idle keep-warm residents that yielded their GPUs, and
    # LIVE deployments whose committed allocation is no longer valid.
    displaced: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


@dataclass
class WaitResult:
    ready: bool
    pending: list[tuple[str, str]] = field(default_factory=list)
    # Endpoints whose engine cannot start (crash-looping): (deployment, endpoint,
    # the engine's own error). Present only when the wait stopped early.
    failures: list[tuple[str, str, str]] = field(default_factory=list)


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
class AccessResult:
    """What a workload needs to reach its endpoints, and what it holds.

    ``endpoints``: the aliases asked for, bundles expanded, in order;
    ``external``: those an external server fulfils (no lease, no deployment);
    ``request_names``: the ``model`` a client sends for each; ``connection``:
    the base URL and credential (``None`` on an in-process backend); ``lease``:
    the one real lease over the managed members, or ``None`` when every member
    is external (the lease never lists an external name); ``acquire``: that
    lease's outcome (readiness, placement). ``published`` is whether the
    publication (the gateway's routes included) completed; ``False`` leaves it
    pending for the next apply. ``front_door_ready`` is ``False`` when an
    external member's front door never accepted its key in time; the lease
    this access took is then already released. ``waited`` is whether
    readiness was checked at all.
    """

    endpoints: list[str]
    external: list[str]
    request_names: dict[str, str]
    connection: ConnectionInfo | None
    lease: Lease | None = None
    acquire: AcquireOutcome | None = None
    front_door_ready: bool | None = None
    published: bool = True
    waited: bool = True

    @property
    def deployments(self) -> list[Deployment]:
        return list(self.acquire.deployments) if self.acquire is not None else []

    @property
    def ready(self) -> bool | None:
        """``True``: verified that every member can be asked now (infer-stack's
        routes for them are published, managed members generate through the
        front door, and the front door accepts its key; the external server
        itself is not asked). ``False``: verified not. ``None``: not checked
        (``wait=False``)."""
        if not self.published or self.front_door_ready is False:
            return False
        if self.acquire is not None and (
                self.acquire.released_on_timeout
                or (self.acquire.wait is not None and not self.acquire.wait.ready)):
            return False
        return True if self.waited else None


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
        backend: ServingBackend,
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
        # Recovery snapshot (see leasing/profile.py): this process's own
        # resolved settings, captured before the backend is switched to the
        # resolved copy, and the snapshot the backend currently renders from.
        #: The first snapshot, rendered from but not yet written (see
        #: :meth:`_sync_profile`).
        self._fresh_profile: dict | None = None
        self._invocation_profile: dict | None = None
        self._applied_profile: dict | None = None
        self._profile_drift_warned = False
        # A recovery snapshot for another backend kind is reported on the
        # first mutation, not here: the advanced explicit publication command
        # must still be able to open a controller to report the incompatibility.
        self._profile_error: Exception | None = None
        # Set only by apply_now(): the operator explicitly re-approves a render
        # that differs from an earlier approved digest.
        self._explicit_apply = False
        # Set when an apply is invoked: whether a failed publication may have
        # changed the runtime (secret rotation reads it; see rotate_gateway_key).
        self._apply_began = False
        self._admission_digest: str | None = None
        #: Whether the last refused admission was for lack of GPUs.
        self._admission_capacity = False
        #: Why residency could not be read, for the refusal that follows.
        self._residency_error: str | None = None
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

    def publication_lock(self):
        """Hold while changing state a render reads outside the ledger (the
        managed ``.env``): the same host-wide lock every publication holds."""
        return self._global_lock()

    def invocation_profile(self) -> dict | None:
        """The recovery profile this invocation resolves to (its settings and
        catalogs), before any switch to a stored one; ``None`` without one."""
        recovery = self.backend.recovery_profile
        return self._invocation_profile or (
            recovery.render_profile() if recovery is not None else None)

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

    def _infeasible_alone(self, deployments, requested: set) -> dict:
        """Which of this lease's deployments cannot be placed even on an idle host.

        Returns ``{deployment_id: reason}``, empty when the whole set fits.
        The check is an optimisation over waiting, never a new failure mode.
        """
        try:
            plan = self.backend.plan_on_idle_host(list(deployments))
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

        The caller MUST hold :meth:`_global_lock`. This writes the backend's
        rendered state (the compose project, the Model manifests) but does not
        bring it up -- that is :meth:`_apply_pending`, under the same lock hold.
        """
        self.ledger.sweep()
        # Residency decides which idle keep-warm deployments are candidates at
        # all; unknown residency fails the render (the change stays pending)
        # rather than guessing which warm models exist.
        residency = self.backend.residency()
        self._backfill_allocations(residency)
        host = self.backend.host_runtime
        self._prepare_network(host)
        desired, placement = self._admission_view(residency)
        if host is not None:
            host.set_adopted(self._prune_adopted(residency))
        before = set(self.backend.observe())
        self.backend.converge(desired, apply=False, placement=placement)
        after = set(self.backend.observe())
        notes = self.backend.placement_notes()
        rec = ReconcileResult(
            realized=sorted(after - before),
            torn_down=sorted(before - after),
            unplaced=sorted(self.backend.last_unplaced),
            placement_errors=list(self.backend.last_errors),
            assignments=dict(self.backend.last_assignments),
            applied=False,
            # The render's own record, the one authority for these facts.
            displaced=list(notes.get('displaced') or ()),
            degraded=list(notes.get('degraded') or ()),
        )
        if host is not None:
            self._adopt_existing(residency, host)
        return rec

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
    # Every acquire, renew and publication goes through admission:
    #   * a LIVE deployment holds a committed allocation (assigned_gpus);
    #   * an IDLE keep-warm deployment is only an optional candidate, and only
    #     while it is uniquely resident; it yields its GPUs to demand and is
    #     never started;
    #   * an acquire is previewed in memory (placement AND render) and commits
    #     its lease together with its allocations, or commits nothing.
    # Where the cluster schedules (KubeAI, ``allocates_gpus`` false) every
    # deployment commits an empty allocation: admission then decides only
    # renderability, and a Pending pod is a wait reason, not an unplaced
    # error. Backends that neither place nor inspect (dry-run, tests) get the
    # same surface from SimpleAdmission.

    def keeps_up(self, deployment: Deployment) -> bool:
        """Whether the reconciler keeps ``deployment`` running.

        LIVE always; IDLE only under ``keep-warm`` (the default policy). An
        IDLE ``stop`` deployment is torn down by the release that idled it
        and stays IDLE in the ledger, so a view that treats every IDLE row as
        running reports a missing container that is exactly as intended.
        """
        if deployment.state == DeploymentState.LIVE:
            return True
        return (deployment.state == DeploymentState.IDLE
                and deployment.spec.get('reclaim', self.reclaim_default) == KEEP_WARM)

    def _stored_state(self, lease_id: str):
        """A lease's state as stored (not virtually expired), or ``None``."""
        lease = self.ledger.get_lease(lease_id)
        return None if lease is None else lease.state

    def _admission_view(
        self, residency, *, overlay=None, virtual_expiry: bool = False
    ):
        """``(desired deployments, PlacementInputs)`` for a render or preview.

        ``overlay`` (an :class:`~infer_stack.leasing.ledger.AcquireOverlay`)
        adds a candidate acquire on top of the ledger, without writing it.
        """
        from .placement import PlacementInputs

        _, deployments = self.ledger.status(virtual_expiry=virtual_expiry)
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
                if not self.keeps_up(deployment):
                    continue
                # Resident, with its GPUs known: every warm unit counts (a
                # KubeAI Model's replicas), a conflict or an unmappable
                # reservation does not. [] where the cluster places.
                gpus = residency.resident_gpus(gid) if residency is not None else None
                if gpus is not None:
                    hints[gid] = gpus
                    optional.append(deployment)
        inputs = PlacementInputs(
            required_ids=set(required), hard=hard, optional_hints=hints,
        )
        return [*required.values(), *optional], inputs

    def _prepare_network(self, host: HostRuntime | None) -> None:
        """Give the host runtime the stable-address table, once a network is migrated.

        Only a backend with containers on this host has a network to stamp
        addresses on; for any other there is nothing to do.
        """
        if host is None:
            return
        config = self.ledger.network_config()
        if config is None:
            host.configure_network(None)
            return
        host.configure_network(
            {'subnet': config['subnet'], 'addresses': self.ledger.service_addresses()},
            on_addresses=self.ledger.add_service_addresses)

    def network_migrate(self, subnet: str, *, force: bool = False) -> ReconcileResult:
        """``infer-stack network migrate``: move the project to stable addresses.

        Operator-invoked. Refused while any lease is ACTIVE unless ``force``,
        because every container, the gateway included, is recreated once.
        Rejects a subnet that overlaps an existing Docker network or host
        route. Changing an already-migrated subnet reassigns every address.
        """
        import ipaddress

        from .profile import ProfileMismatch

        subnet = str(ipaddress.ip_network(subnet))
        host = self.backend.host_runtime
        if host is None:
            raise ProfileMismatch(
                'network migrate moves containers on this host to fixed addresses; '
                'this backend runs none (the compose backend does)')
        with self._global_lock():
            leases, _ = self.ledger.status(virtual_expiry=True)
            active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
            if active and not force:
                raise ProfileMismatch(
                    f'network migrate recreates every container; {len(active)} lease(s) '
                    'are active (release them, or pass --force)'
                )
            clash = host.subnet_clashes(subnet)
            if clash:
                raise ProfileMismatch(f'subnet {subnet} overlaps: {"; ".join(clash)}')
            current = self.ledger.network_config()
            reset = current is not None and current['subnet'] != subnet
            # Preview the migrated render and take approval BEFORE any write:
            # a declined migration must leave the subnet and addresses as they were.
            self._sync_profile(create=True)
            residency = self.backend.residency()
            saved = host.network_table()
            host.configure_network({
                'subnet': subnet,
                'addresses': {} if reset else self.ledger.service_addresses(),
            })
            try:
                desired, inputs = self._admission_view(
                    residency, virtual_expiry=True
                )
                self.backend.preview(desired, inputs, approve=True)
            finally:
                host.configure_network(saved)
            self.ledger.migrate_network(
                subnet=subnet, reset_addresses=reset,
                approved_digest=self.backend.last_preview_digest,
                profile=self._take_fresh_profile(),
            )
            return self._publish()

    def rotate_gateway_key(self, *, force: bool = False) -> ReconcileResult:
        """``infer-stack secrets rotate``: replace the LiteLLM master key.

        Refused while any lease is ACTIVE unless ``force``: its holder
        authenticates with the old key, and the gateway restarts. The new key
        is written, then published like any desired-state change, so the
        gateway (and Open WebUI) are recreated with it.

        A running LiteLLM keeps the key it started with. So when the
        publication fails before its apply began (declined, residency
        unreadable, a render refused) nothing was recreated, and the old key
        goes back into the managed ``.env``: otherwise clients would be handed
        a key the gateway rejects. Once the apply began, the gateway may
        already run the new key; the file keeps it, and the pending
        publication converges the runtime to it on the next apply. An apply
        that returns having not reached the runtime (``ApplyResult.runtime``
        false: an unreadable render, routes that could not be retired) did
        not recreate the gateway either, so the old key goes back too and the
        rotation is refused with the apply's reason: the file must never
        name a key the running gateway rejects.
        """
        from .profile import ProfileMismatch

        front = self.backend.front_door()
        if front is None or not front.litellm:
            raise ProfileMismatch('no LiteLLM gateway to rotate a key for')
        with self._global_lock():
            leases, _ = self.ledger.status(virtual_expiry=True)
            active = [le.id for le in leases if le.state == LeaseState.ACTIVE]
            if active and not force:
                raise ProfileMismatch(
                    f'{len(active)} lease(s) are active and hold the current key '
                    '(release them, or pass --force)'
                )
            self._mark_pending(apply=True)
            replaced = front.rotate_master_key()
            self._apply_began = False
            try:
                rec = self._publish()
            except BaseException:
                if not self._apply_began:
                    front.restore_env(replaced)
                raise
            if rec.applied and not rec.runtime_applied:
                front.restore_env(replaced)
                raise ProfileMismatch(
                    'the key was not changed: the gateway was not recreated ('
                    + (rec.apply_detail or 'the runtime was not reached')
                    + '); retry once `infer-stack apply` succeeds')
            return rec

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
        * ``reclaimed``: idle under ``stop``, and its container is gone, as
          intended (see :meth:`keeps_up`);
        * ``restarting``: resident, but every unit is crash-looping (the one
          serving state taken from
          :func:`~infer_stack.leasing.residency.deployment_health`: it needs
          an operator, and ``leases`` flags it);
        * ``running`` / ``not-running``: whether it is resident (any replica
          warm). A deployment still starting is ``running`` here, since
          startup needs no one; ``status`` shows ``starting``.
        """
        from .profile import profile_drift
        from .residency import ResidencyUnknown, deployment_health

        leases, deployments = self.ledger.status(virtual_expiry=True)
        try:
            notes = self.backend.placement_notes() or {}
        except Exception:  # noqa: BLE001 - a view must always render
            notes = {}
        residency, residency_error = None, None
        try:
            residency = self.backend.residency()
        except ResidencyUnknown as ex:
            residency_error = str(ex)
        degraded = set(notes.get('degraded') or ())
        displaced = set(notes.get('displaced') or ())
        rows = []
        for g in deployments:
            if g.state not in (DeploymentState.LIVE, DeploymentState.IDLE):
                continue
            if residency is None and residency_error is not None:
                condition = 'unknown'
            elif residency is not None and residency.is_conflicted(g.id):
                condition = 'ambiguous'
            elif g.id in degraded and g.state == DeploymentState.LIVE:
                condition = 'degraded'
            elif g.id in displaced and g.state == DeploymentState.IDLE:
                condition = 'displaced'
            elif (g.state == DeploymentState.LIVE and g.assigned_gpus is None
                  and residency is not None
                  and self._gpu_units(g) > 0
                  and residency.unique_unit(g.id) is None):
                condition = 'unresolved'
            elif residency is not None and residency.is_resident(g.id):
                condition = ('restarting'
                             if deployment_health(residency, g.id) == 'restarting'
                             else 'running')
            elif residency is not None:
                condition = 'not-running' if self.keeps_up(g) else 'reclaimed'
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
                                and self._stored_state(le.id) == LeaseState.ACTIVE],
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

    def _adopt_existing(self, residency, host: HostRuntime) -> None:
        """One-time migration: adopt project containers from before ownership labels.

        Runs after the first render on a ledger. A container
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
        rendered = host.rendered_services()
        fingerprints = {name: fp for name, (fp, _) in rendered.items()}
        services = {name: gid for name, (_, gid) in rendered.items()}
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
                            and residency.is_resident(c.deployment_id))
                if not (live_here or resident):
                    continue
            adopted[c.container_id] = {
                'service': c.service, 'fingerprint': fingerprints[c.service]}
        self.ledger.set_adopted_containers(adopted)
        host.set_adopted(adopted)

    def remove_orphans(self, confirm: Callable[[list], bool]) -> list:
        """``gc --orphans``: remove the project's unmanaged containers, with consent.

        Takes a strict snapshot under the lock, lists containers infer-stack
        neither labelled nor adopted, and removes exactly those if ``confirm``
        (shown the list) returns True. Returns the removed containers.
        """
        host = self.backend.host_runtime
        if host is None:
            return []           # nothing of this host's to be an orphan
        with self._global_lock():
            residency = self.backend.residency()
            adopted = self.ledger.adopted_containers() or {}
            orphans = [c for c in residency.all_containers()
                       if not c.labelled and c.container_id not in adopted]
            if not orphans or not confirm(orphans):
                return []
            host.remove_containers([c.container_id for c in orphans])
            return orphans

    def _gpu_units(self, deployment) -> int:
        """GPUs admission must account for: none where the cluster schedules."""
        from .backend import allocates_gpus
        from .placement import required_gpu_count

        return required_gpu_count(deployment) if allocates_gpus(self.backend) else 0

    def _backfill_allocations(self, residency) -> list[str]:
        """Adopt allocations for LIVE deployments that predate them.

        A LIVE deployment with no committed allocation adopts the GPUs of its
        unique warm container. One with no container, several, or unmappable
        GPUs stays unresolved: it keeps running where it is, but no new GPU is
        allocated to anyone until it is released. Returns the unresolved ids.
        """
        unresolved = []
        _, deployments = self.ledger.status()
        for deployment in deployments:
            if deployment.state != DeploymentState.LIVE or deployment.assigned_gpus is not None:
                continue
            if self._gpu_units(deployment) == 0:
                self.ledger.set_allocation(deployment.id, [])
                continue
            # One reservation to adopt: a unique unit, not a replica set.
            resident = residency.unique_unit(deployment.id)
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
        self._admission_capacity = False
        adopted: dict[str, list[int]] = {}
        need: list[str] = []
        for gid, deployment in overlay.deployments.items():
            fresh = gid in overlay.created or gid in overlay.revived
            if not fresh:
                continue
            if self._gpu_units(deployment) == 0:
                continue
            resident = residency.unique_unit(gid) if (
                residency is not None and gid in overlay.revived) else None
            if resident is not None and not resident.all_gpus and resident.gpus:
                adopted[gid] = list(resident.gpus)      # IDLE->LIVE keeps its GPUs
            else:
                need.append(gid)
        if residency is None:
            # The render after the commit needs residency too, so nothing can
            # be admitted without it; refusing here commits nothing.
            why = self._residency_error or 'the runtime did not answer'
            return {}, [
                f'what is running cannot be read right now ({why}), so nothing is '
                'admitted; `infer-stack doctor` checks the runtime'
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
        self._prepare_network(self.backend.host_runtime)
        desired, inputs = self._admission_view(residency, overlay=overlay)
        plan, rendered = self.backend.preview(desired, inputs)
        from .backend import allocates_gpus

        reasons = []
        capacity = False            # did any refusal come from GPU placement?
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
                capacity = True
            elif gid not in plan.assignments:
                why = [e for e in plan.errors if e.startswith(gid)]
                reasons.extend(why or [f'{gid}: could not be placed'])
                # Where the cluster places, a plan without it is a refusal.
                capacity = capacity or allocates_gpus(self.backend)
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
        self._admission_capacity = capacity
        if not reasons:
            # Approval happens now, before anything is committed; the render
            # after the commit produces the same files and does not ask again.
            # Its digest goes into the pending marker, so a recovery after a
            # crash (and perhaps an upgrade) cannot apply something else.
            self.backend.preview(desired, inputs, approve=True)
            self._admission_digest = self.backend.last_preview_digest
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

    def _unresolved_allocations(self, *, exclude: AbstractSet[str] = frozenset()) -> list[str]:
        _, deployments = self.ledger.status()
        return [
            g.id for g in deployments
            if g.state == DeploymentState.LIVE and g.assigned_gpus is None
            and g.id not in exclude and self._gpu_units(g) > 0
        ]

    def set_invocation_catalog(self, catalog) -> None:
        """Update this process's authoritative catalog snapshot.

        Long-lived callers such as the TUI can edit ``catalog.yaml`` without
        rebuilding the controller.  The backend may already be rendering from
        the recovery snapshot, so changing ``backend.catalog`` directly would
        violate the freeze.  Instead update only the invocation profile; the
        next acquire will adopt it under the publication lock using the same
        rules as a fresh CLI process.
        """
        from .profile import catalog_sources

        recovery = self.backend.recovery_profile
        if recovery is None:
            return
        base = dict(self._invocation_profile or recovery.render_profile())
        base['catalogs'] = catalog_sources(catalog)
        self._invocation_profile = base
        self._profile_drift_warned = False

    def _pinned_endpoints(self, residency=None) -> set[str]:
        """Endpoint aliases whose definition a resident workload is running.

        LIVE deployments always count; an IDLE keep-warm deployment counts only
        while its container is actually resident, which is the same rule
        placement uses. Anything else is free to be redefined.
        """
        return set(self._pinned_deployments(residency))

    def _pinned_deployments(self, residency=None) -> dict[str, list[Deployment]]:
        """``{alias: [resident deployments serving it]}`` (see
        :meth:`_pinned_endpoints`)."""
        _, deployments = self.ledger.status(virtual_expiry=True)
        pinned: dict[str, list[Deployment]] = {}
        for deployment in deployments:
            resident = deployment.state == DeploymentState.LIVE or (
                deployment.state == DeploymentState.IDLE and residency is not None
                and residency.is_resident(deployment.id))
            if resident:
                for alias in deployment.served:
                    pinned.setdefault(alias, []).append(deployment)
        return pinned

    def _pinned_changes(self, before: list[dict], after: list[dict],
                        residency=None) -> list[str]:
        """Resident aliases whose meaning ``after`` (catalog sources) changes
        to one their running deployment does not have.

        Pinning is an endpoint rule, not a catalog-conflict case (queue item
        45): an ad-hoc deployment serving ``qwen`` pins ``qwen`` although no
        catalog defines it, so publishing ``qwen -> external`` must refuse
        like redefining a catalog one. A definition that matches what runs
        (same engine and structure) is not a change of meaning.
        """
        from .catalog import CatalogError
        from .profile import CatalogUnion

        pinned = self._pinned_deployments(residency)
        if not pinned:
            return []
        old = CatalogUnion.from_sources(before) if before else None
        new = CatalogUnion.from_sources(after) if after else None
        blocked = []
        for alias in sorted(pinned):
            if new is None or alias not in new.endpoints:
                continue
            try:
                meaning = new.resolve_endpoint(alias)
                if (old is not None and alias in old.endpoints
                        and old.resolve_endpoint(alias).semantic_key()
                        == meaning.semantic_key()):
                    continue
                if meaning.managed and meaning.to_request().compat_key in {
                        d.compat_key for d in pinned[alias]}:
                    continue
            except CatalogError:
                pass
            blocked.append(alias)
        return blocked

    @staticmethod
    def _pinned_refusal(blocked: list[str], what: str) -> Exception:
        from .profile import ProfileMismatch

        return ProfileMismatch(
            f'{what} redefines {", ".join(repr(b) for b in blocked)}, which a '
            'resident deployment is running. Release or evict it '
            f'(`infer-stack evict {blocked[0]}`), then retry; definitions nothing '
            'is running are updated automatically')

    def _profile_quiescent(self, residency=None) -> bool:
        """Whether it is safe to replace the recovery snapshot wholesale.

        Match ``config publish``'s meaningful quiescence boundary: no active
        lease and no managed deployment container.  Gateway/UI containers do
        not hold an endpoint definition and may be reconciled by the next
        publication.  If residency is unavailable we conservatively return
        ``False`` and only compatible catalog additions may advance live.
        """
        leases, _ = self.ledger.status(virtual_expiry=True)
        if any(le.state == LeaseState.ACTIVE for le in leases):
            return False
        if residency is None:
            return False
        try:
            return not any(c.deployment_id for c in residency.all_containers())
        except Exception:  # noqa: BLE001 - unknown residency is not quiescence
            return False

    def _acquire_profile_candidate(self, *, residency=None) -> dict | None:
        """Derive an automatic recovery-snapshot advance for an acquire.

        User configuration remains authoritative; the persisted profile is an
        internal crash-recovery snapshot, not a third configuration surface.

        * Settings (backend, ports, images, gateway placement): adopted from
          the invocation when the stack is quiescent, frozen while workloads
          are resident.
        * Catalogs, the published endpoint definitions: always merged
          (:func:`~infer_stack.leasing.profile.adopt_catalog_sources`): the
          invocation's definitions replace published ones nothing runs; one a
          resident workload runs refuses; unrelated published definitions,
          external endpoints included, stay. Quiescence does not unpublish.
        """
        from .._log import logger
        from .profile import (
            CatalogConflict,
            ProfileMismatch,
            adopt_catalog_sources,
            profile_drift,
        )

        stored = self.ledger.profile()
        invocation = self._invocation_profile
        if invocation is None:
            return None
        if stored is None:
            return dict(invocation)     # the first publication: all of it
        if stored == invocation:
            return None
        if stored.get('backend') != invocation.get('backend'):
            raise ProfileMismatch(
                f"the active recovery snapshot uses {stored.get('backend')!r}, "
                f"but current config selects {invocation.get('backend')!r}; "
                'tear down the old backend before switching'
            )
        drift = profile_drift(stored, invocation)
        # A runbook whose catalog is already a subset of an explicitly seeded
        # union has no drift. Do not compact away sibling runbooks merely
        # because the stack happens to be quiescent.
        if not drift:
            return None
        quiescent = self._profile_quiescent(residency)
        candidate = dict(invocation) if quiescent else dict(stored)
        # Only definitions a resident workload is actually running have to stay
        # frozen; redefining anything else (the normal case while iterating with
        # `catalog endpoint add --force`) replaces it. Nothing else unpublishes.
        pinned = set() if quiescent else self._pinned_endpoints(residency)
        try:
            candidate['catalogs'] = adopt_catalog_sources(
                stored.get('catalogs') or [], invocation.get('catalogs') or [], pinned)
        except CatalogConflict as pinned_ex:
            blocked = sorted(set(pinned_ex.names) & pinned) or sorted(pinned_ex.names)
            raise ProfileMismatch(
                f'the current catalog redefines {", ".join(repr(b) for b in blocked)}, '
                'which a resident deployment is running. Release or evict it '
                f'(`infer-stack evict {blocked[0]}`), then retry; definitions '
                'nothing is running are updated automatically'
            ) from pinned_ex
        blocked = self._pinned_changes(
            stored.get('catalogs') or [], candidate['catalogs'], residency)
        if blocked:
            raise self._pinned_refusal(blocked, 'the current catalog')
        if quiescent:
            return candidate if candidate != stored else None

        deferred = [
            key for key in drift if key != 'catalogs'
        ]
        if deferred and not self._profile_drift_warned:
            self._profile_drift_warned = True
            logger.warning(
                'Current user settings differ from the active recovery snapshot '
                '({}); keeping those global settings frozen while workloads are '
                'resident. Compatible catalog additions still apply now; once the '
                'stack is quiescent, the next acquire adopts current settings '
                'automatically.', ', '.join(deferred),
            )
        return candidate if candidate != stored else None

    def _use_profile_candidate(self, profile: dict) -> None:
        recovery = self.backend.recovery_profile
        if recovery is not None:
            recovery.use_profile(profile)

    def _commit_profile_candidate(self, profile: dict) -> None:
        """Persist an already-previewed acquire profile without publishing alone.

        The acquire that follows creates the publication marker.  A crash between
        this write and that marker therefore leaves only newer recovery inputs,
        never an unrecorded desired-state mutation.  This avoids pairing the
        acquire's approval digest with a lease that has not committed yet.
        """
        self.ledger.set_profile(profile)
        self._profile_error = None
        self._applied_profile = profile

    def _restore_stored_profile(self, stored: dict | None) -> None:
        stored = stored if stored is not None else self._fresh_profile
        if stored is None:
            return
        self._use_profile_candidate(stored)
        self._applied_profile = stored

    def _sync_profile(self, *, create: bool) -> None:
        """Make the backend render from the active recovery snapshot.

        With ``create`` (every mutation, under the lock), a ledger without a
        snapshot renders from this invocation's settings with no published
        endpoints (:attr:`_fresh_profile`). That is not written here: it
        commits with the first publication marker (:meth:`_mark_pending`), and
        an acquire or access commits the invocation's catalogs with its own
        transaction, because writing ``catalogs`` is publishing them. Drift is
        reported once; acquire decides under the same lock whether current
        user config can advance the snapshot. Backends without a render
        profile (null, test fakes) are left alone.
        """
        from .._log import logger
        from .profile import profile_drift

        recovery = self.backend.recovery_profile
        if recovery is None:
            return
        render, use, read = recovery.render_profile, recovery.use_profile, self.ledger.profile
        if create and self._profile_error is not None:
            raise self._profile_error
        stored = read()
        if stored is None and not create:
            return
        if self._invocation_profile is None:
            self._invocation_profile = render()
        if stored is None:
            # Settings only: endpoints are published by the operations that
            # publish them, in their own transaction.
            stored = self._fresh_profile = {**self._invocation_profile, 'catalogs': []}
        elif not self._profile_drift_warned:
            drift = profile_drift(stored, self._invocation_profile)
            if drift:
                self._profile_drift_warned = True
                logger.warning(
                    'Current settings differ from the active recovery snapshot '
                    '({}); existing workloads continue with the frozen snapshot. '
                    'Acquire adopts compatible catalog additions automatically and '
                    'adopts all current settings once the stack is quiescent.',
                    ', '.join(drift),
                )
        if stored != self._applied_profile:
            use(stored)
            self._applied_profile = stored

    def _mark_pending(
        self, *, apply: bool, create_profile: bool = True,
    ) -> dict:
        """Record that desired state is about to change (caller holds the lock).

        Written before the ledger mutation, so a crash between the two leaves at
        worst a redundant marker, never a mutation without one. ``apply`` only
        ever turns ``apply_requested`` on: a staged change never cancels an
        apply already requested (promotion, see the plan's D23).

        A placement scope left in the marker by pre-admission code is dropped:
        a row that code committed without an allocation stays unresolved and is
        never placed, so no render needs that caller's scope.
        """
        if create_profile:
            self._sync_profile(create=True)
        current = self.ledger.publication_pending()
        if current and current.get('placement_context'):
            self.ledger.clear_placement_context()
        return self.ledger.mark_publication_pending(
            apply_requested=apply, profile=self._take_fresh_profile())

    def _take_fresh_profile(self) -> dict | None:
        """The first snapshot to commit with the next marker, or ``None``
        when the ledger already has one."""
        from .._log import logger

        fresh, self._fresh_profile = self._fresh_profile, None
        if fresh is None or self.ledger.profile() is not None:
            return None
        logger.info('Froze the initial recovery snapshot ({} backend); acquire and '
                    'access publish endpoints into it', fresh.get('backend'))
        return fresh

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
        apply_fn = self.backend.apply
        if not marker['apply_requested']:
            rec.publication_pending = True    # staged; never applied here
            return rec
        approved = marker.get('approved_digest')
        rendered = self.backend.last_planned_digest
        # Fail closed: an approved render and no digest for this one is not
        # proof they are the same.
        if approved and rendered != approved:
            if not self._explicit_apply:
                from .profile import ProfileMismatch

                rec.publication_pending = True
                raise ProfileMismatch(
                    'the rendered state differs from what was approved at publication '
                    '(e.g. infer-stack was upgraded in between); review and approve it '
                    'with `infer-stack apply`'
                )
            # `infer-stack apply` approved this render: record it before
            # applying, so a partial apply leaves THIS render approved and an
            # ordinary retry of it proceeds (a later, different one still
            # needs approval).
            if rendered is not None:
                self.ledger.reapprove_render(rendered)
            approved = rendered
        if marker['interrupted']:
            self._wait_for_settled_runtime()
        before = set(self.backend.observe())
        self._apply_began = True
        try:
            outcome = ApplyResult.of(apply_fn())
        except BaseException as ex:
            from .compose import ApplyAborted

            # A killed or failed client does not stop work the daemon already
            # started: the next apply must first wait for the runtime to settle.
            # A refusal (ApplyAborted) started nothing, so it needs no settling.
            self.ledger.mark_publication_pending(
                apply_requested=True, interrupted=not isinstance(ex, ApplyAborted))
            rec.publication_pending = True
            raise
        after = set(self.backend.observe())
        rec.realized = sorted(set(rec.realized) | (after - before))
        rec.torn_down = sorted(set(rec.torn_down) | (before - after))
        rec.applied = True
        rec.runtime_applied = outcome.runtime
        rec.apply_detail = outcome.detail
        if not outcome.complete:
            # The publication is not done, so neither is the approval of its
            # render: a retry re-renders, and a render that drifted (an
            # upgrade in between) must be approved again, not applied.
            rec.publication_pending = True
            logger.warning(
                'apply did not fully take effect ({}); the change stays pending and '
                'the next acquire/release or `infer-stack apply` retries it',
                outcome.detail or ('routes not verified' if outcome.runtime
                                   else 'runtime not reached'))
            return rec
        if approved:
            self.ledger.clear_approved_digest()
        self.ledger.clear_publication_pending(marker['version'])
        rec.publication_pending = False
        return rec

    def _wait_for_settled_runtime(self) -> None:
        """Block (bounded) until two consecutive runtime samples are identical.

        Called before applying on top of an interrupted apply. A sample that
        cannot be read, a container still ``removing``, or a runtime that keeps
        changing past :data:`SETTLE_DEADLINE_S` raises
        :class:`~infer_stack.leasing.backend.RuntimeUnsettled` and leaves the
        change pending. A backend whose ``settle_snapshot`` returns ``None``
        has nothing local to wait for (an in-process backend, or KubeAI with
        its gateway in the cluster); KubeAI with a host gateway returns that
        gateway's Compose containers and is waited on like Compose.
        """
        from .backend import RuntimeUnsettled

        snapshot = self.backend.settle_snapshot
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
            if current is None:
                return                      # nothing local that could still be running
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

        On a ledger with no profile yet (``stack up`` on a fresh stack) it
        publishes the invocation's endpoints first, through the same preview,
        approval and single commit as ``access`` (:meth:`publish_endpoints`), so
        catalog routes exist before any model runs and the first acquire does
        not recreate the gateway. Other mutations (gc, release, evict) never
        publish endpoints.
        """
        self._explicit_apply = True
        try:
            if self.backend.recovery_profile is not None and self.ledger.profile() is None:
                return self.publish_endpoints()
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

        A probe that reports ``fatal`` (the engine is crash-looping, not
        loading) ends the wait at once: waiting out the timeout would hold that
        GPU against every other request while the engine's error goes unseen.
        """
        pairs = [
            (deployment, ep)
            for deployment in deployments
            for ep in sorted(deployment.served)
            if endpoints is None or ep in endpoints
        ]
        deadline = self.clock() + timeout
        last_room = float('-inf')
        while True:
            pending = []
            failures = []
            blocked: dict[str, Deployment] = {}
            for (g, ep) in pairs:
                probe = self.backend.probe_ready(g, ep)
                if probe.ready:
                    continue
                pending.append((g, ep))
                if probe.needs_room:
                    blocked[g.id] = g
                if probe.fatal:
                    failures.append((g.id, ep, probe.detail))
            if not pending:
                return WaitResult(ready=True)
            if failures:
                return WaitResult(
                    ready=False,
                    pending=[(g.id, ep) for g, ep in pending],
                    failures=failures,
                )
            if self.clock() >= deadline:
                return WaitResult(
                    ready=False,
                    pending=[(g.id, ep) for g, ep in pending],
                )
            # After the deadline check: a wait that has given up evicts nothing.
            if blocked and self.clock() - last_room >= ROOM_COOLDOWN_S:
                # A leased model is waiting on capacity: an idle model the
                # backend says could free it gives it up.
                if self._make_room(list(blocked.values())):
                    last_room = self.clock()
            self.sleep(interval)
            pairs = pending

    def _make_room(self, blocked: list[Deployment]) -> str | None:
        """Evict one idle deployment that could free room for ``blocked``.

        The policy is here: only idle deployments (no lease holds them, so a
        leased one is never a victim), the longest idle first, one at a time,
        since the runtime cannot say how much room is needed and every warm
        model kept is a load avoided. Whether a victim could help at all is
        the backend's to say (:meth:`ServingBackend.reclaim_candidates`: the
        same node pool and the resource that is short); when none could,
        nothing is evicted. Returns the victim's id, or ``None``.
        """
        from .._log import logger

        _, deployments = self.ledger.status(virtual_expiry=True)
        idle = sorted((g for g in deployments if g.state == DeploymentState.IDLE),
                      key=lambda g: (g.updated_at, g.id))
        if not idle:
            return None
        useful: set[str] = set()
        for g in blocked:
            useful.update(self.backend.reclaim_candidates(g, idle))
        victims = [g for g in idle if g.id in useful]
        if not victims:
            return None
        victim = victims[0]
        logger.info('making room for leased demand: evicting idle keep-warm {} ({})',
                    victim.id, ', '.join(sorted(victim.served)))
        self.evict([victim.id])
        return victim.id

    def _never_ran(self, deployment_ids: list[str]) -> list[str]:
        """Which of these deployments definitely have no container at all.

        Uses strict residency: a deployment with any container, in any state,
        is kept, and if the runtime cannot be read nothing is reported (so
        nothing warm is ever evicted on a failed look).
        """
        from .residency import ResidencyUnknown

        try:
            snap = self.backend.residency()
        except ResidencyUnknown:
            return []
        return [gid for gid in deployment_ids if not snap.units(gid)]

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
            self.ledger.clear_approved_digest()
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
        result, rec = self._acquire_by_admission(
            owner, requests, ttl_seconds=ttl_seconds, apply=apply,
            wait_for_placement=wait_for_placement,
            placement_timeout=timeout if placement_timeout is None else placement_timeout,
            placement_interval=interval if placement_interval is None else placement_interval,
        )
        return self._finish_acquire(result, rec, apply=apply, wait=wait,
                                    timeout=timeout, interval=interval)

    #: Seconds ``access`` waits for the front door to accept its key, when an
    #: external member has nothing else to wait for.
    FRONT_DOOR_WAIT_S = 120.0

    def access(
        self,
        owner: str,
        endpoints: list,
        *,
        sharing: str | None = None,
        ttl_seconds: float | None = None,
        wait: bool = True,
        timeout: float = 300.0,
        interval: float = 2.0,
        wait_for_placement: bool = False,
    ) -> AccessResult:
        """Make ``endpoints`` reachable: lease the managed ones, publish all.

        ``endpoints`` are :class:`~infer_stack.leasing.endpoints.
        ResolvedEndpoint` s (``catalog.resolve(names)``). The managed members
        take one lease through :meth:`acquire`, whose transaction also
        publishes the invocation's catalogs, external definitions included
        (docs/planning/external-endpoints.md, decision 3). With no managed
        member there is no ledger mutation: :meth:`publish_endpoints` runs the
        same preview, approval and publication without one. External members
        need the LiteLLM front door, a value for their key, and a front door
        that accepts its key (decision 6); nothing is sent to the external
        server. If the front door never answers, the lease this call took is
        released and ``front_door_ready`` is ``False``.
        """
        from .profile import ProfileMismatch

        aliases = [e.alias for e in endpoints]
        external = [e for e in endpoints if not e.managed]
        front = self.backend.front_door() if external else None
        if external:
            if front is None or not front.litellm:
                raise ProfileMismatch(
                    f'{", ".join(repr(e.alias) for e in external)} '
                    f'{"is" if len(external) == 1 else "are"} served by an external '
                    'server, reached only through the LiteLLM front door; this '
                    'backend has none (turn `litellm` on)')
        managed = [e.to_request(sharing_override=sharing) for e in endpoints if e.managed]
        outcome = None
        if managed:
            outcome = self.acquire(
                owner, managed, ttl_seconds=ttl_seconds, wait=wait, timeout=timeout,
                interval=interval, wait_for_placement=wait_for_placement)
            rec = outcome.reconcile
        else:
            rec = self.publish_endpoints()
        result = AccessResult(
            endpoints=aliases, external=[e.alias for e in external],
            request_names=self.backend.request_names(aliases),
            connection=self.backend.connection_info(),
            lease=outcome.lease if outcome is not None else None, acquire=outcome,
            # An external member is reached only through the routes this
            # publication installs: gateway liveness alone does not prove them.
            published=not (external and rec.publication_pending), waited=wait)
        if (front is not None and wait and result.published
                and not (outcome and outcome.released_on_timeout)):
            accepted = front.gateway_accepts(
                front.master_key(), wait=min(timeout, self.FRONT_DOOR_WAIT_S))
            result.front_door_ready = accepted is True
            if not result.front_door_ready and outcome is not None:
                self.release(outcome.lease.id)
        return result

    def publish_endpoints(self) -> ReconcileResult:
        """Publish the invocation's endpoint definitions with no lease.

        The same transaction an acquire runs, minus the ledger mutation:
        under the lock, derive the profile candidate (the invocation's
        catalogs merged into the published union), preview its render with
        approval, then store it with the approved digest and publish. When the
        union already has them it still publishes: the front door may be down
        (a fresh stack, or after ``stack down``), and an unchanged render
        applies as a no-op.
        """
        from .residency import ResidencyUnknown

        recovery = self.backend.recovery_profile
        with self._global_lock():
            self._sync_profile(create=True)
            try:
                residency = self.backend.residency()
            except ResidencyUnknown:
                residency = None
            candidate = self._acquire_profile_candidate(residency=residency)
            if candidate is None or recovery is None:
                self._mark_pending(apply=True)
                return self._publish()
            stored = self.ledger.profile()
            self._use_profile_candidate(candidate)
            try:
                self._prepare_network(self.backend.host_runtime)
                desired, inputs = self._admission_view(residency, virtual_expiry=True)
                self.backend.preview(desired, inputs, approve=True)
            except BaseException:
                self._restore_stored_profile(stored)
                raise
            try:
                # Profile (the published endpoints) and approved marker at once.
                self.ledger.publish_profile(
                    candidate, approved_digest=self.backend.last_preview_digest)
            except BaseException:
                self._restore_stored_profile(stored)
                raise
            self._fresh_profile = None
            self._profile_error = None
            self._applied_profile = candidate
            return self._publish()

    def _acquire_by_admission(
        self, owner, requests, *, ttl_seconds, apply, wait_for_placement,
        placement_timeout, placement_interval,
    ):
        """The acquire: preview in memory, commit only if admissible.

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
                stored_profile = self.ledger.profile()
                leases, _ = self.ledger.status(virtual_expiry=True)
                if any(le.state == LeaseState.EXPIRED and
                       self._stored_state(le.id) == LeaseState.ACTIVE
                       for le in leases):
                    # Maintenance sweep: TTL expiry is a desired-state change.
                    self._mark_pending(apply=True)
                    self.ledger.sweep()
                    self._publish()
                try:
                    residency = self.backend.residency()
                    self._residency_error = None
                except ResidencyUnknown as ex:
                    residency = None
                    self._residency_error = str(ex)
                candidate_profile = self._acquire_profile_candidate(
                    residency=residency
                )
                if candidate_profile is not None:
                    self._use_profile_candidate(candidate_profile)
                try:
                    self.backend.validate_requests(requests)
                    overlay = self.ledger.plan_acquire(
                        requests,
                        resident=residency.is_resident if residency is not None else None)
                    allocations, reasons = self._admit(overlay, residency)
                except BaseException:
                    if candidate_profile is not None:
                        self._restore_stored_profile(stored_profile)
                    raise
                if not reasons:
                    # Allocations are committed with the lease, so no placement
                    # scope needs recording for recovery.
                    self._mark_pending(apply=apply)
                    try:
                        result = self.ledger.acquire(
                            owner, requests, ttl_seconds=ttl_seconds,
                            overlay=overlay, allocations=allocations,
                            # Nothing is applied for --no-apply, so there is no
                            # approval to guard; staged state stays discardable.
                            approved_digest=self._admission_digest if apply else None,
                            # The published endpoints commit with the lease.
                            profile=candidate_profile,
                        )
                    except AdmissionConflict:
                        continue            # the ledger moved; preview again
                    except BaseException:
                        if candidate_profile is not None:
                            self._restore_stored_profile(self.ledger.profile())
                        raise
                    if candidate_profile is not None:
                        self._profile_error = None
                        self._applied_profile = candidate_profile
                    try:
                        rec = self._render()
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
                if candidate_profile is not None:
                    self._restore_stored_profile(stored_profile)
                blocked = sorted(
                    gid for gid in overlay.deployments
                    if gid in overlay.created or gid in overlay.revived
                )
            if not (wait_for_placement and apply):
                raise PlacementError(blocked, reasons, capacity=self._admission_capacity)
            if not checked_feasible:
                checked_feasible = True
                infeasible = self._infeasible_alone(
                    list(overlay.deployments.values()), set(blocked))
                if infeasible:
                    raise PlacementError(sorted(infeasible), sorted(infeasible.values()),
                                         capacity=False)
            if self.clock() + placement_interval > deadline:
                raise PlacementError(blocked, reasons, capacity=self._admission_capacity)
            self.sleep(placement_interval)

    def _finish_acquire(self, result, rec, *, apply, wait, timeout, interval) -> AcquireOutcome:
        """Readiness wait (outside the lock) and the outcome, for both paths."""
        deployments = [self.ledger.get_deployment(g.id) for g in result.deployments]
        deployments = [g for g in deployments if g is not None]
        wait_result = None
        released_on_timeout = False
        if wait and apply:  # nothing to wait on when we only staged the render
            try:
                wait_result = self.wait_ready(
                    deployments,
                    endpoints=set(result.lease.endpoints),
                    timeout=timeout,
                    interval=interval,
                )
            except BaseException:
                # Ctrl-C (or any SIGINT-driven KeyboardInterrupt, or a
                # SystemExit) during the readiness wait used to leave the lease
                # ACTIVE with its deployment LIVE: nothing released it, so the
                # GPU stayed claimed and -- because the stack was no longer
                # quiescent -- every later acquire was refused for redefining a
                # frozen endpoint until the TTL expired. An interrupted wait is
                # an acquire that did not deliver, exactly like a timeout, so it
                # rolls back the same way before the interrupt continues.
                # Measured 2026-09-21: one interrupted `run` blocked every
                # acquire on the host for 22 minutes.
                self.release(result.lease.id)
                raise
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

    # -- the gateway's route registry (routes seed / prune) ----------------

    def _route_gateway(self) -> FrontDoorControl:
        """The front door, or a ProfileMismatch saying there is none."""
        from .profile import ProfileMismatch

        front = self.backend.front_door()
        if front is None or not front.litellm:
            raise ProfileMismatch(
                'the `routes` commands need a LiteLLM gateway (the compose or '
                'kubeai backend, with `litellm` on)')
        return front

    def route_view(self) -> list:
        """The routes the gateway serves now (``routes list``): derived from the
        published catalogs, the placed deployments and any legacy registry,
        exactly as the next render derives them."""
        from .residency import ResidencyUnknown

        self._route_gateway()
        try:
            residency = self.backend.residency()
        except ResidencyUnknown:
            residency = None
        desired, inputs = self._admission_view(residency, virtual_expiry=True)
        return self.backend.routes(desired, inputs)

    def _published_sources(self) -> list[dict]:
        """The published catalog union's sources: the stored profile's, and
        none before the first publication."""
        return list((self.ledger.profile() or {}).get('catalogs') or [])

    def _meanings(self, union) -> dict:
        """``{alias: Meaning}`` for every endpoint of ``union``."""
        from .catalog import CatalogError
        from .routes import Meaning

        if union is None:
            return {}
        routes = {r.alias: r for r in self.backend.catalog_routes(union)}
        meanings = {}
        for name in sorted(union.endpoints):
            try:
                key = union.resolve_endpoint(name).semantic_key()
            except CatalogError:
                continue
            meanings[name] = Meaning(routes.get(name), key)
        return meanings

    def _published_meanings(self) -> dict:
        """What every published alias means: route registry rows, then the
        published union's definitions over them."""
        from .profile import CatalogUnion
        from .routes import Meaning

        meanings = {r.alias: Meaning(r) for r in self._route_gateway().registry_routes()}
        sources = self._published_sources()
        meanings.update(self._meanings(CatalogUnion.from_sources(sources) if sources else None))
        return meanings

    def _store_catalogs(self, catalogs: list[dict]) -> None:
        """Replace the published union (caller holds the lock, marker set)."""
        stored = self.ledger.profile() or self.invocation_profile()
        if stored is None:
            return
        profile = {**stored, 'catalogs': catalogs}
        if profile != stored:
            self._commit_profile_candidate(profile)
            self._use_profile_candidate(profile)

    def plan_route_seed(self, catalogs: Iterable) -> RoutePlan:
        """What publishing these catalogs would add, keep, or redefine.
        Raises :class:`~infer_stack.leasing.profile.CatalogConflict` if they
        disagree among themselves."""
        from .profile import CatalogUnion, catalog_sources
        from .routes import plan_seed

        gateway = self._route_gateway()
        sources = [s for c in catalogs for s in catalog_sources(c)]
        union = CatalogUnion.from_sources(sources) if sources else None
        self._require_keys(gateway, union)
        incoming = self._meanings(union)
        plan = plan_seed(self._published_meanings(), incoming)
        plan.sources = sources
        return plan

    def _require_keys(self, front: FrontDoorControl, catalog) -> None:
        """Refuse a seed when the routes it would publish -- everything
        published plus ``catalog`` -- send a key with no value. Checked before
        the marker: the render refuses too, but only after the seed stored its
        catalogs."""
        from .profile import CatalogUnion

        sources = self._published_sources()
        published = CatalogUnion.from_sources(sources) if sources else None
        front.require_route_keys([*self.backend.catalog_routes(published),
                                  *self.backend.catalog_routes(catalog)])

    def commit_route_seed(self, plan: RoutePlan, *, replace: bool = False
                          ) -> tuple[RoutePlan, ReconcileResult]:
        """Merge the plan's catalogs into the published union and publish;
        redefine conflicts only with ``replace``, and never one a resident
        workload runs. Rechecked under the lock: a conflict that appeared
        since the plan refuses too
        (:class:`~infer_stack.leasing.routes.RouteConflict`) and nothing is
        written. ``replace`` is compare-and-swap: each redefinition replaces
        exactly the meaning the plan showed; one another process changed
        meanwhile refuses (``RouteConflict(changed=True)``)."""
        from .profile import CatalogConflict, ProfileMismatch, adopt_catalog_sources
        from .residency import ResidencyUnknown
        from .routes import RouteConflict, plan_seed

        gateway = self._route_gateway()
        fresh = plan_seed(self._published_meanings(), plan.incoming)
        if fresh.conflicted and not replace:
            raise RouteConflict(sorted(fresh.conflicted))

        def preflight():
            # Under the lock, before the marker: a conflict another process
            # made since the plan refuses without leaving a publication pending.
            now = plan_seed(self._published_meanings(), plan.incoming)
            if now.conflicted and not replace:
                raise RouteConflict(sorted(now.conflicted))
            if replace:
                # Compare-and-swap: replace only what was shown and confirmed.
                shown = {n: old for n, (old, _) in plan.conflicted.items()}
                moved = sorted(n for n, (old, _) in now.conflicted.items()
                               if n not in shown or not (shown[n] == old))
                if moved:
                    raise RouteConflict(moved, changed=True)
            from .profile import CatalogUnion

            self._require_keys(gateway, CatalogUnion.from_sources(plan.sources)
                               if plan.sources else None)
            # A resident alias keeps its meaning, wherever that meaning lived
            # (a catalog, a registry row, a dynamic route): refused before the
            # marker, with or without --replace.
            try:
                residency = self.backend.residency()
            except ResidencyUnknown:
                residency = None
            published = self._published_sources()
            blocked = self._pinned_changes(
                published, adopt_catalog_sources(published, plan.sources, set()),
                residency)
            if blocked:
                raise self._pinned_refusal(blocked, 'routes seed')

        def change():
            now = plan_seed(self._published_meanings(), plan.incoming)
            try:
                residency = self.backend.residency()
            except ResidencyUnknown:
                residency = None
            pinned = self._pinned_endpoints(residency)
            try:
                catalogs = adopt_catalog_sources(
                    self._published_sources(), plan.sources, pinned)
            except CatalogConflict as ex:
                blocked = sorted(set(ex.names) & pinned) or sorted(ex.names)
                raise ProfileMismatch(
                    f'routes seed would redefine {", ".join(map(repr, blocked))}, '
                    'which a resident deployment is running; evict it first '
                    f'(`infer-stack evict {blocked[0]}`)') from ex
            self._store_catalogs(catalogs)
            # A published definition supersedes a registry row of the same name.
            legacy = gateway.route_entries()
            kept = {k: v for k, v in legacy.items() if k not in plan.incoming}
            if kept != legacy:
                gateway.replace_route_entries(kept)
            return now

        return self.publish_change(change, preflight=preflight)

    def _prune_keep(self) -> set[str]:
        """Aliases a prune keeps: the invocation's catalogs', and every one a
        deployment the next render places (or a resident one) serves."""
        from .profile import CatalogUnion
        from .residency import ResidencyUnknown

        mine = (self.invocation_profile() or {}).get('catalogs') or []
        keep = set(CatalogUnion.from_sources(mine).endpoints) if mine else set()
        try:
            residency = self.backend.residency()
        except ResidencyUnknown:
            residency = None
        desired, _ = self._admission_view(residency, virtual_expiry=True)
        keep.update(ep for g in desired for ep in g.served)
        return keep | self._pinned_endpoints(residency)

    def _prunable(self) -> list[str]:
        from .profile import CatalogUnion
        from .routes import plan_prune

        sources = self._published_sources()
        current = set(self._route_gateway().route_entries())
        if sources:
            current |= set(CatalogUnion.from_sources(sources).endpoints)
        return plan_prune(current, self._prune_keep()).dropped

    def plan_route_prune(self) -> RoutePlan:
        """Which aliases a prune unpublishes: every published definition and
        registry row but the invocation's catalogs' and the ones deployments
        serve."""
        return RoutePlan(dropped=self._prunable())

    def commit_route_prune(self, plan: RoutePlan) -> tuple[list[str], ReconcileResult]:
        """Unpublish the plan's aliases that are still unneeded, and publish.
        One that became needed since the plan is kept."""
        from .profile import drop_catalog_names

        gateway = self._route_gateway()
        confirmed = set(plan.dropped)

        def change():
            drop = sorted(confirmed & set(self._prunable()))
            if drop:
                legacy = gateway.route_entries()
                kept = {k: v for k, v in legacy.items() if k not in drop}
                if kept != legacy:
                    gateway.replace_route_entries(kept)
                # A bundle naming an unpublished endpoint goes with it.
                self._store_catalogs(drop_catalog_names(self._published_sources(), set(drop)))
            return drop

        return self.publish_change(change)

    def publish_change(self, change: Callable[[], _T], *,
                       preflight: Callable[[], None] | None = None
                       ) -> tuple[_T, ReconcileResult]:
        """Run ``change`` and publish, as one serialised desired-state mutation.

        For changes to state the render reads outside the ledger (the published
        catalogs and the legacy route registry, via ``routes seed`` / ``routes
        prune``). ``change`` runs under the lock
        after the marker is set, so a crash leaves it pending like any other
        mutation. Returns ``(change's result, reconcile result)``.

        ``preflight`` runs under the lock before the marker is set: a check
        that refuses there (raises) leaves no publication pending.
        """
        with self._global_lock():
            if preflight is not None:
                preflight()
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

        recovery = self.backend.recovery_profile
        if recovery is None:
            raise ProfileMismatch('this backend has no publishable profile')
        use = recovery.use_profile
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
            try:
                snap = self.backend.residency()
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
            # A PURE preview first: the real render persists append-only
            # state (route registry, addresses), which must not happen for
            # a candidate whose publication has not committed.
            previous = self._applied_profile
            use(profile)
            try:
                residency = self.backend.residency()
                self._prepare_network(self.backend.host_runtime)
                desired, inputs = self._admission_view(
                    residency, virtual_expiry=True
                )
                self.backend.preview(desired, inputs, approve=True)
            except BaseException:
                if previous is not None:
                    use(previous)
                raise
            self.ledger.publish_profile(
                profile, approved_digest=self.backend.last_preview_digest)
            self._profile_error = None
            self._applied_profile = profile
            self._invocation_profile = profile
            self._profile_drift_warned = False
            return self._publish()

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
        desired-state change, so it is re-admitted: the IDLE
        deployment adopts its resident GPUs, or is placed fresh; if neither is
        possible the renew fails with :class:`PlacementError` and writes
        nothing. The lease is re-validated as ACTIVE under the lock first.
        """
        from .backend import PlacementError

        fast = self.ledger.renew_if_live(lease_id, ttl_seconds=ttl_seconds)
        if fast is not False:
            return RenewOutcome(fast, [])
        with self._global_lock():
            lease = self.ledger.get_lease(lease_id)
            if lease is None or lease.state != LeaseState.ACTIVE:
                return RenewOutcome(None, [])
            from .residency import ResidencyUnknown

            try:
                residency = self.backend.residency()
                self._residency_error = None
            except ResidencyUnknown as ex:
                residency = None
                self._residency_error = str(ex)
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
                raise PlacementError(list(overlay.revived), reasons,
                                     capacity=self._admission_capacity)
            self._mark_pending(apply=True)
            if self._admission_digest:
                self.ledger.mark_publication_pending(
                    apply_requested=True, approved_digest=self._admission_digest)
            renewed = self.ledger.renew(
                lease_id, ttl_seconds=ttl_seconds, allocations=allocations)
            rec = self._publish()
            return RenewOutcome(renewed, list(overlay.revived), rec)

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
