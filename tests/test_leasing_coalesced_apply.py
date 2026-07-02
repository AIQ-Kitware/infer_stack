"""Render/apply lock split (A) + coalesced apply (B).

The controller splits its critical section in two: a short RENDER lock (ledger
write + compose-file render) and a separate APPLY lock around the slow
``docker compose up``. A monotonic generation in the ledger lets one apply
satisfy every acquirer that rendered before it (coalescing), so N concurrent
acquires need far fewer than N applies, and an acquirer never blocks another's
render while it is applying.

These tests use fakes that share state across separate ``Controller`` instances
(the cross-process shape: distinct sqlite connections + flock handles on one
ledger db), so they exercise the real coordination, not a single object.
"""

from __future__ import annotations

import threading
import time

from infer_stack.leasing import (
    Controller,
    EndpointRequest,
    Ledger,
    SqliteStore,
    vllm_structural,
)
from infer_stack.leasing.backend import Readiness


def _vreq(endpoint: str) -> EndpointRequest:
    return EndpointRequest(
        endpoint=endpoint,
        engine='vllm',
        structural=vllm_structural(model_ref=endpoint),
        capacity={'max_model_len': 2048},
        served={'served_model_name': endpoint},
    )


class SharedStackBackend:
    """Models the one shared compose project across separate controllers.

    ``converge(apply=False)`` (render) writes the desired union to shared
    ``rendered``; ``apply`` brings ``rendered`` "up" (shared ``realized``) and
    counts itself. State is shared + guarded so several backend instances behave
    like several processes driving one docker project.
    """

    def __init__(self, shared: dict, guard: threading.Lock, apply_sleep: float = 0.0):
        self.shared = shared
        self.guard = guard
        self.apply_sleep = apply_sleep
        self.last_unplaced: list[str] = []
        self.last_errors: list[str] = []
        self.last_assignments: dict[str, list[int]] = {}

    def converge(self, desired, *, apply: bool = True) -> None:
        ids = {g.id for g in desired}
        with self.guard:
            self.shared['rendered'] = set(ids)
            if apply:
                self.shared['realized'] = set(ids)
        self.last_assignments = {g.id: [0] for g in desired}

    def apply(self) -> None:
        with self.guard:
            self.shared['apply_calls'] += 1
            self.shared.setdefault('applying', 0)
            self.shared['applying'] += 1
        if self.apply_sleep:
            time.sleep(self.apply_sleep)
        with self.guard:
            self.shared['realized'] = set(self.shared['rendered'])
            self.shared['applying'] -= 1

    def observe(self) -> set:
        with self.guard:
            return set(self.shared['realized'])

    def probe_ready(self, deployment, endpoint) -> Readiness:
        return Readiness(True, 'fake')

    def realize(self, deployment) -> None:  # unused (converge backend)
        pass

    def teardown(self, deployment) -> None:
        pass


def test_acquire_bumps_desired_gen_and_apply_advances_applied_gen(tmp_path):
    """Each acquire bumps desired_gen; the coalesced apply advances applied_gen to
    match, and applied_gen is monotonic (never lowered)."""
    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}
    ledger = Ledger(SqliteStore(db))
    ctl = Controller(ledger, SharedStackBackend(shared, threading.Lock()))

    assert ledger.desired_generation() == 0
    assert ledger.applied_generation() == 0

    ctl.acquire('alice', [_vreq('a')], wait=False)
    g1 = ledger.desired_generation()
    assert g1 >= 1
    assert ledger.applied_generation() == g1  # apply caught up

    ctl.acquire('bob', [_vreq('b')], wait=False)
    g2 = ledger.desired_generation()
    assert g2 > g1
    assert ledger.applied_generation() == g2


def test_applied_generation_is_monotonic(tmp_path):
    db = str(tmp_path / 'ledger.db')
    ledger = Ledger(SqliteStore(db))
    ledger.set_applied_generation(5)
    assert ledger.applied_generation() == 5
    ledger.set_applied_generation(3)  # stale/older apply must not lower it
    assert ledger.applied_generation() == 5
    ledger.set_applied_generation(9)
    assert ledger.applied_generation() == 9


def test_ensure_applied_coalesces_concurrent_waiters(tmp_path):
    """The coalescing core, deterministically: many waiters that all need the
    SAME already-rendered generation collapse to exactly ONE apply. The first to
    win the apply-lock publishes ``applied_gen``; everyone else re-checks and is
    already covered, so they never apply.

    Deterministic (no timing assumptions): the generation is advanced up front, so
    the first apply snapshots the final generation and covers every waiter; the
    apply-lock serializes, so a second winner always re-reads applied_gen >= target
    and breaks without applying.
    """
    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}
    guard = threading.Lock()
    n = 8

    # Pretend N acquirers already rendered: advance desired_gen to N (the
    # compose file content is irrelevant to the count, so leave `rendered` empty).
    seed = Ledger(SqliteStore(db))
    for _ in range(n):
        with seed.store.transaction():
            seed.store.bump_desired_generation()
    g_target = seed.desired_generation()
    assert g_target == n

    barrier = threading.Barrier(n)
    errors: list[str] = []

    def worker() -> None:
        ctl = Controller(
            Ledger(SqliteStore(db)),
            SharedStackBackend(shared, guard, apply_sleep=0.05),
        )
        try:
            barrier.wait()
            ctl._ensure_applied(g_target)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert shared['apply_calls'] == 1, (
        f'{n} waiters for the same generation should coalesce to one apply, '
        f"got {shared['apply_calls']}"
    )
    assert Ledger(SqliteStore(db)).applied_generation() == g_target


def test_concurrent_acquires_all_converge(tmp_path):
    """End-to-end: N concurrent acquires across separate controllers all succeed,
    the ledger fully converges (applied_gen == desired_gen), and every model ends
    up 'running' — no acquirer is left un-applied regardless of how the coalescing
    interleaved."""
    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}
    guard = threading.Lock()
    n = 8
    barrier = threading.Barrier(n)
    results: dict[int, list[str]] = {}
    errors: list[str] = []

    def worker(i: int) -> None:
        ctl = Controller(
            Ledger(SqliteStore(db)),
            SharedStackBackend(shared, guard, apply_sleep=0.02),
        )
        try:
            barrier.wait()
            out = ctl.acquire(f'owner{i}', [_vreq(f'm{i}')], wait=False)
            results[i] = [g.id for g in out.deployments]
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == n
    assert 1 <= shared['apply_calls'] <= n  # sanity: at least one, never per-extra
    final = Ledger(SqliteStore(db))
    assert final.applied_generation() == final.desired_generation()
    all_ids = {gid for ids in results.values() for gid in ids}
    assert shared['realized'] >= all_ids


def test_apply_runs_outside_the_render_lock(tmp_path):
    """The whole point of the split (A): the slow apply must NOT hold the render
    lock, so another caller can render while one is applying.

    Asserted structurally and deterministically (no thread race): when the
    backend's ``apply`` runs, (1) this controller is not holding the render lock
    (``_flock_depth == 0``), and (2) a fresh handle can ``flock`` the render-lock
    file right now with ``LOCK_NB`` — i.e. it is genuinely free for another
    process to grab and render. The old single-lock design would fail both: the
    render flock would still be held throughout the apply.
    """
    import fcntl

    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}
    seen: dict[str, object] = {}
    holder: dict[str, Controller] = {}

    class ProbeBackend(SharedStackBackend):
        def apply(self) -> None:
            ctl = holder['ctl']
            seen['depth'] = ctl._flock_depth
            handle = ctl._open_flock(ctl._lock_path)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen['render_lock_free'] = True
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                seen['render_lock_free'] = False
            finally:
                handle.close()
            super().apply()

    ctl = Controller(Ledger(SqliteStore(db)), ProbeBackend(shared, threading.Lock()))
    holder['ctl'] = ctl
    ctl.acquire('alice', [_vreq('a')], wait=False)

    assert seen.get('depth') == 0, (
        'the render lock is still held during apply — the split did not take '
        f"effect (_flock_depth={seen.get('depth')})"
    )
    assert seen.get('render_lock_free') is True, (
        'the render-lock file could not be acquired during apply — another caller '
        'could not render concurrently'
    )


def test_apply_now_snapshots_generation_before_apply(tmp_path):
    """Regression: apply_now must publish the desired generation captured BEFORE
    the up, like _ensure_applied — a render that lands mid-apply changed files
    this apply never read, and covering it would leave that deployment never
    brought up (applied >= desired suppresses all coalesced applies)."""
    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}

    class MidApplyRender(SharedStackBackend):
        def apply(self) -> None:
            other = Ledger(SqliteStore(db))  # a concurrent acquirer renders now
            with other.store.transaction():
                other.store.bump_desired_generation()
            super().apply()

    ledger = Ledger(SqliteStore(db))
    ctl = Controller(ledger, MidApplyRender(shared, threading.Lock()))
    ctl.apply_now()
    assert ledger.applied_generation() < ledger.desired_generation(), (
        'a render that landed mid-apply was marked covered; its deployment '
        'would never be brought up'
    )


class _NoRoomBackend(SharedStackBackend):
    """Converge fake that can never place an endpoint named ``big``."""

    def converge(self, desired, *, apply: bool = True) -> None:
        placeable = [g for g in desired if 'big' not in g.served]
        self.last_unplaced = [g.id for g in desired if 'big' in g.served]
        super().converge(placeable, apply=apply)


def test_placement_rollback_evicts_and_rerenders(tmp_path):
    """Regression: a failed placement must fully roll back — lease released, the
    never-ran deployments evicted (not left idle-keep-warm, which would pin them
    in the desired set), the placed sibling removed from the on-disk render, and
    the ledger converged (a concurrent waiter can never apply a stale render)."""
    import pytest

    from infer_stack.leasing import DeploymentState, LeaseState
    from infer_stack.leasing.backend import PlacementError

    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}
    ledger = Ledger(SqliteStore(db))
    ctl = Controller(ledger, _NoRoomBackend(shared, threading.Lock()))

    with pytest.raises(PlacementError):
        ctl.acquire('alice', [_vreq('ok'), _vreq('big')], wait=False)

    leases, deployments = ledger.status()
    assert [le.state for le in leases] == [LeaseState.RELEASED]
    assert {g.state for g in deployments} == {DeploymentState.STOPPED}
    # 'ok' was rendered by the failed acquire; the rollback re-render removed it.
    assert shared['rendered'] == set()
    assert shared['realized'] == set()
    assert ledger.applied_generation() == ledger.desired_generation()


def test_rollback_keeps_coalesced_warm_deployment_resident(tmp_path):
    """A failed acquire that coalesced onto a pre-existing warm (idle keep-warm)
    deployment must roll that deployment back to IDLE — not evict the resident
    model someone else may still want warm."""
    import pytest

    from infer_stack.leasing import DeploymentState
    from infer_stack.leasing.backend import PlacementError

    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}
    ledger = Ledger(SqliteStore(db))
    ctl = Controller(ledger, _NoRoomBackend(shared, threading.Lock()))

    warm = ctl.acquire('alice', [_vreq('m')], wait=False)
    mid = warm.deployments[0].id
    ctl.release(warm.lease.id)      # keep-warm: idles but stays resident
    assert mid in shared['realized']

    with pytest.raises(PlacementError):
        ctl.acquire('bob', [_vreq('m'), _vreq('big')], wait=False)

    m = ledger.get_deployment(mid)
    assert m.state == DeploymentState.IDLE      # not STOPPED
    assert mid in shared['realized']            # still resident


def test_converge_aborted_rollback_rerenders(tmp_path):
    """Regression: declining an acquire's compose diff must roll the lease back
    AND re-render, restoring the generation contract — release/evict bump the
    desired generation, so whatever is on disk when the lock drops must not
    contain the declined deployment."""
    import pytest

    from infer_stack.leasing import DeploymentState, LeaseState
    from infer_stack.leasing.backend import ConvergeAborted

    db = str(tmp_path / 'ledger.db')
    shared = {'rendered': set(), 'realized': set(), 'apply_calls': 0}

    class DecliningBackend(SharedStackBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.converge_calls = 0
            self.declined = False

        def converge(self, desired, *, apply: bool = True) -> None:
            self.converge_calls += 1
            if desired and not self.declined:
                self.declined = True
                raise ConvergeAborted('operator declined')
            super().converge(desired, apply=apply)

    ledger = Ledger(SqliteStore(db))
    backend = DecliningBackend(shared, threading.Lock())
    ctl = Controller(ledger, backend)

    with pytest.raises(ConvergeAborted):
        ctl.acquire('alice', [_vreq('a')], wait=False)

    leases, deployments = ledger.status()
    assert [le.state for le in leases] == [LeaseState.RELEASED]
    assert [g.state for g in deployments] == [DeploymentState.STOPPED]
    assert backend.converge_calls == 2  # the declined render + the rollback render
    assert shared['rendered'] == set()
