"""Serialised publication: render and apply under one lock, gated by a marker.

Every desired-state mutation records a durable ``publication_pending`` marker,
mutates the ledger, renders, applies that exact render when an apply was
requested, and clears the marker only after the whole apply succeeded. These
tests cover the crash and failure windows, staged (``--no-apply``) leases, and
the rollback paths.

The fakes share state across separate ``Controller`` instances (the
cross-process shape: distinct sqlite connections and flock handles on one
ledger db), so they exercise the real coordination, not a single object.
"""

from __future__ import annotations

import threading
import time

import pytest

from infer_stack.leasing import (
    Controller,
    EndpointRequest,
    Ledger,
    SqliteStore,
    vllm_structural,
)
from infer_stack.leasing.backend import Readiness



# A worker that dies before reaching the barrier leaves the survivors waiting
# on it forever, and an unbounded join() then hangs the whole run rather than
# failing it -- observed as a CI job stuck for nearly two hours. Both waits are
# bounded, and every worker builds its own objects INSIDE the try so a
# construction failure is recorded instead of silently killing the thread.
THREAD_TIMEOUT_S = 30

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


def _fresh(db, shared, backend_cls=None, **kw):
    backend_cls = backend_cls or SharedStackBackend
    ledger = Ledger(SqliteStore(db))
    return ledger, Controller(ledger, backend_cls(shared, threading.Lock(), **kw))


def _shared():
    return {'rendered': set(), 'realized': set(), 'apply_calls': 0}


def test_acquire_applies_and_clears_the_marker(tmp_path):
    shared = _shared()
    ledger, ctl = _fresh(str(tmp_path / 'ledger.db'), shared)
    out = ctl.acquire('alice', [_vreq('a')], wait=False)
    assert shared['realized'] == {out.deployments[0].id}
    assert ledger.publication_pending() is None
    assert out.reconcile.publication_pending is False


def test_apply_runs_under_the_lock_that_serialises_renders(tmp_path):
    """No render can change the files an apply is reading: the lock is held
    (by this controller) for the whole apply, and another process cannot take it."""
    import fcntl

    db = str(tmp_path / 'ledger.db')
    seen: dict[str, object] = {}
    holder: dict[str, Controller] = {}

    class ProbeBackend(SharedStackBackend):
        def apply(self) -> None:
            ctl = holder['ctl']
            seen['depth'] = ctl._flock_depth
            handle = ctl._open_flock(ctl._lock_path)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen['lock_free'] = True
                fcntl.flock(handle, fcntl.LOCK_UN)
            except OSError:
                seen['lock_free'] = False
            finally:
                handle.close()
            super().apply()

    _, ctl = _fresh(db, _shared(), ProbeBackend)
    holder['ctl'] = ctl
    ctl.acquire('alice', [_vreq('a')], wait=False)
    assert seen == {'depth': 1, 'lock_free': False}


def test_concurrent_acquires_never_overlap_applies_and_all_converge(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    guard = threading.Lock()
    n = 8
    barrier = threading.Barrier(n)
    results: dict[int, list[str]] = {}
    errors: list[str] = []
    overlap = []

    class Checked(SharedStackBackend):
        def apply(self) -> None:
            with self.guard:
                if self.shared.get('applying'):
                    overlap.append(True)
            super().apply()

    def worker(i: int) -> None:
        try:
            ctl = Controller(Ledger(SqliteStore(db)),
                             Checked(shared, guard, apply_sleep=0.02))
            barrier.wait(timeout=THREAD_TIMEOUT_S)
            out = ctl.acquire(f'owner{i}', [_vreq(f'm{i}')], wait=False)
            results[i] = [g.id for g in out.deployments]
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=THREAD_TIMEOUT_S)
    assert not [t for t in threads if t.is_alive()], 'a worker never finished'
    assert not errors, errors
    assert not overlap, 'two applies ran at once'
    assert shared['apply_calls'] == n
    assert shared['realized'] == {g for ids in results.values() for g in ids}
    assert Ledger(SqliteStore(db)).publication_pending() is None


# -- crash windows (plan tests 8-10) -----------------------------------------


def test_crash_after_mutation_before_render_is_applied_by_the_next_operation(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, _ = _fresh(db, shared)
    ledger.mark_publication_pending(apply_requested=True)
    res = ledger.acquire('alice', [_vreq('a')])          # ...and the process dies
    assert shared['rendered'] == set() and shared['realized'] == set()

    ledger2, ctl2 = _fresh(db, shared)                   # a new process
    ctl2.gc()
    assert shared['realized'] == {res.deployments[0].id}
    assert ledger2.publication_pending() is None


def test_crash_after_render_before_apply_is_applied_by_the_next_operation(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, _ = _fresh(db, shared)
    ledger.mark_publication_pending(apply_requested=True)
    res = ledger.acquire('alice', [_vreq('a')])
    SharedStackBackend(shared, threading.Lock()).converge(
        [res.deployments[0]], apply=False)               # rendered; then killed
    assert shared['rendered'] and not shared['realized']

    ledger2, ctl2 = _fresh(db, shared)
    ctl2.apply_now()
    assert shared['realized'] == {res.deployments[0].id}
    assert ledger2.publication_pending() is None


def test_crash_after_apply_before_clear_reapplies_idempotently(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared)
    ctl.acquire('alice', [_vreq('a')], wait=False)
    ledger.mark_publication_pending(apply_requested=True)  # as if never cleared
    realized = set(shared['realized'])

    ctl.gc()
    assert shared['realized'] == realized
    assert shared['apply_calls'] == 2
    assert ledger.publication_pending() is None


# -- staged leases (plan tests 12-15) ----------------------------------------


def test_no_apply_acquire_stays_staged_across_a_reopen(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared)
    out = ctl.acquire('alice', [_vreq('a')], wait=False, apply=False)
    assert out.reconcile.publication_pending is True
    assert shared['rendered'] == {out.deployments[0].id}

    ledger2, ctl2 = _fresh(db, shared)
    ctl2.reconcile(apply=False)                          # `infer-stack render`
    assert shared['realized'] == set() and shared['apply_calls'] == 0
    assert ledger2.publication_pending()['apply_requested'] is False


def test_render_never_applies_even_when_an_apply_is_pending(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared)
    ledger.mark_publication_pending(apply_requested=True)
    ledger.acquire('alice', [_vreq('a')])
    rec = ctl.reconcile(apply=False)
    assert shared['apply_calls'] == 0 and rec.publication_pending is True
    assert ledger.publication_pending()['apply_requested'] is True


def test_a_later_release_applies_a_staged_lease(tmp_path):
    # D23: once any operation requests an apply, the whole pending state is applied.
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared)
    staged = ctl.acquire('alice', [_vreq('a')], wait=False, apply=False)
    other = ledger.acquire('bob', [_vreq('b')])
    ctl.release(other.lease.id)
    assert staged.deployments[0].id in shared['realized']
    assert ledger.publication_pending() is None


def test_apply_now_applies_a_staged_lease_and_clears(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared)
    staged = ctl.acquire('alice', [_vreq('a')], wait=False, apply=False)
    rec = ctl.apply_now()
    assert shared['realized'] == {staged.deployments[0].id}
    assert rec.publication_pending is False
    assert ledger.publication_pending() is None


# -- failed applies (plan tests 16, 20) --------------------------------------


def test_an_apply_that_does_not_take_effect_keeps_the_marker_until_a_retry(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    verdicts = [False, True]

    class RoutesFailOnce(SharedStackBackend):
        def apply(self):
            super().apply()
            return verdicts.pop(0)

    ledger, ctl = _fresh(db, shared, RoutesFailOnce)
    out = ctl.acquire('alice', [_vreq('a')], wait=False)
    assert out.reconcile.publication_pending is True
    assert ledger.publication_pending()['apply_requested'] is True

    rec = ctl.apply_now()
    assert rec.publication_pending is False
    assert ledger.publication_pending() is None


class SettleBackend(SharedStackBackend):
    """Adds a scripted runtime sampler, as ComposeBackend.settle_snapshot."""

    def __init__(self, *args, samples=(), **kw):
        super().__init__(*args, **kw)
        self.samples = list(samples)

    def settle_snapshot(self):
        sample = self.samples.pop(0) if len(self.samples) > 1 else self.samples[0]
        if isinstance(sample, Exception):
            raise sample
        return sample


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _settle_ctl(db, shared, samples, apply_fn=None):
    clock = FakeClock()

    class Backend(SettleBackend):
        def apply(self):
            if apply_fn is not None:
                return apply_fn()
            return super().apply()

    ledger = Ledger(SqliteStore(db))
    backend = Backend(shared, threading.Lock(), samples=samples)
    return ledger, Controller(ledger, backend, clock=clock, sleep=clock.sleep), clock


def _timeout():
    from infer_stack.leasing.backend import BackendTimeout

    raise BackendTimeout('docker compose up timed out after 1800s')


def test_a_timed_out_acquire_leaves_no_active_lease_and_releases_the_lock(tmp_path):
    import fcntl

    from infer_stack.leasing import LeaseState
    from infer_stack.leasing.backend import BackendTimeout

    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl, _ = _settle_ctl(db, shared, [()], apply_fn=_timeout)
    with pytest.raises(BackendTimeout):
        ctl.acquire('alice', [_vreq('a')], wait=False)

    leases, _ = ledger.status()
    assert [le.state for le in leases] == [LeaseState.RELEASED]
    marker = ledger.publication_pending()
    assert marker['apply_requested'] is True and marker['interrupted'] is True
    assert shared['rendered'] == set()        # the rollback re-rendered, never applied
    handle = ctl._open_flock(ctl._lock_path)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)   # raises if still held
    finally:
        handle.close()


def test_after_an_interrupted_apply_the_next_apply_waits_for_a_settled_runtime(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl, _ = _settle_ctl(db, shared, [()], apply_fn=_timeout)
    with pytest.raises(Exception):
        ctl.acquire('alice', [_vreq('a')], wait=False)

    # The daemon is still finishing: states change, then hold still.
    samples = [(('c1', 'created'),), (('c1', 'running'),), (('c1', 'running'),)]
    ledger2, ctl2, clock = _settle_ctl(db, shared, samples)
    ctl2.apply_now()
    assert shared['apply_calls'] == 1
    assert clock.now >= 2 * 2.0                   # two intervals before applying
    assert ledger2.publication_pending() is None  # interrupted cleared with it


@pytest.mark.parametrize('samples', [
    [(('c1', 'running'),), (('c2', 'running'),)],            # never stops changing
    [(('c1', 'removing'),)],                                   # stuck removing
    [RuntimeError('docker ps: permission denied')],            # unreadable
])
def test_an_unsettled_runtime_blocks_the_apply_and_keeps_it_pending(tmp_path, samples):
    from infer_stack.leasing.backend import RuntimeUnsettled

    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, _ = _fresh(db, shared)
    ledger.mark_publication_pending(apply_requested=True, interrupted=True)
    if len(samples) == 2:
        samples = samples * 100
    ledger2, ctl2, clock = _settle_ctl(db, shared, samples)
    with pytest.raises(RuntimeUnsettled):
        ctl2.apply_now()
    assert shared['apply_calls'] == 0
    assert clock.now <= 60.0
    assert ledger2.publication_pending()['interrupted'] is True


def test_no_apply_acquire_never_applies_over_an_older_pending_apply(tmp_path):
    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared)
    ledger.mark_publication_pending(apply_requested=True)
    out = ctl.acquire('alice', [_vreq('a')], wait=False, apply=False)
    assert shared['apply_calls'] == 0
    assert out.reconcile.publication_pending is True
    assert ledger.publication_pending()['apply_requested'] is True


def test_no_apply_rollback_never_applies_over_an_older_pending_apply(tmp_path):
    from infer_stack.leasing.backend import PlacementError

    db = str(tmp_path / 'ledger.db')
    shared = _shared()
    ledger, ctl = _fresh(db, shared, _NoRoomBackend)
    ledger.mark_publication_pending(apply_requested=True)
    with pytest.raises(PlacementError):
        ctl.acquire('alice', [_vreq('big')], wait=False, apply=False)
    assert shared['apply_calls'] == 0


# -- rollback ----------------------------------------------------------------


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
    the publication marker cleared."""
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
    assert ledger.publication_pending() is None


def test_rollback_keeps_coalesced_warm_deployment_resident(tmp_path):
    """A failed acquire that coalesced onto a pre-existing warm (idle keep-warm)
    deployment must roll that deployment back to IDLE — not evict the resident
    model someone else may still want warm."""
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
    AND re-render, so whatever is on disk when the lock drops does not contain
    the declined deployment."""
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


# -- the marker itself -------------------------------------------------------


def test_marker_apply_request_only_turns_on_and_clear_refuses_older_versions(tmp_path):
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    assert ledger.publication_pending() is None
    first = ledger.mark_publication_pending(apply_requested=True)
    second = ledger.mark_publication_pending(apply_requested=False)
    assert second['apply_requested'] is True            # a staged change never cancels
    assert second['version'] == first['version'] + 1
    assert ledger.clear_publication_pending(first['version']) is False
    assert ledger.clear_publication_pending(second['version']) is True
    assert ledger.publication_pending() is None


# -- rollback never evicts on a failed look (strict residency) ---------------


def _warm_then_failed_acquire(tmp_path, residency):
    from infer_stack.leasing.backend import PlacementError

    shared = _shared()

    class Backend(_NoRoomBackend):
        pass

    Backend.residency = lambda self: residency(self)
    ledger, ctl = _fresh(str(tmp_path / 'ledger.db'), shared, Backend)
    warm = ctl.acquire('alice', [_vreq('m')], wait=False)
    ctl.release(warm.lease.id)
    shared['realized'] = set()        # observe() now says "nothing" (as on a docker error)
    with pytest.raises(PlacementError):
        ctl.acquire('bob', [_vreq('m'), _vreq('big')], wait=False)
    return ledger.get_deployment(warm.deployments[0].id)


def test_rollback_does_not_evict_a_warm_deployment_when_residency_is_unknown(tmp_path):
    from infer_stack.leasing import DeploymentState
    from infer_stack.leasing.residency import ResidencyUnknown

    def unknown(backend):
        raise ResidencyUnknown('docker ps failed')

    assert _warm_then_failed_acquire(tmp_path, unknown).state == DeploymentState.IDLE


def test_rollback_keeps_a_deployment_that_has_a_container(tmp_path):
    from infer_stack.leasing import DeploymentState
    from infer_stack.leasing.residency import Container, Residency

    def any_id(backend):
        class Everything(Residency):
            def containers(self, deployment_id):
                return (Container('c1', deployment_id, 'exited'),)
        return Everything({})

    assert _warm_then_failed_acquire(tmp_path, any_id).state == DeploymentState.IDLE


def test_rollback_evicts_a_deployment_with_definitely_no_container(tmp_path):
    from infer_stack.leasing import DeploymentState
    from infer_stack.leasing.residency import Residency

    got = _warm_then_failed_acquire(tmp_path, lambda backend: Residency({}))
    assert got.state == DeploymentState.STOPPED


# -- renew is a desired-state mutator ------------------------------------------


class MutableClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _lapsed_idle_lease(tmp_path, shared, backend_cls=None, **kw):
    """An ACTIVE lease whose TTL lapsed unswept while the deployment went IDLE."""
    clock = MutableClock()
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')), clock=clock)
    ctl = Controller(ledger, (backend_cls or SharedStackBackend)(
        shared, threading.Lock(), **kw))
    a = ledger.acquire('alice', [_vreq('m')], ttl_seconds=100)
    b = ledger.acquire('bob', [_vreq('m')])
    clock.now += 200
    ledger.release(b.lease.id)
    gid = a.deployments[0].id
    assert ledger.get_deployment(gid).state == 'idle'
    return ledger, ctl, a.lease.id, gid


def test_renew_that_revives_an_idle_deployment_publishes(tmp_path):
    shared = _shared()
    ledger, ctl, lease_id, gid = _lapsed_idle_lease(tmp_path, shared)
    out = ctl.renew(lease_id, ttl_seconds=3600)
    assert out.revived_deployment_ids == [gid]
    assert ledger.get_deployment(gid).state == 'live'
    assert shared['apply_calls'] == 1 and gid in shared['realized']
    assert ledger.publication_pending() is None


def test_ttl_only_renew_takes_no_marker_and_runs_no_apply(tmp_path):
    shared = _shared()
    ledger, ctl = _fresh(str(tmp_path / 'ledger.db'), shared)
    lease = ctl.acquire('alice', [_vreq('m')], wait=False).lease
    calls = shared['apply_calls']
    out = ctl.renew(lease.id, ttl_seconds=3600)
    assert out.lease is not None and out.reconcile is None
    assert shared['apply_calls'] == calls
    assert ledger.publication_pending() is None
    assert ctl.renew('lease-nope', ttl_seconds=60).lease is None


def test_a_reviving_renew_waits_for_an_in_flight_apply(tmp_path):
    shared = _shared()
    in_apply, finish = threading.Event(), threading.Event()
    order = []

    class Slow(SharedStackBackend):
        def apply(self):
            if not in_apply.is_set():
                in_apply.set()
                finish.wait(THREAD_TIMEOUT_S)
                order.append('first apply done')
            return super().apply()

    ledger, ctl, lease_id, gid = _lapsed_idle_lease(tmp_path, shared, Slow)
    db = str(tmp_path / 'ledger.db')
    # Step back inside the TTL so t1's own sweep does not expire the lease first;
    # the deployment stays IDLE in the ledger.
    ledger.clock.now -= 150
    t1 = threading.Thread(target=lambda: Controller(
        Ledger(SqliteStore(db), clock=ledger.clock),
        Slow(shared, threading.Lock())).apply_now())
    t1.start()
    assert in_apply.wait(THREAD_TIMEOUT_S)

    def renew():
        Controller(Ledger(SqliteStore(db), clock=ledger.clock),
                   Slow(shared, threading.Lock())).renew(lease_id, ttl_seconds=3600)
        order.append('renew done')

    t2 = threading.Thread(target=renew)
    t2.start()
    time.sleep(0.3)
    assert ledger.get_deployment(gid).state == 'idle'   # blocked on the lock
    finish.set()
    t1.join(THREAD_TIMEOUT_S)
    t2.join(THREAD_TIMEOUT_S)
    assert order == ['first apply done', 'renew done']
    assert ledger.get_deployment(gid).state == 'live'


def test_no_cli_or_tui_code_renews_around_the_controller():
    import pathlib
    import re

    import infer_stack

    root = pathlib.Path(infer_stack.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob('*.py')
        if path.name not in {'ledger.py', 'controller.py'}
        and re.search(r'ledger\.renew\(', path.read_text())
    ]
    assert offenders == []


# -- transitional: rollback under unknown residency keeps a phantom warm candidate --


def test_transitional_unknown_residency_rollback_keeps_an_idle_candidate(tmp_path):
    """Records CURRENT behaviour, which plan step P9 changes.

    A brand-new acquire fails; rollback cannot read residency, so it refuses to
    evict (the safe direction). Until P9 makes IDLE keep-warm deployments
    optional and resident-only, the desired set still contains that IDLE
    deployment, so a later successful apply starts it with no lease behind it.
    After P9 this test must assert the deployment is never started.
    """
    from infer_stack.leasing import DeploymentState
    from infer_stack.leasing.backend import PlacementError
    from infer_stack.leasing.residency import ResidencyUnknown

    shared = _shared()

    class Backend(_NoRoomBackend):
        def residency(self):
            raise ResidencyUnknown('docker ps failed')

    ledger, ctl = _fresh(str(tmp_path / 'ledger.db'), shared, Backend)
    with pytest.raises(PlacementError):
        ctl.acquire('alice', [_vreq('fresh'), _vreq('big')], wait=False)
    fresh = next(g for g in ledger.status()[1] if 'fresh' in g.served)
    assert fresh.state == DeploymentState.IDLE          # not evicted
    ctl.apply_now()
    assert fresh.id in shared['realized']               # started without a lease (P9 fixes)
