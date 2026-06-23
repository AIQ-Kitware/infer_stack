"""Tests for the controller's admission queue (acquire wait_for_placement).

These exercise ``Controller.acquire(wait_for_placement=True)`` against a fake
placement backend with a fixed GPU "budget". The fake mimics a converge-style
backend: it places up to ``budget`` deployments and reports the rest as
``last_unplaced`` (what the Compose backend does when GPUs are full), so the
controller's queue/retry logic can be tested deterministically without GPUs.
"""

from __future__ import annotations

import pytest

from infer_stack.leasing import (
    Controller,
    EndpointRequest,
    Ledger,
    Sharing,
    SqliteStore,
    vllm_structural,
)
from infer_stack.leasing.backend import PlacementError, Readiness


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _id_factory():
    counters: dict[str, int] = {}

    def factory(prefix: str) -> str:
        n = counters.get(prefix, 0)
        counters[prefix] = n + 1
        return f'{prefix}-{n}'

    return factory


class BudgetBackend:
    """Places up to ``budget`` deployments (1 slot each); the rest are unplaced.

    A converge-style fake: ``converge`` recomputes the placed/unplaced split from
    the desired set each call, so freeing demand (a release) lets a queued
    deployment in on the next reconcile.
    """

    def __init__(self, budget: int):
        self.budget = budget
        self.realized: dict[str, object] = {}
        self.last_unplaced: list[str] = []
        self.last_errors: list[str] = []
        self.last_assignments: dict[str, list[int]] = {}

    def converge(self, desired, apply: bool = True) -> None:
        placed = list(desired)[: self.budget]
        unplaced = list(desired)[self.budget :]
        if apply:
            self.realized = {g.id: g for g in placed}
        self.last_assignments = {g.id: [i] for i, g in enumerate(placed)}
        self.last_unplaced = [g.id for g in unplaced]
        self.last_errors = [f'{g.id}: no free GPU' for g in unplaced]

    def observe(self) -> set[str]:
        return set(self.realized)

    def probe_ready(self, deployment, endpoint) -> Readiness:
        return Readiness(deployment.id in self.realized, 'ok')

    def realize(self, deployment) -> None:  # unused by converge backends
        pass

    def teardown(self, deployment) -> None:
        pass


def vreq(endpoint, *, reclaim='stop'):
    return EndpointRequest(
        endpoint=endpoint,
        engine='vllm',
        structural=vllm_structural(model_ref=endpoint),
        capacity={'max_model_len': 32768},
        sharing=Sharing.SHARED,
        spec={'reclaim': reclaim},
        served={'served_model_name': endpoint},
    )


def _make(budget, sleep):
    clock = FakeClock()
    ledger = Ledger(
        SqliteStore(':memory:'), clock=clock, id_factory=_id_factory()
    )
    backend = BudgetBackend(budget=budget)
    ctl = Controller(ledger, backend, clock=clock, sleep=sleep(clock, ledger))
    return ctl, backend, clock, ledger


def test_unplaced_fails_fast_by_default():
    """Without wait_for_placement, a full fleet still fails immediately."""

    def sleep(clock, ledger):
        return lambda dt: clock.advance(dt)

    ctl, backend, _, _ = _make(budget=1, sleep=sleep)
    ctl.acquire('alice', [vreq('A')])  # fills the single slot
    with pytest.raises(PlacementError):
        ctl.acquire('bob', [vreq('B')])  # no wait -> fail fast


def test_acquire_queues_until_a_gpu_frees():
    """wait_for_placement should block until a release frees a slot, then place."""
    state: dict[str, object] = {'a_lease': None, 'released': False}

    def sleep(clock, ledger):
        def _sleep(dt):
            clock.advance(dt)
            # Simulate another job finishing mid-wait: release A, freeing its slot.
            if state['a_lease'] and not state['released']:
                ledger.release(state['a_lease'])
                state['released'] = True

        return _sleep

    ctl, backend, _, _ = _make(budget=1, sleep=sleep)
    a = ctl.acquire('alice', [vreq('A')])
    state['a_lease'] = a.lease.id

    b = ctl.acquire(
        'bob', [vreq('B')], wait_for_placement=True, timeout=100, interval=2
    )
    assert b.deployments[0].id in backend.observe()
    assert state['released'] is True


def test_acquire_queue_times_out_when_never_freed():
    """If no slot ever frees, queueing fails after the placement timeout."""

    def sleep(clock, ledger):
        return lambda dt: clock.advance(dt)  # advance only; never release

    ctl, backend, clock, ledger = _make(budget=1, sleep=sleep)
    ctl.acquire('alice', [vreq('A')])  # fills the slot, never released
    start = clock()
    with pytest.raises(PlacementError):
        ctl.acquire(
            'bob', [vreq('B')], wait_for_placement=True, timeout=10, interval=2
        )
    assert clock() - start >= 10  # actually waited out the timeout
    # The rolled-back lease must not linger as a phantom desired deployment.
    leases, _ = ledger.status()
    assert all(lease.owner != 'bob' or lease.state != 'active' for lease in leases)


if __name__ == '__main__':
    test_unplaced_fails_fast_by_default()
    test_acquire_queues_until_a_gpu_frees()
    test_acquire_queue_times_out_when_never_freed()
    print('All controller admission-queue tests passed.')
