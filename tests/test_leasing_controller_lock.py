"""The controller's cross-process lock serializes the reconcile critical section.

Two callers reconciling at once must not interleave their desired-read +
converge: the second has to wait for the first to finish (and unlock) before it
renders/applies, otherwise it diffs/applies against a target the first just
moved ("last render wins" / stale-diff). The lock lives beside the shared
ledger db; an in-memory ledger has no shared state and degrades to a no-op.
"""

from __future__ import annotations

import threading
import time

from infer_stack.leasing import Controller, Ledger, SqliteStore


class OverlapBackend:
    """A converge-style backend that records max concurrent converges."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()
        self.last_unplaced: tuple = ()
        self.last_errors: tuple = ()
        self.last_assignments: dict = {}

    def converge(self, desired, *, apply: bool = True) -> None:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)  # widen the window an unguarded race would exploit
        with self._guard:
            self.active -= 1

    def observe(self) -> set:
        return set()

    def probe_ready(self, deployment, endpoint):  # pragma: no cover - unused
        from infer_stack.leasing.backend import Readiness

        return Readiness(True, 'fake')


def test_reconcile_is_serialized_across_threads(tmp_path):
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
    backend = OverlapBackend()
    ctl = Controller(ledger, backend)
    assert ctl._lock_path is not None  # file ledger -> real lock

    barrier = threading.Barrier(4)

    def hammer():
        barrier.wait()  # maximize the chance of overlap
        ctl.reconcile()

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert backend.max_active == 1, (
        f'reconcile converges overlapped (max_active={backend.max_active}); '
        'the global lock did not serialize them'
    )


def test_in_memory_ledger_has_no_lock():
    ctl = Controller(Ledger(SqliteStore(':memory:')), OverlapBackend())
    assert ctl._lock_path is None
    # reconcile must still work (no-op lock context)
    ctl.reconcile()
