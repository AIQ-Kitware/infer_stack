"""A keep-warm model without a lease gives way to a model with one.

The rule (decided 2026-09-24): an idle keep-warm deployment -- resident, but
held by no lease -- is always a candidate for eviction when a leased model
needs a resource. Compose admission enforces it when placing. Where the
runtime schedules instead (KubeAI), the probe says `needs_room` and the wait
evicts idle models, the longest idle first, one per cooldown.
"""

from __future__ import annotations

from infer_stack.leasing import (
    Controller,
    DeploymentState,
    Ledger,
    MemoryBackend,
    SqliteStore,
)
from infer_stack.leasing.backend import Readiness
from infer_stack.leasing.controller import ROOM_COOLDOWN_S
from test_leasing_controller import FakeClock, _id_factory, vreq


class OneSlotBackend(MemoryBackend):
    """Room for `slots` models; a model that does not fit waits for room."""

    def __init__(self, slots=1):
        super().__init__(ready=True)
        self.slots = slots

    def probe_ready(self, deployment, endpoint):
        others = [gid for gid in self.realized if gid != deployment.id]
        if deployment.id in self.realized and len(others) >= self.slots:
            return Readiness(False, 'pod: Unschedulable', needs_room=True)
        return super().probe_ready(deployment, endpoint)


def make(slots=1):
    clock = FakeClock()
    ledger = Ledger(SqliteStore(':memory:'), clock=clock, id_factory=_id_factory())
    backend = OneSlotBackend(slots)
    ctl = Controller(ledger, backend, clock=clock, sleep=clock.advance)
    return ctl, ledger, backend, clock


def state(ledger, gid):
    return ledger.get_deployment(gid).state


def test_an_idle_keep_warm_model_gives_way_to_a_leased_one():
    ctl, ledger, backend, _ = make()
    warm = ctl.acquire('a', [vreq('warm')], wait=False)
    ctl.release(warm.lease.id)                          # keep-warm: still resident
    warm_id = warm.deployments[0].id
    assert state(ledger, warm_id) == DeploymentState.IDLE

    out = ctl.acquire('b', [vreq('big')], wait=True, timeout=600, interval=5)

    assert out.wait.ready
    assert state(ledger, warm_id) == DeploymentState.STOPPED
    assert warm_id not in backend.realized


def test_the_longest_idle_goes_first_and_one_at_a_time():
    ctl, ledger, backend, clock = make(slots=2)
    older = ctl.acquire('a', [vreq('older')], wait=False)
    ctl.release(older.lease.id)
    clock.advance(60)
    newer = ctl.acquire('a', [vreq('newer')], wait=False)
    ctl.release(newer.lease.id)

    out = ctl.acquire('b', [vreq('big')], wait=True, timeout=600, interval=5)

    assert out.wait.ready
    assert state(ledger, older.deployments[0].id) == DeploymentState.STOPPED
    # One eviction freed the slot, so the more recently used model stays warm.
    assert state(ledger, newer.deployments[0].id) == DeploymentState.IDLE


def test_evictions_wait_for_the_cooldown():
    ctl, ledger, backend, clock = make(slots=0)          # nothing ever fits
    for name in ('w1', 'w2'):
        out = ctl.acquire('a', [vreq(name)], wait=False)
        ctl.release(out.lease.id)
    big = ctl.acquire('b', [vreq('big')], wait=False)

    ctl.wait_ready(big.deployments, timeout=ROOM_COOLDOWN_S - 1, interval=5)
    stopped = [g for g in ledger.status()[1] if g.state == DeploymentState.STOPPED]
    assert len(stopped) == 1                             # not both within one cooldown


def test_a_leased_model_is_never_evicted_to_make_room():
    ctl, ledger, backend, _ = make()
    held = ctl.acquire('a', [vreq('held')], wait=False)   # still leased
    out = ctl.acquire('b', [vreq('big')], wait=True, timeout=120, interval=5)

    assert not out.wait.ready                             # waits, then times out
    assert state(ledger, held.deployments[0].id) == DeploymentState.LIVE


def test_an_unschedulable_kubeai_pod_asks_for_room(tmp_path):
    from test_leasing_kubeai import FakeHttp, _pod, make_pod_backend, vllm

    be, kubectl = make_pod_backend(tmp_path)
    dep = vllm('grp-big', served='big')
    be.converge([dep])
    kubectl.pods = [_pod('model-big-1', 'grp-big', statuses=False, conditions=[
        {'type': 'PodScheduled', 'status': 'False', 'reason': 'Unschedulable'}])]
    be.http.post = lambda url, **kw: FakeHttp._Resp(503, {'detail': 'not ready'})

    probe = be.probe_ready(dep, 'grp-big')
    assert probe.needs_room and not probe.fatal and not probe.ready
