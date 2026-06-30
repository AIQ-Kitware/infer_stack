"""Tests for the controller: reconcile + readiness against MemoryBackend."""

from __future__ import annotations

from infer_stack.leasing import (
    Catalog,
    Controller,
    DeploymentState,
    EndpointRequest,
    Ledger,
    LeaseState,
    MemoryBackend,
    Sharing,
    SqliteStore,
    ollama_structural,
    vllm_structural,
)


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


def make_controller(*, ready=True, sleep=None):
    clock = FakeClock()
    ledger = Ledger(
        SqliteStore(':memory:'), clock=clock, id_factory=_id_factory()
    )
    backend = MemoryBackend(ready=ready)
    ctl = Controller(
        ledger, backend, sleep=sleep or (lambda dt: clock.advance(dt))
    )
    return ctl, backend, clock


def vreq(endpoint, *, reclaim='keep-warm', sharing=Sharing.SHARED, max_model_len=32768):
    return EndpointRequest(
        endpoint=endpoint,
        engine='vllm',
        structural=vllm_structural(model_ref=endpoint),
        capacity={'max_model_len': max_model_len},
        sharing=sharing,
        spec={'reclaim': reclaim},
        served={'served_model_name': endpoint},
    )


def oreq(endpoint, tag, host='local-ollama'):
    return EndpointRequest(
        endpoint=endpoint,
        engine='ollama',
        structural=ollama_structural(host=host),
        spec={'host': host, 'reclaim': 'keep-warm'},
        served={'model': tag},
        host=host,
    )


def test_acquire_realizes_and_ready():
    ctl, backend, _ = make_controller(ready=True)
    out = ctl.acquire('alice', [vreq('qwen')])
    gid = out.deployments[0].id
    assert backend.realize_calls == [gid]
    assert backend.observe() == {gid}
    assert out.wait.ready is True


def test_coalesced_realize_once():
    ctl, backend, _ = make_controller()
    a = ctl.acquire('alice', [vreq('qwen')])
    b = ctl.acquire('bob', [vreq('qwen')])
    assert a.deployments[0].id == b.deployments[0].id
    assert backend.realize_calls == [a.deployments[0].id]   # realized only once


def test_release_one_keeps_running():
    ctl, backend, _ = make_controller()
    a = ctl.acquire('alice', [vreq('qwen')])
    ctl.acquire('bob', [vreq('qwen')])
    out = ctl.release(a.lease.id)
    assert out.reconcile.torn_down == []
    assert backend.teardown_calls == []
    assert backend.observe() == {a.deployments[0].id}


def test_release_last_keepwarm_stays():
    ctl, backend, _ = make_controller()
    a = ctl.acquire('alice', [vreq('qwen', reclaim='keep-warm')])
    ctl.release(a.lease.id)
    assert backend.teardown_calls == []
    assert backend.observe() == {a.deployments[0].id}        # kept warm


def test_release_last_stop_tears_down():
    ctl, backend, _ = make_controller()
    a = ctl.acquire('alice', [vreq('qwen', reclaim='stop')])
    gid = a.deployments[0].id
    out = ctl.release(a.lease.id)
    assert out.reconcile.torn_down == [gid]
    assert backend.teardown_calls == [gid]
    assert backend.observe() == set()


def test_wait_ready_timeout_releases_lease():
    # A readiness timeout must roll the lease back (release + reconcile) so a
    # never-ready acquire doesn't leave the deployment LIVE pinning a GPU.
    ctl, backend, _ = make_controller(ready=False)
    out = ctl.acquire(
        'alice', [vreq('qwen', reclaim='stop')], timeout=10, interval=2
    )
    gid = out.deployments[0].id
    assert out.wait.ready is False
    assert out.wait.pending == [(gid, 'qwen')]
    assert out.released_on_timeout is True
    # Lease released, stop-policy deployment torn down, GPU freed.
    assert ctl.ledger.get_lease(out.lease.id).state == LeaseState.RELEASED
    assert backend.teardown_calls == [gid]
    assert backend.observe() == set()


def test_wait_ready_timeout_keepwarm_idles_deployment():
    # Same rollback for a keep-warm deployment: the lease is released (no longer
    # ACTIVE) even though the warm container is kept resident as a courtesy.
    ctl, backend, _ = make_controller(ready=False)
    out = ctl.acquire(
        'alice', [vreq('qwen', reclaim='keep-warm')], timeout=10, interval=2
    )
    gid = out.deployments[0].id
    assert out.released_on_timeout is True
    assert ctl.ledger.get_lease(out.lease.id).state == LeaseState.RELEASED
    assert ctl.ledger.get_deployment(gid).state == DeploymentState.IDLE


def test_wait_ready_becomes_ready():
    clock = FakeClock()
    ledger = Ledger(
        SqliteStore(':memory:'), clock=clock, id_factory=_id_factory()
    )
    backend = MemoryBackend(ready=False)
    state = {'n': 0}

    def sleep(dt):
        clock.advance(dt)
        state['n'] += 1
        if state['n'] == 1:
            backend.set_ready(True)

    ctl = Controller(ledger, backend, sleep=sleep)
    out = ctl.acquire('alice', [vreq('qwen')], timeout=100, interval=2)
    assert out.wait.ready is True
    assert state['n'] == 1                               # one poll, then ready


def test_ttl_expiry_reaps_stop_deployment():
    ctl, backend, clock = make_controller()
    a = ctl.acquire(
        'alice', [vreq('qwen', reclaim='stop')], ttl_seconds=10, wait=False
    )
    gid = a.deployments[0].id
    assert backend.observe() == {gid}
    clock.advance(11)
    rec = ctl.reconcile()                                # sweep -> idle -> stop
    assert gid in rec.torn_down
    assert backend.observe() == set()


def test_ttl_expiry_keepwarm_survives():
    ctl, backend, clock = make_controller()
    a = ctl.acquire(
        'alice', [vreq('qwen', reclaim='keep-warm')], ttl_seconds=10, wait=False
    )
    gid = a.deployments[0].id
    clock.advance(11)
    rec = ctl.reconcile()
    assert rec.torn_down == []
    assert backend.observe() == {gid}                   # keep-warm survives TTL


def test_wait_scoped_to_requested_endpoints():
    ctl, backend, _ = make_controller(ready=True)
    a = ctl.acquire('alice', [oreq('qwen-small', 'qwen3.5:4b')])
    gid = a.deployments[0].id
    # one tag on the daemon is unhealthy, but the next caller only needs the other
    backend.set_ready(False, deployment_id=gid, endpoint='qwen-small')
    b = ctl.acquire(
        'bob', [oreq('smollm', 'smollm2:135m')], timeout=10, interval=2
    )
    assert b.deployments[0].id == gid                         # same daemon
    assert b.wait.ready is True                          # only waited for smollm


CATALOG = {
    'models': {'m': {'source': 'hf://org/m'}},
    'endpoints': {
        'e1': {'engine': 'vllm', 'model': 'm', 'runtime': {'max_model_len': 8192}},
        'e2': {
            'engine': 'vllm',
            'model': 'm',
            'runtime': {'max_model_len': 8192},
            'public_name': 'e1',
        },
    },
    'bundles': {'both': ['e1', 'e2']},
}


def test_controller_with_catalog():
    catalog = Catalog.from_dict(CATALOG)
    ctl, backend, _ = make_controller(ready=True)
    out = ctl.acquire('alice', catalog.resolve_names(['both']))
    # e1/e2 are the same deployment identity -> one realized deployment, both served
    assert len(out.deployments) == 1
    assert backend.realize_calls == [out.deployments[0].id]
    assert set(out.deployments[0].served) == {'e1', 'e2'}
    assert out.wait.ready is True
