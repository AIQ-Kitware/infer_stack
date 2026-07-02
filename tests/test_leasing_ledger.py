"""Tests for the backend-agnostic lease ledger (Phase 1 core)."""

from __future__ import annotations

import pytest

from infer_stack.leasing import (
    DeploymentState,
    EndpointRequest,
    LeaseState,
    Ledger,
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


@pytest.fixture
def ledger():
    return Ledger(
        SqliteStore(':memory:'),
        clock=FakeClock(),
        id_factory=_id_factory(),
    )


def vllm_req(
    endpoint,
    *,
    model_ref=None,
    tp=1,
    max_model_len=32768,
    sharing=Sharing.SHARED,
):
    model_ref = model_ref or endpoint
    return EndpointRequest(
        endpoint=endpoint,
        engine='vllm',
        structural=vllm_structural(
            model_ref=model_ref, tensor_parallel_size=tp
        ),
        capacity={'max_model_len': max_model_len} if max_model_len else {},
        sharing=sharing,
        spec={'hf_model_id': model_ref},
    )


def ollama_req(endpoint, *, tag, host='local-ollama'):
    return EndpointRequest(
        endpoint=endpoint,
        engine='ollama',
        structural=ollama_structural(host=host),
        spec={'host': host},
        served={'model': tag},
        host=host,
    )


def test_acquire_creates_live_deployment(ledger):
    res = ledger.acquire('alice', [vllm_req('qwen-coder')])
    assert len(res.deployments) == 1
    deployment = ledger.get_deployment(res.deployments[0].id)
    assert deployment.state == DeploymentState.LIVE
    assert deployment.demand == 1
    assert res.lease.endpoints == ['qwen-coder']


def test_same_model_coalesces(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    b = ledger.acquire('bob', [vllm_req('qwen-coder')])
    assert a.deployments[0].id == b.deployments[0].id
    assert ledger.get_deployment(a.deployments[0].id).demand == 2


def test_release_decrements_then_idles(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    b = ledger.acquire('bob', [vllm_req('qwen-coder')])
    gid = a.deployments[0].id

    r1 = ledger.release(a.lease.id)
    assert r1.idled_deployment_ids == []          # bob still protects it
    assert ledger.get_deployment(gid).demand == 1

    r2 = ledger.release(b.lease.id)
    assert r2.idled_deployment_ids == [gid]
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE
    assert ledger.get_deployment(gid).demand == 0


def test_double_release_is_noop(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    ledger.release(a.lease.id)
    assert ledger.release(a.lease.id).idled_deployment_ids == []


def test_dedicated_does_not_coalesce(ledger):
    shared = ledger.acquire('alice', [vllm_req('qwen-coder')])
    dedi = ledger.acquire(
        'bob', [vllm_req('qwen-coder', sharing=Sharing.DEDICATED)]
    )
    assert shared.deployments[0].id != dedi.deployments[0].id
    assert ledger.get_deployment(dedi.deployments[0].id).sharing == Sharing.DEDICATED
    # both still demand 1 each
    assert ledger.get_deployment(shared.deployments[0].id).demand == 1
    assert ledger.get_deployment(dedi.deployments[0].id).demand == 1


def test_capacity_subsumption(ledger):
    big = ledger.acquire('alice', [vllm_req('qwen-coder', max_model_len=32768)])
    # smaller request fits the bigger deployment -> coalesces
    small = ledger.acquire(
        'bob', [vllm_req('qwen-coder', max_model_len=8192)]
    )
    assert small.deployments[0].id == big.deployments[0].id
    assert ledger.get_deployment(big.deployments[0].id).demand == 2


def test_capacity_insufficient_makes_new_deployment(ledger):
    small = ledger.acquire(
        'alice', [vllm_req('qwen-coder', max_model_len=8192)]
    )
    # bigger request cannot be served by the smaller deployment -> new deployment
    big = ledger.acquire(
        'bob', [vllm_req('qwen-coder', max_model_len=32768)]
    )
    assert big.deployments[0].id != small.deployments[0].id


def test_structural_mismatch_separate_deployments(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder', tp=1)])
    b = ledger.acquire('bob', [vllm_req('qwen-coder', tp=2)])
    assert a.deployments[0].id != b.deployments[0].id


def test_ollama_coalesces_per_daemon(ledger):
    a = ledger.acquire('alice', [ollama_req('qwen-small', tag='qwen3.5:4b')])
    b = ledger.acquire('bob', [ollama_req('smollm', tag='smollm2:135m')])
    # different tags, same daemon config -> one deployment serving both endpoints
    assert a.deployments[0].id == b.deployments[0].id
    deployment = ledger.get_deployment(a.deployments[0].id)
    assert deployment.demand == 2
    assert set(deployment.served) == {'qwen-small', 'smollm'}


def test_ollama_different_host_separate_deployment(ledger):
    a = ledger.acquire('alice', [ollama_req('qwen-small', tag='q', host='h1')])
    b = ledger.acquire('bob', [ollama_req('qwen-small', tag='q', host='h2')])
    assert a.deployments[0].id != b.deployments[0].id


def test_ttl_expiry_stops_protecting(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')], ttl_seconds=3600)
    gid = a.deployments[0].id
    assert ledger.get_deployment(gid).demand == 1

    ledger.clock.advance(3601)               # past the TTL
    # protection lapses immediately for demand purposes...
    assert ledger.get_deployment(gid).demand == 0
    # ...and sweep() materializes the EXPIRED state + idles the deployment
    res = ledger.sweep()
    assert a.lease.id in res.expired_lease_ids
    assert gid in res.idled_deployment_ids
    assert ledger.get_lease(a.lease.id).state == LeaseState.EXPIRED
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE


def test_renew_extends_protection(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')], ttl_seconds=3600)
    ledger.clock.advance(3000)
    ledger.renew(a.lease.id, ttl_seconds=3600)
    ledger.clock.advance(1000)               # would have expired without renew
    assert ledger.get_deployment(a.deployments[0].id).demand == 1
    assert ledger.sweep().expired_lease_ids == []


def test_idle_deployment_is_reused_and_relit(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    gid = a.deployments[0].id
    ledger.release(a.lease.id)
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE

    b = ledger.acquire('bob', [vllm_req('qwen-coder')])
    assert b.deployments[0].id == gid             # reused, not recreated
    assert ledger.get_deployment(gid).state == DeploymentState.LIVE
    assert ledger.get_deployment(gid).demand == 1


def test_single_lease_multiple_endpoints(ledger):
    res = ledger.acquire(
        'alice', [vllm_req('qwen-coder'), vllm_req('reranker')]
    )
    assert len(res.deployments) == 2
    assert sorted(res.lease.endpoints) == ['qwen-coder', 'reranker']
    # one lease protects both -> each deployment demand 1
    for g in res.deployments:
        assert ledger.get_deployment(g.id).demand == 1


def test_reclaimable_deployments(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    assert ledger.reclaimable_deployments() == []
    ledger.release(a.lease.id)
    reclaimable = ledger.reclaimable_deployments()
    assert [g.id for g in reclaimable] == [a.deployments[0].id]


def test_persistence_across_reopen(tmp_path):
    db = tmp_path / 'ledger.db'
    clock = FakeClock()
    led = Ledger(SqliteStore(db), clock=clock, id_factory=_id_factory())
    a = led.acquire('alice', [vllm_req('qwen-coder')])
    led.store.close()

    led2 = Ledger(SqliteStore(db), clock=clock, id_factory=_id_factory())
    lease = led2.get_lease(a.lease.id)
    assert lease is not None and lease.owner == 'alice'
    deployment = led2.get_deployment(a.deployments[0].id)
    assert deployment is not None and deployment.demand == 1


def test_status_snapshot(ledger):
    ledger.acquire('alice', [vllm_req('qwen-coder')])
    ledger.acquire('bob', [ollama_req('smollm', tag='smollm2:135m')])
    leases, deployments = ledger.status()
    assert len(leases) == 2
    assert len(deployments) == 2
    assert {g.engine for g in deployments} == {'vllm', 'ollama'}


def test_renew_refuses_released_and_expired_leases(ledger):
    """Regression: renew must not resurrect a RELEASED/EXPIRED lease — its
    deployments may already be idled/evicted behind it, so a silently re-ACTIVATED
    lease would "protect" nothing that runs; the caller must re-acquire."""
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    gid = a.deployments[0].id
    ledger.release(a.lease.id)
    assert ledger.renew(a.lease.id, ttl_seconds=3600) is None
    assert ledger.get_lease(a.lease.id).state == LeaseState.RELEASED
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE

    b = ledger.acquire('bob', [vllm_req('qwen-coder')], ttl_seconds=100)
    ledger.clock.advance(200)
    ledger.sweep()
    assert ledger.get_lease(b.lease.id).state == LeaseState.EXPIRED
    assert ledger.renew(b.lease.id, ttl_seconds=3600) is None
    assert ledger.get_lease(b.lease.id).state == LeaseState.EXPIRED

    assert ledger.renew('lease-nope', ttl_seconds=3600) is None


def test_renew_revives_idled_deployment_of_active_lease(ledger):
    """An ACTIVE lease whose TTL lapsed unswept can find its deployment idled
    (the other demand vanished while it wasn't protecting). Renewing re-LIVEs
    the deployment and bumps the desired generation so the next apply re-ups it."""
    a = ledger.acquire('alice', [vllm_req('qwen-coder')], ttl_seconds=100)
    b = ledger.acquire('bob', [vllm_req('qwen-coder')])  # infinite, shares gid
    gid = a.deployments[0].id
    ledger.clock.advance(200)      # alice lapses; unswept, so still ACTIVE
    ledger.release(b.lease.id)     # demand hits 0 -> deployment idles
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE
    assert ledger.get_lease(a.lease.id).state == LeaseState.ACTIVE

    g_before = ledger.desired_generation()
    lease = ledger.renew(a.lease.id, ttl_seconds=3600)
    assert lease is not None and lease.state == LeaseState.ACTIVE
    assert ledger.get_deployment(gid).state == DeploymentState.LIVE
    assert ledger.get_deployment(gid).demand == 1
    assert ledger.desired_generation() > g_before  # an apply must re-up it


def test_release_reports_found_and_idempotent(ledger):
    """Regression: a missing lease id must be distinguishable from a release
    that idled nothing, so the CLI can't report success for a typo'd id."""
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    assert ledger.release('lease-nope').found is False
    first = ledger.release(a.lease.id)
    assert first.found and not first.already_released
    again = ledger.release(a.lease.id)  # idempotent (cleanup traps fire twice)
    assert again.found and again.already_released
    assert again.idled_deployment_ids == []


def test_evict_idle_skips_deployment_with_demand(ledger):
    """An IDLE deployment that still has protecting demand (an anomaly, e.g. a
    heartbeat landing in the idle->reclaim window) must never be stopped."""
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    gid = a.deployments[0].id
    # Force the anomalous state directly: IDLE while alice still protects it.
    with ledger.store.transaction():
        ledger.store.set_deployment_state(
            gid, DeploymentState.IDLE, ledger.clock()
        )
    assert ledger.get_deployment(gid).demand == 1
    assert ledger.evict_idle(None) == []
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE
