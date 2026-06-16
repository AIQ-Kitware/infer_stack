"""Tests for the backend-agnostic lease ledger (Phase 1 core)."""

from __future__ import annotations

import pytest

from infer_stack.leasing import (
    EndpointRequest,
    GroupState,
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


def test_acquire_creates_live_group(ledger):
    res = ledger.acquire('alice', [vllm_req('qwen-coder')])
    assert len(res.groups) == 1
    group = ledger.get_group(res.groups[0].id)
    assert group.state == GroupState.LIVE
    assert group.demand == 1
    assert res.lease.endpoints == ['qwen-coder']


def test_same_model_coalesces(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    b = ledger.acquire('bob', [vllm_req('qwen-coder')])
    assert a.groups[0].id == b.groups[0].id
    assert ledger.get_group(a.groups[0].id).demand == 2


def test_release_decrements_then_idles(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    b = ledger.acquire('bob', [vllm_req('qwen-coder')])
    gid = a.groups[0].id

    r1 = ledger.release(a.lease.id)
    assert r1.idled_group_ids == []          # bob still protects it
    assert ledger.get_group(gid).demand == 1

    r2 = ledger.release(b.lease.id)
    assert r2.idled_group_ids == [gid]
    assert ledger.get_group(gid).state == GroupState.IDLE
    assert ledger.get_group(gid).demand == 0


def test_double_release_is_noop(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    ledger.release(a.lease.id)
    assert ledger.release(a.lease.id).idled_group_ids == []


def test_dedicated_does_not_coalesce(ledger):
    shared = ledger.acquire('alice', [vllm_req('qwen-coder')])
    dedi = ledger.acquire(
        'bob', [vllm_req('qwen-coder', sharing=Sharing.DEDICATED)]
    )
    assert shared.groups[0].id != dedi.groups[0].id
    assert ledger.get_group(dedi.groups[0].id).sharing == Sharing.DEDICATED
    # both still demand 1 each
    assert ledger.get_group(shared.groups[0].id).demand == 1
    assert ledger.get_group(dedi.groups[0].id).demand == 1


def test_capacity_subsumption(ledger):
    big = ledger.acquire('alice', [vllm_req('qwen-coder', max_model_len=32768)])
    # smaller request fits the bigger deployment -> coalesces
    small = ledger.acquire(
        'bob', [vllm_req('qwen-coder', max_model_len=8192)]
    )
    assert small.groups[0].id == big.groups[0].id
    assert ledger.get_group(big.groups[0].id).demand == 2


def test_capacity_insufficient_makes_new_group(ledger):
    small = ledger.acquire(
        'alice', [vllm_req('qwen-coder', max_model_len=8192)]
    )
    # bigger request cannot be served by the smaller deployment -> new group
    big = ledger.acquire(
        'bob', [vllm_req('qwen-coder', max_model_len=32768)]
    )
    assert big.groups[0].id != small.groups[0].id


def test_structural_mismatch_separate_groups(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder', tp=1)])
    b = ledger.acquire('bob', [vllm_req('qwen-coder', tp=2)])
    assert a.groups[0].id != b.groups[0].id


def test_ollama_coalesces_per_daemon(ledger):
    a = ledger.acquire('alice', [ollama_req('qwen-small', tag='qwen3.5:4b')])
    b = ledger.acquire('bob', [ollama_req('smollm', tag='smollm2:135m')])
    # different tags, same daemon config -> one group serving both endpoints
    assert a.groups[0].id == b.groups[0].id
    group = ledger.get_group(a.groups[0].id)
    assert group.demand == 2
    assert set(group.served) == {'qwen-small', 'smollm'}


def test_ollama_different_host_separate_group(ledger):
    a = ledger.acquire('alice', [ollama_req('qwen-small', tag='q', host='h1')])
    b = ledger.acquire('bob', [ollama_req('qwen-small', tag='q', host='h2')])
    assert a.groups[0].id != b.groups[0].id


def test_ttl_expiry_stops_protecting(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')], ttl_seconds=3600)
    gid = a.groups[0].id
    assert ledger.get_group(gid).demand == 1

    ledger.clock.advance(3601)               # past the TTL
    # protection lapses immediately for demand purposes...
    assert ledger.get_group(gid).demand == 0
    # ...and sweep() materializes the EXPIRED state + idles the group
    res = ledger.sweep()
    assert a.lease.id in res.expired_lease_ids
    assert gid in res.idled_group_ids
    assert ledger.get_lease(a.lease.id).state == LeaseState.EXPIRED
    assert ledger.get_group(gid).state == GroupState.IDLE


def test_renew_extends_protection(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')], ttl_seconds=3600)
    ledger.clock.advance(3000)
    ledger.renew(a.lease.id, ttl_seconds=3600)
    ledger.clock.advance(1000)               # would have expired without renew
    assert ledger.get_group(a.groups[0].id).demand == 1
    assert ledger.sweep().expired_lease_ids == []


def test_idle_group_is_reused_and_relit(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    gid = a.groups[0].id
    ledger.release(a.lease.id)
    assert ledger.get_group(gid).state == GroupState.IDLE

    b = ledger.acquire('bob', [vllm_req('qwen-coder')])
    assert b.groups[0].id == gid             # reused, not recreated
    assert ledger.get_group(gid).state == GroupState.LIVE
    assert ledger.get_group(gid).demand == 1


def test_single_lease_multiple_endpoints(ledger):
    res = ledger.acquire(
        'alice', [vllm_req('qwen-coder'), vllm_req('reranker')]
    )
    assert len(res.groups) == 2
    assert sorted(res.lease.endpoints) == ['qwen-coder', 'reranker']
    # one lease protects both -> each group demand 1
    for g in res.groups:
        assert ledger.get_group(g.id).demand == 1


def test_reclaimable_groups(ledger):
    a = ledger.acquire('alice', [vllm_req('qwen-coder')])
    assert ledger.reclaimable_groups() == []
    ledger.release(a.lease.id)
    reclaimable = ledger.reclaimable_groups()
    assert [g.id for g in reclaimable] == [a.groups[0].id]


def test_persistence_across_reopen(tmp_path):
    db = tmp_path / 'ledger.db'
    clock = FakeClock()
    led = Ledger(SqliteStore(db), clock=clock, id_factory=_id_factory())
    a = led.acquire('alice', [vllm_req('qwen-coder')])
    led.store.close()

    led2 = Ledger(SqliteStore(db), clock=clock, id_factory=_id_factory())
    lease = led2.get_lease(a.lease.id)
    assert lease is not None and lease.owner == 'alice'
    group = led2.get_group(a.groups[0].id)
    assert group is not None and group.demand == 1


def test_status_snapshot(ledger):
    ledger.acquire('alice', [vllm_req('qwen-coder')])
    ledger.acquire('bob', [ollama_req('smollm', tag='smollm2:135m')])
    leases, groups = ledger.status()
    assert len(leases) == 2
    assert len(groups) == 2
    assert {g.engine for g in groups} == {'vllm', 'ollama'}
