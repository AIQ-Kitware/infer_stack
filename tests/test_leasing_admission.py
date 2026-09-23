"""Admission (plan steps P5, P6, P9): demand gets GPUs, previews commit nothing.

Compose backend over a fake Docker whose ``up`` "runs" every rendered service
and which answers strict residency from the same state.
"""

from __future__ import annotations

import threading

import pytest
import yaml

from infer_stack.hardware import simulate_inventory
from infer_stack.leasing import Catalog, Controller, Ledger, SqliteStore
from infer_stack.leasing.backend import PlacementError
from infer_stack.leasing.compose import ComposeBackend
from infer_stack.leasing.models import DeploymentState, LeaseState
from test_leasing_compose import IMAGES, PORTS, STATE, FakeDocker, FakeHttp


def catalog(**tps):
    return Catalog.from_dict({
        'models': {n: {'source': f'hf://org/{n}'} for n in tps},
        'endpoints': {n: {'engine': 'vllm', 'model': n,
                          'runtime': {'tensor_parallel_size': tp}}
                      for n, tp in tps.items()},
    })


CAT = catalog(big=2, big2=2, one=1, two=1)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def make(tmp_path, *, gpus='2x80', docker=None, clock=None, backend_cls=ComposeBackend,
         catalog_obj=CAT):
    state = tmp_path / 'state'
    docker = docker or FakeDocker()
    backend = backend_cls(state_dir=state, inventory=simulate_inventory(gpus), run=docker,
                          http=FakeHttp(state), images=IMAGES, ports=PORTS, state=STATE,
                          catalog=catalog_obj, litellm=False, ui=False)
    clock = clock or Clock()
    ledger = Ledger(SqliteStore(str(tmp_path / 'ledger.db')), clock=clock)
    return ledger, Controller(ledger, backend, clock=clock, sleep=clock.sleep), docker


def services(ctl):
    return set((yaml.safe_load(ctl.backend.compose_file.read_text()) or {}).get('services') or {})


def acquire(ctl, *names, **kw):
    kw.setdefault('wait', False)
    return ctl.acquire('owner', CAT.resolve_names(list(names)), **kw)


# -- P9: the incident ---------------------------------------------------------------


def test_warm_idle_resident_yields_to_new_demand_without_queueing(tmp_path):   # 28
    ledger, ctl, docker = make(tmp_path)
    a = acquire(ctl, 'big')
    ctl.release(a.lease.id)                                   # keep-warm: still running
    assert len(docker.running) == 1
    b = acquire(ctl, 'big2')                                  # not queued, still admitted
    gid = b.deployments[0].id
    assert ledger.get_deployment(gid).assigned_gpus == [0, 1]
    assert b.reconcile.displaced == [a.deployments[0].id]
    assert len(services(ctl)) == 1 and ledger.get_deployment(a.deployments[0].id).state == 'idle'


def test_idle_keep_warm_without_a_container_is_never_started(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    a = acquire(ctl, 'one')
    ctl.release(a.lease.id)
    docker.running = []                                       # the container went away
    ctl.gc()
    assert services(ctl) == set() and docker.running == []


# -- P6: previews commit nothing ------------------------------------------------------


def test_queued_request_blocked_by_admitted_demand_commits_nothing(tmp_path):      # 29
    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'big')
    with pytest.raises(PlacementError):
        acquire(ctl, 'big2', wait_for_placement=True, placement_timeout=10,
                placement_interval=2)
    leases, deployments = ledger.status()
    assert len(leases) == 1 and len(deployments) == 1


def test_an_unrelated_request_that_fits_is_admitted_while_another_cannot(tmp_path):  # 30
    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'one')
    with pytest.raises(PlacementError):
        acquire(ctl, 'big')
    b = acquire(ctl, 'two')
    assert ledger.get_deployment(b.deployments[0].id).assigned_gpus == [1]


def test_two_processes_contend_for_the_last_gpu_and_exactly_one_wins(tmp_path):    # 31
    ledger, ctl, docker = make(tmp_path)
    acquire(ctl, 'one')
    results, barrier = [], threading.Barrier(2)

    def worker(name):
        _, other, _ = make(tmp_path, docker=docker)
        barrier.wait(10)
        try:
            acquire(other, name)
            results.append('ok')
        except PlacementError:
            results.append('refused')

    threads = [threading.Thread(target=worker, args=(n,)) for n in ('two', 'big')]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(results) == ['ok', 'refused']
    assert len([le for le in ledger.status()[0] if le.state == LeaseState.ACTIVE]) == 2


def test_a_placed_but_unrenderable_candidate_never_creates_a_lease(tmp_path):     # 32
    class Unrenderable(ComposeBackend):
        def preview(self, desired, placement=None, **kw):
            plan, rendered = super().preview(desired, placement, **kw)
            rendered.unrenderable = [g.id for g in desired]
            rendered.errors = ['service name collision']
            return plan, rendered

    ledger, ctl, _ = make(tmp_path, backend_cls=Unrenderable)
    with pytest.raises(PlacementError, match='collision'):
        acquire(ctl, 'one')
    assert ledger.status() == ([], [])


def test_approval_does_not_hold_the_sqlite_write_lock(tmp_path):                     # 33
    ledger, ctl, _ = make(tmp_path)
    first = acquire(ctl, 'one')
    renewed = []

    def approve(planned):
        other = Ledger(SqliteStore(str(tmp_path / 'ledger.db')))
        renewed.append(other.renew_if_live(first.lease.id, ttl_seconds=60))

    ctl.backend._approve_changes = approve
    acquire(ctl, 'two')
    assert renewed and renewed[0] not in (None, False)


def test_a_ledger_change_between_preview_and_commit_retries(tmp_path):              # 34
    calls = []

    class Racy(ComposeBackend):
        def preview(self, desired, placement=None, **kw):
            if kw.get('approve'):
                return super().preview(desired, placement, **kw)
            calls.append(1)
            if len(calls) == 1:
                store = SqliteStore(str(tmp_path / 'ledger.db'))
                with store.transaction():
                    store.bump_admission_state_version()
            return super().preview(desired, placement)

    ledger, ctl, _ = make(tmp_path, backend_cls=Racy)
    out = acquire(ctl, 'one')
    assert len(calls) == 2 and ledger.get_deployment(out.deployments[0].id).assigned_gpus == [0]


def test_unknown_residency_admits_only_resource_neutral_requests(tmp_path):         # 35
    from infer_stack.leasing.residency import ResidencyUnknown

    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'one')

    def unknown():
        raise ResidencyUnknown('docker ps failed')

    ctl.backend.residency = unknown
    with pytest.raises(PlacementError, match='residency is unknown'):
        ctl.acquire('x', CAT.resolve_names(['two']), wait=False, apply=False)
    # Coalescing onto the LIVE, allocated deployment needs no new GPU. Its render
    # still needs residency, so stage it without rendering residency-dependent state.
    overlay = ledger.plan_acquire(CAT.resolve_names(['one']))
    assert ctl._admit(overlay, None) == ({}, [])


# -- P5: allocations and renew ----------------------------------------------------------


def test_release_clears_the_allocation_and_reuse_adopts_resident_gpus(tmp_path):
    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'two')
    a = acquire(ctl, 'one')
    gid = a.deployments[0].id
    gpus = ledger.get_deployment(gid).assigned_gpus
    ctl.release(a.lease.id)
    assert ledger.get_deployment(gid).assigned_gpus is None
    b = acquire(ctl, 'one')                                   # IDLE -> LIVE reuse
    assert b.deployments[0].id == gid
    assert ledger.get_deployment(gid).assigned_gpus == gpus


def test_renew_fast_path_is_lock_free(tmp_path):                                      # 36
    import fcntl

    ledger, ctl, _ = make(tmp_path)
    a = acquire(ctl, 'one', ttl_seconds=100)
    handle = ctl._open_flock(ctl._lock_path)
    fcntl.flock(handle, fcntl.LOCK_EX)          # another process holds the lock
    try:
        out = ctl.renew(a.lease.id, ttl_seconds=500)
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    assert out.lease.expires_at == ledger.clock() + 500 and out.reconcile is None


def test_renew_slow_path_readmits_and_loses_to_an_expired_lease(tmp_path):
    clock = Clock()
    ledger, ctl, docker = make(tmp_path, clock=clock)
    a = acquire(ctl, 'one', ttl_seconds=100)
    b = ledger.acquire('bob', CAT.resolve_names(['one']))    # shares the deployment
    clock.now += 200                                           # a lapses, unswept
    ledger.release(b.lease.id)                                 # deployment idles
    gid = a.deployments[0].id
    assert ledger.get_deployment(gid).state == DeploymentState.IDLE
    out = ctl.renew(a.lease.id, ttl_seconds=3600)
    assert out.revived_deployment_ids == [gid]
    assert ledger.get_deployment(gid).assigned_gpus == [0]     # adopted from residency

    ledger.sweep()                                             # nothing lapsed now
    c = acquire(ctl, 'two', ttl_seconds=10)
    clock.now += 100
    ctl.gc()                                                   # c expires
    assert ctl.renew(c.lease.id, ttl_seconds=60).lease is None


# -- migration backfill (tests 53-55) ---------------------------------------------------


def _legacy_row(ledger, name, **kw):
    """A LIVE deployment committed by pre-allocation code: no assigned_gpus."""
    return ledger.acquire('legacy', CAT.resolve_names([name]), **kw)


def test_upgrade_adopts_the_unique_running_container_of_a_legacy_deployment(tmp_path):
    from infer_stack.leasing.residency import DEPLOYMENT_LABEL

    ledger, ctl, docker = make(tmp_path, gpus='4x80')
    old = _legacy_row(ledger, 'one')
    gid = old.deployments[0].id
    # A container from before ownership labels: Compose service label only.
    docker.add_container('vllm-one', labels={DEPLOYMENT_LABEL: gid}, device_ids=[2])
    ctl.gc()
    assert ledger.get_deployment(gid).assigned_gpus == [2]
    assert list(ledger.adopted_containers()) == [c for c in docker.containers
                                                 if docker.containers[c]['service'] == 'vllm-one']
    assert len(docker.containers) == 1                        # adoption recreated nothing
    acquire(ctl, 'two')                                       # nothing unresolved


@pytest.mark.parametrize('containers', [0, 2])
def test_unresolved_legacy_deployment_is_never_freshly_placed(tmp_path, containers):
    from infer_stack.leasing.residency import DEPLOYMENT_LABEL

    ledger, ctl, docker = make(tmp_path, gpus='4x80')
    old = _legacy_row(ledger, 'one')
    gid = old.deployments[0].id
    for i in range(containers):                               # none, or ambiguous duplicates
        docker.add_container(f'dup{i}', labels={DEPLOYMENT_LABEL: gid}, device_ids=[3])
    ctl.gc()
    assert ledger.get_deployment(gid).assigned_gpus is None
    assert gid not in (ctl.backend._load_sidecar().get('assignments') or {})
    with pytest.raises(PlacementError, match='no committed allocation'):
        acquire(ctl, 'two')
    ctl.release(old.lease.id)                                 # release resolves
    acquire(ctl, 'two')


def test_reservation_from_before_allocations_blocks_until_released(tmp_path):
    from infer_stack.leasing.models import reservation_request

    ledger, ctl, _ = make(tmp_path, gpus='4x80')
    reserved = ledger.acquire('legacy', [reservation_request(1)])
    ctl.gc()
    assert ledger.get_deployment(reserved.deployments[0].id).assigned_gpus is None
    with pytest.raises(PlacementError, match='no committed allocation'):
        acquire(ctl, 'two')
    ctl.release(reserved.lease.id)
    acquire(ctl, 'two')


def test_declined_approval_of_a_coalescing_acquire_changes_nothing(tmp_path):
    from infer_stack.leasing.backend import ConvergeAborted

    cat = Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/m'}},
        # Same model and public name: `b` coalesces onto `a`'s deployment.
        'endpoints': {'a': {'engine': 'vllm', 'model': 'm', 'public_name': 'shared'},
                      'b': {'engine': 'vllm', 'model': 'm', 'public_name': 'shared'}},
    })
    ledger, ctl, _ = make(tmp_path)
    ctl.backend.catalog = None
    first = ctl.acquire('x', cat.resolve_names(['a']), wait=False)
    gid = first.deployments[0].id
    served = dict(ledger.get_deployment(gid).served)
    registry = ctl.backend._registry_file.read_text() if ctl.backend._registry_file.exists() else None

    def decline(planned):
        raise ConvergeAborted('no')

    ctl.backend._approve_changes = decline
    with pytest.raises(ConvergeAborted):
        ctl.acquire('y', cat.resolve_names(['b']), wait=False)    # would add alias b
    assert len(ledger.status()[0]) == 1
    assert ledger.get_deployment(gid).served == served
    now = ctl.backend._registry_file.read_text() if ctl.backend._registry_file.exists() else None
    assert now == registry


def test_a_coalescing_candidate_that_makes_its_deployment_unrenderable_is_refused(tmp_path):
    class Breaks(ComposeBackend):
        def preview(self, desired, placement=None, **kw):
            plan, rendered = super().preview(desired, placement, **kw)
            live = [g.id for g in desired if len(g.served) > 1]
            if live:
                rendered.unrenderable = set(live)
                rendered.errors = ['alias collision']
            return plan, rendered

    cat = Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/m'}},
        # Same model and public name: `b` coalesces onto `a`'s deployment.
        'endpoints': {'a': {'engine': 'vllm', 'model': 'm', 'public_name': 'shared'},
                      'b': {'engine': 'vllm', 'model': 'm', 'public_name': 'shared'}},
    })
    ledger, ctl, _ = make(tmp_path, backend_cls=Breaks)
    ctl.backend.catalog = None
    first = ctl.acquire('x', cat.resolve_names(['a']), wait=False)
    before = ledger.get_deployment(first.deployments[0].id).served
    with pytest.raises(PlacementError, match='alias collision'):
        ctl.acquire('y', cat.resolve_names(['b']), wait=False)
    assert ledger.get_deployment(first.deployments[0].id).served == before
    assert len(ledger.status()[0]) == 1


def test_a_render_failure_after_commit_leaves_no_active_lease(tmp_path):
    from infer_stack.leasing.residency import ResidencyUnknown

    ledger, ctl, _ = make(tmp_path)
    real = ctl.backend.residency
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) > 1:                    # fine for the preview, gone for the render
            raise ResidencyUnknown('docker ps failed')
        return real()

    ctl.backend.residency = flaky
    with pytest.raises(ResidencyUnknown):
        acquire(ctl, 'one')
    assert [le.state for le in ledger.status()[0]] == [LeaseState.RELEASED]


def test_a_new_lease_cannot_coalesce_onto_an_unresolved_legacy_deployment(tmp_path):
    ledger, ctl, _ = make(tmp_path, gpus='4x80')
    old = _legacy_row(ledger, 'one')                          # no allocation, no container
    ctl.gc()
    with pytest.raises(PlacementError, match='unresolved pre-allocation'):
        acquire(ctl, 'one')                                   # would coalesce onto it
    assert len(ledger.status()[0]) == 1
    # Renewing the existing legacy lease is still allowed.
    assert ctl.renew(old.lease.id, ttl_seconds=60).lease is not None


def test_the_admission_approval_digest_is_committed_with_the_lease(tmp_path):
    ledger, ctl, _ = make(tmp_path)
    seen = {}
    real = ledger.acquire

    def spy(*args, **kw):
        assert (ledger.publication_pending() or {}).get('approved_digest') is None
        out = real(*args, **kw)
        seen['marker'] = ledger.publication_pending()     # same transaction as the lease
        return out

    ledger.acquire = spy
    acquire(ctl, 'one')
    assert seen['marker']['approved_digest']
    assert seen['marker']['approved_digest'] == ctl.backend.last_planned_digest
    assert ledger.publication_pending() is None


def test_custom_launch_maps_do_not_change_approval_digest_after_commit(tmp_path):
    """Fresh YAML order and sqlite-sorted order describe one approved render."""
    cat = Catalog.from_dict({
        'models': {'m': {'source': 'hf://org/m'}},
        'endpoints': {'m': {'engine': 'vllm', 'model': 'm', 'runtime': {
            'command': ['single'],
            'env': {
                'PORT': '{port}',
                'SPEC': 'dflash2',
                'CTX': 'huge',
                'PREFIX_CACHE': 1,
                'MAX_LEN': '{max_model_len}',
                'GPU_UTIL': '{gpu_memory_utilization}',
                'EXTRA_ARGS': '--served-model-name={served_model_name}',
            },
            'mounts': {
                '/cache': 'model/cache',
                '/app/models': 'model/weights',
            },
            'max_model_len': 245760,
            'gpu_memory_utilization': 0.93,
        }}},
    })
    ledger, ctl, _ = make(tmp_path, gpus='1x24', catalog_obj=cat)

    out = ctl.acquire('owner', cat.resolve_names(['m']), wait=False)

    assert out.lease.state == LeaseState.ACTIVE
    assert len(ledger.status()[0]) == 1
    assert ledger.publication_pending() is None


def test_a_staged_acquire_records_no_digest_and_release_discards_it(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    staged = acquire(ctl, 'one', apply=False)
    assert ledger.publication_pending()['approved_digest'] is None
    ctl.release_leases([staged.lease.id], evict=True)          # must not hit a digest mismatch
    assert ledger.publication_pending() is None and docker.containers == {}


def test_a_failed_apply_rollback_drops_the_admission_digest(tmp_path):
    from infer_stack.leasing.backend import BackendTimeout

    ledger, ctl, docker = make(tmp_path)
    real = ctl.backend.apply
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise BackendTimeout('up timed out')
        return real()

    ctl.backend.apply = flaky
    ctl.backend.settle_snapshot = lambda: ()
    with pytest.raises(BackendTimeout):
        acquire(ctl, 'one')
    assert ledger.publication_pending()['approved_digest'] is None
    acquire(ctl, 'two')                                          # an ordinary publisher proceeds
    assert ledger.publication_pending() is None
