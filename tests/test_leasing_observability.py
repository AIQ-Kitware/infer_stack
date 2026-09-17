"""Degraded end to end and the health view (plan step P10)."""

from __future__ import annotations

import sqlite3

import pytest

from infer_stack.leasing.backend import PlacementError
from test_leasing_admission import acquire, make


def conditions(ctl):
    return {r['id']: r['condition'] for r in ctl.observe_state()['deployments']}


def test_degraded_is_reported_kept_and_unblocks_on_release(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    a = acquire(ctl, 'one')
    gid = a.deployments[0].id
    ledger.set_allocation(gid, [7])                # its GPU disappeared
    ctl.gc()
    assert conditions(ctl)[gid] == 'degraded'
    assert len(docker.containers) == 1             # neither removed nor restarted
    with pytest.raises(PlacementError):
        acquire(ctl, 'one')                        # coalescing onto it is refused
    ctl.release_leases([a.lease.id], evict=True)
    assert docker.containers == {}                 # released: now removable


def test_displaced_orphans_unknown_and_pending_are_reported(tmp_path):
    ledger, ctl, docker = make(tmp_path)
    a = acquire(ctl, 'big')
    ctl.release(a.lease.id)
    acquire(ctl, 'big2')
    docker.add_container('stray', labels={})
    state = ctl.observe_state()
    assert conditions(ctl)[a.deployments[0].id] == 'displaced'
    assert [o['service'] for o in state['orphans']] == ['stray']
    ledger.mark_publication_pending(apply_requested=False)
    assert ctl.observe_state()['publication_pending']['apply_requested'] is False

    from infer_stack.leasing.residency import ResidencyUnknown

    def unknown():
        raise ResidencyUnknown('docker ps failed')

    ctl.backend.residency = unknown
    state = ctl.observe_state()
    assert state['residency_error'] and set(conditions(ctl).values()) == {'unknown'}


def test_the_health_view_never_writes(tmp_path):
    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'one', ttl_seconds=1)
    ledger.clock.now += 100                        # expired, unswept
    watcher = sqlite3.connect(str(tmp_path / 'ledger.db'))
    before = watcher.execute('PRAGMA data_version').fetchone()[0]
    state = ctl.observe_state()
    assert state['expired_unswept']
    assert watcher.execute('PRAGMA data_version').fetchone()[0] == before


def test_a_refusal_names_the_contested_gpus_and_their_holders(tmp_path):
    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'big')
    with pytest.raises(PlacementError) as info:
        acquire(ctl, 'big2')
    assert 'GPU 0,1' in str(info.value) and 'owner owner' in str(info.value)
