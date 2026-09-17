"""Stable per-service addresses and the upstream check (plan step P7, tests 56-59)."""

from __future__ import annotations

import json

import pytest
import yaml

from infer_stack.leasing.network import NETWORK_NAME, allocate, overlapping_subnets
from infer_stack.leasing.profile import ProfileMismatch
from test_leasing_admission import CAT, acquire, make

SUBNET = '10.123.45.0/28'


@pytest.fixture(autouse=True)
def no_host_routes(monkeypatch):
    import subprocess

    import infer_stack.leasing.network as net

    class Done:
        stdout = ''

    monkeypatch.setattr(net.subprocess, 'run', lambda *a, **k: Done())
    yield subprocess


def addresses(ctl):
    doc = yaml.safe_load(ctl.backend.compose_file.read_text())
    return {name: svc['networks'][NETWORK_NAME]['ipv4_address']
            for name, svc in doc['services'].items()}, doc


def test_migrate_persists_the_subnet_and_skips_reserved_addresses(tmp_path):       # 56
    ledger, ctl, _ = make(tmp_path)
    acquire(ctl, 'one')
    ctl.network_migrate(SUBNET, force=True)
    assert ledger.network_config() == {'subnet': SUBNET}
    table, doc = addresses(ctl)
    assert doc['networks'][NETWORK_NAME]['ipam']['config'] == [{'subnet': SUBNET}]
    assert all(ip not in {'10.123.45.0', '10.123.45.1', '10.123.45.15'} for ip in table.values())
    assert ledger.service_addresses() == table


def test_an_overlapping_subnet_fails_preflight(tmp_path):
    ledger, ctl, docker = make(tmp_path)

    def run(args, **kw):
        if args[:3] == ['docker', 'network', 'ls']:
            return 'n1\n'
        if args[:3] == ['docker', 'network', 'inspect']:
            return json.dumps([{'Name': 'other', 'IPAM': {'Config': [{'Subnet': '10.123.0.0/16'}]}}])
        return docker(args, **kw)

    ctl.backend.run = run
    with pytest.raises(ProfileMismatch, match='overlaps'):
        ctl.network_migrate(SUBNET)
    assert ledger.network_config() is None


def test_a_service_keeps_its_address_and_no_other_service_receives_it(tmp_path):   # 57
    ledger, ctl, _ = make(tmp_path)
    ctl.network_migrate(SUBNET)
    a = acquire(ctl, 'one')
    first, _ = addresses(ctl)
    (svc_a, ip_a), = first.items()
    ctl.release_leases([a.lease.id], evict=True)             # its container is removed
    acquire(ctl, 'two')                                        # a new service arrives
    now, _ = addresses(ctl)
    assert ip_a not in now.values()                           # never handed to another name
    acquire(ctl, 'one')                                        # the old service returns
    again, _ = addresses(ctl)
    assert again[svc_a] == ip_a


def test_migrate_recreates_each_container_once_and_refuses_active_leases(tmp_path):  # 58
    ledger, ctl, docker = make(tmp_path)
    acquire(ctl, 'one')
    with pytest.raises(ProfileMismatch, match='active'):
        ctl.network_migrate(SUBNET)
    before = set(docker.containers)
    ctl.network_migrate(SUBNET, force=True)
    after = set(docker.containers)
    assert len(after) == len(before) and not (after & before)   # each recreated once
    ctl.apply_now()
    assert set(docker.containers) == after                      # and only once


def test_upstream_check_distinguishes_fault_from_not_ready(tmp_path):             # 59
    ledger, ctl, docker = make(tmp_path)
    one = acquire(ctl, 'one')
    two = acquire(ctl, 'two')
    replies = {}

    def run(args, **kw):
        if 'exec' in args:
            url = args[-1]
            return replies[url.split('//')[1].split(':')[0]]
        return docker(args, **kw)

    ctl.backend.run = run
    services = ctl.backend._load_sidecar()['services']
    by_dep = {gid: svc for svc, gid in services.items()}
    replies[by_dep[one.deployments[0].id]] = json.dumps(['two'])          # misrouted
    replies[by_dep[two.deployments[0].id]] = json.dumps({'error': 'refused'})
    result = ctl.backend.upstream_check()
    statuses = {row['deployment']: row['status'] for row in result.values()}
    assert statuses == {one.deployments[0].id: 'routing-fault',
                        two.deployments[0].id: 'not-ready'}
    replies[by_dep[one.deployments[0].id]] = json.dumps(['one'])
    assert ctl.backend.upstream_check()[by_dep[one.deployments[0].id]]['status'] == 'healthy'


def test_allocation_is_append_only():
    table = allocate(SUBNET, {'a': '10.123.45.2'}, ['b', 'a'])
    assert table == {'a': '10.123.45.2', 'b': '10.123.45.3'}


def test_an_unmanaged_holder_of_a_service_address_blocks_it(tmp_path):              # 46b
    from infer_stack.leasing.compose import ApplyAborted

    ledger, ctl, docker = make(tmp_path)
    ctl.network_migrate(SUBNET)
    ctl.backend.network = {'subnet': SUBNET, 'addresses': ledger.service_addresses()}
    docker.add_container('squatter', labels={}, ips=['10.123.45.2'])
    with pytest.raises(ApplyAborted, match='holds address'):
        acquire(ctl, 'one')


def test_a_declined_migration_changes_nothing(tmp_path):
    from infer_stack.leasing.backend import ConvergeAborted

    ledger, ctl, _ = make(tmp_path)
    ctl.network_migrate(SUBNET)
    acquire(ctl, 'one')
    table = ledger.service_addresses()

    def decline(planned):
        raise ConvergeAborted('no')

    ctl.backend._approve_changes = decline
    with pytest.raises(ConvergeAborted):
        ctl.network_migrate('10.123.46.0/28', force=True)
    assert ledger.network_config() == {'subnet': SUBNET}
    assert ledger.service_addresses() == table
    assert ledger.publication_pending() is None


def test_subnet_switch_and_address_reset_are_one_transaction(tmp_path):
    from infer_stack.leasing import Ledger, SqliteStore

    store = SqliteStore(str(tmp_path / 'l.db'))
    ledger = Ledger(store)
    ledger.set_network_config({'subnet': SUBNET})
    ledger.add_service_addresses({'a': '10.123.45.2'})
    real = store._write_marker

    def crash(**kw):
        raise KeyboardInterrupt('killed inside the transaction')

    store._write_marker = crash
    with pytest.raises(KeyboardInterrupt):
        store.migrate_network(subnet='10.123.46.0/28', reset_addresses=True, approved_digest='d')
    store._write_marker = real
    assert ledger.network_config() == {'subnet': SUBNET}          # all or nothing
    assert ledger.service_addresses() == {'a': '10.123.45.2'}


def test_remigrating_ignores_the_route_our_own_network_creates(monkeypatch):
    import infer_stack.leasing.network as net

    class Routes:
        stdout = '10.123.45.0/28 dev br-abc proto kernel scope link\n'

    monkeypatch.setattr(net.subprocess, 'run', lambda *a, **k: Routes())

    def run(args, **kw):
        if args[:3] == ['docker', 'network', 'ls']:
            return 'n1\n'
        return json.dumps([{'Name': NETWORK_NAME, 'IPAM': {'Config': [{'Subnet': SUBNET}]}}])

    assert overlapping_subnets(SUBNET, run) == []
