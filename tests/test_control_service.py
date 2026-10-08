"""Remote command semantics use the same real ledger/controller as local work."""
import json

import pytest

from infer_stack.control.service import ControlService
from infer_stack.leasing import Controller, Ledger, SqliteStore
from infer_stack.leasing.backend import NullBackend
from infer_stack.leasing.catalog import Catalog


@pytest.fixture
def service(tmp_path):
    store = SqliteStore(tmp_path / 'ledger.db')
    catalog = Catalog.from_dict({'models': {'m': {'source': 'hf://org/m'}},
                                 'endpoints': {'q': {'engine': 'vllm', 'model': 'm'}}})
    svc = ControlService(Controller(Ledger(store), NullBackend()), catalog)
    yield svc
    store.close()


def test_idempotent_acquire_and_audited_release(service):
    first = service.acquire('alice', 'key', {'endpoints': ['q']})
    assert first['state'] == 'queued' and first['result'] is None
    assert first == service.acquire('alice', 'key', {'endpoints': ['q']})
    service.step()
    op = service.operations.get(first['id'])
    assert op['state'] == 'succeeded'
    leases, _ = service.controller.ledger.status()
    assert len(leases) == 1 and leases[0].owner == 'alice'
    assert op['result']['lease_id'] == leases[0].id
    assert 'request' not in service.status()['operations'][0]
    assert 'api_key' not in json.dumps(service.status()) or 'api_key_env' in json.dumps(service.status())
    with pytest.raises(PermissionError):
        service.release('bob', 'release-key', leases[0].id)
    release = service.release('alice', 'release-key', leases[0].id)
    service.step()
    assert service.operations.get(release['id'])['state'] == 'succeeded'
    assert service.controller.ledger.get_lease(leases[0].id).state == 'released'
    assert len(service.operations.store._conn.execute('SELECT * FROM control_audit').fetchall()) >= 6


def test_retry_after_commit_before_progress_does_not_acquire_twice(service, monkeypatch):
    operation = service.acquire('alice', 'key', {'endpoints': ['q'], 'dedicated': True})
    transition = service.operations.transition
    def crash(operation_id, state, **kw):
        if state == 'converging':
            raise SystemExit('simulated process death after lease commit')
        return transition(operation_id, state, **kw)
    monkeypatch.setattr(service.operations, 'transition', crash)
    with pytest.raises(SystemExit):
        service.step()
    assert len(service.controller.ledger.status()[0]) == 1
    recovered = ControlService(service.controller, service.catalog)
    recovered.step()
    assert recovered.operations.get(operation['id'])['state'] == 'succeeded'
    assert len(service.controller.ledger.status()[0]) == len(service.controller.ledger.status()[1]) == 1


def test_idempotency_survives_catalog_change_and_rejects_changed_request(service):
    first = service.acquire('alice', 'key', {'endpoints': ['q']})
    service.catalog = Catalog.from_dict({})
    assert first['id'] == service.acquire('alice', 'key', {'endpoints': ['q']})['id']
    with pytest.raises(ValueError, match='different request'):
        service.acquire('alice', 'key', {'endpoints': ['q'], 'dedicated': True})


@pytest.mark.parametrize('payload', [{'endpoints': ['q'], 'image': 'unapproved'},
                                    {'endpoints': ['q'], 'timeout': float('inf')},
                                    {'endpoints': ['q'], 'ttl_seconds': -1},
                                    {'endpoints': []}, {'endpoints': ['q'], 'dedicated': 'yes'}])
def test_bad_remote_request_never_changes_desired_state(service, payload):
    with pytest.raises(ValueError):
        service.acquire('alice', 'key', payload)
    assert not service.operations.all() and not service.controller.ledger.status()[0]


def test_fresh_wait_checks_generation_and_retains_lease_on_failure(service, monkeypatch):
    from infer_stack.leasing.backend import Readiness

    first = service.acquire('alice', 'acquire', {'endpoints': ['q']})
    service.step()
    lease_id = service.operations.get(first['id'])['result']['lease_id']
    monkeypatch.setattr(service.controller.backend, 'probe_ready', lambda *a: Readiness(False, 'engine crashed', fatal=True))
    wait = service.wait('alice', 'wait', {'lease_id': lease_id})
    service.step()
    assert service.operations.get(wait['id'])['state'] == 'failed'
    assert service.controller.ledger.get_lease(lease_id).state == 'active'
    with pytest.raises(PermissionError):
        service.wait('bob', 'other-wait', {'lease_id': lease_id})


def test_readiness_observation_failure_still_enforces_timeout(service, monkeypatch):
    now = [1000.0]
    service.clock = service.operations.clock = lambda: now[0]
    operation = service.acquire('alice', 'key', {'endpoints': ['q'], 'timeout': 1})
    def unreadable(*args):
        raise OSError('backend unavailable')
    monkeypatch.setattr(service.controller.backend, 'probe_ready', unreadable)
    service.step()
    now[0] += 2
    service.step()
    op = service.operations.get(operation['id'])
    assert op['state'] == 'failed'
    assert service.controller.ledger.get_lease(op['result']['lease_id']).state == 'released'


def test_failure_after_commit_releases_acquire(service, monkeypatch):
    operation = service.acquire('alice', 'key', {'endpoints': ['q']})
    def invalid_info():
        raise ValueError('invalid connection descriptor')
    monkeypatch.setattr(service.controller.backend, 'connection_info', invalid_info)
    service.step()
    assert service.operations.get(operation['id'])['state'] == 'failed'
    leases, _ = service.controller.ledger.status()
    assert len(leases) == 1 and leases[0].state == 'released'


def test_concurrent_idempotency_uses_one_durable_operation(service):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        operations = list(executor.map(lambda _: service.acquire('alice', 'same-key', {'endpoints': ['q']}), range(32)))
    assert len({op['id'] for op in operations}) == 1
    assert len(service.operations.all()) == 1


def test_renew_and_operator_commands_are_durable_and_typed(service):
    acquire = service.acquire('alice', 'acquire', {'endpoints': ['q']})
    service.step()
    lease_id = service.operations.get(acquire['id'])['result']['lease_id']
    with pytest.raises(PermissionError):
        service.renew('bob', 'renew', {'lease_id': lease_id, 'ttl_seconds': 60})
    renewal = service.renew('alice', 'renew', {'lease_id': lease_id, 'ttl_seconds': 60})
    service.step()
    renewed = service.operations.get(renewal['id'])
    assert renewed['state'] == 'succeeded'
    expiry = service.controller.ledger.get_lease(lease_id).expires_at
    assert service.renew('alice', 'renew', {'lease_id': lease_id, 'ttl_seconds': 60})['id'] == renewal['id']
    service.step()
    assert service.controller.ledger.get_lease(lease_id).expires_at == expiry
    with pytest.raises(ValueError):
        service.administer('operator', 'command', 'reconcile', {'command': 'uname'})
    reconcile = service.administer('operator', 'reconcile', 'reconcile', {})
    service.step()
    assert service.operations.get(reconcile['id'])['state'] == 'succeeded'


def test_capacity_queue_does_not_block_release(service, monkeypatch):
    from infer_stack.leasing.backend import PlacementError

    first = service.acquire('alice', 'held', {'endpoints': ['q']})
    service.step()
    lease_id = service.operations.get(first['id'])['result']['lease_id']
    original = service.controller.acquire
    def capacity(*args, **kwargs):
        if service.controller.ledger.get_lease(lease_id).state == 'active':
            raise PlacementError([], ['No free GPU'])
        return original(*args, **kwargs)
    monkeypatch.setattr(service.controller, 'acquire', capacity)
    queued = service.acquire('bob', 'queued', {'endpoints': ['q'], 'queue': True})
    release = service.release('alice', 'release', lease_id)
    service.step()  # blocked acquire rotates behind the release
    service.step()  # release creates capacity
    service.step()  # acquire can now progress
    assert service.operations.get(release['id'])['state'] == 'succeeded'
    assert service.operations.get(queued['id'])['state'] == 'succeeded'


def test_bundle_readiness_uses_resolved_endpoint_aliases(service):
    service.catalog = Catalog.from_dict({'models': {'m': {'source': 'hf://org/m'}},
                                         'endpoints': {'q': {'engine': 'vllm', 'model': 'm'}},
                                         'bundles': {'bundle': ['q']}})
    op = service.acquire('alice', 'bundle', {'endpoints': ['bundle']})
    service.step()
    assert service.operations.get(op['id'])['state'] == 'succeeded'
