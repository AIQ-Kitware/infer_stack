"""Catalog approval is atomic, audited, scoped and independent of file edits."""
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from test_control_service import service as service
from test_store_contract import stores as stores

from infer_stack.cli.commands_catalog import (
    CatalogPublishCLI,
    CatalogRevisionsCLI,
)
from infer_stack.control.auth import Identity
from infer_stack.control.catalog import CatalogRevisions, validate_source
from infer_stack.control.client import Client
from infer_stack.control.http import ControlHTTPServer
from infer_stack.control.operations import Operations
from infer_stack.control.service import ControlService
from infer_stack.leasing.catalog import Catalog
from infer_stack.leasing.profile import canonical_digest

ADMIN = Identity('editor', 'catalog-admin')
SOURCE = {'models': {'new': {'source': 'hf://org/new'}},
          'endpoints': {'new': {'model': 'new', 'runtime': {'image': 'approved/image:1'}}}}


def approval(service, source=SOURCE, key='approval', base=None):
    return service.approve_catalog('editor', key,
                                   {'source': source, 'base_revision': base or service.catalog_revisions.active()}, identity=ADMIN)


def test_approved_catalog_survives_restart_without_adopting_changed_files(service):
    original = service.catalog_revisions.active()
    queued = service.acquire('alice', 'old-acquire', {'endpoints': ['q']})
    approved = approval(service)
    service.step()
    service.step()
    assert service.operations.get(approved['id'])['state'] == 'succeeded'
    recovered = ControlService(service.controller, Catalog.from_dict({}))
    assert set(recovered.catalog.endpoints) == {'new'}
    assert recovered.status()['catalog_revision'] == canonical_digest(SOURCE)
    held = recovered.operations.get(queued['id'])
    assert held['result']['catalog_revision'] == original
    assert recovered.catalog_revisions.get(original)['source']['endpoints']['q']['model'] == 'm'
    approved_source = {'source': SOURCE, 'base_revision': original}
    assert recovered.approve_catalog('editor', 'approval', approved_source, identity=ADMIN)['id'] == approved['id']
    with pytest.raises(ValueError, match='different request'):
        recovered.approve_catalog('editor', 'approval', {**approved_source, 'source': {}}, identity=ADMIN)
    # Approval changes new demand; it does not revoke old leases or replace
    # their launch specifications.
    assert recovered.controller.ledger.get_lease(held['result']['lease_id']).state == 'active'


def test_first_explicit_apply_publishes_approved_catalog_after_restart(tmp_path):
    from test_external_endpoints import REMOTE, _ctl, catalog
    from test_leasing_compose import FakeDocker

    source = Catalog.from_dict(catalog(remote=REMOTE))
    ledger, controller = _ctl(tmp_path, source, FakeDocker(), ui=False)
    try:
        service = ControlService(controller, source)
        approved_source = {**source.source, 'endpoints': {**source.source['endpoints'],
                           'remote': {**REMOTE, 'external': {**REMOTE['external'], 'model': 'org/approved'}}}}
        approved = approval(service, source=approved_source)
        service.step()
        assert service.operations.get(approved['id'])['state'] == 'succeeded'
        assert ledger.profile() is None  # approval itself is not realization
        restarted = ControlService(controller, source)
        applied = restarted.administer('operator', 'apply', 'reconcile', {})
        restarted.step()
        assert restarted.operations.get(applied['id'])['state'] == 'succeeded'
        assert controller.backend.catalog.resolve_endpoint('remote').target.model == 'org/approved'
        assert controller.invocation_profile()['catalogs'] == [source.source]
    finally:
        ledger.store.close()


def test_catalog_commit_and_operation_result_rollback_together(service, monkeypatch):
    old = service.catalog_revisions.active()
    op = approval(service)
    transition = service.operations.transition
    def interrupted(operation_id, state, **kwargs):
        if state == 'succeeded':
            raise SystemExit('process interrupted before commit')
        return transition(operation_id, state, **kwargs)
    monkeypatch.setattr(service.operations, 'transition', interrupted)
    with pytest.raises(SystemExit):
        service.step()
    assert service.catalog_revisions.active() == old
    with pytest.raises(KeyError):
        service.catalog_revisions.get(canonical_digest(SOURCE))
    monkeypatch.setattr(service.operations, 'transition', transition)
    recovered = ControlService(service.controller, Catalog.from_dict({}))
    recovered.step()
    assert recovered.operations.get(op['id'])['state'] == 'succeeded'
    assert recovered.catalog_revisions.active() == canonical_digest(SOURCE)
    assert sum(row['state'] == 'succeeded' for row in recovered.operations.audit()) == 1
    assert set(recovered.catalog.endpoints) == {'new'}


def test_concurrent_approvals_require_review_of_the_winning_revision(stores):
    repositories = [CatalogRevisions(stores()), CatalogRevisions(stores())]
    repositories[0].seed({})
    base = repositories[0].active()
    operations = [Operations(repo.store) for repo in repositories]
    pending = [op.submit('editor', f'change-{i}', {'action': 'catalog-approve',
               'client': {'base_revision': base, 'source': source}})
               for i, (op, source) in enumerate(zip(operations, [SOURCE, {'endpoints': {}}]))]
    for op, operation in zip(operations, pending):
        op.transition(operation['id'], 'running')
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(repo.approve, operation, op) for repo, operation, op in zip(repositories, pending, operations)]
        for future in futures:
            future.result()
    final = [operations[0].get(operation['id']) for operation in pending]
    assert sorted(op['state'] for op in final) == ['failed', 'succeeded']
    success = next(op for op in final if op['state'] == 'succeeded')
    assert repositories[0].active() == success['result']['catalog_revision']
    assert next(op for op in final if op['state'] == 'failed')['error']['code'] == 'CatalogConflict'
    assert len(repositories[0].history()['revisions']) == 2


@pytest.mark.parametrize('source', [[], {'models': []}, {'models': {'m': {}}},
                                     {'models': {'m': {'source': 123}}, 'endpoints': {'q': {'model': 'm'}}},
                                     {'endpoints': {'q': {'model': 'missing'}}},
                                     {'endpoints': {'q': {'external': {'api_base': 'http://u:private@example/v1', 'model': 'x'}}}},
                                     {'models': {'m': {'source': float('nan')}}},
                                     {'arbitrary': {'command': 'shell'}}])
def test_invalid_revision_does_not_echo_private_data(source):
    with pytest.raises(ValueError, match='Invalid catalog revision') as error:
        validate_source(source)
    assert 'private' not in str(error.value)


def test_http_and_cli_catalog_approval_have_distinct_privilege(service, tmp_path, capsys):
    registry = tmp_path / 'identities.json'
    entries, clients = [], {}
    for role in ['catalog-admin', 'operator', 'observer', 'lease-user']:
        entries.append({'principal': role, 'role': role, 'token_sha256': hashlib.sha256(role.encode()).hexdigest()})
    registry.write_text(json.dumps({'identities': entries}))
    registry.chmod(0o600)
    server = ControlHTTPServer(service, identity_file=registry, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stop = threading.Event()
    def advance():
        while not stop.wait(.01):
            service.step()
    worker = threading.Thread(target=advance, daemon=True)
    worker.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}'
        for entry in entries:
            token = tmp_path / (entry['role'] + '.token')
            token.write_text(entry['role'])
            token.chmod(0o600)
            clients[entry['role']] = Client(url, token)
        base = service.catalog_revisions.active()
        payload = {'source': SOURCE, 'base_revision': base}
        for role, client in clients.items():
            if role == 'catalog-admin':
                continue
            with pytest.raises(ValueError, match='403'):
                client.request('POST', '/v1/catalog/revisions', payload=payload)
            with pytest.raises(ValueError, match='403'):
                client.request('GET', '/v1/catalog/revisions/' + base)
            assert 'source' not in json.dumps(client.request('GET', '/v1/status'))
        path = tmp_path / 'approved.yaml'
        path.write_text(yaml.safe_dump(SOURCE))
        flags = {'argv': False, 'server_url': url, 'server_token_file': str(tmp_path / 'catalog-admin.token')}
        CatalogRevisionsCLI.main(**flags)
        assert json.loads(capsys.readouterr().out)['active_revision'] == base
        CatalogPublishCLI.main(**flags, source=str(path), base_revision=base, idempotency_key='cli-approval')
        result = json.loads(capsys.readouterr().out)
        assert result['catalog_revision'] == canonical_digest(SOURCE)
        CatalogPublishCLI.main(**flags, source=str(path), base_revision=base, idempotency_key='cli-approval')
        assert json.loads(capsys.readouterr().out)['operation_id'] == result['operation_id']
        CatalogRevisionsCLI.main(**flags, revision=result['catalog_revision'])
        assert json.loads(capsys.readouterr().out)['source'] == SOURCE
        audit = clients['operator'].request('GET', '/v1/audit')
        assert all(row['principal'] == 'catalog-admin' and row['action'] == 'catalog-approve' for row in audit)
    finally:
        stop.set()
        worker.join(5)
        server.shutdown()
        server.server_close()
        thread.join(5)
