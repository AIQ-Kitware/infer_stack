"""Durable remote access preserves the managed/external endpoint contract."""
import json
import threading

import pytest
import yaml
from test_control_service import approve_catalog
from test_external_endpoints import REMOTE, _ctl, catalog
from test_leasing_compose import FakeDocker

from infer_stack.control.auth import Identity
from infer_stack.control.client import Client
from infer_stack.control.http import ControlHTTPServer
from infer_stack.control.service import ControlService
from infer_stack.leasing import Catalog


@pytest.fixture
def external_service(tmp_path):
    approved = Catalog.from_dict(catalog(remote=REMOTE))
    docker = FakeDocker()
    ledger, controller = _ctl(tmp_path, approved, docker, ui=False)
    service = ControlService(controller, approved)
    yield service, docker
    ledger.store.close()


def done(service, operation):
    service.step()
    return service.operations.get(operation['id'])


def test_external_access_needs_no_lease_and_never_generates_upstream(external_service, monkeypatch):
    service, _docker = external_service
    identity = Identity('alice', 'lease-user', ('remote',), max_active_leases=1, max_ttl_seconds=60)
    monkeypatch.setattr(service.controller.backend.http, 'post', lambda *a, **kw: pytest.fail('External access generated a request'))
    result = done(service, service.access('alice', 'external', {'endpoints': ['remote']}, identity=identity))
    assert result['state'] == 'succeeded'
    assert result['result']['lease_id'] is None
    assert result['result']['external'] == ['remote'] and result['result']['models'] == {'remote': 'remote'}
    assert service.controller.ledger.status()[0] == service.controller.ledger.status()[1] == []
    routes = yaml.safe_load((service.controller.backend.state_dir / 'litellm_config.yaml').read_text())['model_list']
    remote = next(row for row in routes if row['model_name'] == 'remote')
    assert remote['litellm_params']['api_base'] == REMOTE['external']['api_base']
    assert remote['litellm_params']['api_key'] == 'os.environ/REMOTE_QWEN_KEY'
    assert 'sk-remote' not in json.dumps(service.status())


def test_mixed_access_is_one_durable_lease_with_restart_recovery(external_service, tmp_path, monkeypatch):
    service, docker = external_service
    accepted = service.access('alice', 'mixed', {'endpoints': ['pair'], 'ttl_seconds': 60, 'dedicated': True})
    transition = service.operations.transition
    def die(operation_id, state, **kwargs):
        if state == 'converging':
            raise SystemExit('Death after lease commit and route publication')
        return transition(operation_id, state, **kwargs)
    monkeypatch.setattr(service.operations, 'transition', die)
    with pytest.raises(SystemExit):
        service.step()
    leases, deployments = service.controller.ledger.status()
    assert len(leases) == len(deployments) == 1 and leases[0].endpoints == ['local']
    changed = Catalog.from_dict(catalog(remote={**REMOTE, 'external': {**REMOTE['external'], 'api_base': 'http://new/v1'}}))
    ledger, controller = _ctl(tmp_path, changed, docker, ui=False)
    try:
        recovered = ControlService(controller, changed)
        result = done(recovered, accepted)
        assert result['state'] == 'succeeded' and result['result']['lease_id'] == leases[0].id
        assert result['result']['endpoints'] == ['local', 'remote']
        assert len(ledger.status()[0]) == len(ledger.status()[1]) == 1
        published = controller.backend.catalog.resolve_endpoint('remote').target
        assert published.api_base == REMOTE['external']['api_base']
        assert controller.invocation_profile()['catalogs'] == [changed.source]
    finally:
        ledger.store.close()


def test_pending_snapshot_survives_catalog_changes_then_new_access_adopts_them(external_service):
    service, _docker = external_service
    payload = {'endpoints': ['remote']}
    accepted = service.access('alice', 'old', payload)
    changed = Catalog.from_dict(catalog(remote={**REMOTE, 'external': {**REMOTE['external'], 'model': 'org/new'}}))
    approve_catalog(service, changed.source)
    assert service.access('alice', 'old', payload)['id'] == accepted['id']
    assert done(service, accepted)['state'] == 'succeeded'
    assert service.controller.backend.catalog.resolve_endpoint('remote').target.model == REMOTE['external']['model']
    newer = done(service, service.access('alice', 'new', payload))
    assert newer['state'] == 'succeeded'
    assert service.controller.backend.catalog.resolve_endpoint('remote').target.model == 'org/new'


def test_queued_managed_acquire_uses_accepted_catalog_on_a_new_controller(external_service, tmp_path):
    service, docker = external_service
    accepted = service.acquire('alice', 'managed', {'endpoints': ['local']})
    expected = service.catalog.resolve_requests(['local'])[0]
    changed_source = catalog(remote=REMOTE)
    changed_source['models']['m']['source'] = 'hf://org/replaced'
    changed = Catalog.from_dict(changed_source)
    ledger, controller = _ctl(tmp_path, changed, docker, ui=False)
    try:
        recovered = ControlService(controller, changed)
        result = done(recovered, accepted)
        assert result['state'] == 'succeeded'
        deployment = ledger.status()[1][0]
        assert deployment.spec['hf_model_id'] == expected.spec['hf_model_id']
        assert controller.invocation_profile()['catalogs'] == [changed.source]
    finally:
        ledger.store.close()


def test_changed_external_route_supersedes_converging_mixed_access(external_service, monkeypatch):
    service, _docker = external_service
    monkeypatch.setattr(service.controller.backend.front_door(), 'gateway_accepts', lambda *a, **kw: None)
    old = done(service, service.access('alice', 'old', {'endpoints': ['pair']}))
    assert old['state'] == 'converging'
    approve_catalog(service, catalog(remote={**REMOTE, 'external': {**REMOTE['external'], 'api_base': 'http://new/v1'}}))
    monkeypatch.setattr(service.controller.backend.front_door(), 'gateway_accepts', lambda *a, **kw: True)
    new = done(service, service.access('alice', 'new', {'endpoints': ['pair']}))
    assert new['state'] == 'succeeded'
    superseded = service.operations.get(old['id'])
    assert superseded['state'] == 'superseded' and superseded['error']['code'] == 'EndpointChanged'
    assert service.controller.ledger.get_lease(old['result']['lease_id']).state == 'released'
    assert service.controller.ledger.get_lease(new['result']['lease_id']).state == 'active'


@pytest.mark.parametrize('names', [['remote'], ['pair']])
def test_gateway_observation_errors_enforce_deadline_and_release_only_managed(external_service, monkeypatch, names):
    service, _docker = external_service
    now = [1000.0]
    service.clock = service.operations.clock = lambda: now[0]
    def unavailable(*args, **kwargs):
        raise OSError('Gateway observation unavailable')
    monkeypatch.setattr(service.controller.backend.front_door(), 'gateway_accepts', unavailable)
    op = done(service, service.access('alice', 'access', {'endpoints': names, 'timeout': 1}))
    assert op['state'] == 'converging'
    now[0] += 2
    failed = done(service, op)
    assert failed['state'] == 'failed'
    if failed['result']['lease_id']:
        assert service.controller.ledger.get_lease(failed['result']['lease_id']).state == 'released'
    else:
        assert service.controller.ledger.status()[0] == []


def test_scope_and_quota_cover_expanded_mixed_bundles(external_service):
    service, _docker = external_service
    restricted = Identity('alice', 'lease-user', ('local',), 1, 60)
    with pytest.raises(PermissionError, match='endpoint'):
        service.access('alice', 'denied', {'endpoints': ['pair'], 'ttl_seconds': 60}, identity=restricted)
    allowed = Identity('alice', 'lease-user', ('local', 'remote'), 1, 60)
    service.access('alice', 'mixed', {'endpoints': ['pair'], 'ttl_seconds': 60}, identity=allowed)
    with pytest.raises(PermissionError, match='quota'):
        service.acquire('alice', 'managed', {'endpoints': ['local'], 'ttl_seconds': 60}, identity=allowed)
    assert service.access('alice', 'external', {'endpoints': ['remote']}, identity=allowed)['state'] == 'queued'
    with pytest.raises(ValueError, match='cannot constrain'):
        service.access('alice', 'placement', {'endpoints': ['pair'], 'node': 'worker'})


def test_real_remote_cli_access_run_and_release_external_env(external_service, tmp_path, capsys, monkeypatch):
    from infer_stack.cli import AccessCLI, ReleaseCLI, RunCLI, commands_leasing

    service, _docker = external_service
    monkeypatch.setattr(commands_leasing, '_open_controller', lambda *a, **kw: pytest.fail('Client opened a local controller'))
    token = tmp_path / 'control.token'
    token.write_text('private-control-token')
    token.chmod(0o600)
    server = ControlHTTPServer(service, token_file=token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stop = threading.Event()
    def process():
        while not stop.wait(.01):
            service.step()
    worker = threading.Thread(target=process, daemon=True)
    worker.start()
    flags = dict(argv=False, server_url=f'http://127.0.0.1:{server.server_port}', server_token_file=str(token))
    env = tmp_path / 'access.env'
    try:
        AccessCLI.main(**flags, names=['remote'], timeout=5, interval=.01, env_file=str(env), json=True)
        descriptor = json.loads(capsys.readouterr().out)
        assert descriptor['lease_id'] is None and descriptor['external'] == ['remote']
        assert 'INFER_STACK_LEASE_ID' not in env.read_text()
        assert 'private-control-token' not in env.read_text() and 'sk-remote' not in env.read_text()
        with pytest.raises(SystemExit) as released:
            ReleaseCLI.main(**flags, env_file=str(env))
        assert released.value.code == 0
        capsys.readouterr()
        import sys
        monkeypatch.setenv('INFER_STACK_LEASE_ID', 'unrelated-parent-lease')
        assert RunCLI.main(**flags, endpoint='remote', timeout=5, interval=.01,
                           command=[sys.executable, '-c', 'import os; assert "INFER_STACK_LEASE_ID" not in os.environ; raise SystemExit(7)']) == 7
        assert service.controller.ledger.status()[0] == []
        observer = Identity('observer', 'observer')
        with pytest.raises(PermissionError):
            service.access(observer.principal, 'observer', {'endpoints': ['remote']}, identity=observer)
        assert Client(flags['server_url'], token).request('GET', '/v1/status')['leases'] == []
    finally:
        stop.set()
        worker.join(5)
        server.shutdown()
        server.server_close()
        thread.join(5)
