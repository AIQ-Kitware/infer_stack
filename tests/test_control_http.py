"""Real HTTP transport, private credentials, rotation, and remote CLI isolation."""
import json
import threading
from types import SimpleNamespace

import pytest
import requests
from test_control_service import service as service

from infer_stack.control.client import Client
from infer_stack.control.http import ControlHTTPServer


@pytest.fixture
def api(service, tmp_path):
    token = tmp_path / 'control.token'
    token.write_text('private-control-token')
    token.chmod(0o600)
    server = ControlHTTPServer(service, token_file=token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield service, token, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_auth_typed_mutations_and_live_token_rotation(api):
    service, token, url = api
    assert requests.get(url + '/v1/status', timeout=5).status_code == 401
    remote = Client(url, token)
    assert remote.request('GET', '/v1/catalog') == [{'alias': 'q', 'protocol': 'chat'}]
    op = remote.request('POST', '/v1/leases', payload={'endpoints': ['q']}, key='stable-key')
    service.step()
    done = remote.wait(op['id'], timeout=5)
    assert done['state'] == 'succeeded'
    assert len(remote.request('GET', '/v1/leases')) == 1
    with pytest.raises(ValueError, match='404'):
        remote.request('GET', '/v1/env')
    token.write_text('rotated-control-token')
    assert requests.get(url + '/v1/status', headers={'Authorization': 'Bearer private-control-token'}, timeout=5).status_code == 401
    assert remote.request('GET', '/v1/status')['leases'][0]['owner'] == 'operator'
    assert 'requests' not in json.dumps(done)
    audit = remote.request('GET', '/v1/audit?limit=2')
    assert len(audit) == 2 and audit[0]['state'] == 'queued'
    assert 'request' not in audit[0] and audit[0]['principal'] == 'operator'
    following = remote.request('GET', '/v1/audit?after=' + str(audit[-1]['id']))
    assert following and all(row['id'] > audit[-1]['id'] for row in following)
    with pytest.raises(ValueError, match='400'):
        remote.request('GET', '/v1/audit?limit=0')
    with pytest.raises(ValueError, match='501'):
        remote.request('GET', '/v1/nodes')


def test_remote_binding_requires_tls_and_private_credentials(service, tmp_path):
    token = tmp_path / 'token'
    token.write_text('token')
    token.chmod(0o600)
    with pytest.raises(ValueError, match='TLS'):
        ControlHTTPServer(service, token_file=token, host='0.0.0.0', port=0)
    token.chmod(0o644)
    with pytest.raises(ValueError, match='private'):
        ControlHTTPServer(service, token_file=token, port=0)
    token.chmod(0o600)
    with pytest.raises(ValueError, match='TLS'):
        Client('http://worker.lab:14043', token)


def test_real_tls_requires_trusted_certificate(service, tmp_path):
    import shutil
    import socket
    import subprocess

    if not shutil.which('openssl'):
        pytest.skip('TLS integration requires openssl')
    key, cert, token = [tmp_path / name for name in ('key.pem', 'cert.pem', 'token')]
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)
    key.chmod(0o600)
    token.write_text('private-token')
    token.chmod(0o600)
    server = ControlHTTPServer(service, token_file=token, port=0, tls_cert=str(cert), tls_key=str(key))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'https://127.0.0.1:{server.server_port}'
    try:
        with socket.create_connection(('127.0.0.1', server.server_port), timeout=5):
            # A peer holds TCP open without starting TLS; other clients serve.
            assert Client(url, token, ca_file=str(cert)).request('GET', '/v1/status')['leases'] == []
        with pytest.raises(requests.exceptions.SSLError):
            requests.get(url + '/v1/status', timeout=5)
        assert Client(url, token, ca_file=str(cert)).request('GET', '/v1/status')['leases'] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_rate_budgets_refill_without_unbounded_identity_memory():
    from infer_stack.control.http import RateLimit

    now = [0.0]
    limits = RateLimit(2, clock=lambda: now[0], max_buckets=2)
    assert limits.allow('alice') and limits.allow('alice')
    assert not limits.allow('alice') and limits.allow('bob')
    now[0] = 30
    assert limits.allow('alice') and not limits.allow('alice')
    assert not limits.allow('charlie') and len(limits.buckets) == 2
    now[0] = 90
    assert limits.allow('charlie') and len(limits.buckets) == 1


def test_control_listener_rate_limit_is_enforced(service, tmp_path):
    token = tmp_path / 'token'
    token.write_text('private-control-token')
    token.chmod(0o600)
    server = ControlHTTPServer(service, token_file=token, port=0, requests_per_minute=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}/v1/status'
        headers = {'Authorization': 'Bearer private-control-token'}
        assert requests.get(url, headers=headers, timeout=5).status_code == 200
        rejected = requests.get(url, headers=headers, timeout=5)
        assert rejected.status_code == 429 and rejected.headers['Retry-After'] == '60'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_timeout_retries_keep_one_idempotency_identity(tmp_path):
    token = tmp_path / 'token'
    token.write_text('token')
    token.chmod(0o600)
    calls = []
    def request(method, url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.Timeout('response lost after accepting mutation')
        return SimpleNamespace(status_code=202, json=lambda: {'id': 'same-operation'})
    remote = Client('http://127.0.0.1:1', token, http=SimpleNamespace(request=request))
    assert remote.request('POST', '/v1/leases', payload={'endpoints': ['q']})['id'] == 'same-operation'
    assert calls[0]['headers']['Idempotency-Key'] == calls[1]['headers']['Idempotency-Key']
    assert calls[0]['verify'] is True and calls[0]['allow_redirects'] is False


def test_remote_cli_never_opens_a_local_ledger(api, monkeypatch, tmp_path, capsys):
    from infer_stack.cli import (
        AccessCLI,
        AcquireCLI,
        ApplyCLI,
        LeasesCLI,
        ReleaseCLI,
        RenewCLI,
        RunCLI,
        WaitCLI,
        commands_leasing,
    )
    from infer_stack.cli.commands_runtime import StatusCLI
    from infer_stack.paths import data_root

    service, token, url = api
    monkeypatch.setattr(commands_leasing, '_open_controller', lambda *a, **kw: pytest.fail('remote CLI opened a local controller'))
    stop = threading.Event()
    def process():
        while not stop.wait(.01):
            service.step()
    worker = threading.Thread(target=process, daemon=True)
    worker.start()
    env = tmp_path / 'lease.env'
    flags = dict(argv=False, server_url=url, server_token_file=str(token))
    try:
        AcquireCLI.main(**flags, names=['q'], timeout=5, interval=.01,
                        idempotency_key='cli-key', env_file=str(env), json=True)
        descriptor = json.loads(capsys.readouterr().out)
        assert descriptor['lease_id'] in env.read_text()
        assert 'private-control-token' not in env.read_text()
        LeasesCLI.main(**flags, json=True)
        assert len(json.loads(capsys.readouterr().out)['leases']) == 1
        StatusCLI.main(**flags)
        assert json.loads(capsys.readouterr().out)['leases'][0]['id'] == descriptor['lease_id']
        WaitCLI.main(**flags, names=['q'], timeout=5, interval=.01, json=True)
        assert json.loads(capsys.readouterr().out)[0]['state'] == 'succeeded'
        RenewCLI.main(**flags, env_file=str(env), ttl='1h')
        assert 'renewed' in capsys.readouterr().out
        ApplyCLI.main(**flags, json=True)
        assert not json.loads(capsys.readouterr().out)['publication_pending']
        ReleaseCLI.main(**flags, env_file=str(env), json=True)
        assert json.loads(capsys.readouterr().out)['released'] == [descriptor['lease_id']]
        AccessCLI.main(**flags, names=['q'], timeout=5, interval=.01, json=True)
        access = json.loads(capsys.readouterr().out)
        assert access['lease_id']
        ReleaseCLI.main(**flags, lease=access['lease_id'])
        capsys.readouterr()
        import sys
        assert RunCLI.main(**flags, endpoint='q', timeout=5, interval=.01,
                           command=[sys.executable, '-c', 'import os; assert os.environ["INFER_STACK_LEASE_ID"]; raise SystemExit(7)']) == 7
        assert all(le['state'] == 'released' for le in service.status()['leases'])
        assert not (data_root() / 'leasing' / 'ledger.db').exists()
    finally:
        stop.set()
        worker.join(5)


def test_named_identities_permissions_ownership_quotas_and_revocation(service, tmp_path):
    import hashlib

    registry = tmp_path / 'identities.json'
    entries, tokens = [], {}
    for principal, role in [('alice', 'lease-user'), ('bob', 'lease-user'), ('observer', 'observer'), ('admin', 'operator')]:
        token = tmp_path / (principal + '.token')
        token.write_text(principal + '-private-token')
        token.chmod(0o600)
        tokens[principal] = token
        entries.append({'principal': principal, 'role': role,
                        'token_sha256': hashlib.sha256(token.read_bytes()).hexdigest(),
                        'max_active_leases': 1 if role == 'lease-user' else None,
                        'allowed_endpoints': ['q'] if role == 'lease-user' else None})
    registry.write_text(json.dumps({'identities': entries}))
    registry.chmod(0o600)
    server = ControlHTTPServer(service, identity_file=registry, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clients = {name: Client(f'http://127.0.0.1:{server.server_port}', token) for name, token in tokens.items()}
    try:
        with pytest.raises(ValueError, match='403'):
            clients['observer'].request('POST', '/v1/leases', payload={'endpoints': ['q']})
        with pytest.raises(ValueError, match='403'):
            clients['alice'].request('POST', '/v1/reconcile', payload={})
        with pytest.raises(ValueError, match='403'):
            clients['alice'].request('POST', '/v1/leases', payload={'endpoints': ['q']})  # finite TTL required
        payload = {'endpoints': ['q'], 'ttl_seconds': 60}
        alice = clients['alice'].request('POST', '/v1/leases', payload=payload, key='alice-key')
        assert clients['alice'].request('POST', '/v1/leases', payload=payload, key='alice-key')['id'] == alice['id']
        with pytest.raises(ValueError, match='403'):
            clients['alice'].request('POST', '/v1/leases', payload=payload, key='over-quota')
        bob = clients['bob'].request('POST', '/v1/leases', payload=payload)
        service.step()
        service.step()
        alice = clients['alice'].wait(alice['id'], timeout=5)
        assert [le['owner'] for le in clients['alice'].request('GET', '/v1/leases')] == ['alice']
        with pytest.raises(ValueError, match='403'):
            clients['bob'].request('GET', '/v1/operations/' + alice['id'])
        with pytest.raises(ValueError, match='403'):
            clients['bob'].request('DELETE', '/v1/leases/' + alice['result']['lease_id'])
        assert len(clients['observer'].request('GET', '/v1/status')['leases']) == 2
        assert {row['principal'] for row in clients['alice'].request('GET', '/v1/audit')} == {'alice'}
        assert {row['principal'] for row in clients['observer'].request('GET', '/v1/audit')} == {'alice', 'bob'}
        registry.write_text(json.dumps({'identities': [row for row in entries if row['principal'] != 'alice']}))
        with pytest.raises(ValueError, match='401'):
            clients['alice'].request('GET', '/v1/status')
        assert clients['bob'].wait(bob['id'], timeout=5)['state'] == 'succeeded'
        audited = {row['principal'] for row in service.operations.store._conn.execute('SELECT principal FROM control_audit')}
        assert audited == {'alice', 'bob'}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_remote_usage_keeps_measurement_provenance(api, tmp_path, capsys):
    from infer_stack.accounting import UsageInterval, UsageStore
    from infer_stack.cli.commands_usage import UsageCLI

    service, token, url = api
    assert Client(url, token).request('GET', '/v1/usage') == []
    service.usage_path = tmp_path / 'usage.db'
    store = UsageStore(service.usage_path)
    try:
        store.append(UsageInterval('window', 0, 10, 'worker', 'deployment', 'replica', 1,
                                   gpu_uuids=('GPU-1',), source='nvidia-smi', samples=((0, 100), (10, 100))))
    finally:
        store.close()
    UsageCLI.main(argv=False, server_url=url, server_token_file=str(token), json=True)
    row, = json.loads(capsys.readouterr().out)
    assert row['energy']['method'] == 'measured' and row['energy']['energy_joules'] == 1000
    assert row['energy']['coverage'] == 1 and row['facts']['gpu_uuids'] == ['GPU-1']
    assert Client(url, token).request('GET', '/v1/usage?node=other') == []
