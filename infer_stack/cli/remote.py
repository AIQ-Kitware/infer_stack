"""CLI transport selection; remote verbs never open a client-side ledger."""
from __future__ import annotations

import json
import os
from pathlib import Path

import kwconf as kw

from ..control.client import Client
from ..leasing.envfile import render_env_file
from ..paths import get_setting
from .context import _apply_path_overrides


class RemoteOptions(kw.Config):
    server_url = kw.Value(None, type=str, alias=['server'], help='Canonical control service URL; otherwise INFER_STACK_SERVER or server_url setting.')
    server_token_file = kw.Value(None, type=str, help='Private file holding control API token; otherwise server_token_file setting.')
    server_ca_file = kw.Value(None, type=str, help='Trusted control TLS CA PEM; defaults to system trust.')


def server_url(config):
    _apply_path_overrides(config)
    return getattr(config, 'server_url', None) or os.environ.get('INFER_STACK_SERVER') or get_setting('server_url')


def client(config):
    token = getattr(config, 'server_token_file', None) or get_setting('server_token_file')
    if not token:
        raise SystemExit('Remote control requires --server-token-file or config set server_token_file PATH')
    return Client(server_url(config), token, ca_file=getattr(config, 'server_ca_file', None) or get_setting('server_ca_file'))


def acquire_descriptor(config, *, names, ttl_seconds, ready=True, access=False):
    if not getattr(config, 'apply', True):
        raise SystemExit('Remote acquisition does not stage client-side manifests; use local acquire --no-apply for staging')
    if getattr(config, 'reserve_gpus', 0):
        raise SystemExit('Remote GPU reservations are not yet supported')
    if getattr(config, 'catalog', None) or getattr(config, 'backend', None) or getattr(config, 'ledger', None):
        raise SystemExit('The control server owns catalog, backend and ledger selection')
    payload = {'endpoints': names, 'ttl_seconds': ttl_seconds, 'timeout': float(config.timeout),
               'dedicated': bool(getattr(config, 'dedicated', False)),
               'queue': bool(getattr(config, 'queue', False))}
    for key in ['node', 'pool']:
        if getattr(config, key, None) is not None:
            payload[key] = getattr(config, key)
    remote = client(config)
    op = remote.request('POST', '/v1/access' if access else '/v1/leases', payload=payload, key=getattr(config, 'idempotency_key', None))
    op = remote.wait(op['id'], ready=ready, timeout=float(config.timeout), interval=float(config.interval))
    result = op['result']
    descriptor = {'schema_version': 1, 'kind': 'infer-stack-endpoint',
                  'operation_id': op['id'], 'lease_id': result['lease_id'],
                  'catalog_revision': result.get('catalog_revision'),
                  'base_url': result['base_url'], 'api_key_env': result['api_key_env'],
                  'external': result.get('external', []), 'published': result.get('published', True),
                  'endpoints': result['models'], 'models': list(result['models'].values())}
    return descriptor, op


def acquire(config, *, names, ttl_seconds, access=False):
    descriptor, op = acquire_descriptor(config, names=names, ttl_seconds=ttl_seconds, ready=bool(config.wait), access=access)
    result = op['result']
    if config.env_file:
        Path(config.env_file).expanduser().write_text(render_env_file(descriptor))
    held = f"lease {result['lease_id']}" if result['lease_id'] else 'access (no lease)'
    print(json.dumps(descriptor, indent=2) if config.json else
          f"{held}: {', '.join(result['endpoints'])}; operation {op['id']} {op['state']}")
    return 0 if descriptor['published'] else 3


def renew(config, *, lease_id, ttl_seconds):
    remote = client(config)
    op = remote.request('POST', '/v1/renewals', payload={'lease_id': lease_id, 'ttl_seconds': ttl_seconds})
    remote.wait(op['id'])
    print(f'renewed {lease_id}')
    return 0


def administer(config, action, payload):
    remote = client(config)
    op = remote.request('POST', '/v1/' + action, payload=payload)
    op = remote.wait(op['id'])
    print(json.dumps(op['result'], indent=2) if getattr(config, 'json', False) else f'{action}: {op["state"]}')
    return 3 if op['result'].get('publication_pending') else 0


def evict(config, *, names):
    if config.all:
        targets = None
    else:
        deployments = client(config).request('GET', '/v1/deployments')
        wanted = set(names)
        targets = [g['id'] for g in deployments if g['state'] == 'idle' and wanted & {g['id'], *g['endpoints']}]
        matched = {name for g in deployments if g['id'] in targets for name in {g['id'], *g['endpoints']}}
        if wanted - matched:
            raise SystemExit('evict: no idle remote deployment for: ' + ', '.join(sorted(wanted - matched)))
    return administer(config, 'evict', {'deployments': targets})


def run(config, *, names, ttl_seconds, command):
    import subprocess

    from ..leasing.envfile import descriptor_env

    descriptor, _op = acquire_descriptor(config, names=names, ttl_seconds=ttl_seconds, access=True)
    try:
        environment = dict(os.environ)
        environment.pop('INFER_STACK_LEASE_ID', None)
        environment.update(descriptor_env(descriptor))
        credential = environment.get(descriptor['api_key_env'])
        if credential is not None:
            environment['OPENAI_API_KEY'] = credential
        return int(subprocess.run(command, env=environment).returncode)
    finally:
        if descriptor['lease_id'] is not None:
            remote = client(config)
            op = remote.request('DELETE', '/v1/leases/' + descriptor['lease_id'])
            remote.wait(op['id'])


def release(config, *, resolve_lease):
    remote = client(config)
    if config.all:
        if config.lease or config.env_file:
            raise SystemExit('release: --all takes no lease/--env-file')
        targets = [le['id'] for le in remote.request('GET', '/v1/leases') if le['state'] == 'active']
    else:
        target = resolve_lease(config, 'release')
        if not target:
            raise SystemExit('release: give a lease id, --env-file, or --all')
        targets = [target]
    for lease_id in targets:
        suffix = '?evict=true' if config.evict else ''
        op = remote.request('DELETE', f'/v1/leases/{lease_id}{suffix}')
        remote.wait(op['id'])
    print(json.dumps({'released': targets}, indent=2) if config.json else f'released {len(targets)} lease(s)')
    return 0


def status(config):
    result = client(config).request('GET', '/v1/status')
    print(json.dumps(result, indent=2))
    return 0


def wait(config, *, names):
    remote = client(config)
    leases = [le for le in remote.request('GET', '/v1/leases') if le['state'] == 'active']
    requested = set(names) if names else {ep for le in leases for ep in le['endpoints']}
    held = {ep for le in leases for ep in le['endpoints']}
    if requested - held:
        raise SystemExit('Endpoints have no active remote lease: ' + ', '.join(sorted(requested - held)))
    results = []
    for lease in leases:
        selected = sorted(requested & set(lease['endpoints']))
        if not selected:
            continue
        op = remote.request('POST', '/v1/waits', payload={'lease_id': lease['id'], 'endpoints': selected,
                                                       'timeout': float(config.timeout)})
        results.append(remote.wait(op['id'], timeout=float(config.timeout), interval=float(config.interval)))
        requested.difference_update(selected)
    print(json.dumps(results, indent=2) if config.json else f'Generation readiness verified for {len(results)} remote lease(s)')
    return 0
