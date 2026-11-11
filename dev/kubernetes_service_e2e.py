#!/usr/bin/env python3
"""Remote-service contract against a quiescent real K3s/KubeAI namespace.

Install the KubeAI chart with a CPU/mock or real serving resource profile first.
Run with PYTHONPATH=. python dev/kubernetes_service_e2e.py --kubeconfig ...
--namespace infer-stack-test-service --base-url http://.../openai/v1
--resource-profile cpu. No host installation or cluster-wide cleanup occurs.
The dedicated namespace must initially contain no infer-stack-managed Models.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import requests
import yaml

from infer_stack.control.client import Client
from infer_stack.control.http import private_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kubeconfig', required=True)
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--resource-profile', default='cpu')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--postgres-dsn-file', help='Private DSN for a disposable PostgreSQL authority; otherwise SQLite.')
    parser.add_argument('--check-scaling', action='store_true', help='Sustain concurrent generation and require two ready cold replicas; use a mock with request latency.')
    args = parser.parse_args()
    kubeconfig = str(Path(args.kubeconfig).resolve())
    def kubectl(*values):
        return subprocess.check_output(['kubectl', '--kubeconfig', kubeconfig, '-n', args.namespace, *values], text=True)
    def models():
        return json.loads(kubectl('get', 'models.kubeai.org', '-l', 'infer-stack/managed=true', '-o', 'json'))['items']
    if models():
        raise SystemExit('Use a quiescent dedicated namespace: managed Models already exist')
    work = Path(tempfile.mkdtemp(prefix='infer-stack-service-e2e-'))
    config, data = work / 'config', work / 'data'
    config.mkdir()
    token = work / 'control.token'
    token.write_text(uuid.uuid4().hex + uuid.uuid4().hex)
    token.chmod(0o600)
    settings = {'backend': 'kubeai', 'kubeai_namespace': args.namespace,
                'kubeai_base_url': args.base_url, 'kubeconfig': kubeconfig, 'litellm': False, 'ui': False}
    (config / 'settings.yaml').write_text(yaml.safe_dump(settings))
    catalog = {'models': {'cold': {'source': 'hf://org/mock-cold'}, 'warm': {'source': 'hf://org/mock-warm'}},
               'endpoints': {
                   'cold': {'engine': 'vllm', 'model': 'cold', 'reclaim': {'policy': 'keep-warm'},
                            'runtime': {'resource_profile': args.resource_profile, 'min_replicas': 0, 'max_replicas': 2, 'target_requests': 1, 'scale_down_delay_seconds': 1}},
                   'warm': {'engine': 'vllm', 'model': 'warm', 'reclaim': {'policy': 'stop'},
                            'runtime': {'resource_profile': args.resource_profile, 'min_replicas': 1, 'max_replicas': 1}}},
               'bundles': {'pair': ['cold', 'warm']}}
    (config / 'catalog.yaml').write_text(yaml.safe_dump(catalog))
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    environment = {**os.environ, 'INFER_STACK_CONFIG_DIR': str(config), 'INFER_STACK_DATA_DIR': str(data)}
    process = None
    postgres = schema = None
    log = (work / 'server.log').open('a')
    remote = Client(f'http://127.0.0.1:{port}', str(token))
    def start():
        nonlocal process
        process = subprocess.Popen([sys.executable, '-m', 'infer_stack', 'server', '--port', str(port), '--token-file', str(token)],
                                   env=environment, stdout=log, stderr=log)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                remote.request('GET', '/v1/status')
                return
            except (ValueError, TimeoutError):
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited; inspect {work / "server.log"}')
                time.sleep(.2)
        raise TimeoutError('Server did not start')
    def stop():
        nonlocal process
        if process is not None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=90)
            process = None
    def mutate(method, path, payload=None, key=None):
        op = remote.request(method, path, payload=payload, key=key)
        return remote.wait(op['id'], timeout=args.timeout, interval=.2)
    def generate(result, endpoint):
        response = requests.post(result['base_url'] + '/chat/completions',
                                 json={'model': result['models'][endpoint], 'max_tokens': 2,
                                       'messages': [{'role': 'user', 'content': 'say ok'}]}, timeout=60)
        response.raise_for_status()
        assert response.json()['choices']
    def eventually(predicate, description):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(1)
        raise AssertionError(description)
    try:
        print(f'Artifacts: {work}', flush=True)
        if args.postgres_dsn_file:
            import psycopg
            from psycopg import sql
            from psycopg.conninfo import conninfo_to_dict, make_conninfo

            dsn = private_text(args.postgres_dsn_file)
            postgres = psycopg.connect(dsn, autocommit=True, connect_timeout=10)
            schema = 'infer_stack_e2e_' + uuid.uuid4().hex
            postgres.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            options = conninfo_to_dict(dsn).get('options', '') + ' -csearch_path=' + schema
            private_dsn = work / 'postgres.dsn'
            private_dsn.write_text(make_conninfo(dsn, options=options))
            private_dsn.chmod(0o600)
            settings['postgres_dsn_file'] = str(private_dsn)
            (config / 'settings.yaml').write_text(yaml.safe_dump(settings))
        start()
        inventory = remote.request('GET', '/v1/nodes')
        assert inventory['source'] == 'kubernetes' and inventory['nodes']
        payload = {'endpoints': ['pair'], 'ttl_seconds': 300, 'timeout': args.timeout}
        acquired = mutate('POST', '/v1/leases', payload, 'pair-key')
        assert acquired['state'] == 'succeeded'
        assert len(models()) == 2
        generate(acquired['result'], 'cold')
        generate(acquired['result'], 'warm')
        audit = remote.request('GET', '/v1/audit')
        assert audit and audit[0]['operation_id'] == acquired['id']
        if args.check_scaling:
            finished = threading.Event()
            def load():
                count = 0
                while not finished.is_set():
                    generate(acquired['result'], 'cold')
                    count += 1
                return count
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                requests_in_flight = [pool.submit(load) for _ in range(8)]
                try:
                    eventually(lambda: any(m['metadata']['name'] == 'cold' and m.get('status', {}).get('replicas', {}).get('ready') == 2 for m in models()),
                               'sustained generation did not create two ready replicas')
                finally:
                    finished.set()
                assert sum(task.result() for task in requests_in_flight) > 0
        stop()
        generate(acquired['result'], 'warm')  # inference survives controller loss
        start()
        assert remote.request('GET', '/v1/audit')[:len(audit)] == audit
        assert mutate('POST', '/v1/leases', payload, 'pair-key')['id'] == acquired['id']
        assert len(remote.request('GET', '/v1/leases')) == 1
        mutate('DELETE', '/v1/leases/' + acquired['result']['lease_id'])
        eventually(lambda: len(models()) == 1, 'stop-policy Model was not pruned')
        eventually(lambda: models()[0].get('status', {}).get('replicas', {}).get('all') == 0,
                   'on-demand Model did not scale back to zero')
        mutate('POST', '/v1/gc', {})
        assert len(models()) == 1  # zero replicas retain logical presence
        reacquired = mutate('POST', '/v1/leases', {'endpoints': ['cold'], 'ttl_seconds': 300, 'timeout': args.timeout})
        generate(reacquired['result'], 'cold')  # an actual request wakes zero
        mutate('DELETE', '/v1/leases/' + reacquired['result']['lease_id'] + '?evict=true')
        eventually(lambda: not models(), 'eviction did not remove the logical Model')
        (work / 'result.json').write_text(json.dumps({'passed': True, 'namespace': args.namespace,
            'checks': ['node inventory', 'audit recovery', 'bundle generation', 'controller-loss inference', 'restart idempotency', 'stop pruning',
                       'scale to zero', 'zero-replica gc preservation', 'wake from zero', 'explicit eviction']
                      + (['scale up under concurrent generation'] if args.check_scaling else [])}, indent=2))
        print('PASS: remote service and real KubeAI lifecycle', flush=True)
    finally:
        if process is not None and process.poll() is None:
            try:
                for lease in remote.request('GET', '/v1/leases'):
                    if lease['state'] == 'active':
                        mutate('DELETE', '/v1/leases/' + lease['id'] + '?evict=true')
                mutate('POST', '/v1/evict', {'deployments': None})
            finally:
                stop()
        log.close()
        if postgres is not None:
            try:
                if schema is not None:
                    from psycopg import sql
                    postgres.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
            finally:
                postgres.close()


if __name__ == '__main__':
    main()
