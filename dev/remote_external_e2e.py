#!/usr/bin/env python3
"""Remote external access through real Docker/LiteLLM; no GPU or model weights.

Run PYTHONPATH=. python dev/remote_external_e2e.py. Requires Docker Compose and
the repository's pinned LiteLLM image (pull it beforehand). Uses a unique
Compose project, temporary authority, and simulator bound to Docker's bridge.
Only this project's resources are removed. Private artifacts remain in /tmp.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import requests

from infer_stack.cli import AccessCLI, RunCLI
from infer_stack.control.client import Client
from infer_stack.control.http import ControlHTTPServer
from infer_stack.control.service import ControlService
from infer_stack.env_utils import write_env_file
from infer_stack.leasing import Catalog, Controller, Ledger, SqliteStore
from infer_stack.leasing.compose import ComposeBackend
from infer_stack.mockserver.server import MockServer


def main():
    bridge = json.loads(subprocess.check_output(['docker', 'network', 'inspect', 'bridge'], text=True))[0]['IPAM']['Config'][0]['Gateway']
    work = Path(tempfile.mkdtemp(prefix='infer-stack-remote-external-'))
    print(f'Artifacts: {work}', flush=True)
    project = 'infer-stack-remote-e2e-' + uuid.uuid4().hex[:12]
    token = work / 'control.token'
    token.write_text(secrets.token_urlsafe(32))
    token.chmod(0o600)
    upstream_key = secrets.token_urlsafe(32)
    mock = MockServer({'models': {'org/mock': {'ability': .9}}, 'require_auth': True, 'api_keys': [upstream_key]}, host=bridge).start()
    approved = Catalog.from_dict({'endpoints': {'remote': {'external': {
        'api_base': mock.url + '/v1', 'model': 'org/mock', 'api_key_env': 'E2E_UPSTREAM_KEY'}}}})
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        gateway_port = reserve.getsockname()[1]
    backend = ComposeBackend(state_dir=work / 'compose', project=project,
                             inventory={'gpu_count': 0, 'gpus': []}, catalog=approved,
                             ui=False, ports={'litellm': gateway_port})
    write_env_file(backend.gateway.env_path, {'E2E_UPSTREAM_KEY': upstream_key})
    server = thread = worker = store = None
    authority, stopped = contextlib.ExitStack(), threading.Event()

    def start():
        nonlocal server, thread, worker, store, authority, stopped
        store = SqliteStore(work / 'ledger.db')
        authority = contextlib.ExitStack()
        authority.enter_context(store.service_lock())
        service = ControlService(Controller(Ledger(store), backend), approved)
        server = ControlHTTPServer(service, token_file=token, port=0)
        stopped = threading.Event()
        def step():
            while not stopped.wait(.1):
                service.step()
        worker = threading.Thread(target=step, daemon=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        thread.start()
        return Client(f'http://127.0.0.1:{server.server_port}', str(token))

    def stop():
        nonlocal server, store
        stopped.set()
        if server is not None:
            server.shutdown()
            server.server_close()
            thread.join(5)
            server = None
        if worker is not None:
            while worker.is_alive():
                worker.join(1)
        authority.close()
        if store is not None:
            store.close()
            store = None

    def generate(descriptor):
        response = requests.post(descriptor['base_url'] + '/chat/completions',
                                 headers={'Authorization': 'Bearer ' + backend.gateway.master_key()},
                                 json={'model': descriptor['endpoints']['remote'], 'max_tokens': 2,
                                       'messages': [{'role': 'user', 'content': 'say ok'}]}, timeout=30)
        response.raise_for_status()
        assert response.json()['choices']
        sent = next(record for record in reversed(mock.recorded) if record['path'].endswith('/chat/completions'))
        headers = {k.lower(): v for k, v in sent['headers'].items()}
        assert headers['authorization'] == 'Bearer ' + upstream_key

    try:
        remote = start()
        env = work / 'access.env'
        flags = dict(argv=False, server_url=remote.url, server_token_file=str(token))
        AccessCLI.main(**flags, names=['remote'], timeout=180, interval=.2,
                       idempotency_key='external-key', env_file=str(env))
        op = remote.request('GET', '/v1/status')['operations'][0]
        descriptor = {'lease_id': op['result']['lease_id'], 'base_url': op['result']['base_url'],
                      'endpoints': op['result']['models']}
        assert descriptor['lease_id'] is None and not mock.recorded
        assert 'INFER_STACK_LEASE_ID' not in env.read_text()
        assert upstream_key not in env.read_text() and token.read_text() not in env.read_text()
        generate(descriptor)
        stop()
        generate(descriptor)  # actual gateway/inference survives control loss
        remote = start()
        flags['server_url'] = remote.url
        AccessCLI.main(**flags, names=['remote'], timeout=180, interval=.2,
                       idempotency_key='external-key', env_file=str(env))
        assert len(remote.request('GET', '/v1/status')['operations']) == 1
        assert remote.request('GET', '/v1/leases') == []
        saved_key = os.environ.get('INFER_STACK_INFERENCE_KEY')
        os.environ['INFER_STACK_INFERENCE_KEY'] = backend.gateway.master_key()
        try:
            script = 'import os, requests; r=requests.post(os.environ["OPENAI_BASE_URL"]+"/chat/completions", headers={"Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]}, json={"model":os.environ["INFER_STACK_ENDPOINT_REMOTE"],"messages":[{"role":"user","content":"hi"}],"max_tokens":2}); r.raise_for_status(); raise SystemExit(7)'
            assert RunCLI.main(**flags, endpoint='remote', timeout=180, interval=.2,
                               command=[sys.executable, '-c', script]) == 7
        finally:
            if saved_key is None:
                os.environ.pop('INFER_STACK_INFERENCE_KEY', None)
            else:
                os.environ['INFER_STACK_INFERENCE_KEY'] = saved_key
        assert remote.request('GET', '/v1/leases') == []
        (work / 'result.json').write_text(json.dumps({'passed': True, 'project': project,
            'checks': ['external publication without generation', 'no lease or credential values in descriptor',
                       'actual alias routing and upstream credential', 'controller-loss inference',
                       'restart idempotency', 'remote run with no lease and preserved exit status']}, indent=2))
        print('PASS: remote external access through real LiteLLM', flush=True)
    finally:
        stop()
        try:
            if backend.compose_file.exists():
                backend.down()
        finally:
            mock.stop()


if __name__ == '__main__':
    main()
