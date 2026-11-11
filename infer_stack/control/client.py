"""Small authenticated control client; retries retain the mutation identity."""
from __future__ import annotations

import ipaddress
import time
import uuid
from urllib.parse import urlsplit

import requests

from .errors import ControlError, ControlTimeout
from .http import private_text


class Client:
    def __init__(self, url: str, token_file: str, *, ca_file=None, http=None):
        parsed = urlsplit(url)
        try:
            loopback = ipaddress.ip_address(parsed.hostname or '').is_loopback
        except ValueError:
            loopback = parsed.hostname == 'localhost'
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ControlError('Control URL must be an absolute HTTP(S) URL without credentials, query or fragment')
        if parsed.scheme != 'https' and not loopback:
            raise ControlError('Remote control clients require TLS')
        self.url, self.token_file, self.verify = url.rstrip('/'), token_file, ca_file or True
        self.http = http or requests.Session()

    def request(self, method: str, resource: str, *, payload=None, key=None):
        try:
            headers = {'Authorization': 'Bearer ' + private_text(self.token_file)}
        except (OSError, ValueError) as ex:
            raise ControlError('Client control credential file is unavailable or not private') from ex
        if method != 'GET':
            headers['Idempotency-Key'] = key or uuid.uuid4().hex
        response = None
        for attempt in range(3):
            try:
                response = self.http.request(method, self.url + resource, json=payload,
                                             headers=headers, timeout=(5, 15), verify=self.verify,
                                             allow_redirects=False)
                break
            except (requests.ConnectionError, requests.Timeout) as ex:
                if attempt == 2:
                    identity = headers.get('Idempotency-Key')
                    recovery = f'; retry the mutation with --idempotency-key={identity}' if identity else ''
                    raise ControlTimeout('Control server did not return a response' + recovery) from ex
        assert response is not None
        try:
            result = response.json()
        except ValueError as ex:
            raise ControlError('Control server returned invalid JSON') from ex
        if response.status_code >= 300:
            raise ControlError(f'Control API ({response.status_code}): {result.get("error", "request failed")}')
        return result

    def wait(self, operation_id: str, *, ready=True, timeout=600, interval=1):
        deadline = time.monotonic() + timeout
        while True:
            op = self.request('GET', f'/v1/operations/{operation_id}')
            if op['state'] == 'succeeded' or (not ready and op['state'] == 'converging'):
                return op
            if op['state'] in {'failed', 'superseded'}:
                raise ControlError(f"Operation {operation_id} {op['state']}: {(op.get('error') or {}).get('message', 'controller rejected the request')}")
            if time.monotonic() >= deadline:
                raise ControlTimeout(f'Operation {operation_id} is still {op["state"]}; query it again to recover the durable result')
            time.sleep(min(interval, max(0, deadline - time.monotonic())))
