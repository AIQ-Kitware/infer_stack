"""Authenticated HTTP transport for the typed single-controller service."""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .auth import Identity, authenticate, identities
from .service import public_operation


class RateLimit:
    def __init__(self, per_minute=600, *, clock=time.monotonic, max_buckets=4096):
        self.rate, self.clock = per_minute, clock
        self.max_buckets = max_buckets
        self.buckets, self.lock = {}, threading.Lock()

    def allow(self, principal):
        if self.rate <= 0:
            return True
        with self.lock:
            now = self.clock()
            if principal not in self.buckets and len(self.buckets) >= self.max_buckets:
                # An idle bucket is fully replenished after a minute. Refuse
                # new identities when full rather than evicting an active
                # budget that an attacker could immediately reset.
                self.buckets = {key: value for key, value in self.buckets.items() if now - value[1] < 60}
                if len(self.buckets) >= self.max_buckets:
                    return False
            tokens, stamp = self.buckets.get(principal, (self.rate, now))
            tokens = min(self.rate, tokens + (now - stamp) * self.rate / 60)
            allowed = tokens >= 1
            self.buckets[principal] = (tokens - 1 if allowed else tokens, now)
            return allowed


def private_text(path: str | Path) -> str:
    file = Path(path).expanduser()
    with file.open() as handle:
        stat = os.fstat(handle.fileno())
        if stat.st_mode & 0o077 or stat.st_uid != os.getuid():
            raise ValueError('Control credential file must be private and owned by the service user')
        value = handle.read().strip()
    if not value:
        raise ValueError('Control credential file is empty')
    return value


class Handler(BaseHTTPRequestHandler):
    server: ControlHTTPServer
    server_version = 'infer-stack-control/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        pass  # URL/request strings are untrusted and may contain credentials

    def _send(self, status: int, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        if status == 429:
            import math
            self.send_header('Retry-After', str(max(1, math.ceil(60 / self.server.rate_limit.rate))))
        self.end_headers()
        self.wfile.write(data)

    def _authenticate(self) -> Identity | None:
        if not self.server.rate_limit.allow('peer:' + self.client_address[0]):
            self._send(429, {'error': 'Control request rate limit exceeded; retry later'})
            return None
        try:
            if self.server.identity_file is not None:
                registry = identities(private_text(self.server.identity_file))
                bearer = self.headers.get('Authorization', '')
                identity = authenticate(bearer.removeprefix('Bearer '), registry) if bearer.startswith('Bearer ') else None
            else:
                assert self.server.token_file is not None
                expected = 'Bearer ' + private_text(self.server.token_file)
                valid = hmac.compare_digest(self.headers.get('Authorization', '').encode(), expected.encode())
                identity = Identity(self.server.principal) if valid else None
        except (OSError, ValueError):
            self._send(503, {'error': 'Control credentials are unavailable'})
            return None
        if identity is None:
            self._send(401, {'error': 'Authentication required'})
            return None
        if not self.server.rate_limit.allow('principal:' + identity.principal):
            self._send(429, {'error': 'Control request rate limit exceeded; retry later'})
            return None
        return identity

    def _body(self):
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Content-Type must be application/json')
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 65536:
            raise ValueError('JSON body must contain 1..65536 bytes')
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError('Incomplete request body')
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError('JSON body must be an object')
        return payload

    def _dispatch(self):
        identity = self._authenticate()
        if identity is None:
            return
        service, principal = self.server.service, identity.principal
        path = urlsplit(self.path).path
        key = self.headers.get('Idempotency-Key', '')
        if self.command != 'GET' and not identity.can_lease:
            raise PermissionError('Read-only identity')
        if self.command == 'POST' and path in {'/v1/reconcile', '/v1/gc', '/v1/evict'} and not identity.operator:
            raise PermissionError('Operator capability required')
        if self.command == 'POST' and path == '/v1/leases':
            self._send(202, service.acquire(principal, key, self._body(), identity=identity))
        elif self.command == 'POST' and path == '/v1/access':
            self._send(202, service.access(principal, key, self._body(), identity=identity))
        elif self.command == 'POST' and path == '/v1/waits':
            self._send(202, service.wait(principal, key, self._body(), operator=identity.operator))
        elif self.command == 'POST' and path == '/v1/renewals':
            payload = self._body()
            ttl = payload.get('ttl_seconds')
            if identity.max_ttl_seconds is not None and (ttl is None or not isinstance(ttl, (int, float)) or ttl > identity.max_ttl_seconds):
                raise PermissionError('Renewal TTL is outside this identity policy')
            self._send(202, service.renew(principal, key, payload, operator=identity.operator))
        elif self.command == 'POST' and path in {'/v1/reconcile', '/v1/gc', '/v1/evict'}:
            self._send(202, service.administer(principal, key, path.removeprefix('/v1/'), self._body()))
        elif self.command == 'DELETE' and path.startswith('/v1/leases/'):
            query = parse_qs(urlsplit(self.path).query)
            if set(query) - {'evict'} or query.get('evict', ['false']) not in [['false'], ['true']]:
                raise ValueError('Release accepts only evict=true or evict=false')
            self._send(202, service.release(principal, key, path.removeprefix('/v1/leases/'), operator=identity.operator,
                                           evict=query.get('evict') == ['true']))
        elif self.command == 'GET' and path in {'/v1/status', '/v1/leases', '/v1/deployments'}:
            status = self._status(identity)
            self._send(200, status if path == '/v1/status' else status[path.removeprefix('/v1/')])
        elif self.command == 'GET' and path == '/v1/catalog':
            catalog = service.catalog
            self._send(200, [{'alias': name, 'protocol': catalog.resolve_endpoint(name).protocol}
                             for name in sorted(catalog.endpoints)
                             if identity.allowed_endpoints is None or name in identity.allowed_endpoints])
        elif self.command == 'GET' and path in {'/v1/nodes', '/v1/audit'}:
            query = parse_qs(urlsplit(self.path).query)
            if path == '/v1/nodes':
                if query:
                    raise ValueError('Node inventory accepts no query options')
                result = service.nodes()
            else:
                if set(query) - {'after', 'limit'} or any(len(v) != 1 for v in query.values()):
                    raise ValueError('Audit accepts after and limit only')
                result = service.operations.audit(principal=principal if identity.role == 'lease-user' else None,
                                                  after=int(query.get('after', ['0'])[0]), limit=int(query.get('limit', ['100'])[0]))
            self._send(200, result)
        elif self.command == 'GET' and path == '/v1/usage':
            rows = service.usage()
            query = parse_qs(urlsplit(self.path).query)
            if set(query) - {'deployment', 'node'} or any(len(values) != 1 for values in query.values()):
                raise ValueError('Usage accepts deployment and node filters only')
            if identity.role == 'lease-user':
                held = {gid for le in self._status(identity)['leases'] for gid in le['deployment_ids']}
                rows = [row for row in rows if row['facts']['deployment'] in held]
            rows = [row for row in rows if all(row['facts'][field] == values[0] for field, values in query.items())]
            self._send(200, rows)
        elif self.command == 'GET' and path.startswith('/v1/operations/'):
            op = service.operations.get(path.removeprefix('/v1/operations/'))
            if op is None:
                raise KeyError(path)
            if identity.role == 'lease-user' and op['principal'] != principal:
                raise PermissionError('Operation belongs to another principal')
            self._send(200, public_operation(op))
        elif self.command == 'GET' and path == '/v1/events':
            # Reconnecting clients receive a current snapshot, so a disconnect
            # cannot lose an operation result. Logs are not used as protocol.
            data = ('event: status\ndata: ' + json.dumps(self._status(identity)) + '\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send(404, {'error': 'Unknown control resource'})

    def _status(self, identity):
        status = self.server.service.status()
        status['identity'] = {'principal': identity.principal, 'role': identity.role,
                              'max_ttl_seconds': identity.max_ttl_seconds,
                              'max_active_leases': identity.max_active_leases}
        if identity.role == 'lease-user':
            status['leases'] = [le for le in status['leases'] if le['owner'] == identity.principal]
            held = {gid for le in status['leases'] for gid in le['deployment_ids']}
            status['deployments'] = [g for g in status['deployments'] if g['id'] in held]
            status['operations'] = [op for op in status['operations'] if op['principal'] == identity.principal]
        return status

    def _handle(self):
        try:
            self._dispatch()
        except PermissionError:
            self._send(403, {'error': 'Permission denied'})
        except KeyError:
            self._send(404, {'error': 'Unknown control resource'})
        except (ValueError, TypeError) as ex:
            self._send(400, {'error': str(ex)})
        except NotImplementedError:
            self._send(501, {'error': 'This backend does not support the requested observation'})
        except Exception:
            from .._log import logger
            logger.exception('Control request failed')
            self._send(503, {'error': 'Control state is unavailable'})

    do_GET = do_POST = do_DELETE = _handle


class ControlHTTPServer(ThreadingHTTPServer):
    # server_close waits for accepted requests before releasing DB authority.
    # Socket/handshake timeouts bound clients that stop sending data.
    daemon_threads = False

    def __init__(self, service, *, token_file: str | Path | None = None, identity_file: str | Path | None = None, principal='operator',
                 host='127.0.0.1', port=14043, tls_cert=None, tls_key=None, max_connections=32, requests_per_minute=600):
        if not 1 <= max_connections <= 1024 or requests_per_minute < 0:
            raise ValueError('Connection limit must be 1..1024 and request rate nonnegative')
        self.connections = threading.BoundedSemaphore(max_connections)
        self.rate_limit = RateLimit(requests_per_minute)
        if bool(token_file) == bool(identity_file):
            raise ValueError('Provide exactly one token file or identity file')
        if identity_file:
            identities(private_text(identity_file))
        else:
            assert token_file is not None
            private_text(token_file)
        try:
            address = ipaddress.ip_address(host)
            loopback = address.is_loopback
            if address.version == 6:
                self.address_family = socket.AF_INET6
        except ValueError:
            loopback = host == 'localhost'
        if bool(tls_cert) != bool(tls_key):
            raise ValueError('Provide both TLS certificate and private key')
        if not loopback and not tls_cert:
            raise ValueError('Off-host control listeners require authenticated TLS')
        self.service, self.token_file, self.identity_file, self.principal = service, token_file, identity_file, principal
        context = None
        if tls_cert:
            assert tls_key is not None
            private_text(tls_key)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(tls_cert, tls_key)
        super().__init__((host, port), Handler)
        self.tls_context = context

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        if self.tls_context:
            # Handshake on the request thread, not inside accept(): a peer
            # that sends no ClientHello must not stop the whole listener.
            connection = self.tls_context.wrap_socket(connection, server_side=True, do_handshake_on_connect=False)
        return connection, address

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()
