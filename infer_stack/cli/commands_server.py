"""Run one canonical controller service with SQLite or PostgreSQL authority."""
from __future__ import annotations

import contextlib
import threading
import time

import kwconf as kw

from ..control.http import ControlHTTPServer
from ..control.service import ControlService
from .commands_leasing import (
    _LeasingCommonMixin,
    _load_catalog,
    _open_controller,
)
from .options import _PathOverridesMixin
from .remote import RemoteOptions, client, server_url


class OperationsCLI(RemoteOptions, _PathOverridesMixin):
    """Recover a durable remote operation or list the visible operations."""

    __command__ = 'operations'
    operation = kw.Value(None, position=1, type=str)
    audit = kw.Value(False, isflag=True, help='Show audited operation transitions instead of operations.')
    after = kw.Value(0, type=int, help='Audit cursor: return records after this sequence ID.')
    limit = kw.Value(100, type=int, help='Maximum audit records (1..1000).')

    @classmethod
    def main(cls, argv=True, **kwargs):
        import json

        config = cls.cli(argv=argv, data=kwargs)
        if not server_url(config):
            raise SystemExit('operations requires a configured control server')
        remote = client(config)
        if config.audit:
            from urllib.parse import urlencode
            if config.operation:
                raise SystemExit('operations: --audit takes no operation ID')
            result = remote.request('GET', '/v1/audit?' + urlencode({'after': config.after, 'limit': config.limit}))
        else:
            result = remote.request('GET', '/v1/operations/' + config.operation) if config.operation else remote.request('GET', '/v1/status')['operations']
        print(json.dumps(result, indent=2))
        return 0


class ServerCLI(_LeasingCommonMixin):
    """Serve typed catalog operations; remote binding requires TLS and a private token.

    This server supports one worker authority. Ordinary local CLI commands share
    its ledger/publication lock. A second server on the same ledger is refused.
    PostgreSQL uses database locks for the same single-authority boundary.
    HA replicas require destination-side fencing before they can be enabled.
    """

    __command__ = 'server'
    catalog = kw.Value(None, type=str, help='Administrator-approved catalog path.')
    host = kw.Value('127.0.0.1', type=str)
    port = kw.Value(14043, type=int)
    token_file = kw.Value(None, type=str, help='Private service-user-owned bearer token file; reloaded for live rotation.')
    identity_file = kw.Value(None, type=str, help='Alternative private JSON identity/role registry with token SHA256 digests.')
    principal = kw.Value('operator', type=str, help='Audit/lease owner represented by this shared operator credential.')
    tls_cert = kw.Value(None, type=str, help='TLS certificate PEM; required for off-host binding.')
    tls_key = kw.Value(None, type=str, help='Private service-user-owned TLS key PEM.')
    postgres_dsn_file = kw.Value(None, type=str, help='Private PostgreSQL DSN file; otherwise local SQLite. Requires infer-stack[postgres].')
    max_connections = kw.Value(32, type=int, help='Bound concurrent control connections, including TLS handshakes.')
    requests_per_minute = kw.Value(600, type=int, help='Request budget per peer and principal; zero disables rate limiting.')

    @classmethod
    def main(cls, argv=True, **kwargs):
        from .._log import logger
        from ..paths import data_root

        config = cls.cli(argv=argv, data=kwargs)
        controller = _open_controller(config, interactive=False)
        store = controller.ledger.store
        server = worker = None
        authority = contextlib.ExitStack()
        stop = threading.Event()
        try:
            authority.enter_context(store.service_lock())
            service = ControlService(controller, _load_catalog(config), usage_path=data_root() / 'usage' / 'records.sqlite3')
            server = ControlHTTPServer(service, token_file=config.token_file, identity_file=config.identity_file, principal=config.principal,
                                       host=config.host, port=config.port, tls_cert=config.tls_cert, tls_key=config.tls_key,
                                       max_connections=config.max_connections, requests_per_minute=config.requests_per_minute)
            def reconcile():
                last_maintenance = float('-inf')
                while not stop.is_set():
                    try:
                        if time.monotonic() - last_maintenance >= 15:
                            last_maintenance = time.monotonic()
                            service.maintain()
                        service.step()
                    except Exception:
                        logger.exception('Control worker will retry after reconciliation failure')
                    stop.wait(1)
            worker = threading.Thread(target=reconcile, name='infer-stack-control-worker', daemon=True)
            worker.start()
            scheme = 'https' if config.tls_cert else 'http'
            print(f'Control service: {scheme}://{config.host}:{server.server_address[1]} (principal={config.principal})', flush=True)
            server.serve_forever(poll_interval=.5)
        except KeyboardInterrupt:
            return 0
        except (OSError, ValueError, RuntimeError) as ex:
            raise SystemExit(f'server: {ex}') from ex
        finally:
            stop.set()
            if server is not None:
                server.server_close()
            if worker is not None:
                # Backend commands have their own time bounds. Retain authority
                # until the last mutation exits, including during shutdown.
                while worker.is_alive():
                    worker.join(timeout=1)
            store.close()
            authority.close()
