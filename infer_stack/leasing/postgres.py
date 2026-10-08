"""PostgreSQL lease authority with serialized transactions and publication.

Install infer-stack[postgres]. One server owns the service lock; automatic HA
takeover is deliberately unavailable until backends fence destination writes.
Native PostgreSQL backups are managed by the database administrator.
"""
from __future__ import annotations

import contextlib
import ipaddress
import threading
from typing import Any

from .store import _SCHEMA, SCHEMA_VERSION, LedgerStore, _Result

WRITE_LOCK = 0x494E465253544B01
PUBLISH_LOCK = 0x494E465253544B02
SERVICE_LOCK = 0x494E465253544B03


class _Connection:
    """Materialize results under the lock, as the SQLite adapter does."""

    def __init__(self, raw, lock):
        self.raw, self.lock = raw, lock

    def execute(self, sql, params=()):
        # SQL templates belong to LedgerStore, never to a remote caller.
        with self.lock, self.raw.cursor() as cursor:
            cursor.execute(sql.replace('?', '%s'), params or None)
            return _Result(cursor.fetchall() if cursor.description else [], cursor.rowcount, None)

    def close(self):
        with self.lock:
            self.raw.close()


class PostgresStore(LedgerStore):
    def __init__(self, dsn: str):
        try:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict
            from psycopg.rows import dict_row
        except ImportError as ex:
            raise RuntimeError('PostgreSQL storage requires pip install "infer-stack[postgres]"') from ex
        options = conninfo_to_dict(dsn)
        hosts = str(options.get('hostaddr') or options.get('host') or '').split(',')
        def local(host):
            if not host or host.startswith('/') or host == 'localhost':
                return True
            try:
                return ipaddress.ip_address(host).is_loopback
            except ValueError:
                return False
        if not all(local(host) for host in hosts) and options.get('sslmode') != 'verify-full':
            raise ValueError('Off-host PostgreSQL requires sslmode=verify-full and a trusted CA')
        self._connect = lambda: psycopg.Connection[dict[str, Any]].connect(dsn, autocommit=True, row_factory=dict_row, connect_timeout=10)
        self._lock = threading.RLock()
        self._publication_tlock = threading.RLock()
        self._publication_depth = 0
        self._publication_connection = None
        self._service_connection = None
        self._conn = _Connection(self._connect(), self._lock)
        try:
            self._ensure_schema()
        except BaseException:
            self.close()
            raise

    def _ensure_schema(self):
        # Bootstrap and migrations use the same serialization point as writes,
        # including first-open races before the server lock is acquired.
        schema = _SCHEMA.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'BIGSERIAL PRIMARY KEY').replace(' REAL', ' DOUBLE PRECISION')
        with self.transaction() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is not None and int(row['value']) != SCHEMA_VERSION:
                raise ValueError('Unsupported PostgreSQL ledger schema version')
            conn.execute(schema)
            conn.execute('ALTER TABLE deployments ADD COLUMN IF NOT EXISTS assigned_gpus TEXT')
            conn.execute("INSERT INTO meta(key,value) VALUES ('schema_version',?) ON CONFLICT(key) DO NOTHING", (str(SCHEMA_VERSION),))

    @contextlib.contextmanager
    def transaction(self):
        with self._lock, self._conn.raw.transaction():
            self._conn.execute('SELECT pg_advisory_xact_lock(?)', (WRITE_LOCK,))
            yield self._conn

    @contextlib.contextmanager
    def service_lock(self):
        """Refuse a second service authority; loss of this session is fatal."""
        if self._service_connection is not None:
            raise RuntimeError('Service authority is already held')
        with self._connect() as connection:
            row = connection.execute('SELECT pg_try_advisory_lock(%s) AS acquired', (SERVICE_LOCK,)).fetchone()
            if row is None or not row['acquired']:
                raise ValueError('Another control server already owns this PostgreSQL authority')
            self._service_connection = connection
            try:
                yield
            finally:
                self._service_connection = None

    @contextlib.contextmanager
    def publication_lock(self):
        """Serialize complete publications across sessions, reentrant in a thread.

        This coordinates healthy sessions. It does not fence an external write
        already in flight after a database/network partition.
        """
        with self._publication_tlock:
            if self._service_connection is not None:
                self._service_connection.execute('SELECT 1')
            outer = self._publication_depth == 0
            if outer:
                connection = self._connect()
                try:
                    connection.execute('SELECT pg_advisory_lock(%s)', (PUBLISH_LOCK,))
                except BaseException:
                    connection.close()
                    raise
                self._publication_connection = connection
            self._publication_depth += 1
            try:
                yield
            finally:
                self._publication_depth -= 1
                if outer:
                    assert self._publication_connection is not None
                    self._publication_connection.close()
                    self._publication_connection = None

    def backup_to(self, path):
        raise ValueError('Use a native PostgreSQL backup; SQLite archive/rotation is unavailable for this store')
