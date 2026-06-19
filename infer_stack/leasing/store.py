"""SQLite-backed persistence for the leasing ledger.

A single sqlite database is the shared store that lets multiple processes /
users coordinate (the redesign replaces the old "everyone re-renders the same
compose file, last render wins" pattern). SQLite is chosen over a file-of-JSON
because it gives atomic reference-count updates and a real write lock for the
read-modify-write coalescing critical section, with zero extra dependencies.

This module is intentionally *low level*: methods execute statements and map
rows to dataclasses, but the multi-step invariants (coalescing, demand-driven
state transitions) live in :mod:`infer_stack.leasing.ledger`, which wraps the
relevant calls in :meth:`SqliteStore.transaction`.

Concurrency model:

* ``isolation_level=None`` -> autocommit; transactions are explicit.
* WAL journal + ``busy_timeout`` so readers don't block writers and a contended
  writer waits rather than failing immediately.
* The coalescing critical section uses ``BEGIN IMMEDIATE`` (via
  :meth:`transaction`) to take the write lock *before* reading, so two
  concurrent ``acquire`` calls cannot both decide "no group exists" and create
  duplicates.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

from .models import DeploymentGroup, Lease, LeaseState

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leases (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    ttl_seconds REAL,
    expires_at REAL,
    heartbeat_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS groups (
    id TEXT PRIMARY KEY,
    compat_key TEXT NOT NULL,
    engine TEXT NOT NULL,
    sharing TEXT NOT NULL,
    capacity TEXT NOT NULL DEFAULT '{}',
    spec TEXT NOT NULL DEFAULT '{}',
    served TEXT NOT NULL DEFAULT '{}',
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id TEXT NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
    endpoint TEXT NOT NULL,
    group_id TEXT NOT NULL REFERENCES groups(id),
    kind TEXT NOT NULL DEFAULT 'endpoint'
);

CREATE INDEX IF NOT EXISTS idx_claims_lease ON claims(lease_id);
CREATE INDEX IF NOT EXISTS idx_claims_group ON claims(group_id);
CREATE INDEX IF NOT EXISTS idx_groups_compat ON groups(compat_key);
"""


def _loads(text: str) -> Any:
    return json.loads(text) if text else None


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(',', ':'), sort_keys=True)


class SqliteStore:
    """Thin sqlite wrapper exposing ledger row operations."""

    def __init__(self, path: str | Path = ':memory:', *, busy_timeout_ms: int = 5000):
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets a long-running process (e.g. the TUI) use
        # this connection from a worker thread for converge-while-monitoring;
        # ``_lock`` serializes write transactions so two threads can't both
        # ``BEGIN IMMEDIATE`` on the one connection. sqlite itself is built
        # serialized, so individual reads across threads are safe.
        self._conn = sqlite3.connect(
            self.path, isolation_level=None, timeout=busy_timeout_ms / 1000,
            check_same_thread=False,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA foreign_keys = ON')
        self._conn.execute(f'PRAGMA busy_timeout = {busy_timeout_ms}')
        if self.path != ':memory:':
            self._conn.execute('PRAGMA journal_mode = WAL')
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self._conn.close()

    # -- transactions ------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Take the write lock up front and commit/rollback atomically.

        ``BEGIN IMMEDIATE`` is what makes the ledger's find-or-create-group
        step race-safe across processes. ``_lock`` adds the same guarantee
        across threads in one process (overlapping ``BEGIN IMMEDIATE`` on a
        shared connection would otherwise raise).
        """
        with self._lock:
            self._conn.execute('BEGIN IMMEDIATE')
            try:
                yield self._conn
                self._conn.execute('COMMIT')
            except Exception:
                self._conn.execute('ROLLBACK')
                raise

    # -- leases ------------------------------------------------------------

    def insert_lease(
        self,
        *,
        lease_id: str,
        owner: str,
        created_at: float,
        ttl_seconds: float | None,
        expires_at: float | None,
        heartbeat_at: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            'INSERT INTO leases(id, owner, state, created_at, ttl_seconds,'
            ' expires_at, heartbeat_at, metadata)'
            ' VALUES(?, ?, ?, ?, ?, ?, ?, ?)',
            (
                lease_id,
                owner,
                LeaseState.ACTIVE,
                created_at,
                ttl_seconds,
                expires_at,
                heartbeat_at,
                _dumps(metadata or {}),
            ),
        )

    def set_lease_state(self, lease_id: str, state: str) -> None:
        self._conn.execute(
            'UPDATE leases SET state = ? WHERE id = ?', (state, lease_id)
        )

    def renew_lease(
        self,
        lease_id: str,
        *,
        ttl_seconds: float | None,
        expires_at: float | None,
        heartbeat_at: float,
    ) -> None:
        self._conn.execute(
            'UPDATE leases SET state = ?, ttl_seconds = ?, expires_at = ?,'
            ' heartbeat_at = ? WHERE id = ?',
            (
                LeaseState.ACTIVE,
                ttl_seconds,
                expires_at,
                heartbeat_at,
                lease_id,
            ),
        )

    def get_lease(self, lease_id: str) -> Lease | None:
        row = self._conn.execute(
            'SELECT * FROM leases WHERE id = ?', (lease_id,)
        ).fetchone()
        return self._row_to_lease(row) if row else None

    def list_leases(self, *, states: tuple[str, ...] | None = None) -> list[Lease]:
        if states:
            placeholders = ','.join('?' for _ in states)
            rows = self._conn.execute(
                f'SELECT * FROM leases WHERE state IN ({placeholders})'
                ' ORDER BY created_at',
                states,
            ).fetchall()
        else:
            rows = self._conn.execute(
                'SELECT * FROM leases ORDER BY created_at'
            ).fetchall()
        return [self._row_to_lease(r) for r in rows]

    def prune(
        self,
        *,
        lease_states: tuple[str, ...] = (),
        group_states: tuple[str, ...] = (),
    ) -> tuple[int, int]:
        """Delete terminal leases/groups (and their claims) from the ledger.

        Claims are removed first so the ``groups.id`` foreign key can't block a
        group deletion; deleting leases also cascades their claims. Returns
        ``(n_leases_deleted, n_groups_deleted)``.
        """
        n_leases = n_groups = 0
        with self.transaction() as conn:
            if lease_states:
                lq = ','.join('?' for _ in lease_states)
                conn.execute(
                    f'DELETE FROM claims WHERE lease_id IN '
                    f'(SELECT id FROM leases WHERE state IN ({lq}))',
                    lease_states,
                )
            if group_states:
                gq = ','.join('?' for _ in group_states)
                conn.execute(
                    f'DELETE FROM claims WHERE group_id IN '
                    f'(SELECT id FROM groups WHERE state IN ({gq}))',
                    group_states,
                )
            if lease_states:
                lq = ','.join('?' for _ in lease_states)
                n_leases = conn.execute(
                    f'DELETE FROM leases WHERE state IN ({lq})', lease_states
                ).rowcount
            if group_states:
                gq = ','.join('?' for _ in group_states)
                n_groups = conn.execute(
                    f'DELETE FROM groups WHERE state IN ({gq})', group_states
                ).rowcount
        return n_leases, n_groups

    def active_leases_past(self, now: float) -> list[Lease]:
        """Active leases whose TTL has elapsed (candidates for expiry)."""
        rows = self._conn.execute(
            'SELECT * FROM leases WHERE state = ? AND expires_at IS NOT NULL'
            ' AND expires_at <= ?',
            (LeaseState.ACTIVE, now),
        ).fetchall()
        return [self._row_to_lease(r) for r in rows]

    def _row_to_lease(self, row: sqlite3.Row) -> Lease:
        claims = self._conn.execute(
            'SELECT endpoint, group_id FROM claims WHERE lease_id = ?'
            ' ORDER BY id',
            (row['id'],),
        ).fetchall()
        return Lease(
            id=row['id'],
            owner=row['owner'],
            state=row['state'],
            created_at=row['created_at'],
            ttl_seconds=row['ttl_seconds'],
            expires_at=row['expires_at'],
            heartbeat_at=row['heartbeat_at'],
            endpoints=[c['endpoint'] for c in claims],
            group_ids=list(dict.fromkeys(c['group_id'] for c in claims)),
        )

    # -- claims ------------------------------------------------------------

    def insert_claim(
        self, *, lease_id: str, endpoint: str, group_id: str, kind: str = 'endpoint'
    ) -> None:
        self._conn.execute(
            'INSERT INTO claims(lease_id, endpoint, group_id, kind)'
            ' VALUES(?, ?, ?, ?)',
            (lease_id, endpoint, group_id, kind),
        )

    # -- groups ------------------------------------------------------------

    def insert_group(self, group: DeploymentGroup) -> None:
        self._conn.execute(
            'INSERT INTO groups(id, compat_key, engine, sharing, capacity,'
            ' spec, served, state, created_at, updated_at)'
            ' VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                group.id,
                group.compat_key,
                group.engine,
                group.sharing,
                _dumps(group.capacity),
                _dumps(group.spec),
                _dumps(group.served),
                group.state,
                group.created_at,
                group.updated_at,
            ),
        )

    def set_group_state(self, group_id: str, state: str, updated_at: float) -> None:
        self._conn.execute(
            'UPDATE groups SET state = ?, updated_at = ? WHERE id = ?',
            (state, updated_at, group_id),
        )

    def update_group_served(
        self, group_id: str, served: dict[str, Any], updated_at: float
    ) -> None:
        self._conn.execute(
            'UPDATE groups SET served = ?, updated_at = ? WHERE id = ?',
            (_dumps(served), updated_at, group_id),
        )

    def get_group(self, group_id: str, *, now: float | None = None) -> DeploymentGroup | None:
        row = self._conn.execute(
            'SELECT * FROM groups WHERE id = ?', (group_id,)
        ).fetchone()
        if row is None:
            return None
        group = self._row_to_group(row)
        if now is not None:
            group.demand = self.demand(group_id, now)
        return group

    def groups_by_compat(
        self, compat_key: str, *, sharing: str, states: tuple[str, ...]
    ) -> list[DeploymentGroup]:
        placeholders = ','.join('?' for _ in states)
        rows = self._conn.execute(
            'SELECT * FROM groups WHERE compat_key = ? AND sharing = ?'
            f' AND state IN ({placeholders}) ORDER BY created_at',
            (compat_key, sharing, *states),
        ).fetchall()
        return [self._row_to_group(r) for r in rows]

    def list_groups(self, *, now: float) -> list[DeploymentGroup]:
        rows = self._conn.execute(
            'SELECT * FROM groups ORDER BY created_at'
        ).fetchall()
        groups = [self._row_to_group(r) for r in rows]
        for group in groups:
            group.demand = self.demand(group.id, now)
        return groups

    def groups_for_lease(self, lease_id: str) -> list[DeploymentGroup]:
        rows = self._conn.execute(
            'SELECT DISTINCT g.* FROM groups g JOIN claims c'
            ' ON c.group_id = g.id WHERE c.lease_id = ?',
            (lease_id,),
        ).fetchall()
        return [self._row_to_group(r) for r in rows]

    def demand(self, group_id: str, now: float) -> int:
        """Number of *protecting* leases referencing ``group_id``.

        A lease protects iff it is ACTIVE and not past its TTL, matching
        :meth:`infer_stack.leasing.models.Lease.is_protecting`.
        """
        row = self._conn.execute(
            'SELECT COUNT(DISTINCT c.lease_id) AS n FROM claims c'
            ' JOIN leases l ON c.lease_id = l.id'
            ' WHERE c.group_id = ? AND l.state = ?'
            ' AND (l.expires_at IS NULL OR l.expires_at > ?)',
            (group_id, LeaseState.ACTIVE, now),
        ).fetchone()
        return int(row['n'])

    def _row_to_group(self, row: sqlite3.Row) -> DeploymentGroup:
        return DeploymentGroup(
            id=row['id'],
            compat_key=row['compat_key'],
            engine=row['engine'],
            sharing=row['sharing'],
            capacity=_loads(row['capacity']) or {},
            spec=_loads(row['spec']) or {},
            served=_loads(row['served']) or {},
            state=row['state'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )
