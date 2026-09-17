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
  concurrent ``acquire`` calls cannot both decide "no deployment exists" and create
  duplicates.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator

from .models import Deployment, Lease, LeaseState

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

CREATE TABLE IF NOT EXISTS deployments (
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
    deployment_id TEXT NOT NULL REFERENCES deployments(id),
    kind TEXT NOT NULL DEFAULT 'endpoint'
);

-- Append-only: a service keeps its address; no other service ever receives it.
CREATE TABLE IF NOT EXISTS service_addresses (
    service TEXT PRIMARY KEY,
    ipv4 TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_claims_lease ON claims(lease_id);
CREATE INDEX IF NOT EXISTS idx_claims_deployment ON claims(deployment_id);
CREATE INDEX IF NOT EXISTS idx_deployments_compat ON deployments(compat_key);
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
        self._busy_timeout_ms = busy_timeout_ms
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA foreign_keys = ON')
        self._conn.execute(f'PRAGMA busy_timeout = {busy_timeout_ms}')
        if self.path != ':memory:':
            # Switching the journal to WAL needs a brief *exclusive* lock, and
            # sqlite returns "database is locked" immediately rather than honoring
            # busy_timeout for this pragma. Several processes opening the SAME
            # fresh ledger at once (e.g. a batch of pipeline jobs all calling
            # `infer-stack acquire`) therefore race here — so retry. Idempotent:
            # once it is WAL, re-running the pragma is a quick no-op.
            self._retry_locked(lambda: self._conn.execute('PRAGMA journal_mode = WAL'))
        self._ensure_schema()

    def _retry_locked(self, fn, *, attempts: int = 100, delay: float = 0.05):
        """Run ``fn``, retrying on a transient "database is locked" from a
        concurrent opener. Re-raises any other error (and the lock error if it
        never clears within ``attempts``)."""
        import time

        for _ in range(attempts - 1):
            try:
                return fn()
            except sqlite3.OperationalError as ex:
                if 'locked' not in str(ex).lower():
                    raise
                time.sleep(delay)
        return fn()  # last attempt: let a persistent lock surface

    def _ensure_schema(self) -> None:
        # CREATE TABLE IF NOT EXISTS also needs the write lock; same concurrent
        # first-open race as the WAL switch, so retry it too.
        self._retry_locked(lambda: self._conn.executescript(_SCHEMA))
        self._retry_locked(self._add_missing_columns)
        # Stamp the version idempotently rather than SELECT-then-INSERT. That
        # read-then-write was a TOCTOU across processes: two CLIs opening the
        # same fresh ledger both saw no row, both inserted, and the loser died
        # with `UNIQUE constraint failed: meta.key`. Retrying could not help --
        # the row exists by then, so the retry fails the same way.
        #
        # Two processes racing to open one ledger is the normal case here, not
        # an edge: it is exactly what the cross-process lock further down
        # exists to serialize, and that lock is taken *after* the store is
        # constructed. ON CONFLICT is the idiom already used for desired_gen
        # and applied_gen below.
        self._retry_locked(
            lambda: self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
                'ON CONFLICT(key) DO NOTHING',
                (str(SCHEMA_VERSION),),
            )
        )

    #: Columns added after the first schema, as (table, column, declaration).
    _ADDED_COLUMNS = (
        ('deployments', 'assigned_gpus', 'TEXT'),
    )

    def _add_missing_columns(self) -> None:
        """Add columns newer code needs to an older ledger (nullable, so safe).

        Two processes can race here; a duplicate-column error from the loser
        is the desired end state.
        """
        for table, column, decl in self._ADDED_COLUMNS:
            have = {r['name'] for r in self._conn.execute(f'PRAGMA table_info({table})')}
            if column in have:
                continue
            try:
                self._conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {decl}')
            except sqlite3.OperationalError as ex:
                if 'duplicate column' not in str(ex).lower():
                    raise

    def close(self) -> None:
        self._conn.close()

    # -- transactions ------------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Take the write lock up front and commit/rollback atomically.

        ``BEGIN IMMEDIATE`` is what makes the ledger's find-or-create-deployment
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

    # -- generation (legacy; unused by the controller) ----------------------
    #
    # Superseded by the publication marker below, which the controller uses to
    # serialise render and apply. The counters are still bumped and kept so an
    # older reader of the same ledger does not break. Their original meaning:
    #   desired_gen  bumped whenever a mutation changes the desired set (a new
    #                deployment, an idled/evicted/expired one). Captured by an
    #                acquirer right after it renders -> "the generation my change
    #                is in".
    #   applied_gen  the floor a successful apply has materialized. An acquirer is
    #                covered once applied_gen >= its captured desired_gen, so one
    #                apply satisfies every waiter that rendered before it.

    def bump_desired_generation(self) -> int:
        """Increment `desired_gen`. MUST be called inside :meth:`transaction`
        (it rides the caller's ``BEGIN IMMEDIATE`` so the bump is atomic with the
        ledger row change). Returns the new value."""
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES ('desired_gen', '1') "
            'ON CONFLICT(key) DO UPDATE SET '
            'value = CAST(meta.value AS INTEGER) + 1'
        )
        # Every demand-changing mutation is also an admission-state change.
        self.bump_admission_state_version()
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'desired_gen'"
        ).fetchone()
        return int(row['value'])

    def desired_generation(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'desired_gen'"
        ).fetchone()
        return int(row['value']) if row else 0

    def applied_generation(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'applied_gen'"
        ).fetchone()
        return int(row['value']) if row else 0

    def set_applied_generation(self, gen: int) -> None:
        """Publish the applied generation. Monotonic: an out-of-order older apply
        can never lower it (the ``WHERE`` guards the upsert)."""
        with self.transaction():
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('applied_gen', ?) "
                'ON CONFLICT(key) DO UPDATE SET value = excluded.value '
                'WHERE CAST(meta.value AS INTEGER) < CAST(excluded.value AS INTEGER)',
                (str(int(gen)),),
            )

    # -- publication marker (serialised publication) -----------------------
    #
    # One row in `meta` records that the desired state has changed and has not
    # yet been fully applied. It is written BEFORE the ledger mutation it
    # announces (intent first), so a crash at any later point leaves it set, and
    # the next applying operation re-renders from the ledger and applies. A crash
    # between writing it and the mutation costs one redundant, idempotent apply.
    #
    #   version          bumped on every mark; a clear names the version it
    #                    applied, so a newer mark is never cleared by an older
    #                    apply (defensive: callers serialise under one lock)
    #   apply_requested  False only for staged changes (`acquire --no-apply`),
    #                    which must not start just because something reopened;
    #                    once True it stays True until cleared

    def meta_json(self, key: str, default=None):
        row = self._conn.execute('SELECT value FROM meta WHERE key = ?', (key,)).fetchone()
        return json.loads(row['value']) if row else default

    def set_meta_json(self, key: str, value) -> None:
        with self.transaction():
            self._conn.execute(
                'INSERT INTO meta(key, value) VALUES (?, ?) '
                'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                (key, json.dumps(value, sort_keys=True)),
            )

    def service_addresses(self) -> dict[str, str]:
        rows = self._conn.execute('SELECT service, ipv4 FROM service_addresses').fetchall()
        return {r['service']: r['ipv4'] for r in rows}

    def add_service_addresses(self, table: dict[str, str]) -> None:
        """Insert new (service, ipv4) rows; existing rows are never changed."""
        with self.transaction():
            current = self.service_addresses()
            for service, ip in table.items():
                if service in current:
                    if current[service] != ip:
                        raise ValueError(f'service {service!r} already has {current[service]}')
                    continue
                self._conn.execute(
                    'INSERT INTO service_addresses(service, ipv4) VALUES (?, ?)', (service, ip))

    def profile(self) -> dict | None:
        """The published render profile, or ``None`` before the first mutation."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'profile'"
        ).fetchone()
        return json.loads(row['value']) if row else None

    def set_profile(self, profile: dict) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('profile', ?) "
                'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                (json.dumps(profile, sort_keys=True),),
            )

    def mark_publication_pending(
        self, *, apply_requested: bool, interrupted: bool = False,
        placement_context: dict | None = None, approved_digest: str | None = None,
    ) -> dict:
        """Record that desired state is changing; return the marker written.

        Both flags only ever turn on until the marker is cleared.
        ``interrupted`` records that an apply was killed mid-flight, so the
        runtime may still be changing underneath the next one.
        ``placement_context`` (an acquire's admission scope, e.g. its
        ``allowed_gpus``) replaces any stored one; ``None`` keeps it.
        """
        with self.transaction():
            current = self._read_publication_pending()
            marker = {
                'version': (current['version'] if current else 0) + 1,
                'apply_requested': bool(apply_requested)
                or bool(current and current['apply_requested']),
                'interrupted': bool(interrupted)
                or bool(current and current['interrupted']),
                'placement_context': placement_context if placement_context is not None
                else (current['placement_context'] if current else None),
                'approved_digest': approved_digest if approved_digest is not None
                else (current['approved_digest'] if current else None),
            }
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES ('publication_pending', ?) "
                'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                (json.dumps(marker, sort_keys=True),),
            )
        return marker

    def publication_pending(self) -> dict | None:
        """The pending marker, or ``None`` when every change has been applied."""
        return self._read_publication_pending()

    def clear_publication_pending(self, version: int) -> bool:
        """Clear the marker if it is not newer than ``version``; True if cleared."""
        with self.transaction():
            current = self._read_publication_pending()
            if current is None or current['version'] > int(version):
                return False
            self._conn.execute("DELETE FROM meta WHERE key = 'publication_pending'")
        return True

    def _read_publication_pending(self) -> dict | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'publication_pending'"
        ).fetchone()
        if not row:
            return None
        marker = json.loads(row['value'])
        return {
            'version': int(marker['version']),
            'apply_requested': bool(marker['apply_requested']),
            'interrupted': bool(marker.get('interrupted', False)),
            'placement_context': marker.get('placement_context'),
            'approved_digest': marker.get('approved_digest'),
        }

    def clear_placement_context(self) -> None:
        """Forget a pending acquire's scope once its render has pinned placement."""
        with self.transaction():
            current = self._read_publication_pending()
            if current is None or current['placement_context'] is None:
                return
            current['placement_context'] = None
            self._conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'publication_pending'",
                (json.dumps(current, sort_keys=True),),
            )

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
        deployment_states: tuple[str, ...] = (),
    ) -> tuple[int, int]:
        """Delete terminal leases/deployments (and their claims) from the ledger.

        Claims are removed first so the ``deployments.id`` foreign key can't block a
        deployment deletion; deleting leases also cascades their claims. Returns
        ``(n_leases_deleted, n_deployments_deleted)``.
        """
        n_leases = n_deployments = 0
        with self.transaction() as conn:
            if lease_states:
                lq = ','.join('?' for _ in lease_states)
                conn.execute(
                    f'DELETE FROM claims WHERE lease_id IN '
                    f'(SELECT id FROM leases WHERE state IN ({lq}))',
                    lease_states,
                )
            if deployment_states:
                gq = ','.join('?' for _ in deployment_states)
                conn.execute(
                    f'DELETE FROM claims WHERE deployment_id IN '
                    f'(SELECT id FROM deployments WHERE state IN ({gq}))',
                    deployment_states,
                )
            if lease_states:
                lq = ','.join('?' for _ in lease_states)
                n_leases = conn.execute(
                    f'DELETE FROM leases WHERE state IN ({lq})', lease_states
                ).rowcount
            if deployment_states:
                gq = ','.join('?' for _ in deployment_states)
                n_deployments = conn.execute(
                    f'DELETE FROM deployments WHERE state IN ({gq})', deployment_states
                ).rowcount
        return n_leases, n_deployments

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
            'SELECT endpoint, deployment_id FROM claims WHERE lease_id = ?'
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
            deployment_ids=list(dict.fromkeys(c['deployment_id'] for c in claims)),
        )

    # -- claims ------------------------------------------------------------

    def insert_claim(
        self, *, lease_id: str, endpoint: str, deployment_id: str, kind: str = 'endpoint'
    ) -> None:
        self._conn.execute(
            'INSERT INTO claims(lease_id, endpoint, deployment_id, kind)'
            ' VALUES(?, ?, ?, ?)',
            (lease_id, endpoint, deployment_id, kind),
        )

    # -- deployments ------------------------------------------------------------

    def insert_deployment(self, deployment: Deployment) -> None:
        self._conn.execute(
            'INSERT INTO deployments(id, compat_key, engine, sharing, capacity,'
            ' spec, served, state, created_at, updated_at, assigned_gpus)'
            ' VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                deployment.id,
                deployment.compat_key,
                deployment.engine,
                deployment.sharing,
                _dumps(deployment.capacity),
                _dumps(deployment.spec),
                _dumps(deployment.served),
                deployment.state,
                deployment.created_at,
                deployment.updated_at,
                None if deployment.assigned_gpus is None
                else _dumps(list(deployment.assigned_gpus)),
            ),
        )

    def set_deployment_state(self, deployment_id: str, state: str, updated_at: float) -> None:
        # Leaving LIVE releases the committed allocation in the same statement.
        self._conn.execute(
            'UPDATE deployments SET state = ?, updated_at = ?,'
            " assigned_gpus = CASE WHEN ? = 'live' THEN assigned_gpus ELSE NULL END"
            ' WHERE id = ?',
            (state, updated_at, state, deployment_id),
        )

    def set_deployment_allocation(
        self, deployment_id: str, gpus: list[int] | None
    ) -> None:
        """Commit (or clear) a LIVE deployment's GPU allocation."""
        self._conn.execute(
            'UPDATE deployments SET assigned_gpus = ? WHERE id = ?',
            (None if gpus is None else _dumps([int(g) for g in gpus]), deployment_id),
        )

    def bump_admission_state_version(self) -> int:
        """Increment ``admission_state_version``; call inside :meth:`transaction`."""
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES ('admission_state_version', '1') "
            'ON CONFLICT(key) DO UPDATE SET '
            'value = CAST(meta.value AS INTEGER) + 1'
        )
        return self.admission_state_version()

    def admission_state_version(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'admission_state_version'"
        ).fetchone()
        return int(row['value']) if row else 0

    def update_deployment_served(
        self, deployment_id: str, served: dict[str, Any], updated_at: float
    ) -> None:
        self._conn.execute(
            'UPDATE deployments SET served = ?, updated_at = ? WHERE id = ?',
            (_dumps(served), updated_at, deployment_id),
        )

    def get_deployment(self, deployment_id: str, *, now: float | None = None) -> Deployment | None:
        row = self._conn.execute(
            'SELECT * FROM deployments WHERE id = ?', (deployment_id,)
        ).fetchone()
        if row is None:
            return None
        deployment = self._row_to_deployment(row)
        if now is not None:
            deployment.demand = self.demand(deployment_id, now)
        return deployment

    def deployments_by_compat(
        self, compat_key: str, *, sharing: str, states: tuple[str, ...]
    ) -> list[Deployment]:
        placeholders = ','.join('?' for _ in states)
        rows = self._conn.execute(
            'SELECT * FROM deployments WHERE compat_key = ? AND sharing = ?'
            f' AND state IN ({placeholders}) ORDER BY created_at',
            (compat_key, sharing, *states),
        ).fetchall()
        return [self._row_to_deployment(r) for r in rows]

    def list_deployments(self, *, now: float) -> list[Deployment]:
        rows = self._conn.execute(
            'SELECT * FROM deployments ORDER BY created_at'
        ).fetchall()
        deployments = [self._row_to_deployment(r) for r in rows]
        for deployment in deployments:
            deployment.demand = self.demand(deployment.id, now)
        return deployments

    def deployments_for_lease(self, lease_id: str) -> list[Deployment]:
        rows = self._conn.execute(
            'SELECT DISTINCT g.* FROM deployments g JOIN claims c'
            ' ON c.deployment_id = g.id WHERE c.lease_id = ?',
            (lease_id,),
        ).fetchall()
        return [self._row_to_deployment(r) for r in rows]

    def demand(self, deployment_id: str, now: float) -> int:
        """Number of *protecting* leases referencing ``deployment_id``.

        A lease protects iff it is ACTIVE and not past its TTL, matching
        :meth:`infer_stack.leasing.models.Lease.is_protecting`.
        """
        row = self._conn.execute(
            'SELECT COUNT(DISTINCT c.lease_id) AS n FROM claims c'
            ' JOIN leases l ON c.lease_id = l.id'
            ' WHERE c.deployment_id = ? AND l.state = ?'
            ' AND (l.expires_at IS NULL OR l.expires_at > ?)',
            (deployment_id, LeaseState.ACTIVE, now),
        ).fetchone()
        return int(row['n'])

    def _row_to_deployment(self, row: sqlite3.Row) -> Deployment:
        return Deployment(
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
            assigned_gpus=_loads(row['assigned_gpus']) if row['assigned_gpus'] else None,
        )
