"""Immutable approved catalog revisions in the control authority database.

Files seed a new service once. Explicit, audited compare-and-swap approvals
advance its catalog thereafter; restarting with different YAML is not approval.
Approval changes eligibility for new requests, never existing serving intent.
"""
from __future__ import annotations

import json
import time
from urllib.parse import urlsplit

from ..leasing.catalog import Catalog
from ..leasing.catalog_edit import SECTIONS
from ..leasing.profile import canonical_digest


def validate_source(source: dict) -> Catalog:
    """Reject malformed/non-JSON input without reflecting private launch data."""
    try:
        if not isinstance(source, dict) or set(source) - set(SECTIONS):
            raise ValueError
        text = json.dumps(source, allow_nan=False)
        if len(text.encode()) > 60000:
            raise ValueError
        for section, entries in source.items():
            if not isinstance(entries, dict):
                raise ValueError
            for name, spec in entries.items():
                if not isinstance(name, str) or not name or len(name) > 253:
                    raise ValueError
                if section == 'bundles':
                    if not isinstance(spec, list) or any(not isinstance(n, str) for n in spec):
                        raise ValueError
                elif not isinstance(spec, dict):
                    raise ValueError
        catalog = Catalog.from_dict(source)
        for endpoint in catalog.endpoints.values():
            catalog.resolve_endpoint(endpoint.name)
            if endpoint.external is not None:
                parts = urlsplit(endpoint.external.api_base)
                if parts.username or parts.password or parts.query or parts.fragment:
                    raise ValueError
        return catalog
    except (ValueError, TypeError, AttributeError, KeyError, OverflowError, RecursionError) as ex:
        raise ValueError('Invalid catalog revision; validate the source locally before approval') from ex


class CatalogRevisions:
    def __init__(self, store, *, clock=time.time):
        self.store, self.clock = store, clock

    @staticmethod
    def _active(connection):
        row = connection.execute("SELECT value FROM meta WHERE key='control_catalog_revision'").fetchone()
        return json.loads(row['value']) if row else None

    def active(self):
        return self._active(self.store._conn)

    def get(self, revision):
        row = self.store._conn.execute('SELECT * FROM control_catalog_revisions WHERE revision=?', (revision,)).fetchone()
        if row is None:
            raise KeyError('Unknown catalog revision')
        return {**dict(row), 'source': json.loads(row['source'])}

    def catalog(self):
        return Catalog.from_dict(self.get(self.active())['source'])

    def history(self):
        # Launch specifications stay private to catalog administrators. Hashes
        # and provenance are safe to include in ordinary status/descriptors.
        rows = self.store._conn.execute('SELECT revision,principal,created_at FROM control_catalog_revisions ORDER BY created_at DESC,revision LIMIT 1000')
        return {'active_revision': self.active(), 'revisions': [dict(row) for row in rows]}

    def seed(self, source):
        validate_source(source)
        with self.store.transaction() as conn:
            if self._active(conn) is None:
                revision = canonical_digest(source)
                self._insert(conn, revision, source, 'bootstrap')
                self._activate(conn, revision)

    def _insert(self, conn, revision, source, principal):
        conn.execute('INSERT INTO control_catalog_revisions VALUES (?,?,?,?) ON CONFLICT(revision) DO NOTHING',
                     (revision, json.dumps(source, sort_keys=True, separators=(',', ':'), allow_nan=False), principal, self.clock()))

    @staticmethod
    def _activate(conn, revision):
        conn.execute("INSERT INTO meta(key,value) VALUES ('control_catalog_revision',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(revision),))

    def approve(self, operation, operations):
        """Catalog activation and terminal operation/audit are one commit.

        A crash cannot leave a successful activation with an operation that
        later retries its stale base revision and misreports failure.
        """
        payload = operation['request']['client']
        source = payload['source']
        revision = canonical_digest(source)
        with self.store.transaction() as conn:
            if self._active(conn) != payload['base_revision']:
                operations.transition(operation['id'], 'failed', connection=conn,
                                      error={'code': 'CatalogConflict', 'message': 'Approved catalog changed; review the current revision and submit again'})
                return
            self._insert(conn, revision, source, operation['principal'])
            self._activate(conn, revision)
            operations.transition(operation['id'], 'succeeded', connection=conn,
                                  result={'catalog_revision': revision, 'previous_revision': payload['base_revision']})
