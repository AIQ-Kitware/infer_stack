"""Durable requests in the same database/backup as the lease authority."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import nullcontext


class Operations:
    """Transactional idempotency and audited operation transitions."""

    def __init__(self, store, *, clock=time.time):
        self.store, self.clock = store, clock

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        for key in ('request', 'result', 'error'):
            if result[key] is not None:
                result[key] = json.loads(result[key])
        return result

    def get(self, operation_id: str):
        return self.decode(self.store._conn.execute('SELECT * FROM control_operations WHERE id=?', (operation_id,)).fetchone())

    def find(self, principal: str, key: str):
        return self.decode(self.store._conn.execute('SELECT * FROM control_operations WHERE principal=? AND idempotency_key=?', (principal, key)).fetchone())

    def all(self):
        return [self.decode(row) for row in self.store._conn.execute('SELECT * FROM control_operations ORDER BY created_at, id')]

    def audit(self, *, principal=None, after=0, limit=100):
        """Stable sequence pagination; no private requests or launch snapshots."""
        if isinstance(after, bool) or not isinstance(after, int) or after < 0 or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError('Audit requires a nonnegative cursor and limit 1..1000')
        query, values = 'SELECT * FROM control_audit WHERE id>?', [after]
        if principal is not None:
            query += ' AND principal=?'
            values.append(principal)
        values.append(limit)
        return [dict(row) for row in self.store._conn.execute(query + ' ORDER BY id LIMIT ?', values)]

    def submit(self, principal: str, key: str, request: dict, *, admit=None) -> dict:
        if not key or len(key) > 200:
            raise ValueError('Idempotency-Key must contain 1..200 characters')
        text = json.dumps(request, sort_keys=True, separators=(',', ':'))
        with self.store.transaction() as conn:
            existing = conn.execute('SELECT * FROM control_operations WHERE principal=? AND idempotency_key=?', (principal, key)).fetchone()
            if existing:
                old = json.loads(existing['request'])
                if (old['action'], old.get('client')) != (request['action'], request.get('client')):
                    raise ValueError('Idempotency key already identifies a different request')
                return self.decode(existing)
            if admit is not None:
                admit(conn)
            operation_id, now = 'op-' + uuid.uuid4().hex, self.clock()
            conn.execute('INSERT INTO control_operations VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)',
                         (operation_id, principal, key, text, 'queued', now, now))
            conn.execute('INSERT INTO control_audit(operation_id,principal,action,timestamp,state) VALUES (?, ?, ?, ?, ?)',
                         (operation_id, principal, request['action'], now, 'queued'))
        return self.get(operation_id)

    def transition(self, operation_id: str, state: str, *, result=None, error=None, connection=None):
        if state not in {'queued', 'running', 'converging', 'succeeded', 'failed', 'superseded'}:
            raise ValueError('Invalid operation state')
        with (self.store.transaction() if connection is None else nullcontext(connection)) as conn:
            row = conn.execute('SELECT * FROM control_operations WHERE id=?', (operation_id,)).fetchone()
            if row is None:
                raise KeyError(operation_id)
            if row['state'] in {'succeeded', 'failed', 'superseded'}:
                raise ValueError('Terminal operations are immutable')
            now = self.clock()
            conn.execute('UPDATE control_operations SET state=?,updated_at=?,result=COALESCE(?,result),error=? WHERE id=?',
                         (state, now, json.dumps(result) if result is not None else None,
                          json.dumps(error) if error is not None else None, operation_id))
            conn.execute('INSERT INTO control_audit(operation_id,principal,action,timestamp,state) VALUES (?, ?, ?, ?, ?)',
                         (operation_id, row['principal'], json.loads(row['request'])['action'], now, state))
        return self.get(operation_id)
