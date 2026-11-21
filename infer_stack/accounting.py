"""Durable usage facts and recomputable energy with explicit measurement gaps.

Intervals are half-open UTC epoch seconds. Samples are total GPU watts for the
record's device set, not whole-host power. Allocation and power are separate
facts; shared leases never receive copies of a deployment's energy.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class UsageInterval:
    id: str
    start: float
    end: float
    node: str
    deployment: str
    replica: str
    gpu_count: int
    gpu_uuids: tuple[str, ...] = ()
    model: str = ''
    endpoint: str = ''
    gpu_product: str = ''
    source: str = ''
    samples: tuple[tuple[float, float], ...] = ()
    max_sample_gap: float = 30
    proxy_watts: float | None = None
    proxy_source: str = ''
    proxy_version: str = ''
    proxy_method: str = 'static_device_class_proxy'
    request_count: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def validate(self) -> None:
        if not all(isinstance(s, str) and s for s in [self.id, self.node, self.deployment, self.replica]):
            raise ValueError('Usage identity, node, deployment and replica are required')
        if not all(not isinstance(n, bool) and math.isfinite(n) for n in [self.start, self.end, self.max_sample_gap]) or self.end <= self.start or self.max_sample_gap <= 0:
            raise ValueError('Usage requires a finite positive interval and sample gap')
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int) or self.gpu_count < 1:
            raise ValueError('gpu_count must be a positive integer')
        if self.gpu_uuids and (len(set(self.gpu_uuids)) != self.gpu_count or len(self.gpu_uuids) != self.gpu_count):
            raise ValueError('GPU UUIDs must be unique and match allocated count')
        if any(not isinstance(value, str) or not value for value in self.gpu_uuids):
            raise ValueError('GPU UUIDs must be nonempty strings')
        if self.samples and (not self.source or not self.gpu_uuids):
            raise ValueError('Measured power requires source and stable GPU UUIDs')
        previous = None
        for stamp, watts in self.samples:
            if not all(not isinstance(n, bool) and math.isfinite(n) for n in [stamp, watts]) or watts < 0 or (previous is not None and stamp <= previous):
                raise ValueError('Power samples require increasing finite times and nonnegative watts')
            previous = stamp
        if self.proxy_watts is not None:
            if isinstance(self.proxy_watts, bool) or not math.isfinite(self.proxy_watts) or self.proxy_watts < 0 or not self.proxy_source or not self.proxy_version:
                raise ValueError('Power proxy requires nonnegative watts, source and version')
            if self.proxy_method not in {'static_device_class_proxy', 'observed_power_proxy'}:
                raise ValueError('Unknown power proxy method')
        for n in [self.request_count, self.input_tokens, self.output_tokens]:
            if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 0):
                raise ValueError('Activity counters must be nonnegative integers or unknown')

    @classmethod
    def from_dict(cls, values: dict) -> UsageInterval:
        data = dict(values)
        data['gpu_uuids'] = tuple(data.get('gpu_uuids') or ())
        data['samples'] = tuple(tuple(row) for row in data.get('samples') or ())
        record = cls(**data)
        record.validate()
        return record


def energy(record: UsageInterval) -> dict:
    """Clip trapezoids to the interval and leave large/unbounded gaps unknown."""
    record.validate()
    duration = record.end - record.start
    measured = coverage = 0.0
    for (t0, p0), (t1, p1) in zip(record.samples, record.samples[1:]):
        if t1 - t0 > record.max_sample_gap:
            continue
        a, b = max(t0, record.start), min(t1, record.end)
        if b <= a:
            continue
        pa = p0 + (p1 - p0) * (a - t0) / (t1 - t0)
        pb = p0 + (p1 - p0) * (b - t0) / (t1 - t0)
        measured += (pa + pb) * 0.5 * (b - a)
        coverage += b - a
    missing = max(0.0, duration - coverage)
    estimated = missing * record.gpu_count * record.proxy_watts if record.proxy_watts is not None else None
    complete = missing <= duration * 1e-12
    method = ('measured' if complete else 'mixed' if coverage and estimated is not None
              else 'measured_partial' if coverage else record.proxy_method if estimated is not None else 'resource_time_only')
    total = measured + (estimated or 0) if coverage or estimated is not None else None
    return {'gpu_seconds': duration * record.gpu_count,
            'gpu_hours': duration * record.gpu_count / 3600,
            'energy_joules': total, 'energy_wh': total / 3600 if total is not None else None,
            'measured_joules': measured if coverage else None, 'estimated_joules': estimated,
            'method': method, 'coverage': coverage / duration,
            'unmeasured_seconds': missing, 'is_complete': complete or estimated is not None,
            'source': record.source, 'proxy_source': record.proxy_source,
            'proxy_version': record.proxy_version, 'proxy_watts': record.proxy_watts,
            'lease_attributed_energy': None}


class UsageStore:
    """Append-only accounting, separate from lease/desired-state authority."""

    def __init__(self, path: str | Path, *, readonly=False):
        self.path = Path(path)
        if readonly:
            from urllib.parse import quote
            self.connection = sqlite3.connect('file:' + quote(str(self.path.resolve())) + '?mode=ro', uri=True)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS usage (id TEXT PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL, node TEXT NOT NULL, replica TEXT NOT NULL, facts TEXT NOT NULL)')
        self.connection.commit()

    def close(self):
        self.connection.close()

    def append(self, record: UsageInterval) -> bool:
        """Identical retries are no-ops; changed IDs or overlaps fail atomically."""
        record.validate()
        facts = json.dumps(asdict(record), sort_keys=True)
        try:
            self.connection.execute('BEGIN IMMEDIATE')
            existing = self.connection.execute('SELECT facts FROM usage WHERE id=?', (record.id,)).fetchone()
            if existing:
                if existing[0] != facts:
                    raise ValueError('Usage ID already records different immutable facts')
                self.connection.commit()
                return False
            rows = self.connection.execute('SELECT facts FROM usage WHERE node=? AND start < ? AND end > ?', (record.node, record.end, record.start))
            for row in rows:
                other = UsageInterval.from_dict(json.loads(row[0]))
                if other.replica == record.replica or set(other.gpu_uuids) & set(record.gpu_uuids):
                    raise ValueError('Overlapping accounting would double count a replica or GPU')
            self.connection.execute('INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?)',
                                    (record.id, record.start, record.end, record.node, record.replica, facts))
            self.connection.commit()
            return True
        except BaseException:
            self.connection.rollback()
            raise

    def records(self) -> list[UsageInterval]:
        return [UsageInterval.from_dict(json.loads(row[0])) for row in
                self.connection.execute('SELECT facts FROM usage ORDER BY start, id')]
