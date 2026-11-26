"""Portable append-only JSONL observations. Reports never rewrite the ledger."""
from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = 'infer-stack-benchmark/1'


def append(path, row):
    import fcntl
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, allow_nan=False) + '\n'
    with path.open('a+b') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0, os.SEEK_END)
        if file.tell():
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b'\n':
                raise ValueError(f'{path}: incomplete last record; preserve a copy and repair before appending')
        file.write(line.encode())
        file.flush()
        os.fsync(file.fileno())


def read(paths):
    """Deduplicate exact copies by ID; refuse conflicting records."""
    import fcntl
    seen = {}
    for path in paths:
        path = Path(path).expanduser()
        with path.open(encoding='utf-8') as file:
            fcntl.flock(file, fcntl.LOCK_SH)
            for num, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row['schema'] != SCHEMA or not isinstance(row['id'], str):
                        raise ValueError('unknown schema or missing ID')
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{path}:{num}: invalid benchmark record: {exc}') from exc
                if row['id'] in seen and seen[row['id']] != row:
                    raise ValueError(f'{path}:{num}: conflicting observation ID {row["id"]}')
                seen[row['id']] = row
    return list(seen.values())
