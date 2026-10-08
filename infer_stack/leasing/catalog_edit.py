"""Canonical file mutation helpers for the serving catalog.

This module is deliberately small.  The catalog parser in :mod:`catalog`
defines what catalog data *means*; this module owns how editable YAML is loaded,
canonicalized, validated, and atomically persisted.  CLI and TUI editors use the
same functions so neither presentation layer becomes a second catalog writer.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from .catalog import Catalog, CatalogError

SECTIONS = ('models', 'endpoints', 'runtime_hosts', 'bundles')


def load_catalog_source(path: str | Path) -> dict[str, Any]:
    """Load editable catalog YAML and materialize the standard sections."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    if not isinstance(data, dict):
        raise SystemExit(f'{path} must contain a YAML mapping at the top level')
    for section in SECTIONS:
        value = data.setdefault(section, {})
        if not isinstance(value, dict):
            raise SystemExit(f'{path}: {section} must be a mapping')
    return data


def validate_catalog_source(data: dict[str, Any]) -> None:
    """Refuse data the runtime catalog parser would reject."""
    try:
        Catalog.from_dict(data)
    except CatalogError as ex:
        raise SystemExit(f'refusing to write an invalid catalog: {ex}') from ex


def canonicalize_catalog_source(data: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical writable spelling of backwards-compatible input.

    Reading remains liberal, but infer-stack never writes the deprecated
    ``public_name`` endpoint key.  This keeps compatibility at the boundary
    without perpetuating two apparent authorities in files infer-stack edits.
    """
    out = copy.deepcopy(data)
    for spec in (out.get('endpoints') or {}).values():
        if not isinstance(spec, dict):
            continue
        if 'public_name' in spec:
            if 'served_name' not in spec:
                spec['served_name'] = spec['public_name']
            spec.pop('public_name', None)
    return out


def dump_catalog_source(data: dict[str, Any]) -> str:
    """Validate and serialize editable catalog data in canonical form."""
    validate_catalog_source(data)
    canonical = canonicalize_catalog_source(data)
    # Validate the exact representation we will publish too.  Today this is
    # equivalent, but keeping the check here makes future migrations safe.
    validate_catalog_source(canonical)
    tidy = {k: v for k, v in canonical.items() if v or k == 'models'}
    return yaml.safe_dump(tidy, sort_keys=False, default_flow_style=False)


def write_catalog_source(path: str | Path, data: dict[str, Any]) -> None:
    """Validate and atomically replace an editable catalog file."""
    path = Path(path)
    text = dump_catalog_source(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def next_indexed_name(existing: Any, base: str) -> str:
    """First free ``{base}-{N}`` (N starts at 1)."""
    n = 1
    while f'{base}-{n}' in existing:
        n += 1
    return f'{base}-{n}'


def slug_alias(text: str) -> str:
    """Make a model/tag string safe as an endpoint alias."""
    out = re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('-')
    return out or text
