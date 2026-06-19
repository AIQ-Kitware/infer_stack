"""Shared CLI helpers: path overrides + inventory resolution.

Trimmed to what the leasing surface needs after the pre-leasing profile world
was removed: ``_apply_path_overrides`` (honour ``--config-dir`` / ``--data-dir``)
and ``effective_inventory`` (honour ``--simulate-hardware`` / ``--allowed-gpus``
so placement/suggest can plan for hardware you don't have in front of you).
"""

from __future__ import annotations

import os
from typing import Any

from ..hardware import detect_inventory, simulate_inventory
from ..paths import set_config_root, set_data_root


def _as_mapping(args: Any) -> dict[str, Any]:
    """Coerce a CLI args object into a plain dict.

    Works for ``None``, ``argparse.Namespace``, and ``scfg.DataConfig``
    instances. Used to side-step name clashes between user-declared fields and
    ``DataConfig`` builtins.
    """
    if args is None:
        return {}
    if hasattr(args, 'asdict'):
        return dict(args.asdict())
    if hasattr(args, '__dict__'):
        return dict(vars(args))
    return dict(args)


def _env_text(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def _parse_allowed_gpus(raw: Any) -> list[int] | None:
    """Parse a comma-separated list of GPU indices, or ``None`` if unset.

    Accepts ints (when the value comes from ``data=`` kwargs in the programmatic
    API), as well as strings of the form ``"1"`` or ``"1,3"``.
    """
    if raw is None or raw == '':
        return None
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [x.strip() for x in str(raw).split(',') if x.strip()]
    try:
        return [int(x) for x in items]
    except (TypeError, ValueError) as ex:
        raise SystemExit(
            f'Invalid --allowed-gpus value {raw!r}: expected a comma-separated '
            f"list of integer GPU indices (e.g. '1' or '1,3'). {ex}"
        )


def _filter_inventory_to_allowed(
    inventory: dict[str, Any], allowed: list[int] | None
) -> dict[str, Any]:
    """Return a new inventory containing only the GPUs whose ``index`` is allowed.

    Real indices are preserved — there is no renumbering.
    """
    if not allowed:
        return inventory
    allowed_set = set(allowed)
    filtered = [
        g for g in inventory.get('gpus', []) if g.get('index') in allowed_set
    ]
    return {'gpu_count': len(filtered), 'gpus': filtered}


def effective_inventory(args: Any | None) -> dict[str, Any] | None:
    """Build the inventory to plan against, honoring CLI / env overrides.

    Returns ``None`` when nothing is constraining the inventory, so the caller
    falls back to ``detect_inventory()`` at plan time.
    """
    overrides = _as_mapping(args)
    spec = overrides.get('simulate_hardware')
    allowed = _parse_allowed_gpus(
        overrides.get('allowed_gpus') or _env_text('INFER_STACK_ALLOWED_GPUS')
    )
    if not spec and allowed is None:
        return None
    base = simulate_inventory(spec) if spec else detect_inventory()
    return _filter_inventory_to_allowed(base, allowed)


def _apply_path_overrides(config: Any) -> None:
    """Honour ``--config-dir`` / ``--data-dir`` from a parsed subcommand config."""
    overrides = _as_mapping(config)
    if overrides.get('config_dir'):
        set_config_root(overrides['config_dir'])
    if overrides.get('data_dir'):
        set_data_root(overrides['data_dir'])
