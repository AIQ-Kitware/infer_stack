"""The shell command that does what a TUI action just did.

The TUI logs one of these after each action, so the dashboard teaches the CLI
instead of hiding it. Each builder returns a command that actually runs; an
action with no CLI counterpart says so rather than inventing one.
"""

from __future__ import annotations

import json
import shlex
from typing import Any

import yaml


def command(*parts: object) -> str:
    """``infer-stack`` plus ``parts``, each quoted for a POSIX shell."""
    return shlex.join(['infer-stack', *(str(p) for p in parts)])


def _kv_value(value: Any) -> str:
    """A ``--runtime KEY=VALUE`` value that YAML-parses back to ``value``."""
    if isinstance(value, str):
        return value if yaml.safe_load(value) == value else json.dumps(value)
    return json.dumps(value)


#: Endpoint fields `catalog endpoint add` can express.
_ENDPOINT_FIELDS = {'engine', 'model', 'host', 'public_name', 'reclaim',
                    'protocol', 'placement', 'runtime'}


def endpoint_add(name: str, entry: dict[str, Any], *, force: bool = False) -> str:
    """``catalog endpoint add`` for ``entry``, as the catalog stores it.

    Example:
        >>> print(endpoint_add('qwen', {'engine': 'vllm', 'model': 'q',
        ...     'placement': {'gpu_indices': [0, 1]},
        ...     'runtime': {'max_model_len': 8192, 'command': ['single']}}, force=True))
        infer-stack catalog endpoint add qwen --model q --gpu 0 1 --runtime max_model_len=8192 'command=["single"]' --force
    """
    parts: list[object] = ['catalog', 'endpoint', 'add', name]
    if entry.get('engine', 'vllm') != 'vllm':
        parts += ['--engine', entry['engine']]
    for key in ('model', 'host', 'public_name', 'protocol'):
        if entry.get(key):
            parts += [f'--{key}', entry[key]]
    policy = (entry.get('reclaim') or {}).get('policy')
    if policy:
        parts += ['--reclaim', policy]
    placement = entry.get('placement') or {}
    if placement.get('min_vram_gib') is not None:
        parts += ['--min_vram_gib', placement['min_vram_gib']]
    if placement.get('gpu_indices'):
        parts += ['--gpu', *placement['gpu_indices']]
    runtime = dict(entry.get('runtime') or {})
    extra = runtime.pop('extra_args', None)
    if extra:
        parts += ['--extra_args', shlex.join(extra) if isinstance(extra, list) else extra]
    if runtime:
        parts += ['--runtime', *(f'{k}={_kv_value(v)}' for k, v in runtime.items())]
    if force:
        parts.append('--force')
    text = command(*parts)
    other = sorted(set(entry) - _ENDPOINT_FIELDS) + sorted(
        f'placement.{k}' for k in placement if k not in ('min_vram_gib', 'gpu_indices'))
    if other:
        text += f'   # then `infer-stack catalog edit` for: {", ".join(other)}'
    return text
