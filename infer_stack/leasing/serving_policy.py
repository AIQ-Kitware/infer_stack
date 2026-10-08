"""Validated replica policy, separate from model launch arguments."""
from __future__ import annotations

FIELDS = ('min_replicas', 'max_replicas', 'target_requests',
          'scale_down_delay_seconds', 'load_balancing')


def policy(runtime: dict) -> dict:
    """Explicit zero selects on-demand residency; omitted bounds stay 1..1."""
    values = {'min_replicas': 1, 'max_replicas': 1}
    values.update({k: runtime[k] for k in FIELDS if k in runtime})
    for name, minimum in [('min_replicas', 0), ('max_replicas', 1),
                          ('target_requests', 1), ('scale_down_delay_seconds', 0)]:
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'runtime.{name} must be an integer >= {minimum}')
    if values['min_replicas'] > values['max_replicas']:
        raise ValueError('runtime.min_replicas must be <= runtime.max_replicas')
    if 'load_balancing' in values and values['load_balancing'] not in {'LeastLoad', 'PrefixHash'}:
        raise ValueError('runtime.load_balancing must be LeastLoad or PrefixHash')
    return values


def identity(runtime: dict) -> dict:
    """Conservative sharing until policy merging exists; preserve legacy keys."""
    values = policy(runtime)
    defaults = {'min_replicas': 1, 'max_replicas': 1}
    return {k: v for k, v in values.items() if v != defaults.get(k)}
