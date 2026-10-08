"""Hard cluster eligibility; Kubernetes chooses the concrete replica placement."""
from __future__ import annotations

import copy
import re

POOL_LABEL = 'infer-stack.ai/pool'
HOSTNAME_LABEL = 'kubernetes.io/hostname'


def validate(placement: dict) -> list[str]:
    errors = []
    for key in ('node', 'pool'):
        if key not in placement:
            continue
        value = placement[key]
        limit = 253 if key == 'node' else 63
        pattern = r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?' if key == 'node' else r'[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?'
        if not isinstance(value, str) or len(value) > limit or not re.fullmatch(pattern, value):
            errors.append(f'placement.{key} must be a valid nonempty Kubernetes {"node name" if key == "node" else "label value"}')
    if placement.get('gpu_indices') and any(placement.get(k) for k in ('node', 'pool')):
        errors.append('local gpu_indices cannot be combined with cluster node/pool placement')
    return errors


def narrow(requests, *, node: str | None = None, pool: str | None = None):
    """Intersect request constraints with approved catalog constraints.

    An exact conflicting key is impossible. Different keys combine conjunctively.
    Copies preserve the immutable catalog and unconstrained legacy identities.
    """
    supplied = {k: v for k, v in [('node', node), ('pool', pool)] if v is not None}
    if not supplied:
        return requests
    if validate(supplied):
        raise ValueError('; '.join(validate(supplied)))
    out = copy.deepcopy(requests)
    for request in out:
        if request.engine != 'vllm':
            raise ValueError('Cluster placement requires a vllm endpoint')
        placement = request.spec.setdefault('placement', {})
        for key, value in supplied.items():
            if key in placement and placement[key] != value:
                raise ValueError(f'Request {key}={value!r} conflicts with approved {key}={placement[key]!r}')
            placement[key] = value
        errors = validate(placement)
        if errors:
            raise ValueError('; '.join(errors))
        request.structural['cluster_placement'] = {k: placement[k] for k in ('node', 'pool') if k in placement}
    return out


def constrained_profile(placement: dict, profiles: dict, nodes: dict, *,
                        explicit: str | None, min_vram: float | None, gpu_count: int) -> str:
    """Select a profile that *enforces* every requested hard constraint.

    Matching today's nodes alone is insufficient: a generic profile could select
    a newly joined ineligible node tomorrow. Require selector implication too.
    Allocatable capacity is total capacity, so occupied eligible nodes can queue.
    """
    required = {}
    if placement.get('pool'):
        required[POOL_LABEL] = placement['pool']
    if placement.get('node'):
        target = nodes.get(placement['node'])
        if target is None:
            raise ValueError(f'Unknown canonical Kubernetes node {placement["node"]!r}')
        hostname = target['labels'].get(HOSTNAME_LABEL)
        if not hostname or sum(n['labels'].get(HOSTNAME_LABEL) == hostname for n in nodes.values()) != 1:
            raise ValueError('Exact node requires a unique kubernetes.io/hostname label')
        required[HOSTNAME_LABEL] = hostname
    eligible = [n for n in nodes.values() if all(n['labels'].get(k) == v for k, v in required.items())
                and n.get('count', 0) >= gpu_count
                and (not min_vram or (n.get('memory_gib') or 0) >= min_vram)]
    if not eligible:
        raise ValueError('No eligible hardware exists for requested node/pool, GPU count and VRAM')
    base, sep, count = (explicit or '').partition(':')
    if sep and (not count.isdigit() or int(count) != gpu_count):
        raise ValueError('Explicit resource profile GPU count conflicts with runtime topology')
    candidates = []
    for name, profile in sorted(profiles.items()):
        if base and name != base:
            continue
        selector = profile.get('nodeSelector') or {}
        if any(str((profile.get(kind) or {}).get('nvidia.com/gpu', '')) != '1' for kind in ('requests', 'limits')):
            continue
        if not all(str(selector.get(k, '')) == v for k, v in required.items()):
            continue
        selected = [n for n in nodes.values() if all(n['labels'].get(k) == str(v) for k, v in selector.items())]
        if not selected or not any(n in eligible for n in selected):
            continue
        if min_vram and any((n.get('memory_gib') or 0) < min_vram for n in selected):
            continue
        candidates.append(name)
    if not candidates:
        raise ValueError('No installed resource profile enforces requested node/pool and VRAM; install an administrator-approved selector profile first')
    return f'{candidates[0]}:{gpu_count}'
