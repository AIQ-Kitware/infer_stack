"""Discover and import individual hardware-sized profiles from the registry.

The registry is not the operator's catalog. Browsing is read-only; importing
adds an exact endpoint and its model, never siblings or unrelated suggestions.
The same code is used by the local TUI and catalog import.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .suggest import builtin_pool, suggest_catalog


def registry_endpoint_names(pool=None) -> set[str]:
    """All names, including optional and hardware-ineligible profiles."""
    pool = builtin_pool() if pool is None else pool
    return {name for model in pool.values() for name in (
        model.name, *(f'{model.name}-{v}' for v in model.endpoint_variants)
    )}


def discovery_catalog(inventory: dict[str, Any], *, pool=None,
                      reserve_display_gpu=False, kubeai=False,
                      all_models=True) -> dict[str, Any]:
    """Return profiles eligible under the existing suggestion rules.

    KubeAI cannot deploy arbitrary custom images/launchers; hide those
    profiles rather than offering a local Docker recipe on a cluster.
    This is a planning check against declared floors, not a startup test.
    """
    frag = suggest_catalog(inventory, pool=pool, all_models=all_models,
                           reserve_display_gpu=reserve_display_gpu)
    if kubeai:
        from ..backends.kubeai import unsupported_runtime_fields

        endpoints = {}
        for name, ep in frag['endpoints'].items():
            if unsupported_runtime_fields(ep.get('runtime') or {}):
                continue
            # Local CUDA indices have no meaning on a cluster.
            placement = ep.get('placement') or {}
            placement.pop('gpu_indices', None)
            if not placement:
                ep.pop('placement', None)
            endpoints[name] = ep
        models = {ep['model'] for ep in endpoints.values()}
        frag = {'models': {n: m for n, m in frag['models'].items() if n in models},
                'endpoints': endpoints}
    return frag


def endpoint_fragment(fragment: dict[str, Any], name: str) -> dict[str, Any]:
    """Extract exactly one endpoint and its referenced model."""
    if name not in fragment['endpoints']:
        raise ValueError(f'profile {name!r} does not fit the selected hardware/backend')
    ep = fragment['endpoints'][name]
    model = ep['model']
    return deepcopy({'models': {model: fragment['models'][model]},
                     'endpoints': {name: ep}})


def profile_choices(inventory: dict[str, Any], *, reserve_display_gpu=False,
                    kubeai=False, pool=None) -> list[dict[str, Any]]:
    """Rows for the registry browser; keep exact, generated catalog fragments."""
    from ..runtime_images import IMAGE_RECIPES

    pool = builtin_pool() if pool is None else pool
    frag = discovery_catalog(inventory, pool=pool, kubeai=kubeai,
                             reserve_display_gpu=reserve_display_gpu)
    normal = discovery_catalog(inventory, pool=pool, kubeai=kubeai,
                               reserve_display_gpu=reserve_display_gpu,
                               all_models=False)['endpoints']
    rows = []
    for name, ep in frag['endpoints'].items():
        model = pool[ep['model']]
        rt = ep.get('runtime') or {}
        place = ep.get('placement') or {}
        image = rt.get('image')
        rows.append({
            'name': name, 'source': model.hf_model_id,
            'tier': model.suggestion_tier if name in normal else 'optional',
            'modalities': ', '.join(model.modalities),
            'context': rt.get('max_model_len'),
            'gpu_count': (int(rt.get('tensor_parallel_size') or 1) *
                          int(rt.get('pipeline_parallel_size') or 1) *
                          int(rt.get('data_parallel_size') or 1)),
            'min_vram_gib': place.get('min_vram_gib'),
            'image': image,
            'managed_build': image in IMAGE_RECIPES,
            'fragment': endpoint_fragment(frag, name),
        })
    priority = {'frontier': 0, 'optional': 1, 'smoke': 2}
    return sorted(rows, key=lambda r: (priority[r['tier']], r['name']))
