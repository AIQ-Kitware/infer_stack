"""Discover and import individual serving profiles from the registry.

Automatic suggestions are hardware-filtered. Explicit discovery is not:
compatibility is advisory, and an operator may import a starting configuration
on another machine or edit it until it works. The normal controller still
owns placement, validation, image preparation and leases.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..hardware import available_gpu_indices
from .suggest import builtin_pool, suggest_catalog


def registry_endpoint_names(pool=None) -> set[str]:
    """All names, including optional and hardware-ineligible profiles."""
    pool = builtin_pool() if pool is None else pool
    return {name for model in pool.values() for name in (
        model.name, *(f'{model.name}-{v}' for v in model.endpoint_variants)
    )}


def discovery_catalog(inventory: dict[str, Any], *, pool=None,
                      reserve_display_gpu=False, kubeai=False,
                      all_models=True, include_unfitting=False) -> dict[str, Any]:
    """Render profiles, optionally including hardware/backend mismatches.

    The default is the existing suggestion filter. Explicit import/browse
    requests the full registry. KubeAI cannot deploy custom launchers; those
    profiles remain visible on demand but are annotated by ``profile_choices``.
    Local CUDA indices never have meaning on a cluster.
    """
    frag = suggest_catalog(inventory, pool=pool, all_models=all_models,
                           reserve_display_gpu=reserve_display_gpu,
                           include_unfitting=include_unfitting)
    if kubeai:
        from ..backends.kubeai import unsupported_runtime_fields

        endpoints = {}
        for name, ep in frag['endpoints'].items():
            if (not include_unfitting
                    and unsupported_runtime_fields(ep.get('runtime') or {})):
                continue
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
        raise ValueError(f'profile {name!r} does not fit the selected hardware/backend or is unknown')
    ep = fragment['endpoints'][name]
    model = ep['model']
    return deepcopy({'models': {model: fragment['models'][model]},
                     'endpoints': {name: ep}})


def _requirements(model, constraints, gpu_count, min_vram):
    """Explain the same model/class gates used by the suggestion compiler."""
    floor = max(float(min_vram or 0), float(model.min_vram_gib_per_replica or 0),
                float(constraints.get('min_vram_gib_per_replica') or 0))
    count = int(constraints.get('preferred_gpu_count') or model.preferred_gpu_count)
    text = f'{max(count, gpu_count)} GPU(s)'
    if floor:
        text += f', at least {floor:g} GiB each'
    ceiling = constraints.get('max_vram_gib_per_replica')
    if ceiling is not None:
        text += f', at most {float(ceiling):g} GiB each'
    low = constraints.get('min_compute_cap')
    if model.requires_ampere:
        low = max(float(low or 0), 8.0)
    high = constraints.get('max_compute_cap')
    if low is not None and high is not None and float(low) == float(high):
        text += f', CUDA compute {float(low):.1f}'
    else:
        if low is not None:
            text += f', CUDA compute >= {float(low):.1f}'
        if high is not None:
            text += f', CUDA compute <= {float(high):.1f}'
    for hints in (model.gpu_name_hints, constraints.get('gpu_name_hints')):
        if hints:
            text += f', GPU name contains one of {", ".join(hints)}'
    return text


def profile_choices(inventory: dict[str, Any], *, reserve_display_gpu=False,
                    kubeai=False, pool=None) -> list[dict[str, Any]]:
    """Every known profile, with its actual fragment and advisory fit status.

    The fitting path is identical to ordinary suggestions; only the fallback
    path renders defaults without host-specific adjustments. Do not synthesize
    fake inventory to trick a gated profile into matching.
    """
    from ..runtime_images import IMAGE_RECIPES

    pool = builtin_pool() if pool is None else pool
    frag = discovery_catalog(inventory, pool=pool, kubeai=kubeai,
                             reserve_display_gpu=reserve_display_gpu,
                             include_unfitting=True)
    fitting = suggest_catalog(inventory, pool=pool, all_models=True,
                              reserve_display_gpu=reserve_display_gpu)['endpoints']
    allowed = set(available_gpu_indices(inventory, reserve_display_gpu))
    gpus = [g for g in (inventory.get('gpus') or []) if g.get('index') in allowed]
    inventory_summary = '; '.join(
        f'GPU {g.get("index")}: {g.get("name") or "unknown"}, '
        f'{float(g.get("memory_gib") or 0):g} GiB, '
        f'compute {g.get("compute_cap") or "unknown"}' for g in gpus
    ) or 'no eligible GPUs detected'
    rows = []
    for name, ep in frag['endpoints'].items():
        model = pool[ep['model']]
        is_default = name == model.name
        variant = {} if is_default else model.endpoint_variants[name[len(model.name) + 1:]]
        constraints = model.default_hardware if is_default else variant
        rt = ep.get('runtime') or {}
        place = ep.get('placement') or {}
        image = rt.get('image')
        gpu_count = (int(rt.get('tensor_parallel_size') or 1) *
                     int(rt.get('pipeline_parallel_size') or 1) *
                     int(rt.get('data_parallel_size') or 1))
        requirements = _requirements(model, constraints, gpu_count, place.get('min_vram_gib'))
        hardware_fit = name in fitting
        unsupported = ()
        if kubeai:
            from ..backends.kubeai import unsupported_runtime_fields
            unsupported = unsupported_runtime_fields(rt)
        warnings = []
        if not hardware_fit:
            warnings.append(f'Outside suggested hardware: requires {requirements}. '
                            f'Available: {inventory_summary}. Import and edit before trying, '
                            'or try the unchanged recipe; normal placement checks still apply.')
        if unsupported:
            warnings.append('Current KubeAI backend cannot launch this recipe '
                            f'({", ".join(unsupported)}). Import is allowed; use Compose '
                            'or a supported cluster configuration to acquire.')
        rows.append({
            'name': name, 'source': model.hf_model_id,
            'tier': model.suggestion_tier if variant.get('suggest_by_default', True) else 'optional',
            'modalities': ', '.join(model.modalities),
            'context': rt.get('max_model_len'),
            'gpu_count': gpu_count,
            'min_vram_gib': place.get('min_vram_gib'),
            'image': image,
            'managed_build': image in IMAGE_RECIPES,
            'hardware_fit': hardware_fit,
            'backend_compatible': not unsupported,
            'requirements': requirements,
            'warnings': warnings,
            'note': model.profile_note if is_default else str(variant.get('profile_note') or ''),
            'fragment': endpoint_fragment(frag, name),
        })
    priority = {'frontier': 0, 'optional': 1, 'smoke': 2}
    return sorted(rows, key=lambda r: (not (r['hardware_fit'] and r['backend_compatible']),
                                      priority[r['tier']], r['name']))
